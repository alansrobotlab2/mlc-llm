"""Stage 1b: int4-dequant + fp16 matmul as two blocks at prim_func root.

Goal: see whether dlight's MatmulFP16Tensorization recognizes the matmul block
when its B operand comes from a dequant producer block in the same prim_func.

Three variants:
  V1 — global fp16 W intermediate:  dequant writes W_fp16[N,K] (alloc_buffer at
       global scope), matmul reads it. We expect dlight to tensorize the matmul
       and leave the dequant as a separate kernel (or a producer the recognizer
       handles via auto_inline).
  V2 — same as V1 but invoke `compute_inline` on the dequant block before
       handing to dlight. If dlight tensorizes the inlined version we get
       single-kernel dequant+MMA without a global temp.
  V3 — call FuseDequantizeMatmulEwise (Relax-level pass) — for reference,
       not used in this prototype.
"""
from __future__ import annotations
import numpy as np
import tvm
from tvm import s_tir
from tvm.s_tir import dlight as dl
from tvm.script import tirx as T


M, N, K = 4096, 1024, 2048
GROUP_SIZE = 32
NUM_ELEM_PER_STORAGE = 8
NUM_STORAGE = K // NUM_ELEM_PER_STORAGE
NUM_GROUP = K // GROUP_SIZE
MAX_INT = 7


@T.prim_func(private=True)
def dequant_matmul_two_block(
    X: T.Buffer((M, K), "float16"),
    W_q: T.Buffer((N, NUM_STORAGE), "uint32"),
    Scale: T.Buffer((N, NUM_GROUP), "float16"),
    O: T.Buffer((M, N), "float16"),
):
    T.func_attr({"tirx.noalias": True})
    W_fp16 = T.alloc_buffer((N, K), "float16")
    for i, j in T.grid(N, K):
        with T.sblock("dequant"):
            vi, vj = T.axis.remap("SS", [i, j])
            shift = T.Cast("uint32", (vj % NUM_ELEM_PER_STORAGE) * 4)
            w_int = T.Cast(
                "float16",
                T.bitwise_and(T.shift_right(W_q[vi, vj // NUM_ELEM_PER_STORAGE], shift),
                              T.uint32(15)),
            )
            W_fp16[vi, vj] = (w_int - T.float16(MAX_INT)) * Scale[vi, vj // GROUP_SIZE]
    for i, j, k in T.grid(M, N, K):
        with T.sblock("compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                O[vi, vj] = T.float16(0.0)
            O[vi, vj] = O[vi, vj] + X[vi, vk] * W_fp16[vj, vk]


def numpy_reference(X_np, W_int4, scale_np):
    scale_full = np.repeat(scale_np, GROUP_SIZE, axis=1)
    W_fp16 = (W_int4.astype(np.float32) - MAX_INT) * scale_full.astype(np.float32)
    O = X_np.astype(np.float32) @ W_fp16.T
    return O.astype(np.float16)


def main():
    target = tvm.target.Target({
        "kind": "cuda", "arch": "sm_87",
        "max_threads_per_block": 1024,
        "max_shared_memory_per_block": 49152,
        "thread_warp_size": 32,
    })
    dev = tvm.cuda(0)

    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
    W_int4 = rng.integers(0, 16, size=(N, K), dtype=np.int8)
    W_packed = np.zeros((N, NUM_STORAGE), dtype=np.uint32)
    for s_idx in range(NUM_ELEM_PER_STORAGE):
        W_packed |= (W_int4[:, s_idx::NUM_ELEM_PER_STORAGE].astype(np.uint32) << (s_idx * 4))
    scale_np = (rng.standard_normal((N, NUM_GROUP), dtype=np.float32).astype(np.float16) * 0.01)
    O_ref = numpy_reference(X_np, W_int4, scale_np)
    print(f"shape M={M} N={N} K={K}, ref O range [{O_ref.min():.3f}, {O_ref.max():.3f}]")

    # ============================================================
    # V1: dlight default (no manual inline) — does it auto-inline?
    # ============================================================
    print("\n=== V1: two blocks at root, no manual inline, dlight default ===")
    f = dequant_matmul_two_block.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    with target:
        mod_v1 = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Reduction(),
                                          dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
    body = str(mod_v1["main"])
    has_wmma = "wmma" in body or "mma_sync" in body
    has_dequant_block = "dequant" in body
    has_W_fp16_alloc = "W_fp16" in body
    print(f"  has wmma: {has_wmma}")
    print(f"  has 'dequant' sblock string: {has_dequant_block}")
    print(f"  has W_fp16 buffer: {has_W_fp16_alloc}")
    with open("/tmp/phase9b_v1_body.txt", "w") as fh:
        fh.write(body)
    print(f"  body written /tmp/phase9b_v1_body.txt ({len(body)} chars)")
    try:
        with target:
            rt = tvm.tirx.build(mod_v1["main"], target=target)
        X_dev = tvm.runtime.tensor(X_np, dev)
        W_dev = tvm.runtime.tensor(W_packed, dev)
        S_dev = tvm.runtime.tensor(scale_np, dev)
        O_dev = tvm.runtime.tensor(np.zeros((M, N), dtype=np.float16), dev)
        for _ in range(3):
            rt(X_dev, W_dev, S_dev, O_dev)
        dev.sync()
        import time
        t0 = time.perf_counter()
        n_iter = 20
        for _ in range(n_iter):
            rt(X_dev, W_dev, S_dev, O_dev)
        dev.sync()
        t_ms = (time.perf_counter() - t0) / n_iter * 1e3
        O_out = O_dev.numpy()
        max_diff = float(np.abs(O_out.astype(np.float32) - O_ref.astype(np.float32)).max())
        rel = max_diff / max(float(np.abs(O_ref.astype(np.float32)).max()), 1e-6)
        flops = 2 * M * N * K
        tflops = flops / (t_ms * 1e-3) / 1e12
        print(f"  ✓ build OK; t={t_ms:.3f} ms, {tflops:.2f} TFLOPS")
        print(f"  parity: max diff {max_diff:.4f}, rel {rel:.2%} {'PASS' if rel < 0.05 else 'FAIL'}")
    except Exception as e:
        print(f"  ✗ build/run FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:200]}")

    # ============================================================
    # V1.5: V1 + manual schedule of leftover dequant block
    # ============================================================
    print("\n=== V1.5: V1 + manual thread-bind on the leftover dequant block ===")
    f = dequant_matmul_two_block.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    with target:
        mod_v1 = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
    sch = s_tir.Schedule(mod_v1)
    try:
        deq_blk = sch.get_sblock("dequant")
        i, j = sch.get_loops(deq_blk)
        fused = sch.fuse(i, j)
        bx, tx = sch.split(fused, factors=[None, 256])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        mod_v15 = sch.mod
        print("  manual schedule OK")
    except Exception as e:
        print(f"  manual schedule FAILED: {e}")
        mod_v15 = None
    if mod_v15 is not None:
        try:
            with target:
                rt = tvm.tirx.build(mod_v15["main"], target=target)
            X_dev = tvm.runtime.tensor(X_np, dev)
            W_dev = tvm.runtime.tensor(W_packed, dev)
            S_dev = tvm.runtime.tensor(scale_np, dev)
            O_dev = tvm.runtime.tensor(np.zeros((M, N), dtype=np.float16), dev)
            for _ in range(3):
                rt(X_dev, W_dev, S_dev, O_dev)
            dev.sync()
            import time
            t0 = time.perf_counter()
            n_iter = 20
            for _ in range(n_iter):
                rt(X_dev, W_dev, S_dev, O_dev)
            dev.sync()
            t_ms = (time.perf_counter() - t0) / n_iter * 1e3
            O_out = O_dev.numpy()
            max_diff = float(np.abs(O_out.astype(np.float32) - O_ref.astype(np.float32)).max())
            rel = max_diff / max(float(np.abs(O_ref.astype(np.float32)).max()), 1e-6)
            flops = 2 * M * N * K
            tflops = flops / (t_ms * 1e-3) / 1e12
            print(f"  ✓ build OK; t={t_ms:.3f} ms, {tflops:.2f} TFLOPS")
            print(f"  parity: max diff {max_diff:.4f}, rel {rel:.2%} {'PASS' if rel < 0.05 else 'FAIL'}")
        except Exception as e:
            print(f"  ✗ build/run FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:200]}")

    # ============================================================
    # V3: inline dequant directly in matmul (no W_fp16 temp)
    # ============================================================
    print("\n=== V3: inline dequant inside matmul block (no W_fp16 temp) ===")

    @T.prim_func(private=True)
    def inline_dequant_matmul(
        X: T.Buffer((M, K), "float16"),
        W_q: T.Buffer((N, NUM_STORAGE), "uint32"),
        Scale: T.Buffer((N, NUM_GROUP), "float16"),
        O: T.Buffer((M, N), "float16"),
    ):
        T.func_attr({"tirx.noalias": True})
        for i, j, k in T.grid(M, N, K):
            with T.sblock("compute"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    O[vi, vj] = T.float16(0.0)
                shift = T.Cast("uint32", (vk % NUM_ELEM_PER_STORAGE) * 4)
                w_int = T.Cast(
                    "float16",
                    T.bitwise_and(T.shift_right(W_q[vj, vk // NUM_ELEM_PER_STORAGE], shift),
                                  T.uint32(15)),
                )
                w_fp16 = (w_int - T.float16(MAX_INT)) * Scale[vj, vk // GROUP_SIZE]
                O[vi, vj] = O[vi, vj] + X[vi, vk] * w_fp16

    f = inline_dequant_matmul.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    with target:
        mod_v3 = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Reduction(),
                                          dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
    body = str(mod_v3["main"])
    has_wmma = "wmma" in body or "mma_sync" in body
    print(f"  has wmma: {has_wmma}")
    with open("/tmp/phase9b_v3_body.txt", "w") as fh:
        fh.write(body)
    print(f"  body written /tmp/phase9b_v3_body.txt ({len(body)} chars)")
    try:
        with target:
            rt = tvm.tirx.build(mod_v3["main"], target=target)
        X_dev = tvm.runtime.tensor(X_np, dev)
        W_dev = tvm.runtime.tensor(W_packed, dev)
        S_dev = tvm.runtime.tensor(scale_np, dev)
        O_dev = tvm.runtime.tensor(np.zeros((M, N), dtype=np.float16), dev)
        for _ in range(3):
            rt(X_dev, W_dev, S_dev, O_dev)
        dev.sync()
        import time
        t0 = time.perf_counter()
        n_iter = 20
        for _ in range(n_iter):
            rt(X_dev, W_dev, S_dev, O_dev)
        dev.sync()
        t_ms = (time.perf_counter() - t0) / n_iter * 1e3
        O_out = O_dev.numpy()
        max_diff = float(np.abs(O_out.astype(np.float32) - O_ref.astype(np.float32)).max())
        rel = max_diff / max(float(np.abs(O_ref.astype(np.float32)).max()), 1e-6)
        flops = 2 * M * N * K
        tflops = flops / (t_ms * 1e-3) / 1e12
        print(f"  ✓ build OK; t={t_ms:.3f} ms, {tflops:.2f} TFLOPS")
        print(f"  parity: max diff {max_diff:.4f}, rel {rel:.2%} {'PASS' if rel < 0.05 else 'FAIL'}")
    except Exception as e:
        print(f"  ✗ build/run FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:200]}")

    # ============================================================
    # V2: manual compute_inline on dequant block before dlight
    # ============================================================
    print("\n=== V2: schedule.compute_inline(dequant) before dlight ===")
    f = dequant_matmul_two_block.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    sch = s_tir.Schedule(mod)
    try:
        deq_blk = sch.get_sblock("dequant")
        sch.compute_inline(deq_blk)
        print("  compute_inline(dequant): OK")
        mod = sch.mod
    except Exception as e:
        print(f"  compute_inline FAILED: {e}")
        return
    with target:
        mod_v2 = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Reduction(),
                                          dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
    body = str(mod_v2["main"])
    has_wmma = "wmma" in body or "mma_sync" in body
    has_W_fp16 = "W_fp16" in body
    print(f"  has wmma: {has_wmma}; has W_fp16 alloc: {has_W_fp16}")
    with open("/tmp/phase9b_v2_body.txt", "w") as fh:
        fh.write(body)
    print(f"  body written /tmp/phase9b_v2_body.txt ({len(body)} chars)")
    try:
        with target:
            rt = tvm.tirx.build(mod_v2["main"], target=target)
        X_dev = tvm.runtime.tensor(X_np, dev)
        W_dev = tvm.runtime.tensor(W_packed, dev)
        S_dev = tvm.runtime.tensor(scale_np, dev)
        O_dev = tvm.runtime.tensor(np.zeros((M, N), dtype=np.float16), dev)
        for _ in range(3):
            rt(X_dev, W_dev, S_dev, O_dev)
        dev.sync()
        import time
        t0 = time.perf_counter()
        n_iter = 20
        for _ in range(n_iter):
            rt(X_dev, W_dev, S_dev, O_dev)
        dev.sync()
        t_ms = (time.perf_counter() - t0) / n_iter * 1e3
        O_out = O_dev.numpy()
        max_diff = float(np.abs(O_out.astype(np.float32) - O_ref.astype(np.float32)).max())
        rel = max_diff / max(float(np.abs(O_ref.astype(np.float32)).max()), 1e-6)
        flops = 2 * M * N * K
        tflops = flops / (t_ms * 1e-3) / 1e12
        print(f"  ✓ build OK; t={t_ms:.3f} ms, {tflops:.2f} TFLOPS")
        print(f"  parity: max diff {max_diff:.4f}, rel {rel:.2%} {'PASS' if rel < 0.05 else 'FAIL'}")
    except Exception as e:
        print(f"  ✗ build/run FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:200]}")


if __name__ == "__main__":
    main()

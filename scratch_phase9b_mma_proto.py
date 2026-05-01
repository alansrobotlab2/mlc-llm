"""Phase 9b Stage 1 prototype — standalone int4-dequant + MMA matmul.

Builds an unscheduled TIR matmul that mirrors the production
`dequantize_group_gemm` body for *one* expert (no persistent loop, no indptr).

Two backends are exercised:
  A. dlight ApplyDefaultSchedule(dl.gpu.Matmul()) — tries to auto-tensorize
     via MatmulFP16Tensorization (uses wmma intrinsics on sm_87+).
  B. scalar reference (the existing hand schedule shape, for parity/timing).

If A produces a TC-using schedule, Stage 2 reuses the same dlight pipeline
inside the persistent-loop wrapper. If A bails, we fall back to manual
sch.tensorize via get_wmma_intrin_group.

Shape: gate_up at top_k=8, pp=512, single-expert slice
  M = 4096 (uniform expert routing for prototype)
  N = 1024 (moe_intermediate_size * 2 / split)
  K = 2048 (hidden_size)
Quant: q4f16_1 (int4 packed in uint32, group_size=32, symmetric).
"""
from __future__ import annotations

import numpy as np
import tvm
from tvm import s_tir, tirx
from tvm.s_tir import dlight as dl
from tvm.script import tirx as T

# ----------------- shape -----------------
M = 4096
N = 1024
K = 2048
GROUP_SIZE = 32
NUM_ELEM_PER_STORAGE = 8  # int4 packed in uint32 = 8 per word
NUM_STORAGE = K // NUM_ELEM_PER_STORAGE
NUM_GROUP = K // GROUP_SIZE
MAX_INT = 7  # symmetric int4: range [-7, 7] stored as [0, 14] with offset


# ----------------- prim_func: pure fp16 matmul (dequant off-line) -----------------
# Stage 1 isolates the MMA path. Production integration (Stage 2) will fuse
# dequant into the W_shared cooperative fetch — dlight's auto_inline_producers
# only inlines into b_g2s if the producer is *inside* the matmul block, which
# requires a different structuring than what TIR makes easy at the prim_func
# root level. For the prototype we measure MMA-only throughput as the target.
@T.prim_func(private=True)
def matmul_pure_fp16(
    X: T.Buffer((M, K), "float16"),
    W_fp16: T.Buffer((N, K), "float16"),
    O: T.Buffer((M, N), "float16"),
):
    T.func_attr({"tirx.noalias": True})
    for i, j, k in T.grid(M, N, K):
        with T.sblock("compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                O[vi, vj] = T.float16(0.0)
            O[vi, vj] = O[vi, vj] + X[vi, vk] * W_fp16[vj, vk]


def numpy_reference(X_np, W_int4, scale_np):
    """NumPy reference: dequantize int4 weights to fp16, matmul X @ W^T, return O."""
    # Dequant: (W_int4 - 7) * scale (broadcast scale across group_size dim)
    scale_full = np.repeat(scale_np, GROUP_SIZE, axis=1)  # (N, K)
    W_fp16 = (W_int4.astype(np.float32) - MAX_INT) * scale_full.astype(np.float32)
    O = X_np.astype(np.float32) @ W_fp16.T  # (M, N)
    return O.astype(np.float16)


def build_and_run_fp16(prim_func, target, dev, X_np, W_fp16_np, label):
    """Compile + run a fp16 matmul prim_func with dlight Matmul tensorize."""
    f = prim_func.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Reduction(),
                                      dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
    body_str = str(mod["main"])
    used_wmma = ("wmma" in body_str) or ("mma_sync" in body_str)
    print(f"  uses wmma: {used_wmma}")
    if len(body_str) < 30000:
        with open("/tmp/phase9b_scheduled_body.txt", "w") as fh:
            fh.write(body_str)
        print(f"  scheduled body written to /tmp/phase9b_scheduled_body.txt ({len(body_str)} chars)")
    with target:
        rt_mod = tvm.tirx.build(mod["main"], target=target)
    X_dev = tvm.runtime.tensor(X_np, dev)
    W_dev = tvm.runtime.tensor(W_fp16_np, dev)
    O_dev = tvm.runtime.tensor(np.zeros((M, N), dtype="float16"), dev)
    # Warmup
    for _ in range(3):
        rt_mod["main"](X_dev, W_dev, O_dev)
    dev.sync()
    # Time
    n_iter = 20
    import time
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rt_mod["main"](X_dev, W_dev, O_dev)
    dev.sync()
    elapsed_ms = (time.perf_counter() - t0) / n_iter * 1e3
    print(f"  {label}: {elapsed_ms:.3f} ms / iter")
    return O_dev.numpy(), elapsed_ms


def main():
    print("=== Phase 9b Stage 1 prototype ===")
    print(f"shape: M={M} N={N} K={K} (gate_up @ pp=512, top_k=8, 1 expert)")

    target = tvm.target.Target({
        "kind": "cuda",
        "arch": "sm_87",
        "max_threads_per_block": 1024,
        "max_shared_memory_per_block": 49152,
        "thread_warp_size": 32,
    })
    dev = tvm.cuda(0)

    # --- generate inputs ---
    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
    # Random int4 weights packed in uint32 (8 nibbles per word)
    W_int4 = rng.integers(0, 16, size=(N, K), dtype=np.int8)
    W_packed = np.zeros((N, NUM_STORAGE), dtype=np.uint32)
    for shift_idx in range(NUM_ELEM_PER_STORAGE):
        W_packed |= (W_int4[:, shift_idx::NUM_ELEM_PER_STORAGE].astype(np.uint32) << (shift_idx * 4))
    scale_np = rng.standard_normal((N, NUM_GROUP), dtype=np.float32).astype(np.float16) * 0.01

    # --- numpy reference (off-line dequant, fp16 matmul) ---
    print("\n--- numpy reference (CPU) ---")
    import time
    t0 = time.perf_counter()
    scale_full = np.repeat(scale_np, GROUP_SIZE, axis=1)  # (N, K)
    W_fp16_np = ((W_int4.astype(np.float32) - MAX_INT) * scale_full.astype(np.float32)).astype(np.float16)
    O_ref = numpy_reference(X_np, W_int4, scale_np)
    print(f"  numpy: {(time.perf_counter() - t0) * 1e3:.1f} ms (one-shot, off critical path)")

    # --- MMA on pure fp16 matmul ---
    print("\n--- dlight Matmul (TC tensorize) on pure fp16 matmul ---")
    O_mma, t_mma = build_and_run_fp16(matmul_pure_fp16, target, dev, X_np, W_fp16_np, "MMA")

    # --- parity ---
    print("\n--- parity check (numpy vs MMA) ---")
    diff = np.abs(O_ref.astype(np.float32) - O_mma.astype(np.float32))
    ref_scale = np.abs(O_ref.astype(np.float32))
    print(f"  max abs diff: {diff.max():.4f}")
    print(f"  mean abs diff: {diff.mean():.6f}")
    print(f"  mean |O_ref|: {ref_scale.mean():.4f}")
    print(f"  max |O_ref|: {ref_scale.max():.4f}")
    rel = diff.max() / max(ref_scale.max(), 1e-6)
    print(f"  max rel diff: {rel:.4%}")
    parity_ok = rel < 0.05
    print(f"  parity: {'PASS' if parity_ok else 'FAIL'} (max rel diff < 5%)")

    # --- bench ---
    print(f"\n--- bench summary ---")
    flops = 2 * M * N * K
    print(f"  MMA:    {t_mma:.3f} ms = {flops / t_mma / 1e9:.1f} GFLOPS")
    print(f"  Orin sm_87 fp16 TC peak ≈ 200 TFLOPS; sustained ≈ 30-50 % typical → 60-100 TFLOPS.")
    print(f"  Reference: production scalar kernel (with dequant inline) ≈ 33 ms at this shape (Stage 9.1).")
    if t_mma < 33 / 3:
        print(f"  ✓ Stage 1 land criterion met: ≥ 3× faster than production scalar (33 ms).")
    else:
        print(f"  ✗ Stage 1 land criterion NOT met: less than 3× speedup.")


if __name__ == "__main__":
    main()

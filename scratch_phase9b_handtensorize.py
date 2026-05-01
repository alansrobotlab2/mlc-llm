"""Stage 2a — hand-tensorize dequant+gemm using wmma intrinsics.

Mirrors production `dequantize_group_gemm` body (X_tile/W_tile/O_tile shared+local
allocation, fp16 dequant feeding into shared, scalar matmul block) but:

  - drops the persistent-loop CTA wrapper (single-expert, Ne=1)
  - hand-applies the wmma cache_read/cache_write/tensorize pattern from
    dlight's MatmulFP16Tensorization

Goal: prove hand-tensorize works on the production-shaped schedule. Once this
lands, Stage 2c adds expert dispatch (indptr scan) and Stage 2d ports into
moe_matmul.py.
"""
from __future__ import annotations
import time
import numpy as np
import tvm
from tvm import s_tir, tirx as tirx
from tvm.script import tirx as T

# --- shape ---
M, N, K = 4096, 1024, 2048      # gate_up @ pp=512, top_k=8, single expert
GROUP_SIZE = 32
NUM_ELEM_PER_STORAGE = 8
NUM_STORAGE = K // NUM_ELEM_PER_STORAGE
NUM_GROUP = K // GROUP_SIZE
MAX_INT = 7

# --- tile config (must satisfy MMA m16n16k16) ---
BLK_M, BLK_N, BLK_K = 16, 128, 32
MICRO = 16  # MMA tile dim

assert BLK_M % MICRO == 0
assert BLK_N % MICRO == 0
assert BLK_K % MICRO == 0


@T.prim_func(private=True)
def dequant_gemm_prod_shape(
    X: T.Buffer((M, K), "float16"),
    W_q: T.Buffer((N, NUM_STORAGE), "uint32"),
    Scale: T.Buffer((N, NUM_GROUP), "float16"),
    O: T.Buffer((M, N), "float16"),
):
    """Mirrors moe_matmul._func with Ne=1 inlined and persistent loop dropped.

    Grid layout: blockIdx.x = (m_tile_idx * tiles_per_row + n_tile_idx). Each
    block computes a BLK_M×BLK_N tile of O. Assumes M % BLK_M == 0 and
    N % BLK_N == 0 (no bounds check); the production kernel will reinstate
    them once the wmma path lands.

    Compute writes directly to O — cache_write in the schedule inserts the
    wmma.accumulator + shared.dyn intermediates following dlight's pattern.
    """
    T.func_attr({"tirx.noalias": True})
    tiles_per_m = M // BLK_M
    tiles_per_n = N // BLK_N
    total_tiles = tiles_per_m * tiles_per_n

    for _bx in T.thread_binding(total_tiles, thread="blockIdx.x"):
        with T.sblock("CTA"):
            bx = T.axis.spatial(total_tiles, _bx)
            m_offset = T.floordiv(bx, tiles_per_n) * BLK_M
            n_offset = T.floormod(bx, tiles_per_n) * BLK_N
            T.reads(X[:, :], W_q[:, :], Scale[:, :])
            T.writes(O[:, :])
            X_tile = T.sblock_alloc_buffer((BLK_M, K), "float16", scope="shared.dyn")
            W_tile = T.sblock_alloc_buffer((BLK_N, K), "float16", scope="shared.dyn")

            for a0, a1 in T.grid(BLK_M, K):
                with T.sblock("X_shared"):
                    i, j = T.axis.remap("SS", [a0, a1])
                    X_tile[i, j] = X[m_offset + i, j]

            for a0, a1 in T.grid(BLK_N, K):
                with T.sblock("W_shared"):
                    i, j = T.axis.remap("SS", [a0, a1])
                    shift = T.Cast("uint32", (j % NUM_ELEM_PER_STORAGE) * 4)
                    w_int = T.Cast(
                        "float16",
                        T.bitwise_and(T.shift_right(W_q[n_offset + i, j // NUM_ELEM_PER_STORAGE], shift),
                                      T.uint32(15)),
                    )
                    W_tile[i, j] = (w_int - T.float16(MAX_INT)) * Scale[n_offset + i, j // GROUP_SIZE]

            for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                with T.sblock("compute"):
                    i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                    with T.init():
                        O[m_offset + i, n_offset + j] = T.float16(0.0)
                    O[m_offset + i, n_offset + j] = (
                        O[m_offset + i, n_offset + j] + X_tile[i, k] * W_tile[j, k]
                    )


def schedule_hand_tensorize(prim_func, target):
    """Manual wmma tensorization mirroring dlight's MatmulFP16Tensorization.

    Order matches dlight's apply():
      1. split (i, j, k) into MMA-shape-aligned (outer, inner) pairs
      2. blockize the inner micro-block
      3. split outer (i, j) for thread/warp distribution; bind blockIdx.* if needed
      4. cooperative fetch of X_tile, W_tile (compute_at(k0))
      5. cache_read("wmma.matrix_a") + compute_at(k1)
      6. cache_read("wmma.matrix_b") + compute_at(k1)
      7. cache_write("wmma.accumulator") + reverse_compute_at(thread_idy)
      8. cache_write("shared.dyn") for accumulator → shared store
      9. decompose_reduction → separate init from k-iteration
     10. tensorize all 5 (load_a, load_b, init, compute, store)
    """
    from tvm.s_tir.tensor_intrin.cuda import get_wmma_intrin_group

    f = prim_func.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    sch = s_tir.Schedule(mod)

    # threading config: BLK_N/MICRO = 8 n-tiles. With TY=8 each warp owns
    # exactly one n-tile, so j_o_in=1 and the cooperative fetch cleanly sits
    # at k_o_o granularity (no warp-level race in the W_tile fetch).
    TY = BLK_N // MICRO  # 8 warps
    WARP = 32
    VEC = 4

    # ---- 1. split inner-MMA loops ----
    main_block = sch.get_sblock("compute")
    i, j, k = sch.get_loops(main_block)
    i_o, i_i = sch.split(i, factors=[None, MICRO])  # BLK_M/16, 16
    j_o, j_i = sch.split(j, factors=[None, MICRO])  # BLK_N/16, 16
    k_o, k_i = sch.split(k, factors=[None, MICRO])  # K/16, 16
    sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)

    # ---- 2. blockize the (i_i, j_i, k_i) micro-block ----
    block_inner = main_block
    block_outer = sch.blockize(i_i)

    # ---- 3. split outer for warp distribution ----
    # Bind j_o (size 8) directly to threadIdx.y. Split k_o into outer × inner.
    k_o_o, k_o_i = sch.split(k_o, factors=[None, BLK_K // MICRO])  # K/BLK_K × BLK_K/16
    sch.reorder(i_o, j_o, k_o_o, k_o_i)
    sch.bind(j_o, "threadIdx.y")
    j_o_ty = j_o

    # software-pipeline annotations skipped — dlight's [0,0,0,0,0,1,1] (7 stages)
    # expects 7 sub-statements at injection time, which requires
    # manifest_shared_memory_local_stage to expand the coop-fetch blocks. Our
    # dequant in W_shared breaks that constraint. Without the pipeline we leave
    # ~30-50% perf on the table; revisit by re-structuring W_shared as
    # dequant-producer + simple-shared-copy two-block sequence.

    # ---- 4. cooperative fetch X_tile, W_tile ----
    # both are at k_o_o granularity (one ko-iter loads BLK_M*BLK_K and BLK_N*BLK_K)
    x_shared = sch.get_sblock("X_shared")
    w_shared = sch.get_sblock("W_shared")

    def _coop(blk):
        sch.compute_at(blk, k_o_o, preserve_unit_loops=True)
        loops = sch.get_loops(blk)[-2:]
        fused = sch.fuse(*loops)
        _, fy, fx, fv = sch.split(fused, factors=[None, TY, WARP, VEC])
        sch.bind(fy, "threadIdx.y")
        sch.bind(fx, "threadIdx.x")
        sch.vectorize(fv)
        # storage_align reduces shared-mem bank conflicts on the fragment loads
        sch.storage_align(blk, 0, axis=-2, factor=16, offset=8)
        # dlight also adds tirx.manifest_shared_memory_local_stage + double_buffer_scope,
        # but those require the block body to be a simple BufferStore — the dequant
        # in W_shared breaks the constraint. Skip both; bench impact TBD.

    _coop(x_shared)
    _coop(w_shared)

    # ---- 5,6. cache_read into wmma fragments, compute_at(k_o_i) ----
    A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
    B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
    sch.compute_at(A_mat, k_o_i)
    sch.compute_at(B_mat, k_o_i)

    # ---- 7,8. cache_write accumulator → shared → global ----
    acc_s2g = sch.cache_write(block_outer, 0, "shared.dyn")
    # storage_align skipped — would target global O, not the shared.dyn cache;
    # without it we may have shared-mem bank conflicts but correctness stands.
    store = sch.cache_write(block_outer, 0, "wmma.accumulator")
    sch.reverse_compute_at(store, j_o_ty)
    sch.reverse_compute_at(acc_s2g, j_o_ty)

    # split store loop into 16-tiles for tensorize
    si, sj = sch.get_loops(store)[-2:]
    si0, si1 = sch.split(si, factors=[None, MICRO])
    sj0, sj1 = sch.split(sj, factors=[None, MICRO])
    sch.reorder(si0, sj0, si1, sj1)

    # acc_s2g cooperative fetch back to global
    asi, asj = sch.get_loops(acc_s2g)[-2:]
    afused = sch.fuse(asi, asj)
    _, afx, afv = sch.split(afused, factors=[None, WARP, VEC])
    sch.bind(afx, "threadIdx.x")
    sch.vectorize(afv)

    # ---- 9. decompose_reduction ----
    block_init_c = sch.decompose_reduction(block_outer, k_o_o)
    block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

    # ---- 10. tensorize ----
    # Use fp16 accumulator (f16f16f16 wmma) — matches our prim_func's O dtype.
    # The fp32-accumulator variant (out_dtype="float32") would require either
    # an explicit fp32 intermediate buffer in the prim_func or upcasting the
    # compute write target. fp16 accumulation costs precision but matches the
    # production schedule's fp16 storage pattern.
    intrin_group = get_wmma_intrin_group(
        load_scope="shared.dyn",
        store_scope="shared.dyn",
        in_dtype="float16",
        out_dtype="float16",
        trans_b=True,
    )

    # load_a: split (i, j) by 16, tensorize inner-i
    ai, aj = sch.get_loops(A_mat)[-2:]
    ai0, ai1 = sch.split(ai, factors=[None, MICRO])
    aj0, aj1 = sch.split(aj, factors=[None, MICRO])
    sch.reorder(ai0, aj0, ai1, aj1)
    sch.unroll(ai0)
    sch.unroll(aj0)
    sch.tensorize(ai1, intrin_group["load_a"])

    # load_b
    bi, bj = sch.get_loops(B_mat)[-2:]
    bi0, bi1 = sch.split(bi, factors=[None, MICRO])
    bj0, bj1 = sch.split(bj, factors=[None, MICRO])
    sch.reorder(bi0, bj0, bi1, bj1)
    sch.unroll(bi0)
    sch.unroll(bj0)
    sch.tensorize(bi1, intrin_group["load_b"])

    # init, compute, store
    sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
    sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
    sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

    return sch.mod


def main():
    target = tvm.target.Target({
        "kind": "cuda", "arch": "sm_87",
        "max_threads_per_block": 1024,
        "max_shared_memory_per_block": 49152,
        "thread_warp_size": 32,
    })
    dev = tvm.cuda(0)

    # inputs
    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
    W_int4 = rng.integers(0, 16, size=(N, K), dtype=np.int8)
    W_packed = np.zeros((N, NUM_STORAGE), dtype=np.uint32)
    for s_idx in range(NUM_ELEM_PER_STORAGE):
        W_packed |= (W_int4[:, s_idx::NUM_ELEM_PER_STORAGE].astype(np.uint32) << (s_idx * 4))
    scale_np = (rng.standard_normal((N, NUM_GROUP), dtype=np.float32).astype(np.float16) * 0.01)
    scale_full = np.repeat(scale_np, GROUP_SIZE, axis=1)
    W_fp16 = ((W_int4.astype(np.float32) - MAX_INT) * scale_full.astype(np.float32)).astype(np.float16)
    O_ref = (X_np.astype(np.float32) @ W_fp16.T.astype(np.float32)).astype(np.float16)

    print(f"shape M={M} N={N} K={K} BLK={BLK_M}x{BLK_N}x{BLK_K}")
    print(f"O_ref range [{O_ref.min():.3f}, {O_ref.max():.3f}]")

    print("\n=== schedule_hand_tensorize ===")
    try:
        mod = schedule_hand_tensorize(dequant_gemm_prod_shape, target)
        body = str(mod["main"])
        has_wmma = "wmma" in body or "mma_sync" in body
        print(f"  has wmma: {has_wmma}")
        with open("/tmp/phase9b_handtensorize.txt", "w") as fh:
            fh.write(body)
        print(f"  body written /tmp/phase9b_handtensorize.txt ({len(body)} chars)")
    except Exception as e:
        print(f"  schedule FAILED: {type(e).__name__}: {str(e)[:300]}")
        import traceback; traceback.print_exc()
        return

    try:
        with target:
            rt = tvm.tirx.build(mod["main"], target=target)
    except Exception as e:
        print(f"  build FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:200]}")
        return

    X_dev = tvm.runtime.tensor(X_np, dev)
    W_dev = tvm.runtime.tensor(W_packed, dev)
    S_dev = tvm.runtime.tensor(scale_np, dev)
    O_dev = tvm.runtime.tensor(np.zeros((M, N), dtype=np.float16), dev)
    for _ in range(3):
        rt(X_dev, W_dev, S_dev, O_dev)
    dev.sync()
    n_iter = 20
    t0 = time.perf_counter()
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


if __name__ == "__main__":
    main()

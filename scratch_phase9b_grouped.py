"""Stage 2c — multi-expert grouped GEMM with hand-tensorize wmma.

Extends Stage 2a's single-expert prototype to handle multiple experts via
indptr scan + grid launch. Each block:
  1. derives its tile_id from blockIdx.x
  2. scans indptr to find (expert_e, m_offset, n_offset)
  3. early-exits if tile_id is past the actual total
  4. runs the hand-tensorized wmma matmul

Persistent loop is dropped (one tile per CTA). The grid is sized to an
upper bound; idle blocks just early-exit.
"""
from __future__ import annotations
import time
import numpy as np
import tvm
from tvm import s_tir
from tvm.script import tirx as T

# --- shape: 4 experts of varying sizes for testing ---
NE = 4
N, K = 1024, 2048
GROUP_SIZE = 32
NUM_ELEM_PER_STORAGE = 8
NUM_STORAGE = K // NUM_ELEM_PER_STORAGE
NUM_GROUP = K // GROUP_SIZE
MAX_INT = 7

BLK_M, BLK_N, BLK_K = 16, 128, 32
MICRO = 16
TILES_PER_N = N // BLK_N

# --- expert routing: variable tokens per expert ---
TOKENS_PER_E = [256, 512, 128, 1024]  # B = sum = 1920
B_TOTAL = sum(TOKENS_PER_E)
# pad to BLK_M boundary for clean tile counts
TILES_PER_E = [(t + BLK_M - 1) // BLK_M for t in TOKENS_PER_E]
TOTAL_TILES = sum(TILES_PER_E) * TILES_PER_N

# upper bound used for grid launch. We use the actual total here since it's
# precomputable from indptr at compile time when sizes are static. For dynamic
# B, the upper bound is `(ceildiv(B, BLK_M) + Ne) * tiles_per_n`.
GRID_SIZE = TOTAL_TILES


def make_prim_func():
    @T.prim_func(private=True)
    def grouped_gemm(
        X: T.Buffer((B_TOTAL, K), "float16"),
        W_q: T.Buffer((NE, N, NUM_STORAGE), "uint32"),
        Scale: T.Buffer((NE, N, NUM_GROUP), "float16"),
        # tile_to_e[bx] = expert index, tile_to_m[bx] = m_offset (in X rows),
        # tile_to_n[bx] = n_offset. Precomputed host-side from indptr.
        tile_to_e: T.Buffer((GRID_SIZE,), "int32"),
        tile_to_m: T.Buffer((GRID_SIZE,), "int32"),
        tile_to_n: T.Buffer((GRID_SIZE,), "int32"),
        O: T.Buffer((B_TOTAL, N), "float16"),
    ):
        T.func_attr({"tirx.noalias": True})

        for _bx in T.thread_binding(GRID_SIZE, thread="blockIdx.x"):
            with T.sblock("CTA"):
                bx = T.axis.spatial(GRID_SIZE, _bx)
                T.reads(X[:, :], W_q[:, :, :], Scale[:, :, :],
                        tile_to_e[:], tile_to_m[:], tile_to_n[:])
                T.writes(O[:, :])

                X_tile = T.sblock_alloc_buffer((BLK_M, K), "float16", scope="shared.dyn")
                W_tile = T.sblock_alloc_buffer((BLK_N, K), "float16", scope="shared.dyn")

                e = tile_to_e[bx]
                m_offset = tile_to_m[bx]
                n_offset = tile_to_n[bx]

                # No early-exit conditional — grid is sized exactly to TOTAL_TILES.
                # When sizes are dynamic the upper-bound grid will require an
                # if-then-else around the writes; we'll add that in production.
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
                            T.bitwise_and(
                                T.shift_right(W_q[e, n_offset + i, j // NUM_ELEM_PER_STORAGE], shift),
                                T.uint32(15),
                            ),
                        )
                        W_tile[i, j] = (w_int - T.float16(MAX_INT)) * Scale[e, n_offset + i, j // GROUP_SIZE]

                for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                    with T.sblock("compute"):
                        i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                        with T.init():
                            O[m_offset + i, n_offset + j] = T.float16(0.0)
                        O[m_offset + i, n_offset + j] = (
                            O[m_offset + i, n_offset + j] + X_tile[i, k] * W_tile[j, k]
                        )

    return grouped_gemm


def schedule_hand_tensorize(prim_func, target):
    from tvm.s_tir.tensor_intrin.cuda import get_wmma_intrin_group

    f = prim_func.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    sch = s_tir.Schedule(mod)

    TY = BLK_N // MICRO
    WARP = 32
    VEC = 4

    main_block = sch.get_sblock("compute")
    i, j, k = sch.get_loops(main_block)
    i_o, i_i = sch.split(i, factors=[None, MICRO])
    j_o, j_i = sch.split(j, factors=[None, MICRO])
    k_o, k_i = sch.split(k, factors=[None, MICRO])
    sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)

    block_inner = main_block
    block_outer = sch.blockize(i_i)

    k_o_o, k_o_i = sch.split(k_o, factors=[None, BLK_K // MICRO])
    sch.reorder(i_o, j_o, k_o_o, k_o_i)
    sch.bind(j_o, "threadIdx.y")

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
        sch.storage_align(blk, 0, axis=-2, factor=16, offset=8)

    _coop(x_shared)
    _coop(w_shared)

    A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
    B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
    sch.compute_at(A_mat, k_o_i)
    sch.compute_at(B_mat, k_o_i)

    acc_s2g = sch.cache_write(block_outer, 0, "shared.dyn")
    store = sch.cache_write(block_outer, 0, "wmma.accumulator")
    sch.reverse_compute_at(store, j_o)
    sch.reverse_compute_at(acc_s2g, j_o)

    si, sj = sch.get_loops(store)[-2:]
    si0, si1 = sch.split(si, factors=[None, MICRO])
    sj0, sj1 = sch.split(sj, factors=[None, MICRO])
    sch.reorder(si0, sj0, si1, sj1)

    asi, asj = sch.get_loops(acc_s2g)[-2:]
    afused = sch.fuse(asi, asj)
    _, afx, afv = sch.split(afused, factors=[None, WARP, VEC])
    sch.bind(afx, "threadIdx.x")
    sch.vectorize(afv)

    block_init_c = sch.decompose_reduction(block_outer, k_o_o)
    block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

    intrin_group = get_wmma_intrin_group(
        load_scope="shared.dyn", store_scope="shared.dyn",
        in_dtype="float16", out_dtype="float16", trans_b=True,
    )

    ai, aj = sch.get_loops(A_mat)[-2:]
    ai0, ai1 = sch.split(ai, factors=[None, MICRO])
    aj0, aj1 = sch.split(aj, factors=[None, MICRO])
    sch.reorder(ai0, aj0, ai1, aj1)
    sch.unroll(ai0); sch.unroll(aj0)
    sch.tensorize(ai1, intrin_group["load_a"])

    bi, bj = sch.get_loops(B_mat)[-2:]
    bi0, bi1 = sch.split(bi, factors=[None, MICRO])
    bj0, bj1 = sch.split(bj, factors=[None, MICRO])
    sch.reorder(bi0, bj0, bi1, bj1)
    sch.unroll(bi0); sch.unroll(bj0)
    sch.tensorize(bi1, intrin_group["load_b"])

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

    print(f"NE={NE}, tokens per expert: {TOKENS_PER_E} (B_TOTAL={B_TOTAL})")
    print(f"tiles per expert: {TILES_PER_E}, total tiles: {TOTAL_TILES}")

    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((B_TOTAL, K), dtype=np.float32).astype(np.float16)
    W_int4 = rng.integers(0, 16, size=(NE, N, K), dtype=np.int8)
    W_packed = np.zeros((NE, N, NUM_STORAGE), dtype=np.uint32)
    for s_idx in range(NUM_ELEM_PER_STORAGE):
        W_packed |= (W_int4[:, :, s_idx::NUM_ELEM_PER_STORAGE].astype(np.uint32) << (s_idx * 4))
    scale_np = (rng.standard_normal((NE, N, NUM_GROUP), dtype=np.float32).astype(np.float16) * 0.01)

    indptr = np.array([0] + list(np.cumsum(TOKENS_PER_E)), dtype=np.int32)
    print(f"indptr: {indptr}")

    # precompute tile_to_e, tile_to_m, tile_to_n
    tile_to_e_np = np.zeros(GRID_SIZE, dtype=np.int32)
    tile_to_m_np = np.zeros(GRID_SIZE, dtype=np.int32)
    tile_to_n_np = np.zeros(GRID_SIZE, dtype=np.int32)
    bx = 0
    for e in range(NE):
        tokens_e = TOKENS_PER_E[e]
        tiles_m = (tokens_e + BLK_M - 1) // BLK_M
        for tm in range(tiles_m):
            for tn in range(TILES_PER_N):
                tile_to_e_np[bx] = e
                tile_to_m_np[bx] = indptr[e] + tm * BLK_M
                tile_to_n_np[bx] = tn * BLK_N
                bx += 1
    assert bx == GRID_SIZE

    # numpy reference: per-expert matmul
    O_ref = np.zeros((B_TOTAL, N), dtype=np.float16)
    for e in range(NE):
        scale_full = np.repeat(scale_np[e], GROUP_SIZE, axis=1)
        W_fp16 = ((W_int4[e].astype(np.float32) - MAX_INT) * scale_full.astype(np.float32)).astype(np.float16)
        m0, m1 = indptr[e], indptr[e + 1]
        O_ref[m0:m1] = (X_np[m0:m1].astype(np.float32) @ W_fp16.T.astype(np.float32)).astype(np.float16)

    pf = make_prim_func()
    print("\n=== schedule + tensorize ===")
    try:
        mod = schedule_hand_tensorize(pf, target)
        body = str(mod["main"])
        print(f"  has wmma: {'wmma' in body or 'mma_sync' in body}")
        with open("/tmp/phase9b_grouped.txt", "w") as fh:
            fh.write(body)
        print(f"  body /tmp/phase9b_grouped.txt ({len(body)} chars)")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
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
    Te_dev = tvm.runtime.tensor(tile_to_e_np, dev)
    Tm_dev = tvm.runtime.tensor(tile_to_m_np, dev)
    Tn_dev = tvm.runtime.tensor(tile_to_n_np, dev)
    O_dev = tvm.runtime.tensor(np.zeros((B_TOTAL, N), dtype=np.float16), dev)

    for _ in range(3):
        rt(X_dev, W_dev, S_dev, Te_dev, Tm_dev, Tn_dev, O_dev)
    dev.sync()
    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rt(X_dev, W_dev, S_dev, Te_dev, Tm_dev, Tn_dev, O_dev)
    dev.sync()
    t_ms = (time.perf_counter() - t0) / n_iter * 1e3

    O_out = O_dev.numpy()
    max_diff = float(np.abs(O_out.astype(np.float32) - O_ref.astype(np.float32)).max())
    rel = max_diff / max(float(np.abs(O_ref.astype(np.float32)).max()), 1e-6)
    flops = 2 * B_TOTAL * N * K
    tflops = flops / (t_ms * 1e-3) / 1e12
    print(f"  ✓ build OK; t={t_ms:.3f} ms, {tflops:.2f} TFLOPS")
    print(f"  parity: max diff {max_diff:.4f}, rel {rel:.2%} {'PASS' if rel < 0.05 else 'FAIL'}")


if __name__ == "__main__":
    main()

"""Stage 2d — production-shape prototype: dispatch tables + v2 GEMM.

Two prim_funcs:
  1. compute_moe_dispatch_tables(indptr) -> (tile_to_e, tile_to_m, tile_to_n)
     Walks indptr, fills 3 int32 lookup tables sized to upper bound.
     Idle entries get sentinel tile_to_e = -1.

  2. dequantize_group_gemm_v2(x, w, scale, indptr, tile_to_e, tile_to_m, tile_to_n)
     Adapted from scratch_phase9b_grouped.py. BLK_M=16. Adds bounds checking
     on X reads + O writes. Skips compute when tile_to_e < 0.

Validation: random shapes (non-aligned tokens-per-expert), parity vs numpy.
"""
from __future__ import annotations
import time
import numpy as np
import tvm
from tvm import s_tir
from tvm.script import tirx as T

import os

NE = int(os.environ.get("V2_NE", "8"))
N, K = 1024, 2048
GROUP_SIZE = 32
NUM_ELEM_PER_STORAGE = 8
NUM_STORAGE = K // NUM_ELEM_PER_STORAGE
NUM_GROUP = K // GROUP_SIZE
MAX_INT = 7

BLK_M, BLK_N, BLK_K = 16, 128, 32
MICRO = 16
TILES_PER_N = N // BLK_N

# Default: small unaligned shape exercises bounds-check.
# Override via V2_TOKENS env (comma-separated) for larger shapes.
_default_tokens = "25,33,17,0,12,48,9,41" if NE == 8 else None
TOKENS_PER_E = [int(x) for x in os.environ.get("V2_TOKENS", _default_tokens).split(",")]
assert len(TOKENS_PER_E) == NE
B_TOTAL = sum(TOKENS_PER_E)


def compute_dispatch_tables_np(indptr_np, B_dim, ne, blk_m, tiles_per_n):
    """Reference (numpy) implementation matching the TIR func behavior."""
    upper = ((B_dim + blk_m - 1) // blk_m + ne) * tiles_per_n
    te = np.full(upper, -1, dtype=np.int32)
    tm = np.zeros(upper, dtype=np.int32)
    tn = np.zeros(upper, dtype=np.int32)
    bx = 0
    for e in range(ne):
        tokens_e = int(indptr_np[e + 1] - indptr_np[e])
        if tokens_e <= 0:
            continue
        tiles_m = (tokens_e + blk_m - 1) // blk_m
        for tmi in range(tiles_m):
            for tni in range(tiles_per_n):
                te[bx] = e
                tm[bx] = int(indptr_np[e]) + tmi * blk_m
                tn[bx] = tni * (N // tiles_per_n)
                bx += 1
    return te, tm, tn, bx


def make_dispatch_func():
    """TIR prim_func that fills (tile_to_e, tile_to_m, tile_to_n) by walking indptr.

    Single CTA, single thread — sized for Ne up to ~128 with at most a few
    thousand output entries. Total wall < 100us, BW-bound on indptr reads.
    """
    @T.prim_func(private=True)
    def _func(
        indptr: T.Buffer((NE + 1,), "int32"),
        var_te: T.handle,
        var_tm: T.handle,
        var_tn: T.handle,
    ):
        T.func_attr({"tirx.is_scheduled": 1, "tirx.noalias": True})
        UPPER = T.int32(is_size_var=True)
        te = T.match_buffer(var_te, (UPPER,), "int32")
        tm = T.match_buffer(var_tm, (UPPER,), "int32")
        tn = T.match_buffer(var_tn, (UPPER,), "int32")

        for _bx in T.thread_binding(1, thread="blockIdx.x"):
            for _tx in T.thread_binding(1, thread="threadIdx.x"):
                with T.sblock("dispatch"):
                    T.reads(indptr[:])
                    T.writes(te[:], tm[:], tn[:])
                    cur = T.sblock_alloc_buffer((1,), "int32", scope="local")
                    cur[0] = 0
                    for ee in range(NE):
                        delta = indptr[ee + 1] - indptr[ee]
                        tiles_m = T.ceildiv(delta, BLK_M)
                        for tmi in T.serial(tiles_m):
                            for tni in T.serial(TILES_PER_N):
                                te[cur[0]] = ee
                                tm[cur[0]] = indptr[ee] + tmi * BLK_M
                                tn[cur[0]] = tni * BLK_N
                                cur[0] += 1
                    # fill remainder with sentinels
                    for bx in T.serial(UPPER - cur[0]):
                        te[cur[0] + bx] = -1
                        tm[cur[0] + bx] = 0
                        tn[cur[0] + bx] = 0

    return _func


def make_gemm_v2_func():
    """v2 GEMM: dispatch-table driven, hand-tensorized wmma matmul.

    Structure: local O_tile accumulator + separate store sblock with bounds
    check on m_offset+i < row_end. Idle blocks (e_v < 0) have row_end=0, so
    no global writes happen. Valid tiles whose last sub-row spills past
    row_end are similarly masked.
    """
    zero_f16 = T.float16(0.0)

    @T.prim_func(private=True)
    def _func(
        var_x: T.handle,
        W_q: T.Buffer((NE, N, NUM_STORAGE), "uint32"),
        Scale: T.Buffer((NE, N, NUM_GROUP), "float16"),
        indptr: T.Buffer((NE + 1,), "int32"),
        var_te: T.handle,
        var_tm: T.handle,
        var_tn: T.handle,
        var_o: T.handle,
    ):
        T.func_attr({"tirx.noalias": True})
        B = T.int32(is_size_var=True)
        UPPER = T.int32(is_size_var=True)
        X = T.match_buffer(var_x, (B, K), "float16")
        out = T.match_buffer(var_o, (B, N), "float16")
        te = T.match_buffer(var_te, (UPPER,), "int32")
        tm = T.match_buffer(var_tm, (UPPER,), "int32")
        tn = T.match_buffer(var_tn, (UPPER,), "int32")

        for _bx in T.thread_binding(UPPER, thread="blockIdx.x"):
            with T.sblock("CTA"):
                bx = T.axis.spatial(UPPER, _bx)
                T.reads(X[:, :], W_q[:, :, :], Scale[:, :, :], indptr[:],
                        te[:], tm[:], tn[:])
                T.writes(out[:, :])

                X_tile = T.sblock_alloc_buffer((BLK_M, K), "float16", scope="shared.dyn")
                W_tile = T.sblock_alloc_buffer((BLK_N, K), "float16", scope="shared.dyn")
                O_tile = T.sblock_alloc_buffer((BLK_M, BLK_N), "float16", scope="shared.dyn")

                e_v = te[bx]
                m_offset = tm[bx]
                n_offset = tn[bx]
                row_end = T.if_then_else(e_v >= 0, indptr[e_v + 1], 0)
                e_safe = T.if_then_else(e_v >= 0, e_v, 0)

                for a0, a1 in T.grid(BLK_M, K):
                    with T.sblock("X_shared"):
                        i, j = T.axis.remap("SS", [a0, a1])
                        X_tile[i, j] = T.if_then_else(
                            m_offset + i < row_end,
                            X[m_offset + i, j],
                            zero_f16,
                        )

                for a0, a1 in T.grid(BLK_N, K):
                    with T.sblock("W_shared"):
                        i, j = T.axis.remap("SS", [a0, a1])
                        shift = T.Cast("uint32", (j % NUM_ELEM_PER_STORAGE) * 4)
                        w_int = T.Cast(
                            "float16",
                            T.bitwise_and(
                                T.shift_right(
                                    W_q[e_safe, n_offset + i, j // NUM_ELEM_PER_STORAGE],
                                    shift,
                                ),
                                T.uint32(15),
                            ),
                        )
                        W_tile[i, j] = (w_int - T.float16(MAX_INT)) * Scale[
                            e_safe, n_offset + i, j // GROUP_SIZE
                        ]

                for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                    with T.sblock("compute"):
                        i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                        with T.init():
                            O_tile[i, j] = zero_f16
                        O_tile[i, j] = O_tile[i, j] + X_tile[i, k] * W_tile[j, k]

                for a0, a1 in T.grid(BLK_M, BLK_N):
                    with T.sblock("store"):
                        i, j = T.axis.remap("SS", [a0, a1])
                        if m_offset + i < row_end:
                            out[m_offset + i, n_offset + j] = O_tile[i, j]

    return _func


def schedule_gemm_v2(prim_func, target):
    """Hand-tensorize schedule for v2:
       compute (writes O_tile shared) ← cache_write to wmma.accumulator
       auto-inserted block writes accumulator → O_tile shared (tensorize wmma_store)
       explicit "store" sblock copies O_tile shared → global out with predicate
    """
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

    # Single cache_write inserts wmma.accumulator between compute and O_tile shared.
    # The auto-inserted block (accumulator → O_tile shared) gets tensorized as wmma_store.
    acc_blk = sch.cache_write(block_outer, 0, "wmma.accumulator")
    sch.reverse_compute_at(acc_blk, j_o)

    si, sj = sch.get_loops(acc_blk)[-2:]
    si0, si1 = sch.split(si, factors=[None, MICRO])
    sj0, sj1 = sch.split(sj, factors=[None, MICRO])
    sch.reorder(si0, sj0, si1, sj1)

    # Schedule the explicit "store" sblock (O_tile shared → out global w/ predicate).
    # Cooperative store: 256 threads (TY=8 × WARP=32) collectively store BLK_M*BLK_N=2048
    # values. Vec=4 → 2048/(8*32*4) = 2 iterations per thread.
    store_block = sch.get_sblock("store")
    sch.reverse_compute_at(store_block, j_o, preserve_unit_loops=True)
    s_loops = sch.get_loops(store_block)[-2:]
    s_fused = sch.fuse(*s_loops)
    # j_o is already 8-wide threadIdx.y. Just split inner over WARP*VEC.
    _, s_fx, s_fv = sch.split(s_fused, factors=[None, WARP, VEC])
    sch.bind(s_fx, "threadIdx.x")
    sch.vectorize(s_fv)

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
    sch.tensorize(sch.get_loops(acc_blk)[-2], intrin_group["store"])
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

    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((B_TOTAL, K), dtype=np.float32).astype(np.float16)
    W_int4 = rng.integers(0, 16, size=(NE, N, K), dtype=np.int8)
    W_packed = np.zeros((NE, N, NUM_STORAGE), dtype=np.uint32)
    for s_idx in range(NUM_ELEM_PER_STORAGE):
        W_packed |= (W_int4[:, :, s_idx::NUM_ELEM_PER_STORAGE].astype(np.uint32) << (s_idx * 4))
    scale_np = (rng.standard_normal((NE, N, NUM_GROUP), dtype=np.float32).astype(np.float16) * 0.01)

    indptr_np = np.array([0] + list(np.cumsum(TOKENS_PER_E)), dtype=np.int32)
    print(f"indptr: {indptr_np}")

    upper = ((B_TOTAL + BLK_M - 1) // BLK_M + NE) * TILES_PER_N
    te_ref, tm_ref, tn_ref, total_ref = compute_dispatch_tables_np(
        indptr_np, B_TOTAL, NE, BLK_M, TILES_PER_N
    )
    print(f"upper={upper}, total_actual={total_ref}, idle={upper-total_ref}")

    # ----- numpy reference matmul -----
    O_ref = np.zeros((B_TOTAL, N), dtype=np.float16)
    for e in range(NE):
        m0, m1 = int(indptr_np[e]), int(indptr_np[e + 1])
        if m1 - m0 == 0:
            continue
        scale_full = np.repeat(scale_np[e], GROUP_SIZE, axis=1)
        W_fp16 = ((W_int4[e].astype(np.float32) - MAX_INT) * scale_full.astype(np.float32)).astype(np.float16)
        O_ref[m0:m1] = (X_np[m0:m1].astype(np.float32) @ W_fp16.T.astype(np.float32)).astype(np.float16)

    # ----- build & run dispatch -----
    print("\n=== dispatch func ===")
    pf_dispatch = make_dispatch_func()
    f = pf_dispatch.with_attr("global_symbol", "main")
    mod_d = tvm.IRModule.from_expr(f)
    try:
        with target:
            rt_d = tvm.tirx.build(mod_d["main"], target=target)
    except Exception as ex:
        print(f"  dispatch build FAILED: {type(ex).__name__}: {ex}")
        import traceback; traceback.print_exc()
        return

    indptr_dev = tvm.runtime.tensor(indptr_np, dev)
    te_dev = tvm.runtime.tensor(np.zeros(upper, dtype=np.int32), dev)
    tm_dev = tvm.runtime.tensor(np.zeros(upper, dtype=np.int32), dev)
    tn_dev = tvm.runtime.tensor(np.zeros(upper, dtype=np.int32), dev)
    rt_d(indptr_dev, te_dev, tm_dev, tn_dev)
    dev.sync()
    te_out = te_dev.numpy()
    tm_out = tm_dev.numpy()
    tn_out = tn_dev.numpy()
    # verify against numpy reference
    ok_te = np.array_equal(te_out, te_ref)
    ok_tm = np.array_equal(tm_out, tm_ref)
    ok_tn = np.array_equal(tn_out, tn_ref)
    print(f"  dispatch parity: te={'OK' if ok_te else 'FAIL'} "
          f"tm={'OK' if ok_tm else 'FAIL'} tn={'OK' if ok_tn else 'FAIL'}")
    if not (ok_te and ok_tm and ok_tn):
        print(f"  te_ref[:20]={te_ref[:20]}\n  te_out[:20]={te_out[:20]}")
        print(f"  tm_ref[:20]={tm_ref[:20]}\n  tm_out[:20]={tm_out[:20]}")
        print(f"  tn_ref[:20]={tn_ref[:20]}\n  tn_out[:20]={tn_out[:20]}")
        return

    # ----- build v2 GEMM -----
    print("\n=== v2 GEMM (schedule + tensorize) ===")
    pf_g = make_gemm_v2_func()
    try:
        mod_g = schedule_gemm_v2(pf_g, target)
        body = str(mod_g["main"])
        print(f"  has wmma: {'wmma' in body or 'mma_sync' in body}")
        with open("/tmp/phase9b_v2_body.txt", "w") as fh:
            fh.write(body)
    except Exception as ex:
        print(f"  schedule FAILED: {type(ex).__name__}: {str(ex)[:300]}")
        import traceback; traceback.print_exc()
        return

    try:
        with target:
            rt_g = tvm.tirx.build(mod_g["main"], target=target)
    except Exception as ex:
        print(f"  build FAILED: {type(ex).__name__}: {str(ex).splitlines()[-1][:200]}")
        return

    X_dev = tvm.runtime.tensor(X_np, dev)
    W_dev = tvm.runtime.tensor(W_packed, dev)
    S_dev = tvm.runtime.tensor(scale_np, dev)
    O_dev = tvm.runtime.tensor(np.zeros((B_TOTAL, N), dtype=np.float16), dev)

    for _ in range(3):
        rt_g(X_dev, W_dev, S_dev, indptr_dev, te_dev, tm_dev, tn_dev, O_dev)
    dev.sync()
    n_iter = 50
    t0 = time.perf_counter()
    for _ in range(n_iter):
        rt_g(X_dev, W_dev, S_dev, indptr_dev, te_dev, tm_dev, tn_dev, O_dev)
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

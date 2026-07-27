#!/usr/bin/env python3
"""vit_flash_tir.py — item 0r's second gate: can TIR reach the rate the probe found?

§22.2 measured the ceiling with hand-written CUDA (`scripts/vit_flash_probe.cu`,
1.77 TFLOP/s, 11.05 ms/layer) and flagged the risk that matters: **break-even needs
1.33 TFLOP/s, and dlight's generated fp32 matmul gets 1.03.** The shipping kernel has
to be TIR (CLAUDE.md), so the open question is not "is flash attention faster" — that
is answered — but "can TIR express it at the rate that makes it faster".

This is the same kernel as the .cu probe, written in TIR against a **symbolic** sequence
length, standing alone: no model, no compile pipeline, no integration. It answers the
rate question and nothing else. If TIR lands near 1.77 TFLOP/s, item 0r is green and the
remaining work is integration. If it lands near dlight's 1.03, item 0r is dead and the
44 ms was never reachable.

Correctness is checked against a NumPy reference at sequence lengths that are not
multiples of the tile, so the partial-tile paths are exercised — the .cu probe's bar.

    source .envrc.local
    python scripts/vit_flash_tir.py
    python scripts/vit_flash_tir.py --seqs 1260,2520,5040
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np  # noqa: E402
import tvm  # noqa: E402
from tvm.script import tirx as T  # noqa: E402

# The cat fixture at the tower's native patch count, and the tower's own config.
H, D, LAYERS = 12, 64, 12
BM, BN = 64, 32          # query rows and key columns per CTA
TX, TY = 16, 16          # thread grid; NT = 256
NT = TX * TY
RM, RN, RO = BM // TY, BN // TX, D // TX     # 4 rows, 2 score cols, 4 output cols
PARTS = NT // BM         # threads cooperating on one softmax row
NEG = -3.0e38            # a finite -inf: exp(NEG - m) underflows to 0, which is what
                         # zeroes `acc` on the first tile without introducing a NaN


@T.prim_func
def flash_attn(var_q: T.handle, var_k: T.handle, var_v: T.handle, var_o: T.handle,
               scale: T.float32):
    T.func_attr({"tir.noalias": True, "global_symbol": "flash_attn"})
    seq = T.int32()
    Q = T.match_buffer(var_q, (H, seq, D), "float32")
    K = T.match_buffer(var_k, (H, seq, D), "float32")
    V = T.match_buffer(var_v, (H, seq, D), "float32")
    O = T.match_buffer(var_o, (H, seq, D), "float32")

    for bx in T.thread_binding(T.ceildiv(seq, BM), thread="blockIdx.x"):
        for by in T.thread_binding(H, thread="blockIdx.y"):
            for tid in T.thread_binding(NT, thread="threadIdx.x"):
                with T.sblock("cta"):
                    T.reads(Q[0:H, 0:seq, 0:D], K[0:H, 0:seq, 0:D], V[0:H, 0:seq, 0:D])
                    T.writes(O[0:H, 0:seq, 0:D])
                    # Padding matches the .cu probe and is deliberate: Qs/Ks are read
                    # down a column, Vs along a row.
                    Qs = T.sblock_alloc_buffer((BM, D + 1), "float32", scope="shared")
                    Ks = T.sblock_alloc_buffer((BN, D + 1), "float32", scope="shared")
                    Vs = T.sblock_alloc_buffer((BN, D), "float32", scope="shared")
                    Ss = T.sblock_alloc_buffer((BM, BN + 1), "float32", scope="shared")
                    rmax = T.sblock_alloc_buffer((BM,), "float32", scope="shared")
                    rsum = T.sblock_alloc_buffer((BM,), "float32", scope="shared")
                    rnew = T.sblock_alloc_buffer((BM,), "float32", scope="shared")
                    rcor = T.sblock_alloc_buffer((BM,), "float32", scope="shared")
                    # Partial reductions. The .cu probe uses a warp butterfly here; TIR
                    # has no portable shuffle, so the partials land in shared and this
                    # kernel pays 6 barriers per tile against the .cu probe's 3.
                    pmax = T.sblock_alloc_buffer((BM, PARTS), "float32", scope="shared")
                    psum = T.sblock_alloc_buffer((BM, PARTS), "float32", scope="shared")
                    acc = T.sblock_alloc_buffer((RM, RO), "float32", scope="local")
                    sreg = T.sblock_alloc_buffer((RM, RN), "float32", scope="local")
                    tmp = T.sblock_alloc_buffer((1,), "float32", scope="local")

                    tx: T.int32 = tid % TX
                    ty: T.int32 = tid // TX
                    row: T.int32 = tid // PARTS
                    part: T.int32 = tid % PARTS

                    for i in T.serial(BM * D // NT):
                        flat: T.int32 = i * NT + tid
                        Qs[flat // D, flat % D] = T.if_then_else(
                            bx * BM + flat // D < seq,
                            Q[by, bx * BM + flat // D, flat % D] * scale,
                            T.float32(0),
                        )
                    if tid < BM:
                        rmax[tid] = T.float32(NEG)
                        rsum[tid] = T.float32(0)
                        # Keep the pad columns inside the touched region, or
                        # CompactBufferAllocation removes them and every strided read
                        # collapses onto one bank. Measured: 0.78 -> see §22.3.
                        Qs[tid, D] = T.float32(0)
                        Ss[tid, BN] = T.float32(0)
                        if tid < BN:
                            Ks[tid, D] = T.float32(0)
                    for i in T.unroll(RM):
                        for j in T.unroll(RO):
                            acc[i, j] = T.float32(0)
                    T.evaluate(T.tvm_storage_sync("shared"))

                    for nb in T.serial(T.ceildiv(seq, BN)):
                        n0: T.int32 = nb * BN
                        for i in T.serial(BN * D // NT):
                            flat: T.int32 = i * NT + tid
                            Ks[flat // D, flat % D] = T.if_then_else(
                                n0 + flat // D < seq,
                                K[by, n0 + flat // D, flat % D], T.float32(0))
                            Vs[flat // D, flat % D] = T.if_then_else(
                                n0 + flat // D < seq,
                                V[by, n0 + flat // D, flat % D], T.float32(0))
                        T.evaluate(T.tvm_storage_sync("shared"))

                        # --- S = (Q*scale) @ K^T for this tile ---
                        for i in T.unroll(RM):
                            for j in T.unroll(RN):
                                sreg[i, j] = T.float32(0)
                        for k in T.serial(D):
                            for i in T.unroll(RM):
                                for j in T.unroll(RN):
                                    sreg[i, j] = sreg[i, j] + Qs[ty * RM + i, k] * Ks[tx * RN + j, k]
                        for i in T.unroll(RM):
                            for j in T.unroll(RN):
                                Ss[ty * RM + i, tx * RN + j] = T.if_then_else(
                                    n0 + tx * RN + j < seq, sreg[i, j], T.float32(NEG))
                        T.evaluate(T.tvm_storage_sync("shared"))

                        # --- online softmax: PARTS threads per row, reduced via shared ---
                        tmp[0] = T.float32(NEG)
                        for c in T.serial(BN // PARTS):
                            tmp[0] = T.max(tmp[0], Ss[row, c * PARTS + part])
                        pmax[row, part] = tmp[0]
                        T.evaluate(T.tvm_storage_sync("shared"))
                        if tid < BM:
                            tmp[0] = rmax[tid]
                            for q in T.serial(PARTS):
                                tmp[0] = T.max(tmp[0], pmax[tid, q])
                            rnew[tid] = tmp[0]
                        T.evaluate(T.tvm_storage_sync("shared"))
                        tmp[0] = T.float32(0)
                        for c in T.serial(BN // PARTS):
                            Ss[row, c * PARTS + part] = T.exp(
                                Ss[row, c * PARTS + part] - rnew[row])
                            tmp[0] = tmp[0] + Ss[row, c * PARTS + part]
                        psum[row, part] = tmp[0]
                        T.evaluate(T.tvm_storage_sync("shared"))
                        if tid < BM:
                            tmp[0] = T.float32(0)
                            for q in T.serial(PARTS):
                                tmp[0] = tmp[0] + psum[tid, q]
                            # exp(NEG - mnew) underflows to 0, which is what zeroes `acc`
                            # on the first tile without introducing a NaN.
                            rcor[tid] = T.exp(rmax[tid] - rnew[tid])
                            rsum[tid] = rsum[tid] * rcor[tid] + tmp[0]
                            rmax[tid] = rnew[tid]
                        T.evaluate(T.tvm_storage_sync("shared"))

                        # --- O = O*corr + P @ V ---
                        for i in T.unroll(RM):
                            for j in T.unroll(RO):
                                acc[i, j] = acc[i, j] * rcor[ty * RM + i]
                        for k in T.serial(BN):
                            for i in T.unroll(RM):
                                for j in T.unroll(RO):
                                    acc[i, j] = acc[i, j] + Ss[ty * RM + i, k] * Vs[k, tx * RO + j]
                        T.evaluate(T.tvm_storage_sync("shared"))

                    for i in T.unroll(RM):
                        if bx * BM + ty * RM + i < seq:
                            for j in T.unroll(RO):
                                O[by, bx * BM + ty * RM + i, tx * RO + j] = (
                                    acc[i, j] / rsum[ty * RM + i])


def reference(q, k, v, scale):
    out = np.empty_like(q)
    for h in range(q.shape[0]):
        s = (q[h] * scale) @ k[h].T
        s = s - s.max(axis=-1, keepdims=True)
        p = np.exp(s)
        out[h] = (p / p.sum(axis=-1, keepdims=True)) @ v[h]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", default="1260,2520,5040")
    p.add_argument("--repeat", type=int, default=10)
    cli = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    mod = tvm.compile(flash_attn, target=target)
    scale = 1.0 / np.sqrt(D)

    # Correctness first: a fast wrong kernel answers nothing.
    for s in (200, 1000):
        rng = np.random.default_rng(0)
        q, k, v = (rng.standard_normal((H, s, D), dtype="float32") * 0.3 for _ in range(3))
        tq, tk, tv = (tvm.runtime.tensor(x, device=dev) for x in (q, k, v))
        to = tvm.runtime.tensor(np.zeros((H, s, D), "float32"), device=dev)
        mod["flash_attn"](tq, tk, tv, to, scale)
        dev.sync()
        err = np.abs(to.numpy() - reference(q, k, v, scale)).max()
        print(f"[verify] seq={s:<5} (partial m- and n-tiles) max abs diff {err:.3e}  "
              f"{'OK' if err < 2e-5 else '*** MISMATCH ***'}")
        if not err < 2e-5:
            sys.exit(1)

    peak = 16 * 128 * 2 * 1.3005e9   # GA10B: 16 SMs x 128 fp32 lanes, §22.2's basis
    print(f"\n{'seq':>6} {'ms/layer':>9} {'TFLOP/s':>9} {'% peak':>8}   "
          f"vs the .cu probe / vs dlight's 1.03")
    for s in (int(x) for x in cli.seqs.split(",")):
        rng = np.random.default_rng(0)
        q, k, v = (rng.standard_normal((H, s, D), dtype="float32") * 0.3 for _ in range(3))
        tq, tk, tv = (tvm.runtime.tensor(x, device=dev) for x in (q, k, v))
        to = tvm.runtime.tensor(np.zeros((H, s, D), "float32"), device=dev)
        ms = mod.mod.time_evaluator("flash_attn", dev, number=1, repeat=cli.repeat)(
            tq, tk, tv, to, scale).median * 1e3
        flop = 2.0 * 2.0 * H * s * s * D
        tf = flop / (ms * 1e-3) / 1e12
        print(f"{s:6d} {ms:9.2f} {tf:9.2f} {100 * flop / (ms * 1e-3) / peak:7.1f}%   "
              f"{tf / 1.77:5.2f}x / {tf / 1.03:5.2f}x")

    print("\nBreak-even against the block it replaces needs 1.33 TFLOP/s (§22.2).")
    print("The hand-written CUDA reference reaches 1.77; dlight's default fp32 matmul 1.03.")


if __name__ == "__main__":
    main()

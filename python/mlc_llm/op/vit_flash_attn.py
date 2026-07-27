"""Flash attention for the Qwen3.5-VL vision tower (workplan item 0r).

Why this exists
---------------
The tower's attention block is three kernels — ``QK^T``, ``softmax`` and ``P@V`` —
whose entire job is to write, re-read and read the same ``(12, 2520, 2520)`` fp32
score matrix, **305 MB**. Together they are 14.71 ms/layer against a 3.67 ms compute
floor (workplan §20.9). Tiling the attention so the scores never leave shared memory
drops the traffic to 27 MB and makes the block compute-bound.

fp32 is not a choice. ``qwen3_vl_vit.py`` records that running this math in fp16
collapses tower parity (max diff 2.03 / rel 39% on the cat fixture), so the fp16
tensor-core path every reference implementation uses is off the table, and there is no
HF kernel to copy — HF runs this tower with eager attention.

What was measured before it was written
---------------------------------------
§22.2 and §22.3, because §20.9's ~89 ms/iter estimate came from quoting cuBLAS's 51%
of fp32 peak for a kernel nobody had written:

- ``scripts/vit_flash_probe.cu`` — the same algorithm in hand-written CUDA, swept over
  six tile shapes, to establish the ceiling: **1.77 TFLOP/s**, 11.05 ms/layer. A
  hand-tiled fp32 GEMM control lands at 1.57 TFLOP/s, i.e. between dlight's 1.03 and
  cuBLAS's 2.69 and closer to dlight. That re-priced the item to ~43-51 ms/iter.
- ``scripts/vit_flash_tir.py`` — this kernel, standalone, verified against NumPy.
  **1.75 TFLOP/s, 0.99x the hand-written CUDA**, against a break-even bar of 1.33.

Two things that cost 2.25x between the first TIR draft and this one, both invisible in
the TIR source and neither of which warns (§22.3, and §22.5 audits the rest of the tree
for them):

1. **A ``scope="local"`` buffer indexed by a loop variable is local *memory*.** Every
   register-tile loop below is ``T.unroll`` so the indices are literals; with
   ``T.serial`` the emitted CUDA is ``float acc[16]`` indexed by a loop var, which is
   DRAM-backed.
2. **``CompactBufferAllocation`` shrinks a buffer to the region actually touched**, so
   the ``+1`` bank-conflict padding below does not exist unless the pad column is
   written. It is written once, deliberately, in the prologue.
"""

import os
from typing import Tuple

from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tirx as T

# Tile shape, from the six-way sweep in scripts/vit_flash_probe.cu. BM64/BN32/16x16 won
# at every sequence length measured (1260, 2520, 5040) and the ranking was identical at
# all three. Note the 8-CTA config loses to this 3-CTA one: occupancy is not the lever
# here, which is the same finding §21.3 reports for the MoE GEMM.
BM, BN = 64, 32
TX, TY = 16, 16
NT = TX * TY

# A finite -inf. exp(NEG - m) underflows to 0, which is what zeroes the accumulator on
# the first tile; a true -inf would make the first correction NaN.
NEG = -3.0e38


def _flash_func(num_heads: int, head_dim: int, scaling: float):
    """Build the PrimFunc for one head configuration.

    ``head_dim`` and ``num_heads`` are baked in (they are compile-time constants for a
    given tower) but the sequence length stays **symbolic** — the tower's patch count
    varies with the image, and §20.8 already refuted pinning it.
    """
    D = head_dim
    RM, RN, RO = BM // TY, BN // TX, D // TX   # rows, score cols, output cols per thread
    PARTS = NT // BM                            # threads cooperating on one softmax row
    assert supported(head_dim), f"head_dim={head_dim} is outside this tile's constraints"

    @T.prim_func(private=True)
    def flash_attn(var_q: T.handle, var_k: T.handle, var_v: T.handle, var_o: T.handle):
        T.func_attr({"tirx.is_scheduled": 1, "tirx.noalias": True})
        seq = T.int32(is_size_var=True)
        Q = T.match_buffer(var_q, (num_heads, seq, D), "float32")
        K = T.match_buffer(var_k, (num_heads, seq, D), "float32")
        V = T.match_buffer(var_v, (num_heads, seq, D), "float32")
        O = T.match_buffer(var_o, (num_heads, seq, D), "float32")

        for bx in T.thread_binding(T.ceildiv(seq, BM), thread="blockIdx.x"):
            for by in T.thread_binding(num_heads, thread="blockIdx.y"):
                for tid in T.thread_binding(NT, thread="threadIdx.x"):
                    with T.sblock("cta"):
                        T.reads(Q[:, :, :], K[:, :, :], V[:, :, :])
                        T.writes(O[:, :, :])
                        # Padding is per-buffer and deliberate: Qs/Ks are read down a
                        # column (stride D+1 puts the TX readers on distinct banks)
                        # while Vs is read along a row as a float4 (no padding, so the
                        # 16 B lanes tile the 32 banks exactly).
                        Qs = T.sblock_alloc_buffer((BM, D + 1), "float32", scope="shared")
                        Ks = T.sblock_alloc_buffer((BN, D + 1), "float32", scope="shared")
                        Vs = T.sblock_alloc_buffer((BN, D), "float32", scope="shared")
                        Ss = T.sblock_alloc_buffer((BM, BN + 1), "float32", scope="shared")
                        rmax = T.sblock_alloc_buffer((BM,), "float32", scope="shared")
                        rsum = T.sblock_alloc_buffer((BM,), "float32", scope="shared")
                        rnew = T.sblock_alloc_buffer((BM,), "float32", scope="shared")
                        rcor = T.sblock_alloc_buffer((BM,), "float32", scope="shared")
                        # Partial reductions. The .cu probe uses a warp butterfly here;
                        # TIR has no portable shuffle, so partials go through shared and
                        # this kernel pays 6 barriers per tile against the probe's 3 —
                        # and still lands within 1% of it.
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
                                Q[by, bx * BM + flat // D, flat % D] * T.float32(scaling),
                                T.float32(0),
                            )
                        if tid < BM:
                            rmax[tid] = T.float32(NEG)
                            rsum[tid] = T.float32(0)
                            # Touch the pad columns, or CompactBufferAllocation removes
                            # them and every strided read collapses onto one bank.
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
                                        sreg[i, j] = (sreg[i, j]
                                                      + Qs[ty * RM + i, k] * Ks[tx * RN + j, k])
                            for i in T.unroll(RM):
                                for j in T.unroll(RN):
                                    Ss[ty * RM + i, tx * RN + j] = T.if_then_else(
                                        n0 + tx * RN + j < seq, sreg[i, j], T.float32(NEG))
                            T.evaluate(T.tvm_storage_sync("shared"))

                            # --- online softmax: PARTS threads per row ---
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
                                        acc[i, j] = (acc[i, j]
                                                     + Ss[ty * RM + i, k] * Vs[k, tx * RO + j])
                            T.evaluate(T.tvm_storage_sync("shared"))

                        for i in T.unroll(RM):
                            if bx * BM + ty * RM + i < seq:
                                for j in T.unroll(RO):
                                    O[by, bx * BM + ty * RM + i, tx * RO + j] = (
                                        acc[i, j] / rsum[ty * RM + i])

    return flash_attn


def supported(head_dim: int) -> bool:
    """Whether this tile shape can cover ``head_dim``.

    The tile is fixed (it won the six-way sweep at ``head_dim=64``, which is what
    Qwen3.5-VL's tower uses), so a tower with a different head dimension has to fall
    back rather than fail: ``RO = head_dim // TX`` has to be a whole multiple of 4 for
    the float4 reads of ``Vs``, and both cooperative loads have to divide evenly across
    the thread block. Checked here rather than asserted at the call site so that
    defaulting this ON cannot break a model it was never measured against.
    """
    return (
        head_dim % TX == 0
        and (head_dim // TX) % 4 == 0
        and BM * head_dim % NT == 0
        and BN * head_dim % NT == 0
    )


def enabled(head_dim: int) -> bool:
    """Default ON since the §23 measurement: ttft −13.5%, gate 184/184 exact.

    ``MLC_QWEN35_VL_FLASH=0`` restores the three-kernel path for an A/B.
    """
    return os.environ.get("MLC_QWEN35_VL_FLASH", "1") == "1" and supported(head_dim)


def flash_attention(q: Tensor, k: Tensor, v: Tensor, scaling: float) -> Tensor:
    """``softmax(q @ k^T * scaling) @ v`` for ``(num_heads, seq, head_dim)`` fp32.

    Replaces the ``QK^T`` / ``softmax`` / ``P@V`` span *and* the ``permute_dims`` that
    fed it, since the transpose is implicit in how ``K`` is read.
    """
    num_heads, _, head_dim = q.shape
    return op.tensor_ir_op(
        _flash_func(int(num_heads), int(head_dim), scaling),
        "vit_flash_attn",
        args=[q, k, v],
        out=Tensor.placeholder(q.shape, "float32"),
    )

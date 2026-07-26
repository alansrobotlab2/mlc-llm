#!/usr/bin/env python3
"""bench_moe_kernel.py — microbench dequantize_group_gemm at 35B-A3B shapes.

Times the TIR kernel directly (skips engine warmup) for fast kernel-fix
iteration. Builds a one-function Relax module that calls the kernel, lowers
through the standard Legalize + dlight + relax.build pipeline, then times via
`time_evaluator`.

Usage:
    .venv/bin/python bench_moe_kernel.py
    .venv/bin/python bench_moe_kernel.py --save base.json
    .venv/bin/python bench_moe_kernel.py --baseline base.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tvm
import tvm.runtime
from tvm import relax
from tvm.s_tir import dlight as dl
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec

from mlc_llm.op.moe_matmul import dequantize_gemv, dequantize_group_gemm
from mlc_llm.op.moe_misc import gating_softmax_topk
from mlc_llm.quantization.quantization import QUANTIZATION
from mlc_llm.quantization.group_quantization import GroupQuantizeLinear
from mlc_llm.quantization.ft_quantization import FTQuantizeLinear


# 35B-A3B q4f16_1 MoE shapes (Qwen3.6-35B-A3B):
#   hidden=2048, moe_intermediate_size=512, num_experts=256, top_k=8
# gate_up_proj: N=2*512=1024 (gate||up), K=2048
# down_proj:    N=2048,                  K=512
# group_size=32 (q4f16_1 default)
#
# B = total active rows = batch * top_k. Decode b=1 → B=8. Prefill b=128 → B=1024.
# At prefill, rows are distributed across all 256 experts (~4 rows/expert avg).
SHAPES = {
    # group_gemm (current decode path): B = top_k active rows
    "gate_up":         dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=8,    spread=False, kind="group_gemm"),
    "down":            dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=8,    spread=False, kind="group_gemm"),
    "gate_up_prefill": dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=1024, spread=True,  kind="group_gemm"),
    "down_prefill":    dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=1024, spread=True,  kind="group_gemm"),
    # Small-batch verify shapes (B = num_tokens * top_k, spread across experts).
    # These are what spec-decode verify pays at γ=1..4 inside the MoE block:
    #   γ=1 (b=2 verify, num_tokens=2): B=16
    #   γ=2 (b=3 verify, num_tokens=3): B=24
    #   γ=3 (b=4 verify, num_tokens=4): B=32
    #   γ=4 (b=5 verify, num_tokens=5): B=40
    # spread=True approximates pessimistic routing (each row goes to a distinct
    # expert; no expert sharing across drafted tokens). Realistic small-batch
    # routing has 30-50% expert overlap which would shift these toward gemv-cost.
    "gate_up_b16":     dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=16,   spread=True,  kind="group_gemm"),
    "down_b16":        dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=16,   spread=True,  kind="group_gemm"),
    "gate_up_b24":     dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=24,   spread=True,  kind="group_gemm"),
    "down_b24":        dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=24,   spread=True,  kind="group_gemm"),
    "gate_up_b32":     dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=32,   spread=True,  kind="group_gemm"),
    "down_b32":        dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=32,   spread=True,  kind="group_gemm"),
    "gate_up_b40":     dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=40,   spread=True,  kind="group_gemm"),
    "down_b40":        dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=40,   spread=True,  kind="group_gemm"),
    "gate_up_b64":     dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=64,   spread=True,  kind="group_gemm"),
    "down_b64":        dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=64,   spread=True,  kind="group_gemm"),
    # gemv (intended decode path; not currently dispatched at b=1)
    "gate_up_gemv":    dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=1,    spread=False, kind="gemv"),
    "down_gemv":       dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=8,    spread=False, kind="gemv"),
    # gemv at multi-token batch — what we'd pay if we dispatched each verify
    # token through a separate gemv call (γ=1 → 2 calls, γ=2 → 3, γ=3 → 4, γ=4 → 5).
    # Each call does B=1 gemv (1 token × top_k=8 active experts).
    # The existing _DequantGemvModule takes B = num_tokens and indptr (1, top_k),
    # so we can also bench "single kernel call processing N tokens through 1 fixed
    # expert set" — useful as a lower bound on the small-batch kernel.
    "gate_up_gemv_b2": dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=2,    spread=False, kind="gemv"),
    "gate_up_gemv_b3": dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=3,    spread=False, kind="gemv"),
    "gate_up_gemv_b5": dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=5,    spread=False, kind="gemv"),
    "down_gemv_b2":    dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=2,    spread=False, kind="gemv"),
    "down_gemv_b3":    dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=3,    spread=False, kind="gemv"),
    "down_gemv_b5":    dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=5,    spread=False, kind="gemv"),
    # topk_softmax (MoE router): single-thread sequential scan at b=1
    "topk_softmax":    dict(Ne=256, N=0,    K=2048, group_size=0,  top_k=8, B=1,    spread=False, kind="topk_softmax"),
    # Dense q4f16_1 GEMVs — the shared expert path (no MoE indptr).
    # Routed via the regular dlight `dl.gpu.GEMV()` schedule.
    # gate||up uses N=2*512=1024 (two cols concatenated); down N=2048.
    "shared_expert_gate_up": dict(Ne=0, N=1024, K=2048, group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    "shared_expert_down":    dict(Ne=0, N=2048, K=512,  group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    # GDN dense matmul (in_proj_qkv): K=2048 hidden -> N=8192. Confirmed from
    # phase4 IR: weight (8192, 256) uint32, scale (8192, 64) fp16. ~9.46 MB/call.
    # v3 production: 72.0 µs/call × 30 calls/tok = 2.16 ms/tok at 73% BW.
    "gdn_in_proj_qkv":       dict(Ne=0, N=8192, K=2048, group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    # Attention o_proj (full-attn + GDN): K=4096 (heads*head_dim*2 with attn_output_gate), N=2048.
    # v3 production: 37.8 µs/call × 40 calls/tok = 1.51 ms/tok at 69% BW.
    "attn_o_proj":           dict(Ne=0, N=2048, K=4096, group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    # lm_head: K=2048 → N=248064 (vocab). Already 90% BW per v3 profile.
    "lm_head":               dict(Ne=0, N=248064, K=2048, group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    # GDN in_proj_z (with silu*multiply epilogue): K=2048, N=4096.
    # v3 production: 42.5 µs/call × 30 calls = 1.27 ms/tok at 62% BW.
    # NOTE: bench harness only fuses dequant+matmul, not the silu/multiply.
    "gdn_in_proj_z":         dict(Ne=0, N=4096, K=2048, group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    # ===== FT (CUTLASS FpAIntB) candidates at g=64 — Phase 2D microbench =====
    # group_size=64 is the production constraint (FineGrainedScaleZeroIterator
    # hard-bakes group_size/64). Compare medians vs the q4f16_1 g=32 dlight
    # baselines above.
    "ft_shared_expert_gate_up": dict(Ne=0, N=1024,   K=2048, group_size=64, top_k=0, B=1, spread=False, kind="ft_dense_gemv"),
    "ft_shared_expert_down":    dict(Ne=0, N=2048,   K=512,  group_size=64, top_k=0, B=1, spread=False, kind="ft_dense_gemv"),
    "ft_gdn_in_proj_qkv":       dict(Ne=0, N=8192,   K=2048, group_size=64, top_k=0, B=1, spread=False, kind="ft_dense_gemv"),
    "ft_attn_o_proj":           dict(Ne=0, N=2048,   K=4096, group_size=64, top_k=0, B=1, spread=False, kind="ft_dense_gemv"),
    "ft_gdn_in_proj_z":         dict(Ne=0, N=4096,   K=2048, group_size=64, top_k=0, B=1, spread=False, kind="ft_dense_gemv"),
    "ft_lm_head":               dict(Ne=0, N=248064, K=2048, group_size=64, top_k=0, B=1, spread=False, kind="ft_dense_gemv"),
    # ===== §9 item 5: is the o_proj/down gap shape-driven or schedule-driven? =====
    # §4.6 has o_proj (N=2048, K=4096) at 74% of the 156 GB/s wall and routed-expert
    # down (N=2048, K=512) at 76%, against 88% for the same schedule at K=2048. The
    # sm_87 branch of dlight's GEMV rule pins TS,TR = 32,16 with no K term at all, so
    # K and the schedule are confounded in every number measured so far. These sweep
    # K at FIXED N=2048 — same output width, same block count, only the reduction
    # depth moves — which separates the two. Pair with MLC_GEMV_TSTR to vary the
    # schedule at a fixed shape.
    "ksweep_n2048_k512":   dict(Ne=0, N=2048, K=512,   group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    "ksweep_n2048_k1024":  dict(Ne=0, N=2048, K=1024,  group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    "ksweep_n2048_k2048":  dict(Ne=0, N=2048, K=2048,  group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    "ksweep_n2048_k4096":  dict(Ne=0, N=2048, K=4096,  group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    "ksweep_n2048_k8192":  dict(Ne=0, N=2048, K=8192,  group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    # N-sweep at fixed K=2048 — the control. If bandwidth tracks N (block count) and
    # not K, the o_proj gap is not about reduction depth at all.
    "nsweep_k2048_n512":   dict(Ne=0, N=512,   K=2048, group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    "nsweep_k2048_n1024":  dict(Ne=0, N=1024,  K=2048, group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    "nsweep_k2048_n2048":  dict(Ne=0, N=2048,  K=2048, group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    "nsweep_k2048_n4096":  dict(Ne=0, N=4096,  K=2048, group_size=32, top_k=0, B=1, spread=False, kind="dense_gemv"),
    # ===== §9 item 0e: the prefill-scale B sweep for the MoE expert GEMM =====
    # §16.7 put `dequantize_group_gemm_v2` + `_v21` at 52.5% of 35B pp512 prefill
    # — 6.9 ms/call at B = 512 tokens × top_k 8 = 4096 rows. Every group_gemm entry
    # above stops at B=64, where §16.4 found the cost flat and concluded "51× worse
    # than gemv"; that conclusion is about decode. These extend the sweep to the
    # shape prefill actually runs, and past it.
    #
    # The sweep is the discriminator. v2's grid is
    #     UPPER = (ceildiv(B, BLK_M=16) + Ne) * (N / BLK_N=128)
    # so the `+ Ne` padding term is a *fixed* block count whose share falls as B
    # grows. Three hypotheses make three different curves:
    #   block-count bound  → ms ∝ UPPER, i.e. ms/row falls steeply with B
    #   weight-BW bound    → ms flat (the 302 MB expert set is read once regardless)
    #   compute bound      → ms ∝ B once every expert is populated
    # B=8192 is past prefill's shape and is here only to extend the lever arm.
    "gate_up_b512":    dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=512,  spread=True, kind="group_gemm"),
    "down_b512":       dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=512,  spread=True, kind="group_gemm"),
    "gate_up_b2048":   dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=2048, spread=True, kind="group_gemm"),
    "down_b2048":      dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=2048, spread=True, kind="group_gemm"),
    # The production prefill shape: 512 tokens × top_k 8.
    "gate_up_b4096":   dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=4096, spread=True, kind="group_gemm"),
    "down_b4096":      dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=4096, spread=True, kind="group_gemm"),
    "gate_up_b8192":   dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=8192, spread=True, kind="group_gemm"),
    "down_b8192":      dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=8192, spread=True, kind="group_gemm"),
    # The same production shape under uniform-random routing instead of a perfectly
    # even division. The kernel built is byte-identical — only the indptr differs —
    # but the real/padding CTA split does not, and that split is what any fix to
    # the padding CTAs would be worth. Also the L2 control for the whole sweep: if
    # `spread=True`'s numbers were an artifact of tidy addressing, these would move.
    "gate_up_b4096_rand": dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=4096, spread="random", kind="group_gemm"),
    "down_b4096_rand":    dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=4096, spread="random", kind="group_gemm"),
}


class _DequantGroupGemmModule(nn.Module):
    def __init__(self, group_size: int):
        super().__init__()
        self.group_size = group_size

    def forward(self, x, w, scale, indptr):
        return dequantize_group_gemm(
            x, w, scale, indptr,
            quantize_dtype="int4",
            indptr_dtype="int32",
            group_size=self.group_size,
        )


class _DequantGemvModule(nn.Module):
    def __init__(self, group_size: int):
        super().__init__()
        self.group_size = group_size

    def forward(self, x, w, scale, indptr):
        return dequantize_gemv(
            x, w, scale, indptr,
            quantize_dtype="int4",
            group_size=self.group_size,
        )


class _TopKSoftmaxModule(nn.Module):
    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        # gating_softmax_topk dispatches to the custom top{k}_softmax kernel
        # when norm_topk_prob=True (the Qwen3.6 default).
        return gating_softmax_topk(x, k=self.top_k, norm_topk_prob=True)


class _DenseGemvModule(nn.Module):
    """Single q4f16_1 dense GEMV (the shared-expert path)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        config = QUANTIZATION["q4f16_1"]
        self.linear = GroupQuantizeLinear(
            in_features=in_features,
            out_features=out_features,
            config=config,
            bias=False,
            out_dtype=None,
        )

    def forward(self, x):
        return self.linear(x)


class _FTDenseGemvModule(nn.Module):
    """Single q4f16_ft_g64 dense GEMV via CUTLASS FpAIntB extern."""

    def __init__(self, in_features: int, out_features: int, group_size: int):
        super().__init__()
        config = QUANTIZATION[f"q4f16_ft_g{group_size}"]
        self.linear = FTQuantizeLinear(
            in_features=in_features,
            out_features=out_features,
            config=config,
            bias=False,
            out_dtype=None,
        )

    def forward(self, x):
        return self.linear(x)


def _arm_cuda_source_dump(path: str) -> None:
    """Capture the CUDA that codegen emits, so "does it use tensor cores" is answerable.

    Via the codegen postproc hook rather than by walking `ex.mod.imported_modules`:
    the built artifact is a `VMExecutable` whose `.mod` exposes neither `type_key`
    nor `imported_modules` here, so the walk finds nothing and reports success.
    Registration is global, so this fires for whichever build runs next.
    """

    @tvm.ffi.register_global_func("tvm_callback_cuda_postproc", override=True)
    def _postproc(code, target):  # pylint: disable=unused-argument
        Path(path).write_text(code)
        print(f"[bench] device source ({len(code)} chars) -> {path}")
        return code


def build_vm(Ne: int, N: int, K: int, group_size: int, top_k: int, B: int,
             spread, kind: str, target, dev, dump_source: str | None = None):
    if dump_source:
        _arm_cuda_source_dump(dump_source)
    if kind == "topk_softmax":
        # gating_softmax_topk takes (B, num_experts) "gate logits" and returns
        # (top-k weights, top-k indices). K here re-purposed as num_experts? No:
        # for this shape we use Ne as num_experts. Inputs: x = (B, Ne).
        mod_spec = {
            "forward": {
                "x": spec.Tensor([B, Ne], "float16"),
            }
        }
        m = _TopKSoftmaxModule(top_k)
    elif kind == "dense_gemv":
        # Plain q4f16_1 Linear: x (B,K) → out (B,N).
        # B=1 (static): mirrors decode-path kernels that specialize seq_len=1.
        # Production uses static-shape kernels (the v2 batch_decode fix pinned
        # batch_size=1, propagating to all decode kernels), so the inner
        # reduction goes through gemv.py — NOT low_batch_gemv. Use static
        # shape here to match production scheduling.
        mod_spec = {
            "forward": {
                "x": spec.Tensor([B, K], "float16"),
            }
        }
        m = _DenseGemvModule(in_features=K, out_features=N)
    elif kind == "ft_dense_gemv":
        # FT path via CUTLASS FpAIntB (libfpA_intB_gemm.so). q_weight is int8
        # storage (2 elts/byte for int4) shape (K, N/2); q_scale is fp16 shape
        # (K/group_size, N).
        mod_spec = {
            "forward": {
                "x": spec.Tensor([B, K], "float16"),
            }
        }
        m = _FTDenseGemvModule(in_features=K, out_features=N, group_size=group_size)
    elif kind == "gemv":
        # gemv: indptr is (1, top_k); x is (B, K) where B in {1, top_k}
        mod_spec = {
            "forward": {
                "x":      spec.Tensor([B, K],                   "float16"),
                "w":      spec.Tensor([Ne, N, K // 8],          "uint32"),
                "scale":  spec.Tensor([Ne, N, K // group_size], "float16"),
                "indptr": spec.Tensor([1, top_k],               "int32"),
            }
        }
        m = _DequantGemvModule(group_size)
    else:
        mod_spec = {
            "forward": {
                "x":      spec.Tensor([B, K],                   "float16"),
                "w":      spec.Tensor([Ne, N, K // 8],          "uint32"),
                "scale":  spec.Tensor([Ne, N, K // group_size], "float16"),
                "indptr": spec.Tensor([Ne + 1],                 "int32"),
            }
        }
        m = _DequantGroupGemmModule(group_size)
    mod, _ = m.export_tvm(spec=mod_spec)
    # Mirror the production lowering pipeline (compiler_pass/pipeline.py)
    # so dequantize+permute_dims+matmul fuse into the same single kernel that
    # runs at decode (`fused_dequantize*_NT_matmul*`).
    from mlc_llm.compiler_pass.fuse_dequantize_transpose import FuseDequantizeTranspose
    from mlc_llm.compiler_pass.fuse_transpose_matmul import FuseTransposeMatmul
    from mlc_llm.compiler_pass.fuse_dequantize_matmul_ewise import FuseDequantizeMatmulEwise
    from mlc_llm.compiler_pass.fuse_ft_dequantize_matmul_epilogue import FuseFTDequantizeEpilogue
    from mlc_llm.compiler_pass.low_batch_specialization import LowBatchGemvSpecialize
    with target:
        mod = FuseFTDequantizeEpilogue()(mod)
        mod = FuseDequantizeTranspose()(mod)
        mod = FuseTransposeMatmul()(mod)
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.AnnotateTIROpPattern()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.FuseOps()(mod)
        mod = relax.transform.FuseTIR()(mod)
        mod = FuseDequantizeMatmulEwise()(mod)
        mod = LowBatchGemvSpecialize()(mod)
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)
    ex = relax.build(mod, target=target)
    return relax.VirtualMachine(ex, dev)


def _upload(arr_np: np.ndarray, dev):
    a = tvm.runtime.empty(arr_np.shape, str(arr_np.dtype), dev)
    a.copyfrom(arr_np)
    return a


def make_inputs(Ne: int, N: int, K: int, group_size: int, top_k: int, B: int,
                spread, kind: str, dev, rng, indptr_np=None):
    if kind == "topk_softmax":
        # Single input: gate logits of shape (B, Ne).
        x_np = rng.standard_normal((B, Ne), dtype="float32").astype(np.float16)
        return [_upload(x_np, dev)]

    if kind == "dense_gemv":
        # q4f16_1 NK layout: q_weight (N, K/8) uint32, q_scale (N, K/group) fp16.
        x_np = rng.standard_normal((B, K), dtype="float32").astype(np.float16)
        w_np = rng.integers(0, 2**32, size=(N, K // 8), dtype=np.uint32)
        scale_np = (rng.standard_normal((N, K // group_size), dtype="float32") * 0.01).astype(np.float16)
        return [_upload(a, dev) for a in (x_np, w_np, scale_np)]

    if kind == "ft_dense_gemv":
        # FT layout: q_weight (K, N/2) int8 (2 int4 packed per byte),
        #            q_scale  (K/group_size, N) fp16.
        # Random values are fine for timing — kernel speed is data-independent.
        x_np = rng.standard_normal((B, K), dtype="float32").astype(np.float16)
        w_np = rng.integers(-128, 128, size=(K, N // 2), dtype=np.int8)
        scale_np = (rng.standard_normal((K // group_size, N), dtype="float32") * 0.01).astype(np.float16)
        return [_upload(a, dev) for a in (x_np, w_np, scale_np)]

    x_np = rng.standard_normal((B, K), dtype="float32").astype(np.float16)
    w_np = rng.integers(0, 2**32, size=(Ne, N, K // 8), dtype=np.uint32)
    scale_np = (rng.standard_normal((Ne, N, K // group_size), dtype="float32") * 0.01).astype(np.float16)
    if indptr_np is None:
        indptr_np = make_indptr(Ne, top_k, B, spread, kind, rng)

    return [_upload(a, dev) for a in (x_np, w_np, scale_np, indptr_np)]


def make_indptr(Ne: int, top_k: int, B: int, spread: bool, kind: str, rng):
    """The routing the timed call runs against.

    Split out of `make_inputs` so the roofline accounting below can count the
    experts and dispatch tiles this exact routing produces, instead of assuming
    `top_k` (which is right at decode and wrong at every prefill shape).
    """
    if kind == "gemv":
        # gemv expects (1, top_k) indptr listing the top-k chosen expert ids.
        return rng.choice(Ne, top_k, replace=False).astype(np.int32).reshape(1, top_k)
    if spread == "random":
        # Every row picks an expert uniformly at random. `spread=True` divides B
        # exactly evenly, which at B=4096/Ne=256 lands 16 rows on every expert —
        # exactly one BLK_M tile each, with zero rounding waste. No real router
        # is that tidy, and the rounding waste is precisely what decides how many
        # of v2's CTAs are padding, so this is the honest version of the shape.
        counts = np.bincount(rng.integers(0, Ne, size=B), minlength=Ne).astype(np.int32)
        indptr_np = np.zeros(Ne + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr_np[1:])
        return indptr_np
    if spread:
        indptr_np = np.zeros(Ne + 1, dtype=np.int32)
        rows_per_expert = np.full(Ne, B // Ne, dtype=np.int32)
        rows_per_expert[: B % Ne] += 1
        np.cumsum(rows_per_expert, out=indptr_np[1:])
        return indptr_np
    assert B == top_k, f"non-spread requires B==top_k (got B={B}, top_k={top_k})"
    indptr_np = np.zeros(Ne + 1, dtype=np.int32)
    active = set(rng.choice(Ne, top_k, replace=False).tolist())
    cum = 0
    for e in range(Ne):
        if e in active:
            cum += 1
        indptr_np[e + 1] = cum
    return indptr_np


# Tile geometry mirrored from `_dequantize_group_gemm_v2` in
# python/mlc_llm/op/moe_matmul.py — keep in sync if BLK_N moves there. BLK_M reads
# the same env var the kernel does, because a mirrored literal reported CTA counts
# for a grid the kernel was not launching as soon as the A/B knob existed.
V2_BLK_M = int(os.environ.get("MLC_MOE_GEMM_V2_BLKM", "16"))
V2_BLK_N = 128


def v2_grid(Ne: int, N: int, B: int, indptr_np) -> dict:
    """Replay v2's dispatch-table construction to count real vs padding CTAs.

    v2 launches `UPPER = (ceildiv(B, BLK_M) + Ne) * tiles_per_n` blocks, where the
    `+ Ne` gives every expert at least one private slack index so the per-expert
    m-tile runs cannot collide. Slack blocks get sentinel `te = -1`, and the
    kernel handles that by zeroing the X tile and predicating off the store — but
    it still loads and dequantizes a full W tile and still runs the full wmma
    matmul, so a padding CTA costs very nearly what a real one does.
    """
    tiles_per_n = N // V2_BLK_N
    ceil = lambda a, b: -(-a // b)  # noqa: E731
    real_m = pad_m = 0
    for e in range(Ne):
        sb = ceil(int(indptr_np[e]), V2_BLK_M) + e
        nb = ceil(int(indptr_np[e + 1]) - int(indptr_np[e]), V2_BLK_M)
        sb_next = ceil(int(indptr_np[e + 1]), V2_BLK_M) + e + 1
        real_m += nb
        pad_m += sb_next - (sb + nb)
    upper_m = ceil(B, V2_BLK_M) + Ne
    return {
        "cta_total": upper_m * tiles_per_n,
        "cta_real": real_m * tiles_per_n,
        "cta_pad": pad_m * tiles_per_n,
        "pad_frac": (pad_m * tiles_per_n) / max(1, upper_m * tiles_per_n),
        "experts_active": int(np.count_nonzero(np.diff(indptr_np))),
    }


# Measured achievable bandwidth on this box (§4.3) — NOT the 204.8 GB/s spec number.
# Every "% of wall" in the workplan is against this.
BW_WALL_GBS = 156.0

# Compute ceilings, DERIVED not measured — quote them as ceilings only.
#   CUDA cores: 2048 lanes × 2 (FMA) × 1300.5 MHz  = 5.33 TFLOP/s fp32
#   Tensor:     16 SM × 2048 FLOP/clk × 1300.5 MHz = 42.6 TFLOP/s fp16 (fp16 accum,
#               the GA10x rate; consistent with AGX Orin's 170 INT8 TOPS sparse
#               = 85 dense = 42.5 fp16). v2 accumulates in fp16, so this is its ceiling.
CUDA_FP32_TFLOPS = 5.33
TENSOR_FP16_TFLOPS = 42.6


def roofline(Ne: int, N: int, K: int, group_size: int, top_k: int, B: int,
             spread: bool, kind: str, median_ms: float, indptr_np=None) -> dict:
    """Bytes and FLOPs the kernel is obliged to do, and what fraction of each wall it reaches.

    Weight traffic dominates at b=1 and is the only term that differs between the
    b=1 shapes: int4 weights at K*N/2 bytes plus fp16 group scales. For the MoE kinds
    the weight term carries the number of *non-empty* experts under the routing
    actually being timed — which is `top_k` at decode but climbs to all `Ne` of them
    at any prefill-scale B, and using `top_k` there would understate the compulsory
    traffic by 32×. Padding-CTA re-reads are deliberately not counted: every one of
    them reads the same expert-0 slice, so they hit in L2 and are a *compute* cost,
    not a bandwidth one.
    """
    if kind == "topk_softmax" or median_ms <= 0:
        return {}
    if kind in ("gemv", "group_gemm"):
        experts = int(np.count_nonzero(np.diff(indptr_np))) if (
            indptr_np is not None and kind == "group_gemm") else top_k
    else:
        experts = 1
    w_bytes = experts * (N * K // 2)                      # int4
    if group_size:
        w_bytes += experts * (N * (K // group_size) * 2)  # fp16 scales
    act_bytes = B * K * 2 + B * N * 2
    total = w_bytes + act_bytes
    gbs = total / (median_ms * 1e-3) / 1e9
    out = {
        "bytes": total, "experts_active": experts,
        "gb_s": gbs, "pct_wall": 100.0 * gbs / BW_WALL_GBS,
    }
    # Useful arithmetic: every active row against its expert's full N×K weight.
    flop = 2.0 * B * N * K
    tflops = flop / (median_ms * 1e-3) / 1e12
    out.update({
        "gflop": flop / 1e9,
        "tflop_s": tflops,
        "pct_cuda_peak": 100.0 * tflops / CUDA_FP32_TFLOPS,
        "pct_tensor_peak": 100.0 * tflops / TENSOR_FP16_TFLOPS,
    })
    return out


def time_kernel(Ne: int, N: int, K: int, group_size: int, top_k: int, B: int,
                spread: bool, kind: str, target, dev, repeats: int, number: int,
                dump_source: str | None = None):
    vm = build_vm(Ne, N, K, group_size, top_k, B, spread, kind, target, dev,
                  dump_source=dump_source)
    rng = np.random.default_rng(seed=42)
    # Drawn once, here, and handed to both the timed call and the accounting below —
    # under `spread="random"` two independent draws would be two different routings,
    # and the CTA split reported would not be the one measured.
    indptr_np = (make_indptr(Ne, top_k, B, spread, kind, np.random.default_rng(seed=7))
                 if kind == "group_gemm" else None)
    inputs = make_inputs(Ne, N, K, group_size, top_k, B, spread, kind, dev, rng,
                         indptr_np=indptr_np)

    for _ in range(3):
        vm["forward"](*inputs)
    dev.sync()

    timer = vm.module.time_evaluator("forward", dev, number=number, repeat=repeats)
    r = timer(*inputs)
    out = {
        "shape": [Ne, N, K, group_size, top_k, B],
        "median_ms": r.median * 1000.0,
        "min_ms":    r.min * 1000.0,
        "std_ms":    r.std * 1000.0,
    }
    if indptr_np is not None:
        out.update(v2_grid(Ne, N, B, indptr_np))
    out.update(roofline(Ne, N, K, group_size, top_k, B, spread, kind,
                        out["median_ms"], indptr_np))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default=None,
                   help="CUDA arch (e.g. sm_87 for Orin). Default: auto-detect.")
    p.add_argument("--baseline", help="Compare median vs this saved JSON.")
    p.add_argument("--save", help="Save current medians to this JSON.")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--number", type=int, default=50)
    p.add_argument("--shapes", default="gate_up,down",
                   help="Subset of " + ",".join(SHAPES.keys()))
    p.add_argument("--dump-source", metavar="PATH",
                   help="Write the emitted CUDA/PTX for the LAST shape to PATH.")
    args = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    v2 = os.environ.get("MLC_MOE_GEMM_V2", "0") == "1"
    print(f"[bench] target={target.export()}")
    # §8's trap in bench form: without the flag the group_gemm shapes silently
    # measure the v1 persistent-loop fallback, which is a different kernel.
    print(f"[bench] MLC_MOE_GEMM_V2={'1 (v2 wmma dispatch-table)' if v2 else '0 (v1 persistent loop)'}")

    names = [s.strip() for s in args.shapes.split(",") if s.strip()]
    results: dict = {}
    for i, name in enumerate(names):
        if name not in SHAPES:
            print(f"unknown shape '{name}'", file=sys.stderr)
            sys.exit(1)
        cfg = SHAPES[name]
        print(f"[bench] {name}  kind={cfg['kind']}  Ne={cfg['Ne']} N={cfg['N']} "
              f"K={cfg['K']} g={cfg['group_size']} top_k={cfg['top_k']} B={cfg['B']}")
        dump = args.dump_source if (args.dump_source and i == len(names) - 1) else None
        r = time_kernel(**cfg, target=target, dev=dev,
                        repeats=args.repeats, number=args.number, dump_source=dump)
        results[name] = r
        bw = (f"  {r['gb_s']:6.1f} GB/s = {r['pct_wall']:5.1f}% of the {BW_WALL_GBS:.0f} wall"
              if "gb_s" in r else "")
        print(f"        median={r['median_ms']:.3f} ms  "
              f"min={r['min_ms']:.3f} ms  std={r['std_ms']:.3f} ms{bw}")
        if "tflop_s" in r:
            print(f"        {r['gflop']:8.2f} GFLOP -> {r['tflop_s']:6.3f} TFLOP/s = "
                  f"{r['pct_tensor_peak']:5.1f}% of the {TENSOR_FP16_TFLOPS:.1f} tensor ceiling "
                  f"({r['pct_cuda_peak']:5.1f}% of the {CUDA_FP32_TFLOPS:.2f} fp32 CUDA-core one), "
                  f"{r['experts_active']} experts touched")
        if v2 and "cta_total" in r:
            print(f"        v2 grid: {r['cta_total']:6d} CTAs, {r['cta_real']:6d} real + "
                  f"{r['cta_pad']:6d} padding ({100.0 * r['pad_frac']:.1f}% wasted), "
                  f"{1e3 * r['median_ms'] / max(1, r['cta_total']):.3f} us/CTA")

    if args.baseline:
        print()
        try:
            base = json.loads(Path(args.baseline).read_text())
        except FileNotFoundError:
            print(f"[bench] baseline missing: {args.baseline}", file=sys.stderr)
            sys.exit(1)
        print(f"=== vs baseline {args.baseline} ===")
        for name, r in results.items():
            if name not in base:
                print(f"  {name}: NEW (no baseline entry)")
                continue
            cur = r["median_ms"]
            ref = base[name]["median_ms"]
            speedup = ref / cur
            delta_pct = (ref - cur) / ref * 100.0
            print(f"  {name}:  {cur:.3f} ms  (baseline {ref:.3f} ms)  "
                  f"{speedup:.2f}× ({delta_pct:+.1f}%)")

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"\n[bench] saved -> {args.save}")


if __name__ == "__main__":
    main()

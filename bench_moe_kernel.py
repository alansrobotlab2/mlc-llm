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


def build_vm(Ne: int, N: int, K: int, group_size: int, top_k: int, B: int,
             spread: bool, kind: str, target, dev):
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
                spread: bool, kind: str, dev, rng):
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

    if kind == "gemv":
        # gemv expects (1, top_k) indptr listing the top-k chosen expert ids.
        chosen = rng.choice(Ne, top_k, replace=False).astype(np.int32)
        indptr_np = chosen.reshape(1, top_k)
    elif spread:
        indptr_np = np.zeros(Ne + 1, dtype=np.int32)
        rows_per_expert = np.full(Ne, B // Ne, dtype=np.int32)
        rows_per_expert[: B % Ne] += 1
        np.cumsum(rows_per_expert, out=indptr_np[1:])
    else:
        assert B == top_k, f"non-spread requires B==top_k (got B={B}, top_k={top_k})"
        indptr_np = np.zeros(Ne + 1, dtype=np.int32)
        active = set(rng.choice(Ne, top_k, replace=False).tolist())
        cum = 0
        for e in range(Ne):
            if e in active:
                cum += 1
            indptr_np[e + 1] = cum

    return [_upload(a, dev) for a in (x_np, w_np, scale_np, indptr_np)]


# Measured achievable bandwidth on this box (§4.3) — NOT the 204.8 GB/s spec number.
# Every "% of wall" in the workplan is against this.
BW_WALL_GBS = 156.0


def bandwidth(Ne: int, N: int, K: int, group_size: int, top_k: int, B: int,
              kind: str, median_ms: float) -> dict:
    """Bytes the kernel is obliged to move, and what fraction of the wall that reaches.

    Weight traffic dominates at b=1 and is the only term that differs between the
    shapes under test: int4 weights at K*N/2 bytes plus fp16 group scales. Activations
    are counted too but are noise at these sizes. For the MoE kinds only the *active*
    experts are touched, so the weight term carries top_k, not Ne.
    """
    if kind == "topk_softmax" or median_ms <= 0:
        return {}
    experts = top_k if kind in ("gemv", "group_gemm") else 1
    w_bytes = experts * (N * K // 2)                      # int4
    if group_size:
        w_bytes += experts * (N * (K // group_size) * 2)  # fp16 scales
    act_bytes = B * K * 2 + B * N * 2
    total = w_bytes + act_bytes
    gbs = total / (median_ms * 1e-3) / 1e9
    return {"bytes": total, "gb_s": gbs, "pct_wall": 100.0 * gbs / BW_WALL_GBS}


def time_kernel(Ne: int, N: int, K: int, group_size: int, top_k: int, B: int,
                spread: bool, kind: str, target, dev, repeats: int, number: int):
    vm = build_vm(Ne, N, K, group_size, top_k, B, spread, kind, target, dev)
    rng = np.random.default_rng(seed=42)
    inputs = make_inputs(Ne, N, K, group_size, top_k, B, spread, kind, dev, rng)

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
    out.update(bandwidth(Ne, N, K, group_size, top_k, B, kind, out["median_ms"]))
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
    args = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    print(f"[bench] target={target.export()}")

    names = [s.strip() for s in args.shapes.split(",") if s.strip()]
    results: dict = {}
    for name in names:
        if name not in SHAPES:
            print(f"unknown shape '{name}'", file=sys.stderr)
            sys.exit(1)
        cfg = SHAPES[name]
        print(f"[bench] {name}  kind={cfg['kind']}  Ne={cfg['Ne']} N={cfg['N']} "
              f"K={cfg['K']} g={cfg['group_size']} top_k={cfg['top_k']} B={cfg['B']}")
        r = time_kernel(**cfg, target=target, dev=dev,
                        repeats=args.repeats, number=args.number)
        results[name] = r
        bw = (f"  {r['gb_s']:6.1f} GB/s = {r['pct_wall']:5.1f}% of the {BW_WALL_GBS:.0f} wall"
              if "gb_s" in r else "")
        print(f"        median={r['median_ms']:.3f} ms  "
              f"min={r['min_ms']:.3f} ms  std={r['std_ms']:.3f} ms{bw}")

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

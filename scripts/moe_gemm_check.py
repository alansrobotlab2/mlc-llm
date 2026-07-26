#!/usr/bin/env python3
"""moe_gemm_check.py — the numerical gate for `dequantize_group_gemm_v2` changes.

Item 0f (§16.10) skips the dispatch table's padding CTAs by giving the reduction
loop a zero trip count when `te[bx] < 0`. The claim behind it is not "close
enough", it is that those CTAs never produced anything observable — so the bar is
**exact equality**, the same bar §16.6's `vb_exact` set for the GDN re-gridding,
not a tolerance.

Builds the kernel twice in one process, `MLC_MOE_GEMM_V2_SKIPPAD` off then on,
runs both on identical inputs and requires `np.array_equal`. Any nonzero
difference means the rewrite skipped a live tile.

Routings matter here and are swept deliberately:
  even   — B divides across Ne exactly; every expert lands on one BLK_M tile, which
           is the *maximum* padding share and the best case for the change
  random — each row picks an expert uniformly; ragged counts, fewer padding CTAs
  B=777  — not a multiple of BLK_M or Ne, so tiles are partially filled and the
           store predicate is doing real work on the last tile of most experts

Usage:
    source .envrc.local
    python scripts/moe_gemm_check.py            # gate + A/B at B=4096
    python scripts/moe_gemm_check.py --quick    # correctness only, no timing
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ.setdefault("MLC_MOE_GEMM_V2", "1")

import numpy as np
import tvm
from tvm import relax
from tvm.s_tir import dlight as dl
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec

import mlc_llm.op.moe_matmul as mm

# 35B-A3B q4f16_1 MoE shapes: hidden 2048, moe_intermediate 512, Ne 256, top_k 8.
NE, GROUP_SIZE = 256, 32
SHAPES = {"gate_up": (1024, 2048), "down": (2048, 512)}


class _Mod(nn.Module):
    def forward(self, x, w, scale, indptr):
        return mm.dequantize_group_gemm(
            x, w, scale, indptr, quantize_dtype="int4",
            indptr_dtype="int32", group_size=GROUP_SIZE,
        )


def build(N: int, K: int, B: int, target, dev):
    mod, _ = _Mod().export_tvm(spec={"forward": {
        "x": spec.Tensor([B, K], "float16"),
        "w": spec.Tensor([NE, N, K // 8], "uint32"),
        "scale": spec.Tensor([NE, N, K // GROUP_SIZE], "float16"),
        "indptr": spec.Tensor([NE + 1], "int32")}})
    from mlc_llm.compiler_pass.fuse_dequantize_transpose import FuseDequantizeTranspose
    from mlc_llm.compiler_pass.fuse_transpose_matmul import FuseTransposeMatmul
    from mlc_llm.compiler_pass.fuse_dequantize_matmul_ewise import FuseDequantizeMatmulEwise
    from mlc_llm.compiler_pass.low_batch_specialization import LowBatchGemvSpecialize
    with target:
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
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
    return relax.VirtualMachine(relax.build(mod, target=target), dev)


def make_inputs(N: int, K: int, B: int, routing: str, dev, seed: int = 0):
    rng = np.random.default_rng(seed)
    if routing == "random":
        counts = np.bincount(rng.integers(0, NE, size=B), minlength=NE).astype(np.int32)
    else:
        counts = np.full(NE, B // NE, dtype=np.int32)
        counts[: B % NE] += 1
    indptr = np.zeros(NE + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    uploaded = []
    for a in (rng.standard_normal((B, K), dtype="float32").astype(np.float16),
              rng.integers(0, 2**32, size=(NE, N, K // 8), dtype=np.uint32),
              (rng.standard_normal((NE, N, K // GROUP_SIZE), dtype="float32") * 0.01).astype(np.float16),
              indptr):
        t = tvm.runtime.empty(a.shape, str(a.dtype), dev)
        t.copyfrom(a)
        uploaded.append(t)
    return uploaded, indptr


def run(vm, args, dev, time_it: bool):
    out = vm["forward"](*args).numpy()
    dev.sync()
    ms = float("nan")
    if time_it:
        for _ in range(3):
            vm["forward"](*args)
        dev.sync()
        ms = vm.module.time_evaluator("forward", dev, number=20, repeat=3)(*args).median * 1e3
    return out, ms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Skip the A/B timing.")
    args_cli = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    print(f"[gate] target=sm_87  MLC_MOE_GEMM_V2={os.environ['MLC_MOE_GEMM_V2']}")

    cases = [(n, B, r) for n in SHAPES
             for B, r in ((8, "even"), (4096, "even"), (4096, "random"), (777, "random"))]
    failures = 0
    for name, B, routing in cases:
        N, K = SHAPES[name]
        args, indptr = make_inputs(N, K, B, routing, dev)
        timed = (B >= 4096) and not args_cli.quick
        os.environ["MLC_MOE_GEMM_V2_SKIPPAD"] = "0"
        base_o, base_ms = run(build(N, K, B, target, dev), args, dev, timed)
        os.environ["MLC_MOE_GEMM_V2_SKIPPAD"] = "1"
        skip_o, skip_ms = run(build(N, K, B, target, dev), args, dev, timed)
        exact = np.array_equal(base_o, skip_o)
        failures += 0 if exact else 1
        verdict = "YES" if exact else f"NO ({int((base_o != skip_o).sum())} elems)"
        speed = f"  {base_ms:7.3f} -> {skip_ms:7.3f} ms  {base_ms / skip_ms:.2f}x" if timed else ""
        print(f"{name:8} B={B:<5} {routing:7} exact={verdict:15}"
              f" experts={int(np.count_nonzero(np.diff(indptr))):3d}{speed}")

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} case(s) not bit-exact)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

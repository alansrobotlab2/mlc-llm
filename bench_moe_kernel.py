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
    # gemv (intended decode path; not currently dispatched at b=1)
    "gate_up_gemv":    dict(Ne=256, N=1024, K=2048, group_size=32, top_k=8, B=1,    spread=False, kind="gemv"),
    "down_gemv":       dict(Ne=256, N=2048, K=512,  group_size=32, top_k=8, B=8,    spread=False, kind="gemv"),
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


def build_vm(Ne: int, N: int, K: int, group_size: int, top_k: int, B: int,
             spread: bool, kind: str, target, dev):
    if kind == "gemv":
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
    with target:
        mod = relax.transform.LegalizeOps()(mod)
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
    return {
        "shape": [Ne, N, K, group_size, top_k, B],
        "median_ms": r.median * 1000.0,
        "min_ms":    r.min * 1000.0,
        "std_ms":    r.std * 1000.0,
    }


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
        print(f"        median={r['median_ms']:.3f} ms  "
              f"min={r['min_ms']:.3f} ms  std={r['std_ms']:.3f} ms")

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

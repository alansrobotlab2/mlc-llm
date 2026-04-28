#!/usr/bin/env python3
"""bench_gdn_kernel.py — microbench gdn_func at 35B-A3B decode shapes.

Times the GatedDeltaNet recurrent TIR kernel directly so kernel rewrites
(register-caching, etc.) can be iterated in seconds rather than via full
compile + e2e bench.

Usage:
    .venv/bin/python bench_gdn_kernel.py
    .venv/bin/python bench_gdn_kernel.py --save baseline_gdn.json
    .venv/bin/python bench_gdn_kernel.py --baseline baseline_gdn.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import tvm
import tvm.runtime
from tvm import relax
from tvm.s_tir import dlight as dl
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec
from tvm.relax.frontend.nn import Tensor as NNTensor
from tvm.relax.frontend.nn import op as nn_op

from mlc_llm.model.qwen35.qwen35_model import create_gated_delta_net_func


# 35B-A3B decode shapes: 32 value heads, K=V=128, batch=1, seq_len=1.
# Same kernel runs for the 0.8B model with 16 heads.
SHAPES = {
    "gdn_decode_35B": dict(num_kh=32, num_vh=32, K=128, V=128, B=1, S=1),
    "gdn_decode_0.8B": dict(num_kh=16, num_vh=16, K=128, V=128, B=1, S=1),
    # Spec-decode verify shapes (γ+1 ≈ 5 tokens at once):
    "gdn_verify_35B_s5": dict(num_kh=32, num_vh=32, K=128, V=128, B=1, S=5),
    # Prefill-ish (longer seq, same kernel):
    "gdn_prefill_35B_s128": dict(num_kh=32, num_vh=32, K=128, V=128, B=1, S=128),
}


class _GdnModule(nn.Module):
    """Wraps create_gated_delta_net_func into a callable nn.Module."""

    def __init__(self, num_kh, num_vh, K, V, dtype="float16"):
        super().__init__()
        self.num_kh = num_kh
        self.num_vh = num_vh
        self.K = K
        self.V = V
        self.dtype = dtype

    def forward(self, q, k, v, gate, beta, state_in):
        b, s = q.shape[0], q.shape[1]
        n_kh, n_vh, K, V = self.num_kh, self.num_vh, self.K, self.V
        out_recurrent, state_out = nn_op.tensor_ir_op(
            create_gated_delta_net_func(
                num_key_heads=n_kh,
                num_value_heads=n_vh,
                key_head_dim=K,
                value_head_dim=V,
                dtype=self.dtype,
            ),
            "gated_delta_net",
            [q, k, v, gate, beta, state_in],
            [
                NNTensor.placeholder([b, s, n_vh, V], "float32"),
                NNTensor.placeholder([b, n_vh, K, V], "float32"),
            ],
        )
        return out_recurrent, state_out


def build_vm(num_kh, num_vh, K, V, B, S, target, dev, dtype="float16"):
    m = _GdnModule(num_kh, num_vh, K, V, dtype=dtype)
    mod_spec = {
        "forward": {
            "q":        spec.Tensor([B, S, num_kh, K],     dtype),
            "k":        spec.Tensor([B, S, num_kh, K],     dtype),
            "v":        spec.Tensor([B, S, num_vh, V],     dtype),
            "gate":     spec.Tensor([B, S, num_vh],        "float32"),
            "beta":     spec.Tensor([B, S, num_vh],        "float32"),
            "state_in": spec.Tensor([B, num_vh, K, V],     "float32"),
        }
    }
    mod, _ = m.export_tvm(spec=mod_spec)
    with target:
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.AnnotateTIROpPattern()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.FuseOps()(mod)
        mod = relax.transform.FuseTIR()(mod)
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)
    ex = relax.build(mod, target=target)
    return relax.VirtualMachine(ex, dev)


def _upload(arr_np, dev):
    a = tvm.runtime.empty(arr_np.shape, str(arr_np.dtype), dev)
    a.copyfrom(arr_np)
    return a


def make_inputs(num_kh, num_vh, K, V, B, S, dev, rng, dtype="float16"):
    q = (rng.standard_normal((B, S, num_kh, K), dtype="float32") * 0.1).astype(dtype)
    k = (rng.standard_normal((B, S, num_kh, K), dtype="float32") * 0.1).astype(dtype)
    v = (rng.standard_normal((B, S, num_vh, V), dtype="float32") * 0.1).astype(dtype)
    gate = (rng.standard_normal((B, S, num_vh), dtype="float32") * 0.05).astype("float32")
    # exp(g) → make positive, near 1
    gate = np.exp(-np.abs(gate)).astype("float32")
    beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((B, S, num_vh), dtype="float32")))).astype("float32")
    state_in = (rng.standard_normal((B, num_vh, K, V), dtype="float32") * 0.1).astype("float32")
    return [_upload(a, dev) for a in (q, k, v, gate, beta, state_in)]


def time_kernel(num_kh, num_vh, K, V, B, S, target, dev, repeats, number, dtype="float16"):
    vm = build_vm(num_kh, num_vh, K, V, B, S, target, dev, dtype=dtype)
    rng = np.random.default_rng(seed=42)
    inputs = make_inputs(num_kh, num_vh, K, V, B, S, dev, rng, dtype=dtype)
    for _ in range(3):
        vm["forward"](*inputs)
    dev.sync()
    timer = vm.module.time_evaluator("forward", dev, number=number, repeat=repeats)
    r = timer(*inputs)
    return {
        "shape": [num_kh, num_vh, K, V, B, S],
        "median_us": r.median * 1e6,
        "min_us":    r.min * 1e6,
        "std_us":    r.std * 1e6,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", help="Compare median vs this saved JSON.")
    p.add_argument("--save", help="Save current medians to this JSON.")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--number", type=int, default=200)
    p.add_argument("--shapes", default="gdn_decode_35B")
    args = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    print(f"[bench] target={target.export()}")

    names = [s.strip() for s in args.shapes.split(",") if s.strip()]
    results = {}
    for name in names:
        cfg = SHAPES[name]
        r = time_kernel(**cfg, target=target, dev=dev,
                        repeats=args.repeats, number=args.number)
        results[name] = r
        line = f"  {name:<28} median={r['median_us']:8.2f} µs  min={r['min_us']:8.2f}  std={r['std_us']:6.2f}"
        if args.baseline:
            try:
                base = json.load(open(args.baseline))
                if name in base:
                    delta = (r["median_us"] - base[name]["median_us"]) / base[name]["median_us"] * 100
                    line += f"  Δ={delta:+.1f}% vs baseline"
            except Exception as e:
                line += f"  (baseline read failed: {e})"
        print(line)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bench] saved to {args.save}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Numerical parity: v2 topk_softmax kernel vs numpy reference.

Builds the kernel via the high-level gating_softmax_topk op (which dispatches
to v2 for Ne=256), runs on random fp16 inputs, compares to numpy.
"""
from __future__ import annotations

import sys

import numpy as np
import tvm
from tvm import relax
from tvm.s_tir import dlight as dl
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec

from mlc_llm.op.moe_misc import gating_softmax_topk


class _M(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return gating_softmax_topk(x, k=self.k, norm_topk_prob=True)


def numpy_topk_softmax(x: np.ndarray, k: int):
    # Pick top-k by raw logit, then softmax-normalize over those k values.
    # Tie-break by smallest index (matches v2's allreduce-min idx).
    B, Ne = x.shape
    out_v = np.zeros((B, k), dtype=np.float16)
    out_i = np.zeros((B, k), dtype=np.int32)
    for b in range(B):
        # argpartition would not break ties consistently, so do it explicitly.
        scores = x[b].astype(np.float32).copy()
        idxs = []
        vals = []
        for _ in range(k):
            best_v = scores.max()
            cand = np.where(scores >= best_v)[0]
            best_i = int(cand.min())
            idxs.append(best_i)
            vals.append(float(scores[best_i]))
            scores[best_i] = -np.inf
        vals_arr = np.array(vals, dtype=np.float32)
        m = vals_arr.max()
        e = np.exp(vals_arr - m)
        w = e / e.sum()
        out_v[b] = w.astype(np.float16)
        out_i[b] = np.array(idxs, dtype=np.int32)
    return out_v, out_i


def main():
    Ne, k = 256, 8
    target = tvm.target.Target.from_device(tvm.cuda(0))
    dev = tvm.cuda(0)

    rng = np.random.default_rng(seed=0)
    failures = 0
    for B in (1, 4, 32, 128):
        x_np = rng.standard_normal((B, Ne), dtype="float32").astype(np.float16)

        m = _M(k)
        mod_spec = {"forward": {"x": spec.Tensor([B, Ne], "float16")}}
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
        vm = relax.VirtualMachine(ex, dev)

        x_t = tvm.runtime.empty(x_np.shape, "float16", dev)
        x_t.copyfrom(x_np)
        v_t, i_t = vm["forward"](x_t)
        v = v_t.numpy()
        i = i_t.numpy()

        ref_v, ref_i = numpy_topk_softmax(x_np, k)

        ok_i = np.array_equal(i, ref_i)
        ok_v = np.allclose(v.astype(np.float32), ref_v.astype(np.float32),
                           rtol=1e-3, atol=1e-4)
        status = "OK " if (ok_i and ok_v) else "FAIL"
        print(f"  B={B:>3d}  Ne={Ne}  k={k}: idx={ok_i}  weights={ok_v}  [{status}]")
        if not ok_i:
            mism = np.where(i != ref_i)
            print(f"    idx mismatches: rows={set(mism[0].tolist())}")
            for r in list(set(mism[0].tolist()))[:3]:
                print(f"      row {r} got={i[r].tolist()} ref={ref_i[r].tolist()}")
            failures += 1
        if not ok_v:
            d = np.abs(v.astype(np.float32) - ref_v.astype(np.float32))
            print(f"    weight max-abs diff = {d.max():.4e}")
            failures += 1

    sys.exit(failures)


if __name__ == "__main__":
    main()

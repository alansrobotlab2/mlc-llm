#!/usr/bin/env python3
"""moe_skippad_ab.py — three-way A/B of item 0f's padding-CTA guard.

  0    no guard at all — every padding CTA runs a full dequant + wmma (§16.8)
  koo  §16.10/§16.11's shipped mechanism: zero trip count on the `k_o_o` reduction
       loop only. A skipped CTA still costs ~20% of a full one — the accumulator
       fill, the accumulator -> O_tile store, and the predicated-off global store
  1    the whole-body `moe_pad_guard` unit loop is zeroed as well, so a padding CTA
       runs nothing

All three must produce byte-identical output; `0` is the reference. The padding
share is a property of the routing, so both routings are swept: `even` is the
50%-padding best case at B=4096 and `random` the ~28% one (§16.8's brackets).

Usage:
    source .envrc.local
    python scripts/moe_skippad_ab.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ.setdefault("MLC_MOE_GEMM_V2", "1")

import numpy as np
import tvm

sys.path.insert(0, os.path.dirname(__file__))
from moe_gemm_check import SHAPES, build, load_real_counts, make_inputs, run  # noqa: E402

MODES = ["0", "koo", "1"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="4096", help="comma-separated batch sizes")
    p.add_argument("--indptr-file", default=None,
                   help=".npz from scripts/moe_expert_histogram.py. The padding share this "
                        "measures is the whole point of the flag, and the synthetic "
                        "routings get it wrong (workplan 17.9); B comes from the file")
    p.add_argument("--indptr-key", default=None, help="a single key from the .npz")
    p.add_argument("--indptr-picks", default="min,med,max",
                   help="which layers to take, ranked by tile count at BLK_M=16")
    cli = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    print(f"[skippad] target=sm_87  modes={MODES}")

    if cli.indptr_file:
        legs = [(int(c.sum()), lbl, c)
                for lbl, c in load_real_counts(cli.indptr_file, cli.indptr_key,
                                               cli.indptr_picks)]
    else:
        legs = [(B, r, None) for B in (int(v) for v in cli.batches.split(","))
                for r in ("even", "random")]

    failures = 0
    for name in SHAPES:
        N, K = SHAPES[name]
        for B, routing, counts in legs:
            args, indptr = make_inputs(N, K, B, routing, dev, counts=counts)
            ref_o, ref_ms, cells = None, None, []
            for mode in MODES:
                os.environ["MLC_MOE_GEMM_V2_SKIPPAD"] = mode
                out, ms = run(build(N, K, B, target, dev), args, dev, True)
                if ref_o is None:
                    ref_o, ref_ms = out, ms
                    cells.append(f"{mode}: {ms:7.3f} ms (ref)")
                    continue
                exact = np.array_equal(ref_o, out)
                failures += 0 if exact else 1
                tag = "exact" if exact else f"DIFF({int((ref_o != out).sum())})"
                cells.append(f"{mode}: {ms:7.3f} ms {ref_ms / ms:5.2f}x {tag}")
            print(f"{name:8} B={B:<5} {routing:16} " + " | ".join(cells))

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} case(s) not bit-exact)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

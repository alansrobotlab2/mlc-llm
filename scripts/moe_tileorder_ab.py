#!/usr/bin/env python3
"""moe_tileorder_ab.py — gate and A/B the v2 GEMM's dispatch-table tile order (item 0j).

Why
---
The real-routing roofline (item 0i) found this kernel issues **2.9x the bytes DRAM
actually supplies** at B=4096 — against 1.4x on the synthetic `even` routing every earlier
measurement used. The excess is L2's job, and how much of it L2 can absorb depends
entirely on the order CTAs visit tiles in.

Within one expert's private CTA range the dispatch table assigns `nb x tiles_per_n` tiles,
and the order is free:

  m-major (shipped)  off = tmi*tiles_per_n + tni
      Consecutive CTAs share `X_tile` and sweep the expert's whole `N x K` weight set,
      then sweep it again for the next row-tile. Each weight slice is re-fetched `nb`
      times with a reuse distance of the expert's entire weight footprint (1 MB for
      gate_up at these shapes) — and several experts are in flight at once against a
      4 MB L2.
  n-major            off = tmi + tni*nb
      Consecutive CTAs share the *weight* slice and vary rows, cutting the reuse
      distance to one slice. `X_tile` is then the re-fetched operand, but it is 2x
      smaller than a weight slice on gate_up and 8x smaller on down.

Both orders assign the identical set of `(e, m, n)` triples to disjoint output regions,
so **the bar is exact equality**, not a tolerance — the same bar item 0f and `BLK_M` are
held to. A difference would mean the permutation dropped or duplicated a tile.

Usage:
    source .envrc.local
    python scripts/moe_tileorder_ab.py --indptr-file tuning/expert_hist_35b.npz
    python scripts/moe_tileorder_ab.py                 # synthetic routings only
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ.setdefault("MLC_MOE_GEMM_V2", "1")
os.environ.setdefault("MLC_MOE_GEMM_V2_SKIPPAD", "1")

import numpy as np
import tvm

sys.path.insert(0, os.path.dirname(__file__))
from moe_gemm_check import SHAPES, build, load_real_counts, make_inputs, run  # noqa: E402

ORDERS = ["m", "n"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="4096,16384", help="comma-separated batch sizes")
    p.add_argument("--blkm", default="16", help="comma-separated BLK_M values to sweep")
    p.add_argument("--hoist", default="0", help="comma-separated HOIST values to sweep")
    p.add_argument("--indptr-file", default=None,
                   help=".npz from scripts/moe_expert_histogram.py. The synthetic routings "
                        "understate the L2 pressure this change targets by 2x, so a real "
                        "one is what decides it (workplan 17.9)")
    p.add_argument("--indptr-key", default=None, help="a single key from the .npz")
    p.add_argument("--indptr-picks", default="med",
                   help="which layers to take, ranked by tile count at BLK_M=16")
    cli = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)

    if cli.indptr_file:
        legs = [(int(c.sum()), lbl, c)
                for lbl, c in load_real_counts(cli.indptr_file, cli.indptr_key,
                                               cli.indptr_picks)]
    else:
        legs = [(B, r, None) for B in (int(v) for v in cli.batches.split(","))
                for r in ("even", "random")]

    blkms = [int(v) for v in cli.blkm.split(",")]
    hoists = cli.hoist.split(",")
    print(f"[tileorder] target=sm_87  orders={ORDERS}  blkm={blkms}  hoist={hoists}")

    failures = 0
    for name in SHAPES:
        N, K = SHAPES[name]
        for B, routing, counts in legs:
            args, indptr = make_inputs(N, K, B, routing, dev, counts=counts)
            for blkm in blkms:
                for hoist in hoists:
                    os.environ["MLC_MOE_GEMM_V2_BLKM"] = str(blkm)
                    os.environ["MLC_MOE_GEMM_V2_HOIST"] = hoist
                    ref_o, ref_ms, cells = None, None, []
                    for order in ORDERS:
                        os.environ["MLC_MOE_GEMM_V2_TILEORDER"] = order
                        out, ms = run(build(N, K, B, target, dev), args, dev, True)
                        if ref_o is None:
                            ref_o, ref_ms = out, ms
                            cells.append(f"{order}: {ms:7.3f} ms (ref)")
                            continue
                        exact = np.array_equal(ref_o, out)
                        failures += 0 if exact else 1
                        tag = "exact" if exact else f"DIFF({int((ref_o != out).sum())})"
                        cells.append(f"{order}: {ms:7.3f} ms {ref_ms / ms:5.2f}x {tag}")
                    print(f"{name:8} B={B:<6} {routing:18} M={blkm:<3} H={hoist} "
                          + " | ".join(cells))

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} case(s) not bit-exact)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

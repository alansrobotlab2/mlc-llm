#!/usr/bin/env python3
"""moe_skiprows_ab.py — gate and A/B skipping all-padding row fragments (item 0k).

Why
---
Item 0i measured a real pp512 prefill at **~23 rows per hit expert**. At `BLK_M=64` a
CTA covers 64 rows, so the expert's last tile carries 23 real rows and 41 padding ones.
The `X_shared` load is already predicated on `row_end`, so those rows cost no DRAM
traffic — but the wmma reduction still runs all `BLK_M/MICRO` row fragments, and the ones
past the real rows reduce nothing but zeros.

**That is the entire reason widening `BLK_M` has a short-prompt cost.** §17.10 measured
`BLK_M=64` at −10.4% on pp128 and concluded no compile-time width is Pareto. Padding-row
compute is the mechanism, and a wider tile creates more of it exactly when experts are
small. If this removes the short-prompt loss, a wide tile becomes shippable as a plain
default and item 0h needs no runtime branch at all.

The bar is **exact equality** — a skipped fragment's global store is predicated off by
`m_offset + i < row_end` regardless of what its accumulator holds, so nothing skipped was
ever observable. This is item 0f's argument applied one level down.

Usage:
    source .envrc.local
    python scripts/moe_skiprows_ab.py --indptr-file tuning/expert_hist_35b.npz
    python scripts/moe_skiprows_ab.py --blkm 16,32,64          # synthetic routings
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

MODES = ["0", "1"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="4096,16384", help="comma-separated batch sizes")
    p.add_argument("--blkm", default="16,32,64", help="comma-separated BLK_M values")
    p.add_argument("--hoist", default="1", help="comma-separated HOIST values")
    p.add_argument("--indptr-file", default=None,
                   help=".npz from scripts/moe_expert_histogram.py. Real routing is what "
                        "decides this: the synthetic `even` leg has no partial tiles at "
                        "all, so it cannot see the effect (workplan 17.9)")
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
    print(f"[skiprows] target=sm_87  blkm={blkms}  hoist={hoists}")

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
                    for mode in MODES:
                        os.environ["MLC_MOE_GEMM_V2_SKIPROWS"] = mode
                        out, ms = run(build(N, K, B, target, dev), args, dev, True)
                        if ref_o is None:
                            ref_o, ref_ms = out, ms
                            cells.append(f"{mode}: {ms:7.3f} ms (ref)")
                            continue
                        exact = np.array_equal(ref_o, out)
                        failures += 0 if exact else 1
                        tag = "exact" if exact else f"DIFF({int((ref_o != out).sum())})"
                        cells.append(f"{mode}: {ms:7.3f} ms {ref_ms / ms:5.2f}x {tag}")
                    print(f"{name:8} B={B:<6} {routing:18} M={blkm:<3} H={hoist} "
                          + " | ".join(cells))

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} case(s) not bit-exact)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

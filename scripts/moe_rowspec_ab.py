#!/usr/bin/env python3
"""moe_rowspec_ab.py — gate and A/B item 0l against item 0k and against neither.

Why
---
Item 0k skipped the row fragments that hold no real rows by rewriting the fragment
loop's *extent* to `ceildiv(real rows, MICRO)`. §18.7 measured it at **0.91x** at
`BLK_M=64` — and at `BLK_M=16`, where the guard can only ever produce extent 1 and is
doing nothing at all, it still lost **5-9%**. That is the whole diagnosis: the cost is
not the skipping, it is that a runtime extent stops the fragment loop unrolling with
constant indices, and every CTA pays it.

Item 0l keeps the extent at the compile-time `BLK_M // MICRO` and predicates the body
instead (`MLC_MOE_GEMM_V2_ROWSPEC=1`). The prediction is a `BLK_M=16` leg at ~1.00x —
the control that 0k fails — and a `BLK_M=64` leg above 1.0x.

Three legs per cell, all built in one process off identical inputs:

    base     SKIPROWS=0 ROWSPEC=0   the reference for both exactness and speed
    skiprows SKIPROWS=1             item 0k, so the two mechanisms are ranked under one
                                    clock state rather than across sessions
    rowspec  ROWSPEC=1              item 0l

The bar is **exact equality** against `base`, not a tolerance: a skipped or predicated
fragment's global store is predicated off by `m_offset + i < row_end` regardless of what
its accumulator holds, so nothing it elides was ever observable.

Usage:
    source .envrc.local
    python scripts/moe_rowspec_ab.py --indptr-file tuning/expert_hist_35b.npz
    python scripts/moe_rowspec_ab.py --blkm 16,32,64          # synthetic routings
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

# (label, SKIPROWS, ROWSPEC). `base` first: it is the reference for every comparison.
MODES = [("base", "0", "0"), ("skiprows", "1", "0"), ("rowspec", "0", "1")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="4096,16384", help="comma-separated batch sizes")
    p.add_argument("--blkm", default="16,32,64", help="comma-separated BLK_M values")
    p.add_argument("--modes", default="base,skiprows,rowspec",
                   help="which legs to run; `base` is always the reference")
    p.add_argument("--indptr-file", default=None,
                   help=".npz from scripts/moe_expert_histogram.py. Real routing is what "
                        "decides this: the synthetic `even` leg has no partial tiles at "
                        "all, so it cannot see the effect (workplan 17.9, 18.2)")
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
    wanted = [m.strip() for m in cli.modes.split(",")]
    modes = [m for m in MODES if m[0] in wanted]
    assert modes and modes[0][0] == "base", "`base` is the reference and must be included"
    # Item 0l predicates rather than shortens, so it needs the hoist to keep the
    # cooperative loads' __syncthreads() outside the predicated region. The hoist is also
    # the configuration every wide-BLK_M number in §18 was taken under.
    os.environ["MLC_MOE_GEMM_V2_HOIST"] = "1"
    print(f"[rowspec] target=sm_87  blkm={blkms}  hoist=1  modes={[m[0] for m in modes]}")

    failures = 0
    for name in SHAPES:
        N, K = SHAPES[name]
        for B, routing, counts in legs:
            args, indptr = make_inputs(N, K, B, routing, dev, counts=counts)
            for blkm in blkms:
                os.environ["MLC_MOE_GEMM_V2_BLKM"] = str(blkm)
                ref_o, ref_ms, cells = None, None, []
                for label, skiprows, rowspec in modes:
                    os.environ["MLC_MOE_GEMM_V2_SKIPROWS"] = skiprows
                    os.environ["MLC_MOE_GEMM_V2_ROWSPEC"] = rowspec
                    out, ms = run(build(N, K, B, target, dev), args, dev, True)
                    if ref_o is None:
                        ref_o, ref_ms = out, ms
                        cells.append(f"{label}: {ms:7.3f} ms (ref)")
                        continue
                    exact = np.array_equal(ref_o, out)
                    failures += 0 if exact else 1
                    tag = "exact" if exact else f"DIFF({int((ref_o != out).sum())})"
                    cells.append(f"{label}: {ms:7.3f} ms {ref_ms / ms:5.2f}x {tag}")
                print(f"{name:8} B={B:<6} {routing:18} M={blkm:<3} " + " | ".join(cells))

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} case(s) not bit-exact)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

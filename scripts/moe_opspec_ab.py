#!/usr/bin/env python3
"""moe_opspec_ab.py — gate and A/B item 0s, the padding rows' operand traffic.

Why
---
§21.3 decomposed `BLK_M=64`'s loss at B=1024, where both widths launch the same CTAs
over the same rows:

    BLK_M=64 total                              0.72x / 0.60x   (gate_up / down)
      occupancy 6 -> 4 CTAs                     0.89x / 0.80x   (instruction-identical
                                                                 shared-pad control)
      residual: the wide tile's own per-CTA work 0.81x / 0.75x

§19.3 had already bounded the padding *multiply* inside that residual at 3% (`ROWSPEC`),
so what is left is the traffic feeding it. Item 0s predicates the two producers:

    a   the A-fragment `load_matrix_sync` -- 4 per k-step at BLK_M=64, 1 at BLK_M=16
    x   the `X_shared` cooperative store  -- ditto; the global load is already
        predicated, the shared store and its `condval` are not

Both are swept separately because ranking them separately is the whole point: §21 exists
because four sections attributed this gap to a mechanism nobody had isolated.

The bar is **exact equality** against the shipped `BLK_M=16`, not a tolerance. A skipped
fragment's accumulator still reaches `O_tile`; its global store is predicated off by
`m_offset + i < row_end`, so nothing elided was ever observable.

⚠️ Do not rank on the synthetic routings alone. §18.2: they are wrong at B=4096
specifically, and B=1024 is the one shape where they are all that exists.

Usage:
    source .envrc.local
    python scripts/moe_opspec_ab.py                                    # B=1024, the 0n shape
    python scripts/moe_opspec_ab.py --indptr-file tuning/expert_hist_35b.npz
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ.setdefault("MLC_MOE_GEMM_V2", "1")
os.environ.setdefault("MLC_MOE_GEMM_V2_SKIPPAD", "1")

import numpy as np  # noqa: E402
import tvm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from moe_gemm_check import SHAPES, build, load_real_counts, make_inputs, run  # noqa: E402

# (label, BLK_M, ROWSPEC, OPSPEC). `m16` first: it is the reference for every comparison,
# for speed and for exactness alike -- BLK_M partitions rows and does not touch the k
# order, so every width is bit-identical (confirmed across §21's runs).
MODES = [
    ("m16", 16, "0", "0"),
    ("rowspec64", 64, "1", "0"),
    ("+a", 64, "1", "a"),
    ("+x", 64, "1", "x"),
    ("+ax", 64, "1", "1"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1024", help="comma-separated batch sizes")
    p.add_argument("--routings", default="even,random")
    p.add_argument("--modes", default=",".join(m[0] for m in MODES))
    p.add_argument("--shapes", default="gate_up,down")
    p.add_argument("--indptr-file", default=None,
                   help=".npz from scripts/moe_expert_histogram.py -- the only routing "
                        "that may be used for ranking at B>=4096 (§18.2)")
    p.add_argument("--indptr-key", default=None)
    p.add_argument("--indptr-picks", default="med")
    p.add_argument("--no-ref-tail", dest="ref_tail", action="store_false", default=True,
                   help="skip the drift control (`jetson_clocks` needs interactive sudo)")
    cli = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    os.environ["MLC_MOE_GEMM_V2_HOIST"] = "1"  # ROWSPEC and OPSPEC both require it

    wanted = [m.strip() for m in cli.modes.split(",")]
    legs = [m for m in MODES if m[0] in wanted]
    assert legs and legs[0][0] == "m16", "`m16` is the reference and must be included"
    if cli.ref_tail:
        legs = legs + [("m16'", *legs[0][1:])]

    if cli.indptr_file:
        cases = [(int(c.sum()), lbl, c) for lbl, c in
                 load_real_counts(cli.indptr_file, cli.indptr_key, cli.indptr_picks)]
    else:
        cases = [(B, r, None) for B in (int(v) for v in cli.batches.split(","))
                 for r in cli.routings.split(",")]

    print(f"[0s] hoist=1  legs={[m[0] for m in legs]}  "
          f"routing={'real' if cli.indptr_file else 'synthetic'}")

    failures = 0
    for name in cli.shapes.split(","):
        N, K = SHAPES[name]
        for B, routing, counts in cases:
            args, indptr = make_inputs(N, K, B, routing, dev, counts=counts)
            ref_o, ref_ms, wide_ms, cells = None, None, None, []
            for label, blkm, rowspec, opspec in legs:
                os.environ["MLC_MOE_GEMM_V2_BLKM"] = str(blkm)
                os.environ["MLC_MOE_GEMM_V2_ROWSPEC"] = rowspec
                os.environ["MLC_MOE_GEMM_V2_OPSPEC"] = opspec
                out, ms = run(build(N, K, B, target, dev), args, dev, True)
                if ref_o is None:
                    ref_o, ref_ms = out, ms
                    cells.append(f"{label} {ms:7.3f}ms (ref)")
                    continue
                if label == "rowspec64":
                    wide_ms = ms
                exact = np.array_equal(ref_o, out)
                failures += 0 if exact else 1
                tag = "exact" if exact else f"DIFF({int((ref_o != out).sum())})"
                # Two denominators, because they answer different questions: vs `m16` is
                # "should this ship", vs `rowspec64` is "did item 0s do anything".
                gain = f" [{wide_ms / ms:4.2f}x]" if wide_ms and label.startswith("+") else ""
                cells.append(f"{label} {ms:7.3f}ms {ref_ms / ms:5.2f}x{gain} {tag}")
            experts = int(np.count_nonzero(np.diff(indptr)))
            print(f"{name:8} B={B:<6} {routing:18} experts={experts:3d} | " + " | ".join(cells))

    print("\n[…x] in brackets is against `rowspec64`, i.e. what item 0s itself bought.")
    print(f"{'PASS' if failures == 0 else f'FAIL ({failures} case(s) not bit-exact)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

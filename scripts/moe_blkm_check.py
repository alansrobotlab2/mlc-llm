#!/usr/bin/env python3
"""moe_blkm_check.py — gate and A/B the v2 GEMM's row-blocking factor `BLK_M`.

§16.9 measured `MLC_MOE_GEMM_V2_BLKM=32/64` at 0.64x/0.39x and diagnosed it: the
`i_o` row-fragment loop sits *outside* the `k_o_o` loop the cooperative shared
loads are `compute_at`-ed to, so every extra row-fragment re-runs the whole
`BLK_N x K` weight dequant. Widening `BLK_M` then buys nothing and costs a
proportional amount of dequant.

This script exists to measure the fix. `BLK_M` changes only how output rows are
grouped into CTAs — the reduction over `K` is split identically at every value —
so **the bar is exact equality against `BLK_M=16`**, not a tolerance, exactly as
`moe_gemm_check.py` gates item 0f.

Usage:
    source .envrc.local
    python scripts/moe_blkm_check.py                    # gate + time 16/32/64
    python scripts/moe_blkm_check.py --blkm 16,32       # a subset
    python scripts/moe_blkm_check.py --quick            # correctness only
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
from moe_gemm_check import (  # noqa: E402
    NE, SHAPES, build, load_real_counts, make_inputs, run,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blkm", default="16,32,64", help="comma-separated BLK_M values")
    p.add_argument("--quick", action="store_true", help="skip the timing pass")
    p.add_argument("--batches", default="4096", help="comma-separated batch sizes to time")
    p.add_argument("--hoist", default="0", help="comma-separated MLC_MOE_GEMM_V2_HOIST values "
                                                "(item 0h). Every combination is swept")
    p.add_argument("--indptr-file", default=None,
                   help=".npz from scripts/moe_expert_histogram.py. Replaces the synthetic "
                        "even/random routings with measured ones and takes B from them; "
                        "REQUIRED for any BLK_M conclusion, since the synthetics got item "
                        "0h's sign wrong (workplan 17.9)")
    p.add_argument("--indptr-key", default=None,
                   help="a single key from the .npz (e.g. prose_len2048); default is all")
    p.add_argument("--indptr-picks", default="min,med,max",
                   help="which layers to take, ranked by tile count at BLK_M=16")
    cli = p.parse_args()

    blkms = [int(v) for v in cli.blkm.split(",")]
    assert blkms[0] == 16, "BLK_M=16 is the exactness reference; keep it first"
    batches = [int(v) for v in cli.batches.split(",")]
    hoists = cli.hoist.split(",")
    # The reference must be the shipped config: BLK_M=16 with the hoist off. At BLK_M=16
    # i_o has extent 1, so hoist on/off are the same kernel there — which is itself worth
    # asserting, and the sweep does by including (16, 1) when --hoist includes it.
    assert hoists[0] == "0", "HOIST=0 is the reference; keep it first"

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    print(f"[blkm] target=sm_87  SKIPPAD={os.environ['MLC_MOE_GEMM_V2_SKIPPAD']}  blkm={blkms}")

    if cli.indptr_file:
        # Measured routing (item 0i). B comes from the histogram, so --batches is ignored:
        # a real pp512 prefill is B=4096 *and* a particular shape of raggedness, and the
        # two cannot be varied independently. §17.9 is why the synthetic legs are not the
        # thing to quote — keep them only if --routings asks for them.
        real = load_real_counts(cli.indptr_file, cli.indptr_key, cli.indptr_picks)
        cases = [(n, int(c.sum()), lbl, c) for n in SHAPES for lbl, c in real]
        print(f"[blkm] real routing from {cli.indptr_file}: "
              + ", ".join(f"{lbl} (B={int(c.sum())})" for lbl, c in real))
    else:
        cases = [(n, B, r, None) for n in SHAPES for B in batches for r in ("even", "random")]
    failures = 0
    for name, B, routing, counts in cases:
        N, K = SHAPES[name]
        args, indptr = make_inputs(N, K, B, routing, dev, counts=counts)
        ref_o, ref_ms, row = None, None, []
        for hoist in hoists:
            for blkm in blkms:
                os.environ["MLC_MOE_GEMM_V2_BLKM"] = str(blkm)
                os.environ["MLC_MOE_GEMM_V2_HOIST"] = hoist
                label = f"M={blkm}/H={hoist}"
                try:
                    out, ms = run(build(N, K, B, target, dev), args, dev, not cli.quick)
                except Exception as exc:  # a config that will not schedule is a result
                    row.append(f"{label}: FAIL {type(exc).__name__}")
                    failures += 1
                    continue
                if ref_o is None:
                    ref_o, ref_ms = out, ms
                    row.append(f"{label}: " + ("(ref)" if cli.quick else f"{ms:7.3f} ms (ref)"))
                    continue
                exact = np.array_equal(ref_o, out)
                failures += 0 if exact else 1
                tag = "exact" if exact else f"DIFF({int((ref_o != out).sum())})"
                timing = "" if cli.quick else f"{ms:7.3f} ms {ref_ms / ms:5.2f}x "
                row.append(f"{label}: {timing}{tag}")
        print(f"{name:8} B={B:<5} {routing:7} experts={int(np.count_nonzero(np.diff(indptr))):3d}\n    "
              + "\n    ".join(row))

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} case(s) not bit-exact)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

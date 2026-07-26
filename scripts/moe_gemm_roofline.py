#!/usr/bin/env python3
"""moe_gemm_roofline.py — roofline the v2 GEMM's *real* CTAs, separately from its padding ones.

§16.8 measured this kernel against both walls, but it did so before item 0f skipped
the padding CTAs (§16.11) and before `BLK_K` went 32 -> 64 (§17.3), so every "% of
wall" figure in the workplan for `dequantize_group_gemm_v2` describes a kernel that no
longer exists. This answers the question the §17 handoff left: with padding nearly
free, how close are the CTAs that do real work?

Method. The kernel's cost is `n_real*c_real + n_pad*c_skip`. Rather than assume a
ratio between them — §16.10 measured `c_pad/c_real = 0.933` for *unguarded* padding
CTAs and §17.2 found that constant does not carry to guarded ones — both coefficients
are fit by least squares over a sweep of (B, routing), which moves `n_real` and
`n_pad` semi-independently. `n_real` is computed exactly from the routing, the same
way `_dispatch_func` does it. The fit residual is reported: if the two-parameter model
does not describe the data, nothing downstream of it is trustworthy.

`c_real * n_real` is then the time the useful CTAs took, and the bytes and FLOPs they
were obliged to move are known exactly, giving their true share of each wall.

Bytes are reported two ways and the difference matters:
  issued — what the CTAs ask for: `n_real * (W_tile + Scale + X_tile + O_tile)`. X is
           re-read once per n-tile and W once per m-tile, so this exceeds DRAM traffic
  unique — what DRAM must supply: each hit expert's weights once, X once, O once.
           The gap between the two is what L2 has to absorb.

Usage:
    source .envrc.local
    python scripts/moe_gemm_roofline.py                  # both shapes, BLK_K 32 and 64
    python scripts/moe_gemm_roofline.py --blkk 64        # just the shipped config
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
from moe_gemm_check import GROUP_SIZE, NE, SHAPES, build, make_inputs, run  # noqa: E402

# Same walls every other number in the workplan is quoted against (bench_moe_kernel.py).
BW_WALL_GBS = 156.0
TENSOR_FP16_TFLOPS = 42.6
BLK_M, BLK_N = 16, 128


def cta_counts(indptr: np.ndarray, B: int, N: int) -> tuple[int, int]:
    """Real and padding CTA counts, mirroring `_dispatch_func`'s tiling exactly."""
    counts = np.diff(indptr)
    tiles_per_n = N // BLK_N
    n_real = int(np.ceil(counts / BLK_M).sum()) * tiles_per_n
    n_total = (-(-B // BLK_M) + NE) * tiles_per_n
    return n_real, n_total - n_real


def bytes_and_flops(indptr: np.ndarray, B: int, N: int, K: int, n_real: int) -> dict:
    counts = np.diff(indptr)
    hit = int(np.count_nonzero(counts))
    per_cta = (
        BLK_N * K // 2                      # W_tile, int4
        + BLK_N * (K // GROUP_SIZE) * 2     # Scale, fp16
        + BLK_M * K * 2                     # X_tile, fp16
        + BLK_M * BLK_N * 2                 # O_tile store, fp16
    )
    unique = (
        hit * (N * K // 2 + N * (K // GROUP_SIZE) * 2)  # each hit expert's weights once
        + B * K * 2                                      # X once
        + B * N * 2                                      # O once
    )
    return {
        "issued": n_real * per_cta,
        "unique": unique,
        "flops": n_real * 2 * BLK_M * BLK_N * K,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blkk", default="32,64", help="comma-separated BLK_K values")
    p.add_argument("--batches", default="1024,2048,4096,8192,16384")
    p.add_argument("--ref-batch", type=int, default=4096,
                   help="batch to report the roofline at (4096 = pp512's 512 tok x top-8)")
    cli = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    batches = [int(v) for v in cli.batches.split(",")]

    for blkk in (int(v) for v in cli.blkk.split(",")):
        os.environ["MLC_MOE_GEMM_V2_BLKK"] = str(blkk)
        print(f"\n{'=' * 78}\nBLK_K={blkk}  (SKIPPAD={os.environ['MLC_MOE_GEMM_V2_SKIPPAD']})\n{'=' * 78}")
        for name, (N, K) in SHAPES.items():
            rows = []
            for B in batches:
                for routing in ("even", "random"):
                    args, indptr = make_inputs(N, K, B, routing, dev)
                    _, ms = run(build(N, K, B, target, dev), args, dev, True)
                    n_real, n_pad = cta_counts(indptr, B, N)
                    rows.append((B, routing, n_real, n_pad, ms, indptr))

            # Least squares: ms ~= n_real*c_real + n_pad*c_skip
            A = np.array([[r[2], r[3]] for r in rows], dtype=float)
            y = np.array([r[4] for r in rows], dtype=float)
            (c_real, c_skip), *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ np.array([c_real, c_skip])
            resid = np.abs(pred - y) / y

            print(f"\n{name}  N={N} K={K}")
            print(f"  fit: c_real={c_real * 1e3:.4f} us  c_skip={c_skip * 1e3:.4f} us  "
                  f"c_skip/c_real={c_skip / c_real:.1%}   "
                  f"residual max {resid.max():.2%} median {np.median(resid):.2%}")

            for B, routing, n_real, n_pad, ms, indptr in rows:
                if B != cli.ref_batch:
                    continue
                t_real = n_real * c_real
                bf = bytes_and_flops(indptr, B, N, K, n_real)
                gbs_i = bf["issued"] / (t_real * 1e-3) / 1e9
                gbs_u = bf["unique"] / (t_real * 1e-3) / 1e9
                tflops = bf["flops"] / (t_real * 1e-3) / 1e12
                print(f"  B={B} {routing:7} n_real={n_real:6} n_pad={n_pad:6} "
                      f"total={ms:7.3f} ms  real={t_real:7.3f} ms ({100 * t_real / ms:4.1f}%)")
                print(f"      unique {bf['unique'] / 1e6:7.1f} MB -> {gbs_u:6.1f} GB/s "
                      f"= {100 * gbs_u / BW_WALL_GBS:5.1f}% of the {BW_WALL_GBS:.0f} GB/s wall")
                print(f"      issued {bf['issued'] / 1e6:7.1f} MB -> {gbs_i:6.1f} GB/s "
                      f"({bf['issued'] / bf['unique']:.2f}x unique; the excess is L2's job)")
                print(f"      {tflops:6.2f} TFLOP/s = {100 * tflops / TENSOR_FP16_TFLOPS:4.1f}% "
                      f"of the {TENSOR_FP16_TFLOPS} TFLOP/s fp16 tensor ceiling")


if __name__ == "__main__":
    main()

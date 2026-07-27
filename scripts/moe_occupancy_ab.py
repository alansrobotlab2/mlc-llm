#!/usr/bin/env python3
"""moe_occupancy_ab.py — item 0n: name the wide tile's per-CTA cost.

The question (workplan §19.3, §19.8)
------------------------------------
At B=1024 every hit expert holds fewer than `BLK_M` rows at *both* widths, so
`BLK_M=16` and `BLK_M=64` launch the **same real CTAs** over the **same rows** and
fetch the **same weight tiles**. `BLK_M=64` is nonetheless 30-40% slower. Everything
that differs is per-CTA, and §19.8 named three candidates off the emitted CUDA:

  (a) dynamic shared memory 20.25 kB -> 27.0 kB, "8 resident CTAs become 6"
  (b) the `X_shared` cooperative store: 4 fragments per k-step instead of 1
  (c) `load_matrix_sync` of the X fragments: 4 per k-step instead of 1

This settles it without `ncu`, which is blocked on this box (§8), by measuring the
*resource* side directly and then perturbing the one resource that binds.

What it measures
----------------
1. **Static occupancy, from ptxas.** Dumps the generated CUDA per `BLK_M`, runs
   `nvcc -Xptxas -v` on it for the register count, reads the dynamic shared-memory
   bytes out of the built module's launch parameters, and computes resident CTAs
   under all three sm_87 limits (threads, registers, shared). This is what makes
   candidate (a) checkable rather than arguable: the shared-memory limit is only the
   binding one if it is *below* the other two.

2. **A register-budget perturbation.** Re-compiles the identical CUDA under
   `-maxrregcount`, which changes occupancy and nothing else -- same source, same
   schedule, same traffic. If forcing the wide tile back to the narrow tile's
   occupancy recovers the gap, the term is register pressure; if it does not (or
   spills eat it), the term is the per-k-step work of (b)/(c).

Usage:
    source .envrc.local
    python scripts/moe_occupancy_ab.py                      # the item-0n question
    python scripts/moe_occupancy_ab.py --batches 1024,4096  # add the prefill shape
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ.setdefault("MLC_MOE_GEMM_V2", "1")
os.environ.setdefault("MLC_MOE_GEMM_V2_SKIPPAD", "1")

import numpy as np  # noqa: E402
import tvm  # noqa: E402
import tvm_ffi  # noqa: E402
from tvm.contrib import nvcc  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from moe_gemm_check import SHAPES, build, load_real_counts, make_inputs, run  # noqa: E402

# sm_87 (Orin, GA10B) per-SM limits. Registers are allocated per warp in units of
# 256, which is why the register limit is not simply 65536 / (regs * threads).
SM87 = dict(max_threads=1536, max_regs=65536, max_shared=164 * 1024, reg_alloc_unit=256)
CTA_THREADS = 256  # BLK_N // MICRO warps of 32; independent of BLK_M

# L1 and shared are one 192 kB array per SM, split at one of a fixed set of boundaries.
# TVM does not set a carveout, so the driver picks the smallest one that holds the
# resident CTAs' shared memory -- and whatever is left is L1. That matters here because
# the W_q cooperative fetch deliberately issues duplicate-address loads (see the VEC=4
# note in moe_matmul.py): they are L1 requests, not DRAM traffic, so this kernel's cost
# moves with L1 capacity. Any shared-memory perturbation therefore changes *two* things,
# and a leg that does not hold the carveout fixed cannot attribute its own result.
SM87_UNIFIED = 192 * 1024
CARVEOUT_KB = (0, 8, 16, 32, 64, 100, 132, 164)

# CC 8.x reserves 1 kB of shared memory per thread block for system use, so a CTA costs
# `shared + 1024` against both the occupancy limit and the carveout. **Measured here, not
# taken on faith**: sweeping the pad in 256 B steps puts the 132 -> 164 kB carveout step
# between pad=768 and pad=1024 (1.00x -> 0.79x, a sharp 21% cliff). Without the
# reservation the step would sit between 1792 and 2048. Leaving it out is what made the
# first pass of this script mis-attribute a pure L1 effect to occupancy.
SM87_CTA_RESERVED = 1024


def carveout_for(shared_per_sm: int) -> tuple[int, int]:
    """(shared carveout, L1 bytes) the driver must select to hold `shared_per_sm`."""
    for kb in CARVEOUT_KB:
        if kb * 1024 >= shared_per_sm:
            return kb * 1024, SM87_UNIFIED - kb * 1024
    raise ValueError(f"{shared_per_sm} B exceeds the 164 kB shared carveout")


def _extra_nvcc_options() -> list[str]:
    v = os.environ.get("MOE_OCC_MAXRREG", "")
    return [f"-maxrregcount={int(v)}"] if v else []


# The occupancy knob. `-maxrregcount` is *not* one: ptxas rewrites the instruction stream
# to hit the budget, so a register leg changes work and occupancy together. A static
# `__shared__` array is meant to change neither the instructions nor the traffic -- only
# how much SM shared memory is left for the next resident CTA.
#
# "Meant to" is doing work in that sentence, which is why `--branch-control` exists. The
# array has to be kept alive against nvcc's optimiser somehow, and the obvious way -- a
# never-taken `if` referencing it -- is itself a perturbation. `keep=asm` materialises the
# address through an empty `asm volatile` instead: no branch, no memory clobber, one
# address register. Run both and require them to agree before believing any pad leg.
_PAD_ANCHOR = "extern __shared__ uchar buf_dyn_shmem[];"
_KEEP = os.environ.get("MOE_OCC_KEEP", "asm")


def _inject_shared_pad(code: str, nbytes: int) -> str:
    """`nbytes > 0` pads; `nbytes < 0` injects only the keep-alive, as its own control."""
    decl = f"  __shared__ char occ_pad[{abs(nbytes)}];\n" if nbytes > 0 else ""
    if _KEEP == "branch":
        # Guarded by a runtime kernel argument, not by `threadIdx`, so nvcc cannot prove
        # it dead. Never executes: `UPPER` is a grid extent and is always positive.
        src = "occ_pad[threadIdx.x]" if nbytes > 0 else "threadIdx.x"
        keep = f"  if (UPPER == -987654321) {{ buf_dyn_shmem[0] = (uchar){src}; }}\n"
    else:
        sym = "occ_pad" if nbytes > 0 else "buf_dyn_shmem"
        keep = f'  asm volatile("" :: "l"({sym}));\n'
    head, sep, tail = code.partition(_PAD_ANCHOR)
    if not sep:
        raise SystemExit("shared-pad anchor not found in the generated CUDA")
    # Only the v2 GEMM kernel declares the dynamic buffer, so one substitution is enough;
    # `partition` guarantees we hit the first (and only) occurrence per emitted module.
    return head + _PAD_ANCHOR + "\n" + decl + keep + tail


@tvm_ffi.register_global_func("tvm_callback_cuda_compile", override=True)
def _compile_with_options(code, target):  # pylint: disable=unused-argument
    """Same default path as TVM's own callback, plus the two 0n perturbations."""
    pad = int(os.environ.get("MOE_OCC_SHPAD", "0"))
    if pad:
        code = _inject_shared_pad(code, pad)
    return nvcc.compile_cuda(
        code, target_format="fatbin", compiler="nvcc", options=_extra_nvcc_options() or None
    )


def resident_ctas(regs: int, shared_bytes: int) -> dict:
    """Resident CTAs per SM under each sm_87 limit, and which one binds."""
    by_threads = SM87["max_threads"] // CTA_THREADS
    warps = CTA_THREADS // 32
    per_warp = -(-(regs * 32) // SM87["reg_alloc_unit"]) * SM87["reg_alloc_unit"]
    by_regs = (SM87["max_regs"] // per_warp) // warps
    by_shared = SM87["max_shared"] // (shared_bytes + SM87_CTA_RESERVED)
    limits = {"by_threads": by_threads, "by_regs": by_regs, "by_shared": by_shared}
    ctas = min(limits.values())
    binding = ",".join(sorted(k[3:] for k, v in limits.items() if v == ctas))
    return dict(ctas=ctas, binding=binding, **limits)


def ptxas_registers(src: str) -> int:
    """Register count for `dequantize_group_gemm_v2_kernel`, straight from ptxas."""
    pad = int(os.environ.get("MOE_OCC_SHPAD", "0"))
    if pad:
        src = _inject_shared_pad(src, pad)
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        cmd = ["nvcc", "-arch=sm_87", "-O3", "-cubin", "-o", "/dev/null",
               "-Xptxas", "-v", *_extra_nvcc_options(), path]
        out = subprocess.run(cmd, capture_output=True, text=True).stderr
    finally:
        os.unlink(path)
    # Two entry functions are emitted; take the one after the v2 kernel's banner.
    tail = out.split("dequantize_group_gemm_v2_kernel")[-1]
    m = re.search(r"Used (\d+) registers", tail)
    spill = re.search(r"(\d+) bytes spill stores", tail)
    # ptxas only prints `smem` when there is static shared memory, i.e. on a pad leg.
    smem = re.search(r"(\d+) bytes smem", tail)
    if not m:
        raise SystemExit(f"could not parse ptxas output:\n{out}")
    return (int(m.group(1)),
            int(spill.group(1)) if spill else 0,
            int(smem.group(1)) if smem else 0)


def device_source(mod, depth: int = 0):
    if depth > 4:
        return None
    try:
        src = mod.inspect_source()
    except Exception:  # pylint: disable=broad-except
        src = None
    if src and "__global__" in src:
        return src
    subs = getattr(mod, "imports", [])
    for sub in (subs() if callable(subs) else subs):
        got = device_source(sub, depth + 1)
        if got:
            return got
    return None


def dyn_shared_bytes(src: str, blk_m: int, blk_n: int = 128, blk_k: int = 64) -> int:
    """Dynamic shared bytes for one CTA.

    The schedule storage-aligns both tiles to a 72-half row stride
    (`storage_align(factor=16, offset=8)` at BLK_K=64), and `O_tile` aliases into the
    W_tile allocation. Cross-checked against the emitted source's largest
    `buf_dyn_shmem` index below, so a schedule change cannot silently invalidate it.
    """
    stride = blk_k + 8
    halves = blk_m * stride + blk_n * stride
    halves = max(halves, blk_m * stride + blk_m * blk_n)  # O_tile if it is the larger
    return halves * 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", default="1024", help="comma-separated batch sizes")
    p.add_argument("--routings", default="even,random")
    p.add_argument("--blkm", default="16,32,64")
    p.add_argument("--maxrreg", default="",
                   help="comma-separated -maxrregcount values to add as extra legs "
                        "on the widest BLK_M (e.g. 40,48). NOT an occupancy-only knob: "
                        "ptxas reschedules to hit the budget")
    p.add_argument("--shpad", default="",
                   help="comma-separated static __shared__ pad sizes in bytes, added as "
                        "extra legs on the NARROWEST BLK_M. This is the occupancy-only "
                        "control: identical instructions, fewer resident CTAs")
    p.add_argument("--shapes", default="gate_up,down")
    p.add_argument("--indptr-file", default=None,
                   help=".npz from scripts/moe_expert_histogram.py. §18.2: the synthetic "
                        "routings are wrong at B=4096 specifically. B=1024 (item 0n's "
                        "shape) is not covered by the histogram, so it stays synthetic")
    p.add_argument("--indptr-key", default=None)
    p.add_argument("--indptr-picks", default="med")
    p.add_argument("--branch-control", action="store_true", default=True,
                   help="add a leg carrying the keep-alive but no pad array")
    p.add_argument("--no-branch-control", dest="branch_control", action="store_false")
    p.add_argument("--ref-tail", action="store_true", default=True,
                   help="re-measure the reference leg last, as a drift control")
    p.add_argument("--no-ref-tail", dest="ref_tail", action="store_false")
    cli = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    blkms = [int(v) for v in cli.blkm.split(",")]
    rregs = [int(v) for v in cli.maxrreg.split(",") if v.strip()]
    shpads = [int(v) for v in cli.shpad.split(",") if v.strip()]

    # (label, BLK_M, HOIST, ROWSPEC, maxrregcount-or-None, shared-pad bytes)
    legs = []
    for m in blkms:
        # The hoist is inert at BLK_M=16 (i_o extent 1) and is the configuration every
        # wide-BLK_M number in §18/§19 was taken under, so it is on for all of them.
        legs.append((f"M{m}", m, "1", "1" if m > 16 else "0", None, 0))
    for r in rregs:
        m = max(blkms)
        legs.append((f"M{m}/rreg{r}", m, "1", "1" if m > 16 else "0", r, 0))
    if cli.branch_control:
        # The keep-alive with *no* array: whatever this leg costs is the instrument, not
        # the perturbation, and every pad ratio has to be read against it.
        m = min(blkms)
        legs.append((f"M{m}/keep", m, "1", "1" if m > 16 else "0", None, -1))
    for pad in shpads:
        m = min(blkms)
        legs.append((f"M{m}/pad{pad // 1024}k", m, "1", "1" if m > 16 else "0", None, pad))
    if cli.ref_tail:
        # `jetson_clocks` needs an interactive sudo (§8), so a run can drift under DVFS
        # and the reference leg is always measured first. Re-measuring it last turns that
        # from an assumption into a number: anything other than ~1.00x means the ordering
        # is contaminating every ratio in the row.
        head = legs[0]
        legs.append((head[0] + "'", *head[1:]))

    print(f"[0n] target=sm_87  CTA={CTA_THREADS} threads  "
          f"limits: {SM87['max_threads']} thr, {SM87['max_regs']} regs, "
          f"{SM87['max_shared'] // 1024} kB shared per SM")

    # ---- pass 1: static resources, no timing ----
    print("\n--- static occupancy (ptxas + schedule) ---")
    print(f"{'leg':14} {'regs':>5} {'spill':>6} {'shared':>8} "
          f"{'by_thr':>7} {'by_reg':>7} {'by_shm':>7} {'CTAs/SM':>8} {'occ':>6} "
          f"{'shm/SM':>8} {'L1':>6}  binds")
    static = {}
    for label, m, hoist, rowspec, rreg, pad in legs:
        os.environ["MLC_MOE_GEMM_V2_BLKM"] = str(m)
        os.environ["MLC_MOE_GEMM_V2_HOIST"] = hoist
        os.environ["MLC_MOE_GEMM_V2_ROWSPEC"] = rowspec
        os.environ["MOE_OCC_MAXRREG"] = str(rreg) if rreg else ""
        os.environ["MOE_OCC_SHPAD"] = str(pad)
        N, K = SHAPES["gate_up"]
        vm = build(N, K, 1024, target, dev)
        src = device_source(vm.module)
        regs, spill, smem = ptxas_registers(src)
        assert smem >= max(pad, 0), f"pad did not land: ptxas reports {smem} B static"
        shared = dyn_shared_bytes(src, m) + max(pad, 0)
        occ = resident_ctas(regs, shared)
        per_sm = (shared + SM87_CTA_RESERVED) * occ["ctas"]
        carve, l1 = carveout_for(per_sm)
        static[label] = dict(regs=regs, spill=spill, shared=shared, l1=l1, **occ)
        print(f"{label:14} {regs:5d} {spill:6d} {shared/1024:7.2f}k "
              f"{occ['by_threads']:7d} {occ['by_regs']:7d} {occ['by_shared']:7d} "
              f"{occ['ctas']:8d} {occ['ctas'] * CTA_THREADS / SM87['max_threads']:5.0%} "
              f"{per_sm/1024:7.1f}k {l1/1024:5.0f}k  {occ['binding']}")

    # ---- pass 2: timing, and bit-exactness against the shipped M=16 leg ----
    print("\n--- timing (bit-exactness vs the shipped M16 leg) ---")
    failures = 0
    if cli.indptr_file:
        cases = [(int(c.sum()), lbl, c) for lbl, c in
                 load_real_counts(cli.indptr_file, cli.indptr_key, cli.indptr_picks)]
    else:
        cases = [(B, r, None) for B in (int(v) for v in cli.batches.split(","))
                 for r in cli.routings.split(",")]
    for name in cli.shapes.split(","):
        N, K = SHAPES[name]
        for B, routing, counts in cases:
            args, indptr = make_inputs(N, K, B, routing, dev, counts=counts)
            ref_o, ref_ms, cells = None, None, []
            for label, m, hoist, rowspec, rreg, pad in legs:
                os.environ["MLC_MOE_GEMM_V2_BLKM"] = str(m)
                os.environ["MLC_MOE_GEMM_V2_HOIST"] = hoist
                os.environ["MLC_MOE_GEMM_V2_ROWSPEC"] = rowspec
                os.environ["MOE_OCC_MAXRREG"] = str(rreg) if rreg else ""
                os.environ["MOE_OCC_SHPAD"] = str(pad)
                out, ms = run(build(N, K, B, target, dev), args, dev, True)
                if ref_o is None:
                    ref_o, ref_ms = out, ms
                    cells.append(f"{label} {ms:7.3f}ms (ref)")
                    continue
                exact = np.array_equal(ref_o, out)
                failures += 0 if exact else 1
                tag = "exact" if exact else f"DIFF({int((ref_o != out).sum())})"
                cells.append(f"{label} {ms:7.3f}ms {ref_ms / ms:5.2f}x {tag}")
            experts = int(np.count_nonzero(np.diff(indptr)))
            print(f"{name:8} B={B:<6} {routing:18} experts={experts:3d} | "
                  + " | ".join(cells))

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} not bit-exact)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

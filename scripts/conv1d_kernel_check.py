#!/usr/bin/env python3
"""Numerical unit gate for the fused in-place causal conv1d kernel.

Answers one question the end-to-end gates cannot: *is the kernel wrong, or merely
rounded differently?* Those two look identical from a token-diff, and on the 35B at
q4f16_1 they are routinely confused -- near-ties are dense enough there that any change
in fp16 accumulation order flips tokens and cascades (workplan §6.2, §10).

That confusion actually happened. `create_causal_conv1d_func_inplace` came out
byte-identical to the copy path on the 0.8B (conv_dim=6144) and diverged on all 5
prompts on the 35B (conv_dim=8192). This script showed the kernel is correct at both
widths and every sequence shape, so the divergence is dlight scheduling the 4-tap
reduction in a different order at the wider shape -- not an indexing bug.

Method: compute the reference in fp64 from the same definition the TE path uses, and
require the kernel to land within fp16 rounding of it. Three things are checked, and
only the first is allowed any tolerance at all:

  1. conv output      -- fp16 rounding of a 4-term sum, so ~1 ulp
  2. the new state    -- a pure copy, no arithmetic, so BIT-EXACT or the shift is wrong
  3. every other history slot -- must be untouched, or the ring write is addressing
                                 the wrong slot, which is the §11 failure mode that is
                                 invisible under prefix_cache_mode=disable

Runs in seconds on any CUDA device and needs no model, no weights and no engine.
"""
from __future__ import annotations

import argparse

import numpy as np
import tvm

from mlc_llm.model.qwen35.qwen35_model import create_causal_conv1d_func_inplace

# (conv_dim, label). 0.8B is 2*16*128 + 16*128; 35B is 2*16*128 + 32*128.
WIDTHS = [(6144, "0.8B"), (8192, "35B-A3B")]
# decode, the sub-kernel-width shapes where the state/qkv split straddles, and prefill
SEQ_LENS = [1, 2, 3, 4, 17, 512]

FP16_EPS = 9.8e-4
REL_TOL = 5e-3  # a few ulp of headroom on a 4-term fp16 sum


def reference(qkv, weight, state, seq_len, ks):
    """fp64 reference, straight from the TE definition in qwen35_model.py."""
    ks_m1 = ks - 1
    cat = np.concatenate([state.astype(np.float64), qkv.astype(np.float64)], axis=1)
    out = np.zeros((qkv.shape[0], seq_len, qkv.shape[2]), dtype=np.float64)
    for si in range(seq_len):
        for kk in range(ks):
            out[:, si, :] += cat[:, si + kk, :] * weight[:, 0, kk].astype(np.float64)
    return out, cat[:, seq_len : seq_len + ks_m1, :]


def run_shape(conv_dim, seq_len, ks, batch, max_hist, hist_slot_id, seed):
    rng = np.random.default_rng(seed)
    dev = tvm.cuda(0)
    f = create_causal_conv1d_func_inplace(conv_dim=conv_dim, kernel_size=ks, dtype="float16")
    mod = tvm.compile(tvm.IRModule({"main": f}), target="cuda")

    qkv = (rng.standard_normal((batch, seq_len, conv_dim)) * 0.5).astype(np.float16)
    weight = (rng.standard_normal((conv_dim, 1, ks)) * 0.5).astype(np.float16)
    storage = (rng.standard_normal((batch, max_hist, ks - 1, conv_dim)) * 0.5).astype(np.float16)
    seq_slot = np.arange(batch, dtype=np.int32)
    hist_slot = np.full((batch,), hist_slot_id, dtype=np.int32)

    state_in = storage[seq_slot, hist_slot].copy()
    ref_out, ref_state = reference(qkv, weight, state_in, seq_len, ks)

    args = [
        tvm.runtime.tensor(x, dev)
        for x in (qkv, weight, storage, seq_slot, hist_slot,
                  np.zeros((batch, seq_len, conv_dim), np.float16))
    ]
    mod["main"](*args)
    dev.sync()

    got_out = args[5].numpy().astype(np.float64)
    got_storage = args[2].numpy()
    hist_out = (hist_slot + 1) % max_hist
    got_state = got_storage[seq_slot, hist_out].astype(np.float64)

    scale = max(float(np.abs(ref_out).max()), 1e-6)
    rel = float(np.abs(got_out - ref_out).max()) / scale
    state_err = float(np.abs(got_state - ref_state).max())
    touched = {int(hist_slot[0]), int(hist_out[0])}
    others = [h for h in range(max_hist) if h not in touched]
    clobber = float(
        np.abs(got_storage[:, others].astype(np.float64) - storage[:, others].astype(np.float64)).max()
    ) if others else 0.0
    return rel, state_err, clobber


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-size", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2, help=">1 exercises per-batch slot indexing")
    ap.add_argument("--max-history", type=int, default=64, help="64 = radix; 1 = disable")
    ap.add_argument("--hist-slot", type=int, default=7, help="non-zero exercises the ring")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"fp16 eps ~ {FP16_EPS:.1e}; rounding-order differences stay at that scale.")
    print(f"batch={args.batch} max_history={args.max_history} hist_slot={args.hist_slot}\n")
    bad = 0
    for conv_dim, label in WIDTHS:
        for seq_len in SEQ_LENS:
            rel, se, cl = run_shape(conv_dim, seq_len, args.kernel_size, args.batch,
                                    args.max_history, args.hist_slot % args.max_history,
                                    args.seed)
            fail = rel > REL_TOL or se != 0.0 or cl != 0.0
            bad += fail
            print(f"  {label:<8} conv_dim={conv_dim:5d} seq_len={seq_len:4d}  "
                  f"out_rel={rel:.2e}  state_err={se:.1e}  other_slots={cl:.1e}"
                  f"{'   <-- FAIL' if fail else ''}")
    if bad:
        raise SystemExit(f"\n{bad} shape(s) FAILED")
    print("\nALL SHAPES PASS — output within fp16 rounding, state bit-exact, ring clean")


if __name__ == "__main__":
    main()

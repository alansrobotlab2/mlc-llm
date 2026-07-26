#!/usr/bin/env python3
"""Numerical unit gate for the fused in-place GatedDeltaNet recurrence kernels.

The recurrent counterpart to `conv1d_kernel_check.py`, and it exists for the same reason:
on the 35B a token-diff cannot tell "the ring is misindexed" from "the reduction rounded
differently", and the history path is only reachable under `prefix_cache_mode=radix`,
which is the mode no bench harness sets (workplan §13, §14.1).

Two kernels are gated, and the second is the point of the script:

  * `create_gated_delta_net_func_inplace`              -- decode, one ring advance
  * `create_gated_delta_net_func_with_history_inplace` -- prefill under radix, a
                                                          per-position scatter that wraps

Three checks per shape, and the tolerance budget is deliberately lopsided:

  1. output vs the **copy-path kernel** -- BIT-EXACT. Both kernels run the same two passes
     in the same order over the same fp32 registers, so anything but 0 here means the
     fusion changed the arithmetic, not just where the state lives.
  2. the ring content -- BIT-EXACT against a numpy scatter of the copy path's own
     per-position history tensor. This is the check that catches an off-by-one in
     `(hist + 1 + t) % max_hist`.
  3. every slot the scatter must not reach -- untouched, byte for byte. A kernel that
     writes the right values into the wrong slots passes 1 and fails this.

A fourth, looser check runs the recurrence in fp64 from the mathematical definition, so
the gate still has teeth if the copy-path kernel is itself wrong. Only that one gets a
tolerance.

The interesting shapes are `seq_len > max_history`: a real prefill chunk is 512-2048
positions against 64 ring slots, so the flush wraps and slot `hist_slot` -- the one the
recurrence loaded its state from -- is itself overwritten mid-kernel.

Runs in seconds on any CUDA device. No model, no weights, no engine.
"""
from __future__ import annotations

import argparse

import numpy as np
import tvm

from mlc_llm.model.qwen35.qwen35_model import (
    create_gated_delta_net_func_inplace,
    create_gated_delta_net_func_with_history,
    create_gated_delta_net_func_with_history_inplace,
)

# (num_key_heads, num_value_heads, label). Head dims are 128 on both models; the 35B
# runs GVA with 2 value heads per key head, which is what exercises `heads_per_group`.
HEAD_CONFIGS = [(16, 16, "0.8B"), (16, 32, "35B-A3B")]
# 1 is decode; 63/64/65 straddle the default max_history so the skip guard turns on
# exactly once across them; 128 and 512 are wrapped prefill chunks.
SEQ_LENS = [1, 2, 5, 17, 63, 64, 65, 128, 512]

HEAD_DIM = 128
REL_TOL = 5e-3  # fp32 accumulation over a 128-term dot, compounded across the sequence


def make_inputs(rng, batch, seq_len, n_kh, n_vh, K, V):
    """Inputs in the ranges the model actually produces them in.

    `gate` is `exp(g)` with `g < 0` so the recurrence decays rather than growing without
    bound, and `beta` is a sigmoid output. Feeding uniform noise instead makes the fp64
    comparison meaningless -- the delta rule amplifies, and the tolerance would be
    measuring the test's own conditioning.
    """
    q = (rng.standard_normal((batch, seq_len, n_kh, K)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((batch, seq_len, n_kh, K)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((batch, seq_len, n_vh, V)) * 0.5).astype(np.float16)
    gate = np.exp(-rng.random((batch, seq_len, n_vh)) * 0.1).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((batch, seq_len, n_vh))))).astype(np.float32)
    return q, k, v, gate, beta


def reference_fp64(q, k, v, gate, beta, state_in, n_kh, n_vh, K, V):
    """The recurrence in fp64, straight from the definition the TIR kernels implement.

    Only the output is returned. The per-position state history is deliberately not
    materialized in fp64 -- at a 512-position chunk it is gigabytes, and the bit-exact
    comparison against the copy path already covers the state.
    """
    batch, seq_len = q.shape[0], q.shape[1]
    heads_per_group = n_vh // n_kh
    scale = 1.0 / np.sqrt(K)
    S = state_in.astype(np.float64).copy()  # (batch, n_vh, K, V)
    out = np.zeros((batch, seq_len, n_vh, V), dtype=np.float64)
    kh = np.arange(n_vh) // heads_per_group
    for t in range(seq_len):
        kt = k[:, t][:, kh].astype(np.float64)  # (batch, n_vh, K)
        qt = q[:, t][:, kh].astype(np.float64)
        vt = v[:, t].astype(np.float64)  # (batch, n_vh, V)
        g = gate[:, t].astype(np.float64)[..., None, None]
        S *= g
        dot_sk = np.einsum("bhk,bhkv->bhv", kt, S)
        coef = beta[:, t].astype(np.float64)[..., None] * (vt - dot_sk)  # (batch, n_vh, V)
        S += kt[..., None] * coef[:, :, None, :]
        out[:, t] = np.einsum("bhk,bhkv->bhv", qt, S) * scale
    return out


def _compile(func):
    return tvm.compile(tvm.IRModule({"main": func}), target="cuda")


def _copy_path(q, k, v, gate, beta, state_in, n_kh, n_vh, K, V, dev):
    """Run the shipped `gdn_func_history` kernel — the trusted reference for check 1."""
    batch, seq_len = q.shape[0], q.shape[1]
    mod = _compile(
        create_gated_delta_net_func_with_history(
            num_key_heads=n_kh, num_value_heads=n_vh, key_head_dim=K,
            value_head_dim=V, dtype="float16",
        )
    )
    out = np.zeros((batch, seq_len, n_vh, V), np.float32)
    hist = np.zeros((batch, seq_len, n_vh, K, V), np.float32)
    args = [tvm.runtime.tensor(x, dev)
            for x in (q, k, v, gate, beta, state_in, out, hist)]
    mod["main"](*args)
    dev.sync()
    return args[6].numpy(), args[7].numpy()


def _verdict(got_storage, ref_storage, written, batch, max_hist):
    """Split the storage verdict into 'slots that should have changed' vs 'the rest'.

    Reported separately on purpose: a single "storage differs" number would let a
    ring-addressing bug hide behind a legitimate write.
    """
    state_err, clobber = 0.0, 0.0
    for si in range(batch):
        for h in range(max_hist):
            d = float(np.abs(got_storage[si, h].astype(np.float64)
                             - ref_storage[si, h].astype(np.float64)).max())
            if (si, h) in written:
                state_err = max(state_err, d)
            else:
                clobber = max(clobber, d)
    return state_err, clobber


def run_shape(n_kh, n_vh, seq_len, batch, max_hist, hist_slot_id, seed, history):
    K = V = HEAD_DIM
    rng = np.random.default_rng(seed)
    dev = tvm.cuda(0)

    q, k, v, gate, beta = make_inputs(rng, batch, seq_len, n_kh, n_vh, K, V)
    storage = (rng.standard_normal((batch, max_hist, n_vh, K, V)) * 0.2).astype(np.float32)
    seq_slot = np.arange(batch, dtype=np.int32)
    hist_slot = np.full((batch,), hist_slot_id, dtype=np.int32)
    state_in = storage[seq_slot, hist_slot].copy()

    ref_out, ref_hist = _copy_path(q, k, v, gate, beta, state_in, n_kh, n_vh, K, V, dev)
    fp64_out = reference_fp64(q, k, v, gate, beta, state_in, n_kh, n_vh, K, V)

    maker = (create_gated_delta_net_func_with_history_inplace if history
             else create_gated_delta_net_func_inplace)
    mod = _compile(maker(num_key_heads=n_kh, num_value_heads=n_vh,
                         key_head_dim=K, value_head_dim=V, dtype="float16"))
    out = np.zeros((batch, seq_len, n_vh, V), np.float32)
    args = [tvm.runtime.tensor(x, dev)
            for x in (q, k, v, gate, beta, storage, seq_slot, hist_slot, out)]
    mod["main"](*args)
    dev.sync()
    got_out = args[8].numpy()
    got_storage = args[5].numpy()

    # Expected ring content: apply the copy path's own per-position history through the
    # same scatter `create_set_with_history_func` performs, in ascending t so the
    # last write wins — which is what makes a wrapped ring well-defined.
    expected = storage.copy()
    written = set()
    for bi in range(batch):
        if history:
            for t in range(seq_len):
                slot = (int(hist_slot[bi]) + 1 + t) % max_hist
                expected[int(seq_slot[bi]), slot] = ref_hist[bi, t]
                written.add((int(seq_slot[bi]), slot))
        else:
            slot = (int(hist_slot[bi]) + 1) % max_hist
            expected[int(seq_slot[bi]), slot] = ref_hist[bi, seq_len - 1]
            written.add((int(seq_slot[bi]), slot))

    scale = max(float(np.abs(ref_out).max()), 1e-6)
    bitexact = float(np.abs(got_out.astype(np.float64) - ref_out.astype(np.float64)).max())
    fp64_rel = float(np.abs(got_out.astype(np.float64) - fp64_out).max()) / scale
    state_err, clobber = _verdict(got_storage, expected, written, batch, max_hist)
    return bitexact, fp64_rel, state_err, clobber


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2, help=">1 exercises per-batch slot indexing")
    ap.add_argument("--max-history", type=int, default=64, help="64 = radix; 1 = disable")
    ap.add_argument("--hist-slot", type=int, default=7, help="non-zero exercises the ring")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-lens", type=str, default="",
                    help="comma-separated override for the default sweep")
    args = ap.parse_args()

    seq_lens = ([int(s) for s in args.seq_lens.split(",")] if args.seq_lens else SEQ_LENS)
    print(f"batch={args.batch} max_history={args.max_history} hist_slot={args.hist_slot}")
    print("out_bit / state_err must be exactly 0; fp64_rel is the only tolerance.\n")
    bad = 0
    for label, history in (("decode (state_id=0, single ring advance)", False),
                           ("history path (per-position scatter, radix default)", True)):
        print(f"{label}:")
        for n_kh, n_vh, name in HEAD_CONFIGS:
            for seq_len in seq_lens:
                if not history and seq_len != 1:
                    continue  # the decode kernel only ever sees seq_len 1
                be, rel, se, cl = run_shape(n_kh, n_vh, seq_len, args.batch,
                                            args.max_history,
                                            args.hist_slot % args.max_history,
                                            args.seed, history)
                fail = be != 0.0 or se != 0.0 or cl != 0.0 or rel > REL_TOL
                bad += fail
                wrap = " WRAP" if history and seq_len > args.max_history else ""
                print(f"  {name:<8} n_vh={n_vh:3d} seq_len={seq_len:4d}  "
                      f"out_bit={be:.1e}  fp64_rel={rel:.2e}  state_err={se:.1e}  "
                      f"other_slots={cl:.1e}{wrap}{'   <-- FAIL' if fail else ''}")
        print()
    if bad:
        raise SystemExit(f"{bad} shape(s) FAILED")
    print("ALL SHAPES PASS — output bit-exact vs the copy path, ring bit-exact, slots clean")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Kernel-level A/B for the history-path GatedDeltaNet recurrence, in TIR.

`scripts/gdn_recurrence_probe.cu` (§16.2) answered the design question in standalone CUDA:
splitting the `K` reduction across lanes is worth 2.24-2.79x. This script measures the same
thing on the kernels MLC actually compiles, which differs from the probe in two ways that
matter:

  * it includes the **per-position ring flush** the probe deliberately left out. The probe's
    header calls its own numbers an upper bound for exactly this reason -- the flush is pure
    stores and blending a bandwidth term into a latency measurement would have muddied the
    design question. Here it is part of the kernel under test, so this is the number to
    renormalize a prefill estimate against.
  * it runs TVM's codegen rather than hand-written CUDA, so the index arithmetic, the
    unrolling and the register allocation are whatever ptxas makes of TIR.
  * it uses the **real grid**, which is `(num_value_heads, batch)` -- 16 blocks on the 0.8B
    but **32 on the 35B**. The probe hardcodes `n_kh = 16` for its grid, so it models the
    0.8B only, and that is what made §16.2's 2.79x wrong for the 35B: at 32 blocks the
    unsplit kernel already fits 2 blocks/SM and starts at twice the occupancy the probe
    assumed. Splitting is worth 1.21x there, not 2.79x, and `k_split=2` is a regression.
    See §16.5.

⚠️ **This is a kernel bench, not a prefill prediction.** Per §15.2, an end-to-end estimate
has to be renormalized against a trace of the actual A/B baseline; `gdn_func_history_inplace`
was 95.6 ms of a 95.6 + 19.3 + ... ms pp512 budget (§15.6), so Amdahl applies to the rest.
Use `--trace-share` to fold in that share and print a bounded whole-prefill number.

⚠️ **Run exactly one of these at a time, and verify that before trusting a number.** A split
kernel costs ptxas 30-42 s, so a sweep spends most of its wall clock compiling and looks
idle; it is easy to conclude a run has died and start another on top of it. That happened
on 2026-07-26b and produced a table that had to be thrown away. `ps -C python` does **not**
find these -- the process shows as `timeout NNNN python ...`, so the executable-name match
misses it. Use `pgrep -af gdn_kernel_bench`.

The state buffer is written in place and re-read across timing iterations, so the recurrence
walks toward its own fixed point over a long run. That is fine here and deliberately not
reset: the kernel has no data-dependent control flow, so its cost does not depend on the
values, and re-uploading 500 MB of state between iterations would dominate what we are
timing. `gdn_kernel_check.py` is where values are adjudicated.

  python scripts/gdn_kernel_bench.py                      # 0.8B and 35B head configs
  python scripts/gdn_kernel_bench.py --k-splits 1,4 --seq-lens 512
"""
from __future__ import annotations

import argparse

import numpy as np
import tvm

from mlc_llm.model.qwen35.qwen35_model import (
    create_gated_delta_net_func_with_history_inplace,
    create_gated_delta_net_func_with_history_inplace_ksplit,
)

# Same configs the numerical gate uses: head dims are 128 on both models, and the 35B runs
# GVA with 2 value heads per key head.
HEAD_CONFIGS = [(16, 16, "0.8B"), (16, 32, "35B-A3B")]
HEAD_DIM = 128


def build(n_kh, n_vh, k_split, v_block=0):
    common = dict(
        num_key_heads=n_kh,
        num_value_heads=n_vh,
        key_head_dim=HEAD_DIM,
        value_head_dim=HEAD_DIM,
        dtype="float16",
    )
    func = (
        create_gated_delta_net_func_with_history_inplace(**common)
        if k_split == 1
        else create_gated_delta_net_func_with_history_inplace_ksplit(
            k_split=k_split, v_block=v_block, **common
        )
    )
    return tvm.compile(tvm.IRModule({"main": func}), target="cuda")


def parse_configs(k_splits, v_blocks):
    """(k_split, v_block) pairs to time. v_block is inert at k_split=1 (no such parameter)."""
    configs = []
    for ks in k_splits:
        for vb in ([0] if ks == 1 else v_blocks):
            if (ks, vb) not in configs:
                configs.append((ks, vb))
    return configs


def label(ks, vb):
    return f"ks{ks}" if not vb else f"ks{ks}/v{vb}"


def make_args(rng, batch, seq_len, n_kh, n_vh, max_hist, dev):
    """Inputs in the ranges the model produces, per `gdn_kernel_check.make_inputs`."""
    K = V = HEAD_DIM
    q = (rng.standard_normal((batch, seq_len, n_kh, K)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((batch, seq_len, n_kh, K)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((batch, seq_len, n_vh, V)) * 0.5).astype(np.float16)
    gate = np.exp(-rng.random((batch, seq_len, n_vh)) * 0.1).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((batch, seq_len, n_vh))))).astype(np.float32)
    storage = (rng.standard_normal((batch, max_hist, n_vh, K, V)) * 0.2).astype(np.float32)
    seq_slot = np.arange(batch, dtype=np.int32)
    hist_slot = np.full((batch,), 7 % max_hist, dtype=np.int32)
    out = np.zeros((batch, seq_len, n_vh, V), np.float32)
    return [tvm.runtime.tensor(x, dev)
            for x in (q, k, v, gate, beta, storage, seq_slot, hist_slot, out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1, help="1 is the shipped prefill geometry")
    ap.add_argument("--max-history", type=int, default=64, help="64 = radix")
    ap.add_argument("--seq-lens", type=str, default="1,128,512,2048")
    ap.add_argument("--k-splits", type=str, default="1,2,4,8")
    ap.add_argument("--v-blocks", type=str, default="0",
                    help="comma-separated value columns per block (item 0d); 0 = one block per "
                         "head. Crossed with --k-splits; block is v_block*k_split threads, so "
                         "this is what makes k_split=8 reachable below 1024 threads")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace-share", type=float, default=0.0,
                    help="this kernel's share of traced prefill GPU time (e.g. 0.239 for the "
                         "0.8B pp512 figure in §14.2); prints an Amdahl-bounded whole-prefill "
                         "speedup alongside the kernel speedup")
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    k_splits = [int(s) for s in args.k_splits.split(",")]
    v_blocks = [int(s) for s in args.v_blocks.split(",")]
    configs = parse_configs(k_splits, v_blocks)
    if 1 not in k_splits:
        raise SystemExit("--k-splits must include 1; it is the baseline every ratio is against")
    dev = tvm.cuda(0)

    print(f"batch={args.batch} max_history={args.max_history} iters={args.iters}")
    print("blocks = n_vh x (V / v_block) x batch;  threads/block = v_block x k_split "
          "(v_block 0 means V=128)")
    print("ms per kernel call; 'x' is vs ks1 at the same shape.\n")

    for n_kh, n_vh, name in HEAD_CONFIGS:
        mods = {c: build(n_kh, n_vh, *c) for c in configs}
        head = "".join(f"{label(*c) + ' ms':>13}" for c in configs)
        ratio = "".join(f"{label(*c) + ' x':>10}" for c in configs if c[0] != 1)
        print(f"{name} (n_kh={n_kh}, n_vh={n_vh}; blocks = n_vh x V/v_block x batch)")
        print(f"{'seq_len':>8}{head}{ratio}")
        for seq_len in seq_lens:
            times = {}
            for c in configs:
                rng = np.random.default_rng(args.seed)
                tensors = make_args(rng, args.batch, seq_len, n_kh, n_vh, args.max_history, dev)
                mods[c]["main"](*tensors)  # warm up: first call pays JIT + module load
                dev.sync()
                timer = mods[c].mod.time_evaluator(
                    "main", dev, number=args.iters, repeat=3, min_repeat_ms=0
                )
                # Median over repeats, per `bench_moe_kernel.time_kernel`.
                times[c] = timer(*tensors).median * 1e3
            base = times[(1, 0)]
            cells = "".join(f"{times[c]:13.4f}" for c in configs)
            ratios = "".join(f"{base / times[c]:10.2f}" for c in configs if c[0] != 1)
            print(f"{seq_len:8d}{cells}{ratios}")
            if args.trace_share:
                # Amdahl on the traced share: the rest of prefill is unchanged.
                sh = args.trace_share
                bounded = "".join(
                    f"{1.0 / (1.0 - sh + sh * times[c] / base):10.3f}"
                    for c in configs if c[0] != 1
                )
                print(f"{'  ^ e2e':>8}{'':{13 * len(configs)}}{bounded}")
        print()

    if args.trace_share:
        print(f"'e2e' applies Amdahl at share={args.trace_share:.3f} of traced prefill GPU time. "
              "It is\nstill an estimate: renormalize against a trace of the real A/B (§15.2).")


if __name__ == "__main__":
    main()

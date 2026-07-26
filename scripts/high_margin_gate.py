#!/usr/bin/env python3
"""A deterministic state gate for models whose only reference is a *different*
quantization — i.e. the 35B (workplan-cuda-13.md §9 item 0b, §6.2).

Two problems make the existing 35B gates unable to adjudicate a state change,
and this script fixes both.

**Problem 1 — cascade.** `validate.py --greedy-parity` free-runs both sides. One
near-tie flip and the two sequences are in different contexts forever, so every
later token is incomparable. §6.2 measured 1/50, 15/50, 2/50, 5/50 against the
fp8 reference; those numbers say almost nothing about correctness because they
are dominated by where the *first* flip happened. Fix: **teacher forcing.** At
position i, MLC is asked for one token given the reference's exact prefix
`prompt + ref_tokens[:i]`. Every position is then scored in the same context the
reference saw, and a flip at position 3 cannot contaminate position 4.

**Problem 2 — no defensible bar.** `>=48/50` is calibrated for fp16-vs-fp16;
applying it to 4-bit-vs-8-bit is the category error §6 already flags. Fix:
**score only where the reference had margin.** The capture step records
`logprob(top1) - logprob(top2)` at every position, and the check step asserts
agreement only at positions where that margin clears `--tau`. A wide-margin
position that flips is a bug; a near-tie that flips is quantization. The prompt
set is built to produce mostly wide margins (exact-continuation sequences,
arithmetic, rote lists, verbatim copy) — §6.2's point 3, that prompt 5 is the
only 50/50 prompt because it is the only high-margin one, is the whole thesis.

Note what this does *not* claim: it is still a tier-2 check (§6.1). It cannot
prove the port was right to begin with — only that a change did not alter what
the model computes at positions where the answer is not in doubt. For a change
that is bit-exact by construction, `scripts/greedy_snapshot.py` is stronger and
needs no reference at all.

Usage:

    # capture (once per reference model) — ~40 min for the 35B fp8 shim
    python scripts/high_margin_gate.py --capture \\
        --model Qwen/Qwen3.6-35B-A3B-FP8 \\
        --out tuning/high_margin_ref_35b_fp8.json

    # check a lib against it — run BOTH prefix-cache modes (§6)
    for m in disable radix; do
      python scripts/high_margin_gate.py --check tuning/high_margin_ref_35b_fp8.json \\
          --model-dir dist/qwen3_6-35B-A3B-q4f16_1 \\
          --model-lib dist/qwen3_6-35B-A3B-q4f16_1/lib.so --prefix-cache-mode $m
    done

Exits nonzero if any position with reference margin >= --tau disagrees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Exact-continuation sequences, arithmetic, rote lists and verbatim copy. Chosen
# so the top-1 logit gap is wide almost everywhere — that is the property that
# survives a change of quantization, and the only property worth gating on.
# Deliberately raw completions, not chat-templated: a chat turn's first token is
# a formatting choice, which is exactly the kind of near-tie this set avoids.
HIGH_MARGIN_PROMPTS = [
    "1, 1, 2, 3, 5, 8, 13, 21,",
    "2, 4, 8, 16, 32, 64, 128, 256,",
    "1, 4, 9, 16, 25, 36, 49, 64,",
    "10, 20, 30, 40, 50, 60, 70, 80,",
    "3 x 1 = 3\n3 x 2 = 6\n3 x 3 = 9\n3 x 4 = 12\n3 x 5 =",
    "Monday, Tuesday, Wednesday, Thursday,",
    "January, February, March, April, May,",
    "a, b, c, d, e, f, g, h, i,",
    "Repeat the list exactly.\nList: alpha bravo charlie delta echo foxtrot golf hotel\n"
    "List: alpha bravo charlie delta echo foxtrot golf hotel\nList: alpha",
    "Chemical symbols: hydrogen H, helium He, lithium Li, beryllium Be, "
    "boron B, carbon C, nitrogen N, oxygen",
]

DEFAULT_N = 40
DEFAULT_TAU = 2.0  # nats; margin 2.0 == top-1 is ~7.4x more likely than top-2
TAU_SWEEP = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


# ──────────────────────────────────────────────────────────────────────────────
# Capture: HF reference tokens + per-position top1/top2 margin
# ──────────────────────────────────────────────────────────────────────────────


def run_capture(args: argparse.Namespace) -> None:
    import torch

    from validate import load_hf  # installs the fp8 software-dequant shim if needed

    out_path = Path(args.out)
    if out_path.exists() and not args.regen:
        sys.exit(f"[capture] {out_path} exists; pass --regen to rebuild")

    model, tokenizer = load_hf(args.model, args.dtype, args.device)
    model.eval()

    prompts = HIGH_MARGIN_PROMPTS[: args.num_prompts]
    results = []
    t_start = time.perf_counter()

    for pi, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
        prompt_ids = input_ids[0].tolist()
        tokens: list[int] = []
        margins: list[float] = []
        runners_up: list[int] = []

        for _ in range(args.num_tokens):
            with torch.no_grad():
                logits = model(input_ids, use_cache=False).logits[0, -1].float()
            # Re-run the full sequence each step. Same reason as validate.py:
            # with use_cache=False, feeding only the new token would wipe both
            # the prompt context and the GatedDeltaNet recurrent state.
            lp = torch.log_softmax(logits, dim=-1)
            top2 = torch.topk(lp, 2)
            tok = int(top2.indices[0])
            tokens.append(tok)
            margins.append(float(top2.values[0] - top2.values[1]))
            runners_up.append(int(top2.indices[1]))
            input_ids = torch.cat(
                [input_ids, torch.tensor([[tok]], device=args.device)], dim=1
            )

        wide = sum(m >= args.tau for m in margins)
        print(
            f"[capture] {pi + 1}/{len(prompts)} "
            f"{wide}/{len(margins)} positions at margin>={args.tau} "
            f"(median {sorted(margins)[len(margins) // 2]:.2f}) "
            f"in {time.perf_counter() - t0:.1f}s"
        )
        print(f"           {prompt!r}\n           -> {tokenizer.decode(tokens)!r}")

        results.append(
            {
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "tokens": tokens,
                "token_texts": [tokenizer.decode([t]) for t in tokens],
                "margins": margins,
                "runner_up": runners_up,
                "text": tokenizer.decode(tokens),
            }
        )

    payload = {
        "model_id": args.model,
        "dtype": args.dtype,
        "num_tokens": args.num_tokens,
        "capture_tau": args.tau,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    all_margins = [m for r in results for m in r["margins"]]
    print(f"\n[capture] wrote {out_path} in {time.perf_counter() - t_start:.0f}s")
    print(f"[capture] {len(all_margins)} positions total. Margin distribution:")
    for t in TAU_SWEEP:
        n = sum(m >= t for m in all_margins)
        print(f"           margin >= {t:>4}: {n:4d}/{len(all_margins)} ({100 * n / len(all_margins):5.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# Check: teacher-forced MLC argmax at every position
# ──────────────────────────────────────────────────────────────────────────────


def run_check(args: argparse.Namespace) -> None:
    ref = json.loads(Path(args.check).read_text())
    print(
        f"[check] reference {args.check} "
        f"(model={ref['model_id']!r}, {len(ref['results'])} prompts x {ref['num_tokens']} tokens)"
    )

    from mlc_llm import MLCEngine
    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve.config import EngineConfig

    t0 = time.perf_counter()
    engine = MLCEngine(
        model=args.model_dir,
        model_lib=args.model_lib,
        device=args.device,
        mode="interactive",
        engine_config=EngineConfig(prefix_cache_mode=args.prefix_cache_mode),
    )
    print(
        f"[check] engine up in {time.perf_counter() - t0:.1f}s "
        f"(prefix_cache_mode={args.prefix_cache_mode})"
    )

    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=1)

    def forced_next(token_ids: list[int], rid: str) -> str:
        """MLC's greedy next token, as text, given exactly `token_ids`."""
        out = ""
        for deltas in engine._generate(token_ids, gen_cfg, request_id=rid):
            for d in deltas:
                out += d.delta_text
        return out

    def forced_prefix(prompt_ids: list[int], tokens: list[int], i: int) -> list[int]:
        """The prefix fed at position `i`, or a deliberately wrong one.

        `--negative-control stale1` feeds a prefix one token short while still
        scoring against the reference's token for position `i`. That is exactly
        what an off-by-one in recurrent-state history indexing looks like from
        the outside: the right prompt, a state one step stale. A gate that
        cannot fail this cannot detect the bug class it exists for.
        """
        if args.negative_control == "stale1" and i >= 1:
            return prompt_ids + tokens[: i - 1]
        return prompt_ids + tokens[:i]

    per_prompt = []
    n_wide_total = n_wide_bad = n_narrow_total = n_narrow_bad = 0
    # pass rate at each tau, so the choice of --tau is visible rather than asserted
    sweep = {t: [0, 0] for t in TAU_SWEEP}  # tau -> [scored, mismatched]

    for pi, r in enumerate(ref["results"]):
        prompt_ids = r["prompt_ids"]
        n = min(args.num_tokens, len(r["tokens"])) if args.num_tokens else len(r["tokens"])
        mism = []
        t0 = time.perf_counter()

        for i in range(n):
            expected = r["token_texts"][i]
            got = forced_next(forced_prefix(prompt_ids, r["tokens"], i), f"hm-{pi}-{i}")
            margin = r["margins"][i]
            ok = got == expected
            for t, acc in sweep.items():
                if margin >= t:
                    acc[0] += 1
                    acc[1] += not ok
            if margin >= args.tau:
                n_wide_total += 1
                n_wide_bad += not ok
            else:
                n_narrow_total += 1
                n_narrow_bad += not ok
            if not ok:
                mism.append({"pos": i, "margin": margin, "expected": expected, "got": got})

        wide_bad = [m for m in mism if m["margin"] >= args.tau]
        status = "PASS" if not wide_bad else "FAIL"
        n_wide = sum(m >= args.tau for m in r["margins"][:n])
        print(
            f"\n[check] prompt {pi + 1}: [{status}] "
            f"{n_wide - len(wide_bad)}/{n_wide} wide-margin positions agree "
            f"({len(mism)} total mismatches over {n} positions, "
            f"{time.perf_counter() - t0:.1f}s)"
        )
        print(f"         {r['prompt']!r}")
        for m in mism[: args.show]:
            tag = "WIDE" if m["margin"] >= args.tau else "tie "
            print(
                f"         {tag} pos {m['pos']:3d} margin {m['margin']:7.3f} "
                f"expected {m['expected']!r} got {m['got']!r}"
            )
        if len(mism) > args.show:
            print(f"         ... and {len(mism) - args.show} more")
        per_prompt.append({"wide": n_wide, "wide_bad": len(wide_bad), "mismatches": mism})

    engine.terminate()

    print("\n" + "=" * 72)
    print(f"[check] margin sweep ({args.model_lib}, prefix_cache_mode={args.prefix_cache_mode})")
    print("         tau   scored  mismatch   agreement")
    for t in TAU_SWEEP:
        scored, bad = sweep[t]
        rate = 100 * (scored - bad) / scored if scored else float("nan")
        print(f"        {t:>5}   {scored:5d}     {bad:5d}     {rate:6.2f}%")
    print(
        f"\n[check] at tau={args.tau}: {n_wide_total - n_wide_bad}/{n_wide_total} wide-margin "
        f"positions agree; near-ties {n_narrow_total - n_narrow_bad}/{n_narrow_total} "
        f"(informational — near-ties are where quantization legitimately differs)"
    )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "model_lib": args.model_lib,
                    "prefix_cache_mode": args.prefix_cache_mode,
                    "tau": args.tau,
                    "wide_total": n_wide_total,
                    "wide_bad": n_wide_bad,
                    "narrow_total": n_narrow_total,
                    "narrow_bad": n_narrow_bad,
                    "sweep": {str(k): v for k, v in sweep.items()},
                    "per_prompt": per_prompt,
                },
                indent=2,
            )
        )
        print(f"[check] wrote {args.json_out}")

    if n_wide_bad > args.allow:
        sys.exit(
            f"\n[check] FAIL — {n_wide_bad} wide-margin disagreement(s), "
            f"allowance {args.allow}. These are not quantization near-ties."
        )
    print("\n[check] PASS")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true", help="build the reference from an HF model")
    mode.add_argument("--check", metavar="REF_JSON", help="check an MLC lib against a reference")

    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8", help="HF model id (--capture)")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--out", default="tuning/high_margin_ref.json", help="reference path (--capture)")
    p.add_argument("--regen", action="store_true", help="overwrite an existing reference")
    p.add_argument("--num-prompts", type=int, default=len(HIGH_MARGIN_PROMPTS))
    p.add_argument(
        "--num-tokens", type=int, default=DEFAULT_N,
        help="positions per prompt; on --check, 0 means 'all in the reference'",
    )

    p.add_argument("--model-dir", help="compiled MLC model dir (--check)")
    p.add_argument("--model-lib", help="explicit .so — never glob (--check)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--prefix-cache-mode", default="radix", choices=["disable", "radix"],
        help="run BOTH on any recurrent-state change; 'disable' is blind to a whole "
             "bug class (workplan §6). Teacher forcing under 'radix' also exercises "
             "the history path hard: every position extends the previous by one token.",
    )
    p.add_argument("--tau", type=float, default=DEFAULT_TAU, help="margin (nats) to score at")
    p.add_argument("--allow", type=int, default=0, help="tolerated wide-margin disagreements")
    p.add_argument("--show", type=int, default=6, help="mismatches printed per prompt")
    p.add_argument("--json-out", default=None)
    p.add_argument(
        "--negative-control", default=None, choices=["stale1"],
        help="deliberately break the check to prove it is not vacuous. 'stale1' "
             "feeds a prefix one token short — the outward signature of an "
             "off-by-one in history-state indexing. Expect a loud FAIL.",
    )

    args = p.parse_args()
    if args.capture:
        run_capture(args)
    else:
        if not args.model_dir or not args.model_lib:
            sys.exit("--check needs --model-dir and --model-lib")
        run_check(args)


if __name__ == "__main__":
    main()

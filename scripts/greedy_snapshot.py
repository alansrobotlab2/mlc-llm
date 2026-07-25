#!/usr/bin/env python3
"""Byte-exact before/after gate for refactors that must not change numerics.

Captures greedy token IDs for a fixed prompt set into JSON, then diffs a later
run against it. This is the right gate for a change that is *bit-exact by
construction* (a weight-layout merge, a kernel fusion, a scheduling change):
"same lib, same prompts, identical token IDs" is stronger than a tolerance-based
HF comparison and needs no reference model at all.

It is NOT a substitute for `validate.py --greedy-parity` against HuggingFace —
that is what catches a port being wrong in the first place. Use this to prove a
*refactor* of already-validated code changed nothing.

Capture a baseline before touching code:
    python scripts/greedy_snapshot.py --model-dir dist/qwen3_6-35B-A3B-q4f16_1 \\
        --model-lib dist/qwen3_6-35B-A3B-q4f16_1/lib.so \\
        --out tuning/greedy_35b_before.json

Then after the change:
    python scripts/greedy_snapshot.py --model-dir ... --model-lib ... \\
        --compare tuning/greedy_35b_before.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Fixed set. Mixed prompt shapes on purpose: a short high-margin factual, a
# generation task with many near-ties, an arithmetic prompt (near-ties in the
# logits are where quantization noise shows up first), and a long-ish prompt so
# the run covers more than one prefill tile.
PROMPTS = [
    "The capital of France is",
    "Write a one-sentence definition of a transformer neural network:",
    "Q: What is the largest planet in our solar system? A:",
    "Compute the first ten Fibonacci numbers, comma separated:",
    "Explain in two sentences why memory bandwidth, not compute, "
    "limits single-batch LLM decoding on an embedded GPU:",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--model-lib", required=True,
                   help="explicit .so — never glob, a dist dir may hold lib.so and lib_nofi.so")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--out", default=None, help="write snapshot JSON here")
    p.add_argument("--compare", default=None, help="diff against this snapshot and exit nonzero on mismatch")
    args = p.parse_args()

    if not args.out and not args.compare:
        sys.exit("pass --out to capture a baseline or --compare to check against one")

    from mlc_llm import MLCEngine
    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve.config import EngineConfig

    t0 = time.perf_counter()
    print(f"[snap] loading {args.model_dir} / {Path(args.model_lib).name}", flush=True)
    engine = MLCEngine(
        model=args.model_dir,
        model_lib=args.model_lib,
        device=args.device,
        mode="interactive",
        # radix prefix cache + GDN rnn_state deadlock; see qwen3_5.md §9
        engine_config=EngineConfig(prefix_cache_mode="disable"),
    )
    print(f"[snap] engine up in {time.perf_counter() - t0:.1f}s", flush=True)

    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    tok = engine.tokenizer

    results = []
    for i, prompt in enumerate(PROMPTS):
        text = ""
        for deltas in engine._generate(prompt, gen_cfg, request_id=f"snap-{i}"):
            for d in deltas:
                if d.delta_text:
                    text += d.delta_text
        ids = tok.encode(text) if tok is not None else None
        results.append({"prompt": prompt, "text": text, "ids": ids})
        print(f"[snap] {i}: {text[:70]!r}...", flush=True)

    engine.terminate()

    snap = {
        "model_dir": args.model_dir,
        "model_lib": args.model_lib,
        "max_tokens": args.max_tokens,
        "results": results,
    }

    if args.out:
        Path(args.out).write_text(json.dumps(snap, indent=2))
        print(f"\n[snap] wrote {args.out}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        if base["max_tokens"] != args.max_tokens:
            sys.exit(f"max_tokens differs: baseline {base['max_tokens']} vs {args.max_tokens}")
        bad = 0
        for i, (b, n) in enumerate(zip(base["results"], results)):
            if b["text"] == n["text"]:
                print(f"  prompt {i}: IDENTICAL")
                continue
            bad += 1
            # first divergent character, so the report says where it split
            j = next((k for k in range(min(len(b["text"]), len(n["text"])))
                      if b["text"][k] != n["text"][k]), min(len(b["text"]), len(n["text"])))
            print(f"  prompt {i}: DIVERGES at char {j}")
            print(f"    baseline: {b['text'][max(0, j - 40):j + 40]!r}")
            print(f"    current : {n['text'][max(0, j - 40):j + 40]!r}")
        if bad:
            sys.exit(f"\n{bad}/{len(results)} prompts diverged — the change is NOT bit-exact")
        print(f"\nall {len(results)} prompts byte-identical")


if __name__ == "__main__":
    main()

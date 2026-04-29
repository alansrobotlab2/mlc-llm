"""Run a single engine, output 5 prompts × N greedy tokens to a JSON file.

Used as a subprocess from scratch_phase6_parity_compare.py to avoid mixing
two engines in the same Python process (causes hangs after engine.terminate()).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

PROMPTS = [
    "The capital of France is",
    "Quantum mechanics is the branch of physics that",
    "def fibonacci(n):\n    ",
    "Once upon a time in a small village,",
    "The integral of x squared from 0 to 1 equals",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-lib", required=True)
    parser.add_argument("--n-tokens", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from mlc_llm import MLCEngine
    print(f"loading engine from {args.model_dir}", flush=True)
    t0 = time.time()
    eng = MLCEngine(args.model_dir, mode="interactive", model_lib=args.model_lib)
    print(f"loaded dt={time.time()-t0:.1f}s", flush=True)

    outs = []
    for i, p in enumerate(PROMPTS):
        t0 = time.time()
        out = ""
        for tok in eng.chat.completions.create(
            stream=True,
            messages=[{"role": "user", "content": p}],
            max_tokens=args.n_tokens,
            temperature=0.0,
        ):
            delta = tok.choices[0].delta.content
            if delta:
                out += delta
        outs.append(out)
        snippet = out[:80] + ("…" if len(out) > 80 else "")
        print(f"  [{i}] dt={time.time()-t0:.1f}s : {snippet!r}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"prompts": PROMPTS, "outputs": outs}, f, indent=2)
    print(f"saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()

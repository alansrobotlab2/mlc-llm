"""Phase 7 parity: greedy decode 5 prompts x 50 tokens, fp16-TIR vs mxfp4 KV.

Mirrors scratch_phase6_parity.py — same prompts, same gate.
"""
from __future__ import annotations

import argparse
import sys
import time

PROMPTS = [
    "The capital of France is",
    "Quantum mechanics is the branch of physics that",
    "def fibonacci(n):\n    ",
    "Once upon a time in a small village,",
    "The integral of x squared from 0 to 1 equals",
]


def _say(msg: str) -> None:
    print(msg, flush=True, file=sys.stdout)
    sys.stdout.flush()


def run_engine(model_dir: str, lib_path: str, n_tokens: int) -> list[str]:
    from mlc_llm import MLCEngine
    eng = MLCEngine(model_dir, mode="interactive", model_lib=lib_path)
    outs: list[str] = []
    for i, p in enumerate(PROMPTS):
        t0 = time.time()
        out = ""
        for tok in eng.chat.completions.create(
            stream=True,
            messages=[{"role": "user", "content": p}],
            max_tokens=n_tokens,
            temperature=0.0,
        ):
            delta = tok.choices[0].delta.content
            if delta:
                out += delta
        outs.append(out)
        snippet = out[:80] + ("…" if len(out) > 80 else "")
        _say(f"  [{i}] dt={time.time()-t0:.1f}s : {snippet!r}")
    eng.terminate()
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--ref-lib", required=True)
    parser.add_argument("--cmp-dir", required=True)
    parser.add_argument("--cmp-lib", required=True)
    parser.add_argument("--n-tokens", type=int, default=50)
    args = parser.parse_args()

    _say(f"[parity] {len(PROMPTS)} prompts × {args.n_tokens} tokens")
    _say(f"  ref:  {args.ref_dir}")
    _say(f"  cmp:  {args.cmp_dir}")

    _say("\n[ref] loading engine...")
    t0 = time.time()
    ref_outs = run_engine(args.ref_dir, args.ref_lib, args.n_tokens)
    _say(f"[ref] total dt={time.time()-t0:.1f}s")

    _say("\n[cmp] loading engine...")
    t0 = time.time()
    cmp_outs = run_engine(args.cmp_dir, args.cmp_lib, args.n_tokens)
    _say(f"[cmp] total dt={time.time()-t0:.1f}s")

    _say("\n=== PARITY ===")
    n_pass = 0
    for i in range(len(PROMPTS)):
        if ref_outs[i] == cmp_outs[i]:
            n_pass += 1
            _say(f"[{i}] EXACT match ({len(ref_outs[i])} chars)")
        else:
            lcp = 0
            for a, b in zip(ref_outs[i], cmp_outs[i]):
                if a == b:
                    lcp += 1
                else:
                    break
            _say(f"[{i}] DIVERGE at char {lcp} (ref={len(ref_outs[i])} cmp={len(cmp_outs[i])})")
            _say(f"     ref: {ref_outs[i]!r}")
            _say(f"     cmp: {cmp_outs[i]!r}")
    _say(f"\n=== {n_pass}/{len(PROMPTS)} EXACT (Phase 7 gate is >=2/5; 4-bit drift expected) ===")


if __name__ == "__main__":
    main()

"""Tiny smoke: load engine + 1 prompt × 10 tokens. Confirms a pre-Phase-7
lib still boots against the rebuilt TVM runtime (which now passes a new
is_mxfp4_kv bool into the PagedKVCache constructor, dispatched server-side
from the StringImm dtype_kv arg).
"""
from __future__ import annotations

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-lib", required=True)
    parser.add_argument("--n-tokens", type=int, default=10)
    args = parser.parse_args()

    from mlc_llm import MLCEngine

    print(f"[smoke] loading {args.model_lib}", flush=True)
    t0 = time.time()
    eng = MLCEngine(args.model_dir, mode="interactive", model_lib=args.model_lib)
    print(f"[smoke] engine load dt={time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    out = ""
    for tok in eng.chat.completions.create(
        stream=True,
        messages=[{"role": "user", "content": "The capital of France is"}],
        max_tokens=args.n_tokens,
        temperature=0.0,
    ):
        delta = tok.choices[0].delta.content
        if delta:
            out += delta
    print(f"[smoke] gen dt={time.time() - t0:.1f}s", flush=True)
    print(f"[smoke] out: {out!r}", flush=True)

    if not out.strip():
        print("[smoke] FAIL: empty output", flush=True)
        sys.exit(1)
    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    main()

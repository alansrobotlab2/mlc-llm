#!/usr/bin/env python3
"""Parameterized decode profiler for ncu/nsys.

Unlike repo profile_decode.py this takes explicit --model-dir/--model-lib
(the 35B dist dir holds both lib.so and lib_nofi.so, so glob("*.so")[0] is unsafe)
and puts a distinct NVTX range around steady-state decode only.

  ncu --nvtx --nvtx-include "decode-steady/" --metrics <...> \
      python prof_decode35.py --model-dir ... --model-lib ...
"""
from __future__ import annotations

import argparse

try:
    import torch.cuda.nvtx as nvtx
except Exception:  # torch optional
    class _N:
        @staticmethod
        def range_push(*a, **k):
            pass

        @staticmethod
        def range_pop(*a, **k):
            pass

    nvtx = _N()


def drain(engine, prompt, gen_cfg, request_id, n_tokens, mark_after):
    """Generate n_tokens; open an NVTX range once `mark_after` tokens have landed
    so the marked region is steady-state decode, not prefill."""
    n = 0
    marked = False
    for d in engine._generate(prompt, gen_cfg, request_id=request_id):
        for delta in d:
            if delta.delta_text:
                n += 1
                if not marked and n >= mark_after:
                    nvtx.range_push("decode-steady")
                    marked = True
        if n >= n_tokens:
            break
    if marked:
        nvtx.range_pop()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-lib", required=True)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--mark-after", type=int, default=8)
    ap.add_argument("--prefix-cache-mode", default="disable", choices=["disable", "radix"])
    args = ap.parse_args()

    from mlc_llm import MLCEngine
    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve.config import EngineConfig

    engine = MLCEngine(
        model=args.model_dir,
        model_lib=args.model_lib,
        device="cuda:0",
        mode="interactive",
        engine_config=EngineConfig(prefix_cache_mode=args.prefix_cache_mode),
    )

    base = ("The quick brown fox jumps over the lazy dog. " * 200)

    def make_prompt(salt: str) -> str:
        # Under radix an identical prompt makes the second request a full cache hit, so
        # the trace would hold no prefill at all. Salt keeps the shared prefix at ~0.
        text = (f"Archive record {salt}. " + base) if args.prefix_cache_mode != "disable" else base
        # trim to roughly prompt_len tokens by characters; exactness not required
        return text[: args.prompt_len * 4]

    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=args.tokens)

    n = drain(engine, make_prompt("warm"), gen_cfg, "warmup", args.tokens, 10**9)
    print(f"[profile] warmup: {n} tokens")

    n = drain(engine, make_prompt("prof"), gen_cfg, "prof", args.tokens, args.mark_after)
    print(f"[profile] profiled: {n} tokens")

    engine.terminate()


if __name__ == "__main__":
    main()

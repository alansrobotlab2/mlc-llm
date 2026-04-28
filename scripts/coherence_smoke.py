#!/usr/bin/env python3
"""Coherence smoke for the 35B-A3B v3 lib (post parallel topk_softmax).

Uses the same engine pattern as bench_mlc.py (interactive mode +
prefix_cache_mode='disable') to avoid the radix-cache + GDN deadlock.
Greedy-decodes 80 tokens for 3 prompts and prints the text so a human can
sanity-check that the generations are coherent English.

Usage:
    .venv/bin/python scripts/coherence_smoke.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="dist/qwen3_6-35B-A3B-q4f16_1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-tokens", type=int, default=80)
    args = p.parse_args()

    from mlc_llm import MLCEngine
    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve.config import EngineConfig

    model_dir = Path(args.model_dir)
    lib_path = str(next(model_dir.glob("*.so")))

    import time as _time
    t0 = _time.perf_counter()
    print(f"[smoke] {_time.strftime('%H:%M:%S')} loading engine from {model_dir} on {args.device}", flush=True)
    engine = MLCEngine(
        model=str(model_dir),
        model_lib=lib_path,
        device=args.device,
        mode="interactive",
        engine_config=EngineConfig(prefix_cache_mode="disable"),
    )
    print(f"[smoke] {_time.strftime('%H:%M:%S')} engine constructed in {_time.perf_counter()-t0:.1f}s", flush=True)

    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    prompts = [
        "The capital of France is",
        "Write a one-sentence definition of a transformer neural network:",
        "Q: What is the largest planet in our solar system? A:",
    ]

    for i, prompt in enumerate(prompts):
        ts = _time.strftime('%H:%M:%S')
        t1 = _time.perf_counter()
        print(f"\n--- prompt {i} @ {ts} ---", flush=True)
        print(f"PROMPT: {prompt}", flush=True)
        text = ""
        n_chunks = 0
        for delta_outputs in engine._generate(prompt, gen_cfg, request_id=f"smoke-{i}"):
            n_chunks += 1
            for delta in delta_outputs:
                if delta.delta_text:
                    text += delta.delta_text
        dt = _time.perf_counter() - t1
        print(f"OUTPUT ({dt:.1f}s, {n_chunks} chunks): {text}", flush=True)

    engine.terminate()
    print(f"\n[smoke] {_time.strftime('%H:%M:%S')} done", flush=True)


if __name__ == "__main__":
    main()

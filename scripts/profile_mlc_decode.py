#!/usr/bin/env python3
"""Profiler-bounded decode harness for nsys / ncu.

Loads an MLC engine, runs one untimed warmup pass, then brackets a single
timed decode pass with cudaProfilerStart/Stop so external profilers can
restrict capture to the steady-state decode window (skipping the multi-minute
engine warmup and the params load).

Usage with nsys:
  nsys profile -o decode_trace --capture-range=cudaProfilerApi \\
      --capture-range-end=stop --trace=cuda,nvtx,osrt \\
      .venv/bin/python scripts/profile_mlc_decode.py \\
      --model-dir dist/qwen3_6-35B-A3B-q4f16_1 --pp 128 --tg 64
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path


def _cudart():
    return ctypes.CDLL("libcudart.so")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pp", type=int, default=128)
    p.add_argument("--tg", type=int, default=64)
    p.add_argument("--warmup-tg", type=int, default=8)
    args = p.parse_args()

    from mlc_llm import MLCEngine
    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve.config import EngineConfig
    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir)
    so_files = list(model_dir.glob("*.so"))
    if not so_files:
        print(f"[profile] no .so under {model_dir}", file=sys.stderr)
        sys.exit(1)
    lib_path = str(so_files[0])

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tokenizer.encode(filler, add_special_tokens=False)
    while len(ids) < args.pp:
        ids = ids + ids
    prompt = tokenizer.decode(ids[: args.pp])

    print(f"[profile] loading engine device={args.device}")
    engine = MLCEngine(
        model=str(model_dir),
        model_lib=lib_path,
        device=args.device,
        mode="interactive",
        engine_config=EngineConfig(prefix_cache_mode="disable"),
    )

    # Untimed warmup (kernel selection, JIT, KV cache populate)
    print(f"[profile] warmup pass tg={args.warmup_tg}")
    warmup_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=args.warmup_tg)
    for _ in engine._generate(prompt, warmup_cfg, request_id="warmup"):
        pass

    # Timed pass under cudaProfilerStart / Stop
    print(f"[profile] timed pass tg={args.tg} (cudaProfilerStart)")
    cudart = _cudart()
    cudart.cudaProfilerStart()
    t0 = time.perf_counter()
    n = 0
    ttft = None
    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=args.tg)
    for delta_outputs in engine._generate(prompt, gen_cfg, request_id="timed"):
        for delta in delta_outputs:
            if delta.delta_text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    total = time.perf_counter() - t0
    cudart.cudaProfilerStop()
    print(f"[profile] cudaProfilerStop")

    decode_t = total - (ttft or 0.0)
    n_dec = max(n - 1, 1)
    print(f"[profile] tokens={n} ttft={ttft*1000:.1f}ms decode={decode_t*1000:.1f}ms "
          f"pp_tps={args.pp/(ttft or 1):.2f} tg_tps={n_dec/decode_t:.2f}")
    engine.terminate()


if __name__ == "__main__":
    main()

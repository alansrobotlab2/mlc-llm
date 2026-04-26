#!/usr/bin/env python3
"""Minimal decode profiler harness.

Loads the q4f16_g16e model, runs a short prefill+decode trace with NVTX
markers around the steady-state decode region. Run under nsys:

  nsys profile -t cuda,nvtx --cuda-graph-trace=node \
      -o /tmp/qwen35_decode -f true \
      .venv/bin/python profile_decode.py

Then:
  nsys stats --report=cuda_gpu_kern_sum --format=csv /tmp/qwen35_decode.nsys-rep
"""
from __future__ import annotations
from pathlib import Path

try:
    import torch.cuda.nvtx as nvtx
except Exception:
    class _N:
        @staticmethod
        def range_push(*a, **k): pass
        @staticmethod
        def range_pop(*a, **k): pass
    nvtx = _N()


DECODE_TOKENS = 64


def run(engine, prompt: str, gen_cfg, request_id: str, label: str) -> int:
    nvtx.range_push(label)
    n = 0
    for d in engine._generate(prompt, gen_cfg, request_id=request_id):
        for delta in d:
            if delta.delta_text:
                n += 1
        if n >= DECODE_TOKENS:
            break
    nvtx.range_pop()
    return n


def main() -> None:
    model_dir = Path("dist/qwen3_5-0.8B-q4f16_g16e")
    lib = next(model_dir.glob("*.so"))

    from mlc_llm import MLCEngine
    from mlc_llm.protocol.generation_config import GenerationConfig

    print(f"[profile] Loading {model_dir}")
    engine = MLCEngine(
        model=str(model_dir),
        model_lib=str(lib),
        device="cuda:0",
        mode="interactive",
    )

    prompt = "The quick brown fox jumps over the lazy dog."
    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=DECODE_TOKENS)

    n_warm = run(engine, prompt, gen_cfg, "warmup", "warmup-full-pass")
    print(f"[profile] warmup: {n_warm} tokens")

    n = run(engine, prompt, gen_cfg, "prof", "profiled-prefill-and-decode")
    print(f"[profile] profiled: {n} tokens")

    engine.terminate()


if __name__ == "__main__":
    main()

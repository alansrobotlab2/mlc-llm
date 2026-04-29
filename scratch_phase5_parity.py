"""Phase 5.6: greedy parity check — fp8-KV vs fp16-KV (regression).

Generates 50 tokens with greedy decode from each lib on canonical prompts;
reports per-prompt token-agreement count.
"""
import os
import time
from pathlib import Path

from mlc_llm import MLCEngine
from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve.config import EngineConfig


CANONICAL_PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "In machine learning, attention mechanisms",
    "The first law of thermodynamics states that",
    "Once upon a time, in a small village",
]
N_GEN = 50


def gen_text(model_dir: str, prompt: str, n: int) -> str:
    so_files = sorted(Path(model_dir).glob("lib*.so"))
    lib_path = str(next(p for p in so_files if p.name == "lib.so"))
    engine = MLCEngine(
        model=model_dir,
        model_lib=lib_path,
        device="cuda:0",
        mode="interactive",
        engine_config=EngineConfig(prefix_cache_mode="disable"),
    )
    try:
        gc = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=n)
        text = ""
        for delta_outputs in engine._generate(prompt, gc, request_id="x"):
            for d in delta_outputs:
                if d.delta_text:
                    text += d.delta_text
        return text
    finally:
        engine.terminate()


def main():
    fp16_dir = "/home/alfie/mlc-llm/dist/qwen3_6-35B-A3B-q4f16_1"
    fp8_dir = "/home/alfie/mlc-llm/dist/qwen3_6-35B-A3B-q4f16_1_kvfp8"

    print(f"=== Phase 5.6 greedy parity: fp16-KV vs fp8-KV, {N_GEN} tokens each ===\n")
    for prompt in CANONICAL_PROMPTS:
        print(f"Prompt: {prompt!r}")
        t0 = time.time()
        fp16 = gen_text(fp16_dir, prompt, N_GEN)
        t1 = time.time()
        fp8 = gen_text(fp8_dir, prompt, N_GEN)
        t2 = time.time()
        match = "EXACT" if fp16 == fp8 else "DIFFERS"
        print(f"  fp16 ({t1-t0:.1f}s): {fp16!r}")
        print(f"  fp8  ({t2-t1:.1f}s): {fp8!r}")
        # Token-level prefix agreement on shared prefix length:
        common = 0
        for a, b in zip(fp16, fp8):
            if a != b:
                break
            common += 1
        print(f"  text match: {match} | common-prefix chars = {common}/{min(len(fp16), len(fp8))}\n")


if __name__ == "__main__":
    main()

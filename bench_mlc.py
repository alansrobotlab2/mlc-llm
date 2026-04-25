#!/usr/bin/env python3
"""
MLC bench harness — matches llama-bench output format.

Two metrics:
  pp_tps  : prompt processing (prefill) tokens/sec — derived from time-to-first-token
  tg_tps  : token generation (decode) tokens/sec — derived from steady-state per-token deltas

Run:
  source .envrc.local && .venv/bin/python bench_mlc.py \\
      --model-dir dist/qwen3_6-35B-A3B-q4f16_1 \\
      --device cuda:0 \\
      --pp 512 --tg 128 --runs 3
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pp", type=int, default=512, help="Prompt length in tokens")
    p.add_argument("--tg", type=int, default=128, help="Tokens to generate")
    p.add_argument("--runs", type=int, default=3, help="Repetitions (median reported)")
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args()


def build_prompt(tokenizer, target_len: int) -> str:
    # Use deterministic filler text. Keep tokenizing until we hit target length.
    # Avoids special tokens / chat-template confusion.
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tokenizer.encode(filler, add_special_tokens=False)
    if len(ids) < target_len:
        # Extend
        filler = filler * (target_len // len(ids) + 2)
        ids = tokenizer.encode(filler, add_special_tokens=False)
    ids = ids[:target_len]
    return tokenizer.decode(ids), len(ids)


def time_run(engine, prompt: str, gen_cfg, request_id: str) -> tuple[float, float, int, int]:
    """Returns (ttft_s, total_s, n_decode_tokens, n_prefill_tokens_seen)."""
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    for delta_outputs in engine._generate(prompt, gen_cfg, request_id=request_id):
        for delta in delta_outputs:
            if delta.delta_text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_tokens += 1
    total = time.perf_counter() - t0
    return ttft, total, n_tokens


def main() -> None:
    args = parse_args()

    try:
        from mlc_llm import MLCEngine
        from mlc_llm.protocol.generation_config import GenerationConfig
    except ImportError:
        print("[mlc] cannot import mlc_llm.", file=sys.stderr)
        sys.exit(1)

    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir)
    so_files = list(model_dir.glob("*.so"))
    if not so_files:
        print(f"[mlc] No .so under {model_dir}", file=sys.stderr)
        sys.exit(1)
    lib_path = str(so_files[0])

    print(f"[mlc] Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print(f"[mlc] Loading engine: device={args.device} lib={lib_path}")
    engine = MLCEngine(
        model=str(model_dir),
        model_lib=lib_path,
        device=args.device,
        mode="interactive",
    )

    prompt, prompt_len = build_prompt(tokenizer, args.pp)
    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=args.tg)
    print(f"[mlc] Prompt length {prompt_len} tokens; generating {args.tg} tokens; {args.runs} runs (after {args.warmup} warmup)")

    # Warmup
    for w in range(args.warmup):
        ttft, total, ntok = time_run(engine, prompt, gen_cfg, f"warmup-{w}")
        print(f"[mlc] warmup {w}: ttft={ttft*1000:.1f}ms total={total*1000:.1f}ms tokens={ntok}")

    # Measured runs
    pp_tps_samples = []
    tg_tps_samples = []
    total_samples = []
    for r in range(args.runs):
        ttft, total, ntok = time_run(engine, prompt, gen_cfg, f"run-{r}")
        decode_t = total - ttft
        # n_decode_tokens excludes the first token (which was bundled with prefill)
        n_decode = max(ntok - 1, 1)
        pp_tps = prompt_len / ttft if ttft > 0 else float("inf")
        tg_tps = n_decode / decode_t if decode_t > 0 else float("inf")
        pp_tps_samples.append(pp_tps)
        tg_tps_samples.append(tg_tps)
        total_samples.append(total)
        print(f"[mlc] run {r}: ttft={ttft*1000:.1f}ms decode={decode_t*1000:.1f}ms tokens={ntok}  "
              f"pp_tps={pp_tps:.2f}  tg_tps={tg_tps:.2f}")

    engine.terminate()

    print()
    print(f"=== MLC q4f16_1 — pp={args.pp} tg={args.tg} runs={args.runs} ===")
    print(f"  pp_tps: median={statistics.median(pp_tps_samples):.2f}  min={min(pp_tps_samples):.2f}  max={max(pp_tps_samples):.2f}")
    print(f"  tg_tps: median={statistics.median(tg_tps_samples):.2f}  min={min(tg_tps_samples):.2f}  max={max(tg_tps_samples):.2f}")
    print(f"  wall:   median={statistics.median(total_samples)*1000:.1f}ms")


if __name__ == "__main__":
    main()

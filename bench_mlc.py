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
    p.add_argument(
        "--pp",
        default="512",
        help="Prompt length in tokens. Comma-separated to bench multiple contexts in one engine.",
    )
    p.add_argument("--tg", type=int, default=128, help="Tokens to generate")
    p.add_argument("--runs", type=int, default=3, help="Repetitions (median reported)")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--label", default="MLC", help="Label for the summary header")
    p.add_argument("--json-out", help="Optional path to write JSON summary {pp: {pp_tps, tg_tps}}.")
    p.add_argument("--baseline", help="Path to baseline JSON; print delta vs baseline.")
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
        from mlc_llm.serve.config import EngineConfig
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

    # Disable radix prefix caching: hybrid GDN models cannot roll back rnn_state
    # after a multi-token prefill, so the engine's PopN-on-prefix-match path crashes.
    print(f"[mlc] Loading engine: device={args.device} lib={lib_path}")
    engine = MLCEngine(
        model=str(model_dir),
        model_lib=lib_path,
        device=args.device,
        mode="interactive",
        engine_config=EngineConfig(prefix_cache_mode="disable"),
    )

    pp_values = [int(x) for x in args.pp.split(",") if x.strip()]
    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=args.tg)

    summaries: list[tuple[int, float, float]] = []  # (pp, pp_tps_median, tg_tps_median)

    for pp_target in pp_values:
        prompt, prompt_len = build_prompt(tokenizer, pp_target)
        print(f"\n[mlc] pp={prompt_len}  tg={args.tg}  runs={args.runs} (warmup={args.warmup})")

        for w in range(args.warmup):
            ttft, total, ntok = time_run(engine, prompt, gen_cfg, f"warmup-pp{prompt_len}-{w}")
            print(f"[mlc]   warmup {w}: ttft={ttft*1000:.1f}ms total={total*1000:.1f}ms tokens={ntok}")

        pp_tps_samples = []
        tg_tps_samples = []
        for r in range(args.runs):
            ttft, total, ntok = time_run(engine, prompt, gen_cfg, f"run-pp{prompt_len}-{r}")
            decode_t = total - ttft
            n_decode = max(ntok - 1, 1)
            pp_tps = prompt_len / ttft if ttft > 0 else float("inf")
            tg_tps = n_decode / decode_t if decode_t > 0 else float("inf")
            pp_tps_samples.append(pp_tps)
            tg_tps_samples.append(tg_tps)
            print(f"[mlc]   run {r}: ttft={ttft*1000:.1f}ms decode={decode_t*1000:.1f}ms "
                  f"tokens={ntok}  pp_tps={pp_tps:.2f}  tg_tps={tg_tps:.2f}")

        summaries.append((prompt_len, statistics.median(pp_tps_samples), statistics.median(tg_tps_samples)))

    engine.terminate()

    print()
    print(f"=== {args.label} — tg={args.tg} runs={args.runs} ===")
    print(f"{'pp':>8}  {'pp_tps':>10}  {'tg_tps':>10}")
    for pp, pp_tps, tg_tps in summaries:
        print(f"{pp:>8}  {pp_tps:>10.2f}  {tg_tps:>10.2f}")

    if args.baseline:
        import json as _json
        try:
            base = _json.loads(Path(args.baseline).read_text())
        except FileNotFoundError:
            print(f"[mlc] baseline missing: {args.baseline}", file=sys.stderr)
            sys.exit(1)
        print()
        print(f"=== vs baseline {args.baseline} ===")
        print(f"{'pp':>8}  {'pp_tps':>14}  {'tg_tps':>14}")
        for pp, pp_tps, tg_tps in summaries:
            entry = base.get(str(pp))
            if entry is None:
                print(f"{pp:>8}  {pp_tps:>14.2f}  {tg_tps:>14.2f}  (NEW)")
                continue
            ref_pp = entry.get("pp_tps", float("nan"))
            ref_tg = entry.get("tg_tps", float("nan"))
            d_pp = (pp_tps - ref_pp) / ref_pp * 100.0 if ref_pp else float("nan")
            d_tg = (tg_tps - ref_tg) / ref_tg * 100.0 if ref_tg else float("nan")
            print(f"{pp:>8}  {pp_tps:>8.2f}({d_pp:+5.1f}%)  {tg_tps:>8.2f}({d_tg:+5.1f}%)")

    if args.json_out:
        import json
        Path(args.json_out).write_text(
            json.dumps({str(pp): {"pp_tps": pp_tps, "tg_tps": tg_tps}
                        for pp, pp_tps, tg_tps in summaries}, indent=2)
        )


if __name__ == "__main__":
    main()

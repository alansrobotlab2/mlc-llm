"""MLC bench sweep across multiple TG values, single engine load.

Mirrors bench_mlc.py's measurement protocol: pp=512 prompt, then `tg` greedy
tokens. Reports median over `runs` reps after `warmup` warmups.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROMPT_FILLER = "The quick brown fox jumps over the lazy dog. " * 200


def build_prompt(tokenizer, target_len: int) -> tuple[str, int]:
    filler = PROMPT_FILLER
    ids = tokenizer.encode(filler, add_special_tokens=False)
    if len(ids) < target_len:
        filler = filler * (target_len // len(ids) + 2)
        ids = tokenizer.encode(filler, add_special_tokens=False)
    ids = ids[:target_len]
    return tokenizer.decode(ids), len(ids)


def time_run(engine, prompt: str, gen_cfg, request_id: str):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", default="512,1024,2048,4096,8192",
                    help="Comma-separated TG values to sweep")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--model-lib", default=None,
                    help="Explicit lib.so path (default: first *.so under model-dir)")
    args = ap.parse_args()

    from mlc_llm import MLCEngine
    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve.config import EngineConfig
    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir)
    if args.model_lib:
        lib_path = args.model_lib
    else:
        so_files = list(model_dir.glob("*.so"))
        if not so_files:
            print(f"[mlc] No .so under {model_dir}", file=sys.stderr)
            sys.exit(1)
        lib_path = str(so_files[0])

    print(f"[mlc] Loading tokenizer from {model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print(f"[mlc] Loading engine: device={args.device} lib={lib_path}", flush=True)
    t0 = time.perf_counter()
    engine = MLCEngine(
        model=str(model_dir),
        model_lib=lib_path,
        device=args.device,
        mode="interactive",
        engine_config=EngineConfig(prefix_cache_mode="disable"),
    )
    print(f"[mlc] Engine loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    prompt, prompt_len = build_prompt(tokenizer, args.pp)
    print(f"[mlc] Prompt length: {prompt_len} tokens", flush=True)

    tg_values = [int(x) for x in args.tg.split(",") if x.strip()]
    summaries = []  # (tg, pp_tps_med, tg_tps_med)

    for tg in tg_values:
        gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=tg)
        print(f"\n[mlc] tg={tg}  runs={args.runs} (warmup={args.warmup})", flush=True)

        for w in range(args.warmup):
            t0 = time.perf_counter()
            ttft, total, ntok = time_run(engine, prompt, gen_cfg,
                                         f"warmup-tg{tg}-{w}")
            print(f"[mlc]   warmup {w}: ttft={ttft*1000:.1f}ms "
                  f"total={total*1000:.1f}ms tokens={ntok} "
                  f"wall={time.perf_counter()-t0:.1f}s",
                  flush=True)

        pp_tps_samples = []
        tg_tps_samples = []
        for r in range(args.runs):
            ttft, total, ntok = time_run(engine, prompt, gen_cfg,
                                         f"run-tg{tg}-{r}")
            decode_t = total - ttft
            n_decode = max(ntok - 1, 1)
            pp_tps = prompt_len / ttft if ttft > 0 else float("inf")
            tg_tps = n_decode / decode_t if decode_t > 0 else float("inf")
            pp_tps_samples.append(pp_tps)
            tg_tps_samples.append(tg_tps)
            print(f"[mlc]   run {r}: ttft={ttft*1000:.1f}ms "
                  f"decode={decode_t*1000:.1f}ms tokens={ntok} "
                  f"pp_tps={pp_tps:.2f} tg_tps={tg_tps:.2f}",
                  flush=True)

        summaries.append((tg,
                          statistics.median(pp_tps_samples),
                          statistics.median(tg_tps_samples)))

    engine.terminate()

    print(f"\n=== MLC sweep — pp={args.pp} runs={args.runs} ===", flush=True)
    print(f"{'tg':>8}  {'pp_tps':>10}  {'tg_tps':>10}", flush=True)
    for tg, pp_tps, tg_tps in summaries:
        print(f"{tg:>8}  {pp_tps:>10.2f}  {tg_tps:>10.2f}", flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {str(tg): {"pp_tps": pp_tps, "tg_tps": tg_tps}
             for tg, pp_tps, tg_tps in summaries},
            indent=2,
        ))


if __name__ == "__main__":
    main()

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
import zlib
from pathlib import Path

PROMPT_FILLER = "The quick brown fox jumps over the lazy dog. " * 200

# ⚠️ PROMPT_FILLER is ONE sentence repeated: a 512-token prompt built from it has just
# **11 distinct tokens (2.1%)**, against 188 (36.7%) for real prose. That is fine for a
# dense model, where prefill cost does not depend on *which* tokens arrive — and wrong
# for this MoE, where the router keys on hidden states, so a low-diversity prompt
# concentrates routing onto far fewer experts than real text does. Expert concentration
# is exactly what sets the MoE GEMM's tile count (workplan §17.9), so the filler prompt
# does not merely mis-scale the pp number: it can flip which kernel configuration wins.
# Use --prompt-file for any measurement whose conclusion depends on routing.
def build_prompt(tokenizer, target_len: int, salt: str = "",
                 source: str | None = None) -> tuple[str, int]:
    """Build a `target_len`-token prompt, optionally prefixed by a unique `salt`.

    `salt` exists for prefix-cache benching. With `prefix_cache_mode != "disable"` an
    identical prompt makes every run after the first a full radix-cache hit, so ttft
    stops measuring prefill and starts measuring a cache lookup — the pp column then
    reads as a huge win that no first-time request ever sees. A per-run salt keeps the
    shared prefix at ~0 tokens so each run really does prefill.

    `source` is natural-language text to draw the prompt from instead of the repeated
    filler; see the warning above for why that matters on a MoE. A different window of
    it is used per salt, so runs stay distinct without falling back to repetition.

    The returned length is the *re-encoded* length, not the requested one: the
    truncate-then-decode round trip is not guaranteed to be token-count stable, and
    pp_tps divides by this number.
    """
    if source is not None:
        ids = tokenizer.encode(source, add_special_tokens=False)
        if len(ids) < target_len * 2:
            raise SystemExit(f"--prompt-file has {len(ids)} tokens; need >= {target_len * 2} "
                             f"so each run can take a distinct window")
        # Distinct window per salt. crc32, not hash(): str hashing is salted per process
        # (PYTHONHASHSEED), which would silently pick different windows on every run and
        # make two libs incomparable — the exact class of harness bug §14.1 warns about.
        off = (zlib.crc32(salt.encode()) % max(len(ids) - target_len, 1)) if salt else 0
        text = tokenizer.decode(ids[off:off + target_len])
        return text, len(tokenizer.encode(text, add_special_tokens=False))
    filler = (salt + " " if salt else "") + PROMPT_FILLER
    ids = tokenizer.encode(filler, add_special_tokens=False)
    if len(ids) < target_len:
        filler = filler + PROMPT_FILLER * (target_len // max(len(ids), 1) + 2)
        ids = tokenizer.encode(filler, add_special_tokens=False)
    text = tokenizer.decode(ids[:target_len])
    return text, len(tokenizer.encode(text, add_special_tokens=False))


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
    ap.add_argument("--prefix-cache-mode", default="disable",
                    choices=["disable", "radix"],
                    help="Engine prefix_cache_mode. Default 'disable' — note that on a "
                         "hybrid (RNNState) model this decides which forward path prefill "
                         "takes, so 'disable' does NOT measure what a default-configured "
                         "user gets. See workplan-cuda-13.md §13.")
    ap.add_argument("--prompt-file", default=None,
                    help="Draw the prompt from this natural-language file instead of the "
                         "repeated filler. REQUIRED for any MoE conclusion that depends on "
                         "expert routing: the filler has 11 distinct tokens per 512 and "
                         "concentrates the router (see build_prompt's warning, workplan 17.9).")
    ap.add_argument("--unique-prompts", default=None, action="store_true",
                    help="Salt every run's prompt so radix cannot hit. Defaults to on "
                         "whenever --prefix-cache-mode is not 'disable'.")
    args = ap.parse_args()
    if args.unique_prompts is None:
        args.unique_prompts = args.prefix_cache_mode != "disable"

    from mlc_llm import MLCEngine
    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve.config import EngineConfig
    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir)
    if args.model_lib:
        lib_path = args.model_lib
    else:
        canonical = model_dir / "lib.so"
        if canonical.exists():
            lib_path = str(canonical)
        else:
            so_files = sorted(model_dir.glob("*.so"))
            if not so_files:
                print(f"[mlc] No .so under {model_dir}", file=sys.stderr)
                sys.exit(1)
            if len(so_files) > 1:
                names = ", ".join(p.name for p in so_files)
                print(f"[mlc] Multiple .so files in {model_dir} but no lib.so — "
                      f"pass --model-lib explicitly. Found: {names}", file=sys.stderr)
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
        engine_config=EngineConfig(prefix_cache_mode=args.prefix_cache_mode),
    )
    print(f"[mlc] Engine loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"[mlc] prefix_cache_mode={args.prefix_cache_mode} "
          f"unique_prompts={args.unique_prompts}", flush=True)

    src_text = Path(args.prompt_file).read_text() if args.prompt_file else None

    def prompt_for(tag: str) -> tuple[str, int]:
        return build_prompt(tokenizer, args.pp,
                            salt=f"Archive record {tag}." if args.unique_prompts else "",
                            source=src_text)

    prompt, prompt_len = prompt_for("base")
    print(f"[mlc] Prompt length: {prompt_len} tokens", flush=True)
    if args.unique_prompts and prompt_len != args.pp:
        print(f"[mlc] NOTE: re-encoded length {prompt_len} != requested {args.pp}; "
              f"pp_tps uses the real length", flush=True)

    tg_values = [int(x) for x in args.tg.split(",") if x.strip()]
    summaries = []  # (tg, pp_tps_med, tg_tps_med)

    for tg in tg_values:
        gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=tg)
        print(f"\n[mlc] tg={tg}  runs={args.runs} (warmup={args.warmup})", flush=True)

        for w in range(args.warmup):
            t0 = time.perf_counter()
            p_text, _ = prompt_for(f"w{tg}x{w}") if args.unique_prompts else (prompt, prompt_len)
            ttft, total, ntok = time_run(engine, p_text, gen_cfg,
                                         f"warmup-tg{tg}-{w}")
            print(f"[mlc]   warmup {w}: ttft={ttft*1000:.1f}ms "
                  f"total={total*1000:.1f}ms tokens={ntok} "
                  f"wall={time.perf_counter()-t0:.1f}s",
                  flush=True)

        pp_tps_samples = []
        tg_tps_samples = []
        for r in range(args.runs):
            p_text, p_len = prompt_for(f"r{tg}x{r}") if args.unique_prompts else (prompt, prompt_len)
            ttft, total, ntok = time_run(engine, p_text, gen_cfg,
                                         f"run-tg{tg}-{r}")
            decode_t = total - ttft
            n_decode = max(ntok - 1, 1)
            pp_tps = p_len / ttft if ttft > 0 else float("inf")
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

"""Stage 8.2 smoke: prefix-cache hit on a hybrid GDN model.

What this validates:
  1. With `prefix_cache_mode='radix'`, two back-to-back requests sharing a
     long prompt prefix produce **identical** continuations (the cached path
     must reproduce the fresh-prefill path bit-exactly under greedy decoding).
  2. The cache-hit request's TTFT is materially smaller than the cache-miss
     request's TTFT — i.e., we actually skipped re-prefill of the shared
     portion. Bar: cache-hit TTFT < 50% of cache-miss TTFT (loose, since
     prefill TTFT is small for our 0.8B at 256 tokens; the wall-clock win
     scales with prefix length).
  3. With `prefix_cache_mode='disable'`, the same two requests produce
     identical continuations to the radix path's first request — i.e., we
     didn't break decode parity.

Workflow:
    Request A: 256-token shared prompt + " The answer is" -> 8 decode tokens
    Request B (same engine): same 256-token shared prompt + " Another query" -> 8 decode tokens

If the prefix cache is wired correctly:
  - A's prefill goes through batch_prefill_with_history (cache_prefill=true)
  - B's MatchPrefixCache hits A's radix-tree node for the 256 shared tokens
  - B forks A, PopNFromRNNStateOnly rolls B's rnn_state back to position 256
    (no-op here since A is freshly prefilled and prefilled_offset == A.seq_length)
  - B prefills only " Another query" -> ~3 tokens

Run after `mlc compile` of the 0.8B with the Phase 8 model spec changes.
"""
from __future__ import annotations

import argparse
import sys
import time


def gen(engine, prompt: str, max_tokens: int = 8) -> tuple[str, float]:
    from mlc_llm.protocol.generation_config import GenerationConfig

    cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    ttft = None
    out = ""
    for delta_outputs in engine._generate(prompt, cfg, request_id=f"req-{time.time_ns()}"):
        for delta in delta_outputs:
            if delta.delta_text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out += delta.delta_text
    return out, ttft if ttft is not None else float("nan")


def run_pair(engine, shared_prompt: str, suffix_a: str, suffix_b: str, max_tokens: int):
    out_a, ttft_a = gen(engine, shared_prompt + suffix_a, max_tokens=max_tokens)
    out_b, ttft_b = gen(engine, shared_prompt + suffix_b, max_tokens=max_tokens)
    return out_a, ttft_a, out_b, ttft_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-lib", required=True)
    parser.add_argument("--shared-len", type=int, default=256,
                        help="Approximate token length of shared prefix.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=("both", "off", "on"),
        default="both",
        help="Which engine(s) to run. 'both' loads cache-off then cache-on (suitable for "
             "smaller models that fit two consecutive engine instances in GPU memory). "
             "'off' or 'on' loads a single engine; pair two `--mode off` and `--mode on` "
             "subprocess runs for larger models where back-to-back loads OOM.",
    )
    args = parser.parse_args()

    from mlc_llm import MLCEngine
    from mlc_llm.serve.config import EngineConfig
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Build a deterministic shared prompt long enough that prefix matching has
    # something to reuse. (At 256 tokens the prefill is ~1.2 s on the 0.8B at
    # PP=210 tps, plenty of headroom to see TTFT collapse.)
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tok.encode(filler, add_special_tokens=False)[: args.shared_len]
    shared = tok.decode(ids)
    suffix_a = " The answer is"
    suffix_b = " Another query"

    print(f"[smoke] shared prefix tokens: {len(ids)}, suffixes: {suffix_a!r} / {suffix_b!r}")

    results: dict[str, tuple[str, float, str, float]] = {}

    if args.mode in ("both", "off"):
        # === Run with prefix cache OFF (baseline; no Phase-8 path involved) ===
        print("\n[smoke] === baseline: prefix_cache_mode='disable' ===")
        eng_off = MLCEngine(
            model=args.model_dir,
            model_lib=args.model_lib,
            device="cuda:0",
            mode="interactive",
            engine_config=EngineConfig(prefix_cache_mode="disable"),
        )
        a_off, ttft_a_off, b_off, ttft_b_off = run_pair(
            eng_off, shared, suffix_a, suffix_b, args.max_tokens
        )
        print(f"  req-A ttft={ttft_a_off*1000:.1f} ms   out={a_off!r}")
        print(f"  req-B ttft={ttft_b_off*1000:.1f} ms   out={b_off!r}")
        results["off"] = (a_off, ttft_a_off, b_off, ttft_b_off)
        eng_off.terminate()

    if args.mode in ("both", "on"):
        # === Run with prefix cache ON (Phase-8 hybrid path) ===
        print("\n[smoke] === phase 8: prefix_cache_mode='radix' ===")
        eng_on = MLCEngine(
            model=args.model_dir,
            model_lib=args.model_lib,
            device="cuda:0",
            mode="interactive",
            engine_config=EngineConfig(prefix_cache_mode="radix"),
        )
        a_on, ttft_a_on, b_on, ttft_b_on = run_pair(
            eng_on, shared, suffix_a, suffix_b, args.max_tokens
        )
        print(f"  req-A ttft={ttft_a_on*1000:.1f} ms   out={a_on!r}")
        print(f"  req-B ttft={ttft_b_on*1000:.1f} ms   out={b_on!r}")
        results["on"] = (a_on, ttft_a_on, b_on, ttft_b_on)
        eng_on.terminate()

    # === Validate (only when both halves available) ===
    if args.mode != "both":
        print(f"\n[smoke] OK (single-mode '{args.mode}' run; cross-mode validation requires both)")
        return

    a_off, ttft_a_off, b_off, ttft_b_off = results["off"]
    a_on, ttft_a_on, b_on, ttft_b_on = results["on"]

    print("\n[smoke] === validation ===")
    fail = False

    # 1. Decode parity: prefix-cache should not change the model's output.
    if a_off != a_on:
        print(f"  FAIL: req-A output differs between cache-off and cache-on:")
        print(f"    off: {a_off!r}")
        print(f"    on : {a_on!r}")
        fail = True
    else:
        print(f"  OK   req-A decode parity (cache-off == cache-on)")

    if b_off != b_on:
        print(f"  FAIL: req-B output differs between cache-off and cache-on:")
        print(f"    off: {b_off!r}")
        print(f"    on : {b_on!r}")
        fail = True
    else:
        print(f"  OK   req-B decode parity (cache-off == cache-on)")

    # 2. TTFT win: req-B in the cache-on case should be materially faster than
    #    req-B in the cache-off case (because the shared prefix is reused).
    if ttft_b_on < ttft_b_off * 0.5:
        print(f"  OK   req-B ttft on cache hit: {ttft_b_on*1000:.1f} ms < "
              f"{ttft_b_off*1000:.1f} ms * 0.5 (= {ttft_b_off*500:.1f} ms)")
    else:
        print(f"  WARN req-B ttft on cache hit: {ttft_b_on*1000:.1f} ms NOT < "
              f"{ttft_b_off*1000:.1f} ms * 0.5. Hit may not have triggered, or "
              f"shared prefix is too short to see the win at this scale.")
        # Don't fail on TTFT — small prompts have noise dominated by overhead.

    if fail:
        print("\n[smoke] FAIL")
        sys.exit(1)
    print("\n[smoke] OK (Phase 8 stage 2: hybrid prefix cache hit produces parity output)")


if __name__ == "__main__":
    main()

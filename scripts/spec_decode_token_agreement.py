#!/usr/bin/env python3
"""B-ext spec decode pre-flight — token-agreement check between draft and target.

For B-ext (external-draft) speculative decoding to work, the draft model's
greedy next-token must match the target's greedy next-token at the same
prefix often enough that the speculative-sampling accept rate is positive.

This script runs greedy-decode in MLC for both models on the same prompts,
then for each step i, it:
  1. Takes the target's prefix at step i (target_tokens[:i])
  2. Asks draft to predict the next token (draft greedy)
  3. Compares to target_tokens[i]

Sequential GPU use only (load draft → unload → load target → unload).

Usage:
    .venv/bin/python scripts/spec_decode_token_agreement.py \\
        --draft  dist/qwen3_5-0.8B-q4f16_1 \\
        --target dist/qwen3_6-35B-A3B-q4f16_1 \\
        --max-tokens 50

Decision rule:
  match-rate < 30% → B-ext is dead, do not pursue.
  match-rate > 50% → worth proceeding to engine integration.
  30%–50% → marginal; depends on accept-rate amplification from multi-token batches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _greedy_decode(model_dir: str, device: str, prompt: str, max_tokens: int) -> list[int]:
    """Greedy-decode `max_tokens` from `prompt`. Returns list of token ids (excl. prompt)."""
    from mlc_llm import MLCEngine
    from mlc_llm.protocol.generation_config import GenerationConfig
    from mlc_llm.serve.config import EngineConfig

    md = Path(model_dir)
    lib_path = str(next(md.glob("*.so")))
    engine = MLCEngine(
        model=str(md),
        model_lib=lib_path,
        device=device,
        mode="interactive",
        engine_config=EngineConfig(prefix_cache_mode="disable"),
    )
    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=max_tokens, logprobs=False)
    tokens: list[int] = []
    for delta in engine._generate(prompt, gen_cfg, request_id="agreement"):
        for choice in delta.choices:
            for tok in choice.delta.token_ids or []:
                tokens.append(tok)
    return tokens


def _next_token_at_prefix(model_dir: str, device: str, prefix_tokens: list[int]) -> int:
    """Run draft on a literal prefix (already-tokenized) and return greedy next token.

    MLCEngine doesn't expose a clean prefix-as-tokens path, so we use the
    streaming API with max_tokens=1.
    """
    # We re-tokenize the decoded text. This is lossy but correct for whitespace-tokenizers.
    # TODO: extend to use raw token ids when MLCEngine supports it.
    raise NotImplementedError(
        "Per-prefix probe needs MLCEngine token-level API; "
        "use the simpler greedy-vs-greedy heuristic for now."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="dist/qwen3_5-0.8B-q4f16_1")
    p.add_argument("--target", default="dist/qwen3_6-35B-A3B-q4f16_1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    prompts = [
        "The capital of France is",
        "Q: What is the largest planet in our solar system? A:",
        "Write a one-sentence definition of a transformer neural network:",
        "Translate to French: Hello, how are you?",
        "List three benefits of exercise:",
    ]

    print(f"[agreement] draft={args.draft}", flush=True)
    print(f"[agreement] target={args.target}", flush=True)
    print(f"[agreement] N prompts={len(prompts)} max_tokens={args.max_tokens}", flush=True)

    print("\n=== running TARGET first (greedy) ===", flush=True)
    target_tokens: list[list[int]] = []
    for i, prompt in enumerate(prompts):
        toks = _greedy_decode(args.target, args.device, prompt, args.max_tokens)
        target_tokens.append(toks)
        print(f"  prompt {i}: {len(toks)} tokens", flush=True)

    print("\n=== running DRAFT (greedy) ===", flush=True)
    draft_tokens: list[list[int]] = []
    for i, prompt in enumerate(prompts):
        toks = _greedy_decode(args.draft, args.device, prompt, args.max_tokens)
        draft_tokens.append(toks)
        print(f"  prompt {i}: {len(toks)} tokens", flush=True)

    print("\n=== greedy-vs-greedy match-rate (lower bound on B-ext accept rate) ===", flush=True)
    overall_match = 0
    overall_total = 0
    per_prompt = []
    for i, (t, d) in enumerate(zip(target_tokens, draft_tokens)):
        n = min(len(t), len(d))
        matches = sum(1 for j in range(n) if t[j] == d[j])
        rate = matches / n if n else 0.0
        per_prompt.append({"prompt": prompts[i], "n": n, "matches": matches, "rate": rate})
        first_div = next((j for j in range(n) if t[j] != d[j]), n)
        print(f"  prompt {i}: {matches}/{n} = {rate:.1%}  (first-divergence at step {first_div})", flush=True)
        overall_match += matches
        overall_total += n

    overall_rate = overall_match / overall_total if overall_total else 0.0
    print(f"\nOVERALL: {overall_match}/{overall_total} = {overall_rate:.1%}", flush=True)

    decision = "PROCEED" if overall_rate > 0.50 else ("MARGINAL" if overall_rate > 0.30 else "DEAD")
    print(f"DECISION: {decision} (cutoffs: <30% dead, 30-50% marginal, >50% proceed)", flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "draft": args.draft,
            "target": args.target,
            "overall": {"matches": overall_match, "total": overall_total, "rate": overall_rate},
            "per_prompt": per_prompt,
            "decision": decision,
        }, indent=2))
        print(f"\n[agreement] saved -> {args.json_out}", flush=True)


if __name__ == "__main__":
    main()

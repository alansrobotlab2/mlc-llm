#!/usr/bin/env python3
"""Long-prompt bit-exactness gate on the history path (§14's 3133-token check, redone).

A single-chunk prefill never advances `history_slot_id` mid-prompt. A prompt longer than
`prefill_chunk_size` does: chunk 2 loads from the slot chunk 1's scatter left behind, so a
ring off-by-one that a 512-token prompt cannot see becomes reachable. Under radix only.

Usage: --lib A --out a.json, then --lib B --compare a.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--model-lib", required=True)
ap.add_argument("--prefix-cache-mode", default="radix")
ap.add_argument("--prefill-chunk", type=int, default=2048)
ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--out")
ap.add_argument("--compare")
args = ap.parse_args()

from mlc_llm import MLCEngine  # noqa: E402
from mlc_llm.serve.config import EngineConfig  # noqa: E402

model_dir = pathlib.Path(args.model_dir)
model_name = json.loads((model_dir / "mlc-chat-config.json").read_text())["model_type"]

# ~3100 tokens of deterministic, non-repeating text. Non-repeating matters: a repetitive
# prompt is exactly the case where a corrupted recurrent state still produces plausible
# text, because every chunk looks like every other one.
para = (
    "The {n}th observation concerns the transport of heat through a lattice of {n} atoms, "
    "where the coupling constant is {k} and the boundary is held at temperature {t}. "
    "Under these conditions the flux is not stationary, and the correction term of order "
    "{n} must be retained in the expansion. "
)
prompt = "".join(para.format(n=i, k=i * 3 + 1, t=273 + i) for i in range(1, 90))

engine = MLCEngine(
    model=str(model_dir),
    model_lib=args.model_lib,
    device="cuda:0",
    mode="interactive",
    engine_config=EngineConfig(
        max_num_sequence=2,
        max_total_sequence_length=8192,
        prefill_chunk_size=args.prefill_chunk,
        prefix_cache_mode=args.prefix_cache_mode,
    ),
)
resp = engine.completions.create(
    prompt=prompt, model=model_name, max_tokens=args.max_tokens,
    temperature=0.0, top_p=1.0, stream=False, extra_body={"ignore_eos": True},
)
text = resp.choices[0].text
n_tok = resp.usage.prompt_tokens
print(f"[long] prompt_tokens={n_tok} chunk={args.prefill_chunk} "
      f"chunks={-(-n_tok // args.prefill_chunk)} mode={args.prefix_cache_mode}")
print(f"[long] {text[:120]!r}...")

if args.out:
    pathlib.Path(args.out).write_text(json.dumps({"n_tok": n_tok, "text": text}))
    print(f"[long] wrote {args.out}")
if args.compare:
    ref = json.loads(pathlib.Path(args.compare).read_text())
    if ref["n_tok"] != n_tok:
        sys.exit(f"[long] FAIL: prompt tokenized differently ({ref['n_tok']} vs {n_tok})")
    if ref["text"] != text:
        j = next((i for i, (a, b) in enumerate(zip(ref["text"], text)) if a != b),
                 min(len(ref["text"]), len(text)))
        sys.exit(f"[long] FAIL: diverges at char {j}\n  ref: {ref['text'][j:j+60]!r}\n"
                 f"  got: {text[j:j+60]!r}")
    print(f"[long] PASS — byte-identical over {n_tok} prompt tokens, "
          f"{-(-n_tok // args.prefill_chunk)} prefill chunks")

# The engine keeps a live background thread; a plain return hangs at interpreter
# shutdown. Same class of trap as the three in workplan §11.
sys.stdout.flush()
os._exit(0)

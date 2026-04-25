#!/usr/bin/env python3
"""
Stage 6 diagnostic: re-run prompt 1 step-by-step and dump top-5 logits at every
step. Focus on step 33, where MLC and HF diverged (HF picked ' fashion', MLC
picked ' vibrant'). Goal: measure the rank-1 vs rank-2 logit gap. If it's tiny
(within fp16 rounding range) the divergence is a benign tie-flip; if it's wide
there's a real numerical issue.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
DEVICE = "cuda:0"
PROMPT = "The capital of France is"
N_STEPS = 50
FOCUS_STEP = 33  # 0-indexed; the divergent step
TOPK = 5


@torch.inference_mode()
def main() -> None:
    print(f"[diag] Loading {MODEL_ID} on {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    ).eval().half()

    input_ids = tok.encode(PROMPT, return_tensors="pt").to(DEVICE)
    print(f"[diag] Prompt tokens: {input_ids[0].tolist()}")

    for step in range(N_STEPS):
        out = model(input_ids, use_cache=False)
        logits = out.logits[0, -1].float()  # vocab
        probs = torch.softmax(logits, dim=-1)
        top = logits.topk(TOPK)
        top_ids = top.indices.tolist()
        top_logits = top.values.tolist()
        top_probs = probs[top.indices].tolist()

        next_id = top_ids[0]
        gap_logit = top_logits[0] - top_logits[1]
        gap_prob = top_probs[0] - top_probs[1]
        ratio = top_probs[0] / max(top_probs[1], 1e-30)

        marker = "  <<< FOCUS" if step == FOCUS_STEP else ""
        print(
            f"[step {step:2d}] picked={next_id:6d} {tok.decode([next_id])!r:18s} "
            f"gap_logit={gap_logit:+.4f} gap_prob={gap_prob:+.4f} ratio={ratio:.2f}x{marker}"
        )
        if step == FOCUS_STEP or step in (FOCUS_STEP - 1, FOCUS_STEP + 1):
            for r, (tid, lg, pr) in enumerate(zip(top_ids, top_logits, top_probs)):
                print(f"           rank{r+1}: id={tid:6d} {tok.decode([tid])!r:18s} logit={lg:+.4f} prob={pr:.4f}")

        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)

    full_text = tok.decode(input_ids[0, -N_STEPS:].tolist())
    print(f"\n[diag] Full generated text: {full_text!r}")


if __name__ == "__main__":
    main()

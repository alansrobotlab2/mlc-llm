"""Production runner for Qwen3.5-0.8B with EAGLE self-speculative decoding.

Recommended config for the 0.8B on Orin AGX after worklog cont. 12:
  - target: dist/qwen3_5-0.8B-q0f16
  - draft:  dist/qwen3_5-0.8B-q0f16-mtp-draft  (the trained MTP head as
            an EAGLE-style draft; 99.5 → 120.5 tps γ=4 after B.6 + dlight patch)
  - spec_draft_length=4: 4.71 average accepted tokens per round, byte-identical
    output to target_only on the canonical prompts.

Single-instance interactive serving; max_batch_size=1.
"""

import argparse
import time

from mlc_llm.serve import EngineConfig, MLCEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="dist/qwen3_5-0.8B-q0f16")
    ap.add_argument("--draft", default="dist/qwen3_5-0.8B-q0f16-mtp-draft")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--draft-length", type=int, default=4)
    args = ap.parse_args()

    engine_config = EngineConfig(
        additional_models=[(args.draft, f"{args.draft}/lib.so")],
        speculative_mode="eagle",
        spec_draft_length=args.draft_length,
        max_total_sequence_length=8192,
        # max_num_sequence must be >= spec_draft_length + 1 so EAGLE's CanPrefill
        # admission doesn't reject single-batch requests.
        max_num_sequence=args.draft_length + 1,
        prefill_chunk_size=1024,
    )

    engine = MLCEngine(
        model=args.target,
        model_lib=f"{args.target}/lib.so",
        engine_config=engine_config,
        mode="interactive",
    )

    print(f"--- Qwen3.5-0.8B + MTP draft, γ={args.draft_length} ---", flush=True)
    t0 = time.time()
    out = engine.chat.completions.create(
        messages=[{"role": "user", "content": args.prompt}],
        model="qwen3_5",
        max_tokens=args.max_tokens,
        temperature=0.0,
        stream=False,
    )
    elapsed = time.time() - t0
    print(out.choices[0].message.content, flush=True)
    print(f"\n--- generated in {elapsed:.2f}s ---", flush=True)

    metrics = engine.metrics()
    m = metrics.model_dump() if hasattr(metrics, "model_dump") else metrics
    sd = m.get("spec_decode", {})
    print(f"decode tps:    {m.get('decode_tokens_per_s', 0):.1f}")
    if "accept_len" in sd:
        max_step = max(int(k.split("=")[1].rstrip("}")) for k in sd["accept_len"])
        print(f"avg accept_len at step {max_step}: {sd['accept_len'][f'accept_len{{step={max_step}}}']:.2f}")
    if "accept_rate" in sd:
        print("step-1 accept rate:", sd["accept_rate"].get("accept_rate{step=1}", "n/a"))
    engine.terminate()


if __name__ == "__main__":
    main()

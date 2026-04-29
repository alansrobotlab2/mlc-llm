"""Smoke test for 35B-A3B EAGLE self-spec with the qwen3_5_moe MTP draft.

Sister of `spec_smoke.py` (which targets the 0.8B). Loads the recompiled v6
target (with EAGLE-compat methods) + the new MTP draft, runs a tiny generation,
prints engine metrics so we can see accept rate / decode time.
"""

import argparse
import time

from mlc_llm.serve import EngineConfig, MLCEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        default="dist/qwen3_6-35B-A3B-q4f16_1",
        help="MLC target model directory (mlc-chat-config.json + lib.so + params)",
    )
    ap.add_argument(
        "--draft",
        default="dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft",
        help="MTP draft model directory",
    )
    ap.add_argument("--draft-length", type=int, default=1)
    ap.add_argument(
        "--prompt", default="The capital of France is", help="Plain completion prompt"
    )
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument(
        "--mode",
        default="spec",
        choices=["spec", "target_only"],
        help="spec: EAGLE w/ MTP draft. target_only: skip draft for baseline.",
    )
    args = ap.parse_args()

    spec_factor = max(1, args.draft_length + 1)
    if args.mode == "spec":
        engine_config = EngineConfig(
            additional_models=[(args.draft, f"{args.draft}/lib.so")],
            speculative_mode="eagle",
            spec_draft_length=args.draft_length,
            max_total_sequence_length=4096,
            max_num_sequence=spec_factor,
            prefill_chunk_size=512,
        )
    else:
        engine_config = EngineConfig(
            max_total_sequence_length=4096,
            max_num_sequence=1,
            prefill_chunk_size=512,
        )

    engine = MLCEngine(
        model=args.target,
        model_lib=f"{args.target}/lib.so",
        engine_config=engine_config,
        mode="interactive",
    )

    print(f"--- engine constructed (mode={args.mode}, gamma={args.draft_length}) ---", flush=True)
    t0 = time.time()
    out = engine.completions.create(
        prompt=args.prompt,
        model="qwen3_5_moe",
        max_tokens=args.max_tokens,
        temperature=0.0,
        stream=False,
        extra_body={"ignore_eos": True},
    )
    elapsed = time.time() - t0
    text = out.choices[0].text
    print(text, flush=True)
    print(f"--- generated in {elapsed:.2f}s ---", flush=True)

    print("--- engine metrics ---", flush=True)
    metrics = engine.metrics()
    print(metrics.model_dump_json(indent=2) if hasattr(metrics, "model_dump_json") else metrics)

    engine.terminate()


if __name__ == "__main__":
    main()

"""Smoke test: drive Qwen3.5-0.8B through EAGLE self-spec with the MTP draft.

Loads target (no-MTP, q0f16) + draft (MTP head, q0f16), runs a tiny generation,
prints engine metrics so we can see accept rate / decode time.
"""

import argparse

from mlc_llm.serve import EngineConfig, MLCEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        default="dist/qwen3_5-0.8B-q0f16",
        help="MLC target model directory (mlc-chat-config.json + lib.so + params)",
    )
    ap.add_argument(
        "--draft",
        default="dist/qwen3_5-0.8B-q0f16-mtp-draft",
        help="MTP draft model directory",
    )
    ap.add_argument("--draft-length", type=int, default=4)
    ap.add_argument(
        "--prompt", default="The capital of France is", help="Plain completion prompt"
    )
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    # EAGLE's CanPrefill admission rejects when num_running_rsentries *
    # (spec_draft_length + 1) > max_num_sequence (batch_prefill_base.cc:283-290),
    # so max_num_sequence must be >= spec_draft_length + 1 even for batch=1.
    spec_factor = args.draft_length + 1
    engine_config = EngineConfig(
        additional_models=[(args.draft, f"{args.draft}/lib.so")],
        speculative_mode="eagle",
        spec_draft_length=args.draft_length,
        max_total_sequence_length=4096,
        max_num_sequence=spec_factor,
        prefill_chunk_size=512,
    )

    engine = MLCEngine(
        model=args.target,
        model_lib=f"{args.target}/lib.so",
        engine_config=engine_config,
    )

    print("--- engine constructed; starting generation ---", flush=True)
    out = engine.chat.completions.create(
        messages=[{"role": "user", "content": args.prompt}],
        model="qwen3_5",
        max_tokens=args.max_tokens,
        temperature=0.0,
        stream=False,
    )
    print(out.choices[0].message.content, flush=True)

    print("--- engine metrics ---", flush=True)
    metrics = engine.metrics()
    print(metrics.model_dump_json(indent=2) if hasattr(metrics, "model_dump_json") else metrics)

    engine.terminate()


if __name__ == "__main__":
    main()

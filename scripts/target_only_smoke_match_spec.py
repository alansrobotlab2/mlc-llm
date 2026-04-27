"""Target alone with the SAME engine config as spec_smoke — rule out config differences."""

from mlc_llm.serve import EngineConfig, MLCEngine


def main():
    target = "dist/qwen3_5-0.8B-q0f16"
    engine_config = EngineConfig(
        max_total_sequence_length=4096,
        max_num_sequence=5,
        prefill_chunk_size=512,
    )
    engine = MLCEngine(
        model=target,
        model_lib=f"{target}/lib.so",
        engine_config=engine_config,
    )
    print("--- engine constructed; starting generation ---", flush=True)
    out = engine.chat.completions.create(
        messages=[{"role": "user", "content": "The capital of France is"}],
        model="qwen3_5",
        max_tokens=32,
        temperature=0.0,
        stream=False,
    )
    print(out.choices[0].message.content, flush=True)
    engine.terminate()


if __name__ == "__main__":
    main()

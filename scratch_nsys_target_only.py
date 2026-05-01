"""Profile target-only decode on the 35B-A3B fp16 lib. nsys captures only the
profiled pass (warmup is excluded via cudaProfilerStart/Stop).

Goal for 2a: measure what fraction of decode time is spent in the GDN
recurrent-state kernels (`gdn_func_kernel`) and the linear-attn supporting
kernels (`depthwise_conv1d*`, the related fused gate/sigmoid/concat kernels).
Compare against MoE/MHA kernels.

Run via:
  source .envrc.local && nsys profile -o /tmp/nsys_target_only \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda --force-overwrite=true \
      .venv/bin/python scratch_nsys_target_only.py
"""
import ctypes

from mlc_llm.serve import EngineConfig, MLCEngine

target = "dist/qwen3_6-35B-A3B-q4f16_1"
engine = MLCEngine(
    model=target,
    model_lib=f"{target}/lib.so",
    engine_config=EngineConfig(
        max_total_sequence_length=4096,
        max_num_sequence=1,
        prefill_chunk_size=512,
    ),
    mode="interactive",
)
print("--- engine constructed (target-only) ---", flush=True)

# Warmup
print("--- warmup ---", flush=True)
engine.completions.create(
    prompt="The capital of France is",
    model="qwen3_5_moe",
    max_tokens=8,
    temperature=0.0,
    stream=False,
    extra_body={"ignore_eos": True},
)

cudart = ctypes.CDLL("libcudart.so")
print("--- profiled pass cudaProfilerStart ---", flush=True)
cudart.cudaProfilerStart()
out = engine.completions.create(
    prompt="The capital of France is",
    model="qwen3_5_moe",
    max_tokens=32,
    temperature=0.0,
    stream=False,
    extra_body={"ignore_eos": True},
)
cudart.cudaProfilerStop()
print("--- cudaProfilerStop ---", flush=True)
print(out.choices[0].text, flush=True)
engine.terminate()

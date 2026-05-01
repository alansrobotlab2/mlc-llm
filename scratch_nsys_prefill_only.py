"""Profile a single 512-token prefill on the 35B-A3B fp16 lib.

Phase 9 Stage 9.1 deliverable. Companion to scratch_nsys_target_only.py
(decode profile) — same engine/lib, same warmup discipline, but the profiled
window is wrapped around a single completion request with a 512-token prompt
and max_tokens=1, so the captured kernel-time is ~99% prefill (one decode
step at ~18 ms is unavoidable but trivial vs ~2.5 s of prefill at 207 tps).

Reuse scratch_nsys_bucket.py to bucket the resulting nsys-rep — the existing
regexes already cover the prefill kernel families (`batch_prefill_paged_kv`,
`dequantize_group_gemm`, `topk|cumsum|get_indices|moe_sum`, `scatter_output`).

Run via:
  source .envrc.local && nsys profile -o /tmp/nsys_prefill_only \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda --cuda-graph-trace=node --force-overwrite=true \
      .venv/bin/python scratch_nsys_prefill_only.py

Then bucket:
  .venv/bin/python scratch_nsys_bucket.py /tmp/nsys_prefill_only.nsys-rep
"""
import ctypes

from mlc_llm.serve import EngineConfig, MLCEngine
from transformers import AutoTokenizer

target = "dist/qwen3_6-35B-A3B-q4f16_1"
PP = 512  # prefill length
PROMPT_FILLER = "The quick brown fox jumps over the lazy dog. " * 200


def build_prompt(tokenizer, target_len: int) -> str:
    filler = PROMPT_FILLER
    ids = tokenizer.encode(filler, add_special_tokens=False)
    if len(ids) < target_len:
        filler = filler * (target_len // len(ids) + 2)
        ids = tokenizer.encode(filler, add_special_tokens=False)
    return tokenizer.decode(ids[:target_len])


tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
prompt = build_prompt(tokenizer, PP)
prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
print(f"--- prompt: {prompt_len} tokens (target {PP}) ---", flush=True)

engine = MLCEngine(
    model=target,
    model_lib=f"{target}/lib.so",
    engine_config=EngineConfig(
        max_total_sequence_length=4096,
        max_num_sequence=1,
        prefill_chunk_size=PP,
        prefix_cache_mode="disable",
    ),
    mode="interactive",
)
print("--- engine constructed (prefill-only) ---", flush=True)

# Warmup: one full prefill + a few decode tokens so the cudagraph cache and
# any one-shot JIT paths are hot before the profiled window.
print("--- warmup ---", flush=True)
engine.completions.create(
    prompt=prompt,
    model="qwen3_5_moe",
    max_tokens=4,
    temperature=0.0,
    stream=False,
    extra_body={"ignore_eos": True},
)

cudart = ctypes.CDLL("libcudart.so")
print("--- profiled pass cudaProfilerStart ---", flush=True)
cudart.cudaProfilerStart()
out = engine.completions.create(
    prompt=prompt,
    model="qwen3_5_moe",
    max_tokens=1,
    temperature=0.0,
    stream=False,
    extra_body={"ignore_eos": True},
)
cudart.cudaProfilerStop()
print("--- cudaProfilerStop ---", flush=True)
print(f"first decoded token: {out.choices[0].text!r}", flush=True)
print(f"prompt_tokens={out.usage.prompt_tokens} "
      f"completion_tokens={out.usage.completion_tokens}", flush=True)
engine.terminate()

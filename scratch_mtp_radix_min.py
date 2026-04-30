"""Minimal MTP + prefix_cache=radix repro on 35B with longer prompt."""
import time

from mlc_llm.serve import EngineConfig, MLCEngine
from transformers import AutoTokenizer

target = "dist/qwen3_6-35B-A3B-q4f16_1"
draft = "dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft"

tok = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
filler = "The quick brown fox jumps over the lazy dog. " * 200
ids = tok.encode(filler, add_special_tokens=False)[:256]
shared = tok.decode(ids)
print(f"prompt token len: {len(ids)}", flush=True)

cfg = EngineConfig(
    additional_models=[(draft, f"{draft}/lib.so")],
    speculative_mode="eagle",
    spec_draft_length=1,
    max_total_sequence_length=4096,
    max_num_sequence=2,
    prefill_chunk_size=512,
    prefix_cache_mode="radix",
)

engine = MLCEngine(
    model=target,
    model_lib=f"{target}/lib.so",
    engine_config=cfg,
    mode="interactive",
)
print("--- engine constructed γ=1 prefix_cache=radix ---", flush=True)

t0 = time.time()
out = engine.completions.create(
    prompt=shared + " The answer is",
    model="qwen3_5_moe",
    max_tokens=8,
    temperature=0.0,
    stream=False,
    extra_body={"ignore_eos": True},
)
print(f"--- req-A generated in {time.time()-t0:.2f}s ---", flush=True)
print(repr(out.choices[0].text), flush=True)

t0 = time.time()
out = engine.completions.create(
    prompt=shared + " Another query",
    model="qwen3_5_moe",
    max_tokens=8,
    temperature=0.0,
    stream=False,
    extra_body={"ignore_eos": True},
)
print(f"--- req-B generated in {time.time()-t0:.2f}s ---", flush=True)
print(repr(out.choices[0].text), flush=True)

engine.terminate()

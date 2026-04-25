# Project: Implement Qwen3.5-0.8B (Gated DeltaNet) in MLC-LLM

## Goal
Add support for Qwen3.5-0.8B to MLC-LLM. Validate numerical parity with the
PyTorch (HuggingFace transformers) reference. Do NOT attempt the 35B-A3B model
until 0.8B passes Stage 5.

## Reference implementations (read-only)
- vLLM: ../vllm/vllm/model_executor/models/qwen3_next.py  
- flash-linear-attention: ../flash-linear-attention/fla/layers/gated_deltanet.py  
- HuggingFace transformers source for Qwen3.5 (in site-packages)

## Structural template
- python/mlc_llm/model/qwen3_moe/  → copy and modify

## New code lives in
- python/mlc_llm/model/qwen3_next/qwen3_next_model.py
- python/mlc_llm/model/qwen3_next/qwen3_next_loader.py
- python/mlc_llm/model/qwen3_next/qwen3_next_quantization.py
- Register in python/mlc_llm/model/model.py

## Validation harness
- ./validate.py — runs PyTorch reference and MLC side by side
- ./reference_outputs.pt — cached PyTorch outputs (regenerate if model changes)

## Stages (do not skip)
1. Baseline: MLC running Qwen2.5-0.5B successfully
2. Reference harness for Qwen3.5-0.8B in PyTorch
3. Skeleton MLC model that runs (stub DeltaNet)
4. Real DeltaNet, layer-by-layer numerical validation
5. End-to-end generation parity (greedy decode, 50 tokens, ≥48 match)

## Numerical tolerance
- Per-layer hidden states: rtol=1e-3, atol=1e-3 (fp16)
- Generated tokens: at least 48/50 match in greedy decode

## Constraints
- All development on 0.8B. The 35B-A3B is out of scope until Stage 5.
- Validate each DeltaNet sub-step (conv1d, normalize, gate, state update)
  against PyTorch separately before composing them.
- When stuck on a numerical mismatch >1 hour, add per-tensor logging
  to BOTH the PyTorch and MLC paths and diff intermediate values.

## Things to NOT do
- Do not attempt to make it fast. Correctness first.
- Do not write custom CUDA kernels. Use Relax/TIR primitives.
- Do not touch quantization until Stage 5 passes in fp16.
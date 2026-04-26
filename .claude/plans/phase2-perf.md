# Plan: Phase 2 — Decode Perf for Qwen3.5-0.8B on Orin AGX

## Context

Correctness phase (Stages 0–6 in [worklog.md](../../worklog.md)) shipped 2026-04-25 on a separate Blackwell sm_120 dev box. The actual deployment target is **Orin AGX** (sm_87 Ampere, 64 GB LPDDR5 unified memory at ~204 GB/s). The Phase 2 plan now targets that hardware.

The dense Qwen3.5-0.8B path passes 50/50 greedy parity with HF fp16; the MoE Qwen3.6-35B-A3B path passes 4/5 prompts at 50/50 + 1/5 at 33/50 (single fp16-noise-floor flip). Both correctness baselines were established on the prior box; the same MLC source tree is reused here.

**Why a fresh plan:** the Blackwell plan compared against `unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S` on a 1.7 TB/s discrete GPU. Neither the model nor the perf reasoning transfers. Orin AGX has ~8× less memory bandwidth, ~4× less FLOPs, and can't usefully fit the 35B model alongside the OS + activations.

## Hardware reality

| | Orin AGX (this box) | Blackwell sm_120 (prior plan) |
|---|---|---|
| Compute capability | sm_87 (Ampere) | sm_120 (Blackwell) |
| Memory | 64 GB **unified** LPDDR5, ~204 GB/s peak | 32 GB GDDR7, ~1700 GB/s |
| Tensor cores | 64, fp16/bf16/int8 (no fp8/fp4) | 5th-gen, fp4/fp8 native |
| Power envelope | 15W / 30W / 50W / MAXN (~60W) | unconstrained |
| FlashInfer cache | sm_87 in upstream prebuilts | sm_120 was missing |
| 35B-A3B as target? | **No** — won't fit usefully | yes, was the gate |

**Implication:** decode on Orin is even more memory-bandwidth-bound than on Blackwell. Fused dequant work matters *more*, but the absolute target tps will be much lower. CUDA-graph capture and per-step launch overhead matter *more* (Ampere's per-launch cost is a larger share of a smaller per-step budget).

## Goal

**Single load-bearing target:** decode `tg128` tps for **Qwen3.5-0.8B q4f16_1**, single batch, sm_87, MAXN power, fan locked. Acceptance gate: same model in **llama.cpp + Q4_K_S GGUF**, same conditions.

- **Bar to beat (parity):** llama.cpp Q4_K_S decode tps on Orin AGX (TBD — measure in Phase 2A).
- **Goal:** beat llama.cpp decode by **≥30%**. (Smaller margin than the Blackwell ask: the headroom on Orin is real but the BW ceiling is hard.)

35B-A3B is **out of scope on Orin**. If we later want a larger acceptance gate, the candidate is a 4–7B dense model (Qwen3-4B or Qwen3.5-3B) — pick after 0.8B work plateaus.

## Phase 2A — Setup + baseline (gating)

Nothing in Phase 2B starts until A is green and bench numbers are committed.

### A.1 — Lock the box for repeatable benches
```bash
sudo nvpmodel -m 0          # MAXN
sudo jetson_clocks          # pin max GPU/CPU/EMC clocks
# fan: manual + max via /sys/devices/pwm-fan/* or jtop
```
Record these in every bench log header. Numbers from a thermal-throttled run are worse than no numbers.

### A.2 — Toolchain
- **mlc_llm**: no aarch64+sm_87 wheel on PyPI. Either build from source against `3rdparty/tvm`, or use the MLC nightly aarch64 index if it exists. Confirm `nvcc -arch=sm_87` + cuDNN paths.
- **llama.cpp**: built at `../llama.cpp/build/bin/llama-bench`, configured `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=87`, `GGML_CUDA_FA=ON`. **Note `GGML_CUDA_FA_ALL_QUANTS=OFF`** — verify whether enabling it changes Q4_K_S decode before benching.
- **Python venv** with torch (already 2.8 + CUDA 12.6 system-wide), transformers (4.51.3), huggingface_hub (0.36.2), and the local `python/` source on `PYTHONPATH`.

### A.3 — Get models
- `Qwen/Qwen3.5-0.8B` (HF safetensors, ~1.6 GB fp16) → `~/.cache/huggingface/`
- `unsloth/Qwen3.5-0.8B-GGUF`, file `Q4_K_S` (~500 MB) → `dist/gguf/`

### A.4 — Compile MLC q4f16_1
```
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<rev>/
python -m mlc_llm convert_weight "$SNAP" --quantization q4f16_1 -o dist/qwen3_5-0.8B-q4f16_1
python -m mlc_llm gen_config   "$SNAP" --quantization q4f16_1 --conv-template qwen3_5 -o dist/qwen3_5-0.8B-q4f16_1
python -m mlc_llm compile dist/qwen3_5-0.8B-q4f16_1 --device cuda -o dist/qwen3_5-0.8B-q4f16_1/lib.so
```
Capture compile-time warnings around fused-quant codegen on sm_87, don't ignore.

### A.5 — Fix `bench_mlc.py`'s prefill measurement
Same bug as the Blackwell plan: `engine._generate()` yields the first delta before GPU prefill completes, so prefill tps is fake. Replace with non-streaming `chat.completions.create(...)` + `usage.prefill_tokens_per_sec` if MLC populates it; otherwise instrument with `cudaEventRecord` around prefill / first-decode-step. **Acceptance:** reported `prefill_tps` × token_count yields a wall-clock that's longer than one decode step.

### A.6 — Baseline bench (5 runs each, 60s warmup, MAXN+jetson_clocks)
- llama.cpp: `llama-bench -m Qwen3.5-0.8B-Q4_K_S.gguf -p 512 -n 128 -r 5`
- MLC: `bench_mlc.py` against `dist/qwen3_5-0.8B-q4f16_1/`, same `pp=512, tg=128, runs=5`
- Capture for both: decode tps (median + stdev), prefill tps, peak resident memory (`tegrastats`), peak GPU power.

Save logs as `bench_llamacpp_0.8B_orin.log`, `bench_mlc_q4_0.8B_orin.log`. **These are the regression baseline for every fix that follows.**

## Phase 2B — Profile-driven Tier-1 (ranked, profile supersedes)

Run `nsys profile --gpu-metrics-devices all` against one decode step. Bin time into:

1. Dense MLP `gate_up_proj` / `down_proj` GEMM (and the dequant cost feeding it)
2. Full-attention QKV + paged-KV access (FlashInfer or TIR fallback — verify which on sm_87)
3. GatedDeltaNet TIR kernel (18/24 layers in 0.8B)
4. Per-step launch overhead

Whichever ranks #1 gets fixed first. Below is the *suspected* order; the profile reorders it.

### B.1 — Fused dequant + matmul for q4f16_1 (suspected #1)
llama.cpp's `mul_mat_q` reads Q4_K once and fuses dequant into the matmul tile; MLC's q4f16_1 dequantizes to fp16 then matmuls, doubling LPDDR5 traffic. On a 204 GB/s ceiling, that's the dominant cost.

Two paths:
- **B.1a — Try `q4f16_ft` (or AWQ, GPTQ) if MLC carries fused kernels for it on sm_87.** Verify by reading `python/mlc_llm/quantization/` and checking the codegen path. Re-bench before writing any kernel.
- **B.1b — Fused TIR kernel for q4f16_1.** Tile-load Q4 into shared mem, dequant in-register, matmul in one pass. Targets `gate_up_proj` and `down_proj` first (largest weights, hottest path).

Re-run greedy parity (≥48/50) after any quant-scheme switch — a tps win that breaks parity is worthless.

### B.2 — CUDA graph capture for decode
On Orin, per-step launch overhead is a *larger* fraction of decode wall-clock than on Blackwell. Confirm whether MLC captures the decode graph; if not, enable. Each step is fixed-shape (single token), canonical case for capture.

Acceptance: ≥1.3× decode tps. Re-bench llama.cpp side-by-side — llama.cpp on Orin already uses graph capture for some paths, so part of MLC's gap may be exactly this.

### B.3 — FlashInfer sm_87
Verify FlashInfer is actually being used (check compile log for the same cache warning the Blackwell run hit, just with `87f/` instead of `120f/`). sm_87 is in upstream prebuilts so should be a non-issue, but verify. If it's falling back to TIR paged-KV, that's free perf.

## Phase 2C — Speculative (only if 2B leaves a gap)

### C.1 — GatedDeltaNet TIR tune for sm_87
The kernel at `python/mlc_llm/model/qwen35/qwen35_model.py::create_gated_delta_net_func` was correctness-first, never tuned. On sm_87: split-K, vectorized 16-byte loads, tensor-core fragments where the conv1d / state-update math allows, shared-memory tiling for the recurrent state. High effort — only start if the profile puts GDN in the top 3.

### C.2 — INT8 weight + fp16 act
sm_87 has full-rate int8 tensor cores. MLC has W8A16 paths in the quant config. Smaller model footprint → less LPDDR5 pressure. Worth a one-shot try if the W4 fused path doesn't close the gap.

### C.3 — Engine knob audit
`prefill_chunk_size`, `kv_cache_page_size`, `max_total_sequence_length`. Cheap to walk through; low expected return.

### C.4 — Power-mode sweep
Bench at MAXN, 50W, 30W to characterize the perf/W curve. Useful for the user (Orin is power-constrained in deployment) even if it doesn't move the headline tps.

## Stop conditions

- **Win:** 0.8B decode ≥ 1.3× llama.cpp Q4_K_S baseline on Orin AGX MAXN.
- **Diminishing returns:** three consecutive Tier-2/3 items each <10% on the headline number.
- **Roofline ceiling:** profile shows >80% of decode time is LPDDR5 weight reads at >180 GB/s actual (close to the 204 GB/s peak). At that point only quantization gets us further.

## Re-bench cadence

After every landed fix: full bench against the same conditions, append to `worklog.md` with date + the change that landed. Don't stack un-validated optimizations.

## Risks / known unknowns

- **mlc_llm wheel on aarch64+sm_87.** May need to build TVM + mlc_llm from source. Budget a half-day if so.
- **Thermal throttling under sustained bench.** Orin AGX without active cooling will throttle within a minute of full-power decode. Verify with `tegrastats` during long runs.
- **Unified memory contention.** Anything else running on the box (browser, IDE, other processes) eats into the 204 GB/s shared with the GPU. Bench with a quiet system; document what was running.
- **0.8B is small enough that some kernels become launch-overhead-bound.** That makes graph capture (B.2) potentially more impactful than fused-dequant (B.1) — the profile in 2A.6 will tell us which.

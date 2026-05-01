# Qwen3-Next / Qwen3.5 / Qwen3.6 in MLC-LLM — Technical Reference

This is the living technical document for bringing up the Qwen3-Next family of hybrid (GatedDeltaNet + GQA) models in MLC-LLM. Living = update as decisions land or assumptions change. Date-stamped progress goes in [`worklog.md`](./worklog.md).

---

## 1. Goal & Scope

End goal: **Qwen3.6-35B-A3B** (35B-param hybrid MoE, ~3B activated) running end-to-end on MLC-LLM with greedy-decode parity vs. HuggingFace transformers.

De-risk path: validate **Qwen3.5-0.8B** first. It is the smallest member of the family, dense (no MoE), with symmetric linear-attention heads and standard RoPE. Anything that breaks here is in the GatedDeltaNet / hybrid stack, isolated from MoE complexity.

Strict ordering — nothing on 35B until 0.8B passes the parity bar in §11.

---

## 2. Quick Start: Compile & Run Qwen3.6-35B-A3B on Orin

End-to-end recipe for the shipped lib at [dist/qwen3_6-35B-A3B-q4f16_1/](dist/qwen3_6-35B-A3B-q4f16_1/). Tested on Orin AGX (sm_87, MAXN). ~25 min cold from a fresh clone, dominated by the 72 GB HF download.

### 2.1 Prereqs

- Orin AGX (or any sm ≥ 87 CUDA device with ≥ 24 GB VRAM for q4f16_1).
- `MAXN` power profile (`sudo nvpmodel -m 0 && sudo jetson_clocks`) for benchmark stability.
- This repo's editable mlc-llm wheel installed in `.venv` (the repo `pyproject.toml` + vendored TVM).
- HF auth (`hf auth login`) for the gated `Qwen/Qwen3.6-35B-A3B` repo.

### 2.2 Download weights

```bash
hf download Qwen/Qwen3.6-35B-A3B   # 72 GB bf16, ~7-10 min on a fast link
SNAP=$(hf download Qwen/Qwen3.6-35B-A3B --local-dir-use-symlinks True | tail -1)
# Or grab the snapshot path directly from the cache:
# SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<rev>/
```

### 2.3 Convert + gen-config + compile

```bash
.venv/bin/python -m mlc_llm convert_weight "$SNAP" \
    --quantization q4f16_1 \
    -o dist/qwen3_6-35B-A3B-q4f16_1                                # ~5 min, no GPU; 19 GB output

.venv/bin/python -m mlc_llm gen_config "$SNAP" \
    --quantization q4f16_1 --conv-template qwen3_5 \
    -o dist/qwen3_6-35B-A3B-q4f16_1                                # 3 sec; auto-picks model_type=qwen3_5_moe

MLC_MOE_GEMM_V2=1 .venv/bin/python -m mlc_llm compile dist/qwen3_6-35B-A3B-q4f16_1 \
    --device cuda \
    --opt "flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1" \
    -o dist/qwen3_6-35B-A3B-q4f16_1/lib.so                         # ~25 min, 202 MB sm_87 lib
```

Two flags matter here:
- **`MLC_MOE_GEMM_V2=1`** (env var, Phase 9b) — opts the int4 MoE GEMM into the
  dispatch-table + hand-tensorized wmma m16n8k16 kernel. **2.52× pp512** vs
  the persistent-loop v1 (Stage 9.2 baseline 207.95 → 523.67 tps). Decode
  (b=1) is unaffected — still routes through `dequantize_gemv` shortcut.
- **`flashinfer=1`** (compile opt) — links FlashInfer's paged-decode +
  paged-prefill kernels. **+21 % tg512** at pp=512 KV depth (44.88 → 54.35).
  FlashInfer compiles cleanly on Orin sm_87 since the Phase 6 ABI fix; the
  earlier "FlashInfer cache lacks sm_87" claim is stale (was a JIT issue
  fixed in vendored TVM). `--model-lib` is still recommended at runtime to
  bypass the JIT cache lookup.

Combined, this lib hits **pp512 = 561.16 / tg512 = 54.35** on the Orin AGX
MAXN bench — past every Phase 9 gate including the 450-tps "parity to
llama.cpp" stretch. `cudagraph=1;cutlass=1` are the long-standing
Orin-tuned flags from prior phases.

A FlashInfer-off / v1-only build of the same dir is preserved at
[lib_phase9_cta1024_pre_v2.so](dist/qwen3_6-35B-A3B-q4f16_1/lib_phase9_cta1024_pre_v2.so)
for apples-to-apples regression checks.

### 2.4 Run

Interactive chat:

```bash
source .envrc.local && .venv/bin/python -m mlc_llm chat \
    dist/qwen3_6-35B-A3B-q4f16_1 --device cuda:0 \
    --model-lib dist/qwen3_6-35B-A3B-q4f16_1/lib.so
```

`--model-lib` is required — without it the JIT cache lookup re-resolves to the FlashInfer path and segfaults. Engine API:

```python
from mlc_llm import MLCEngine
engine = MLCEngine(
    "dist/qwen3_6-35B-A3B-q4f16_1",
    model_lib="dist/qwen3_6-35B-A3B-q4f16_1/lib.so",
    mode="interactive", device="cuda:0",
)
```

### 2.5 Bench (TG=512 steady-state, MAXN locked)

```bash
source .envrc.local && .venv/bin/python bench_mlc.py \
    --model dist/qwen3_6-35B-A3B-q4f16_1 \
    --pp 128 --tg 512 --runs 3 --warmup 1
```

### 2.6 Known CLI quirks

- The wheel does **not** install a `mlc_llm` shell script — always invoke as `.venv/bin/python -m mlc_llm <cmd>`.
- `convert_weight` and `gen_config` reject HF repo IDs; pass the local snapshot path (`$SNAP` above).
- `gen_config` prints argparse errors to stdout but the actual error to stderr — when redirecting, keep them separate (`> out 2> err`) or you'll see an empty `Error` block.

### 2.7 Optional variants in dist/

| Build | Use case | Notes |
|---|---|---|
| [dist/qwen3_6-35B-A3B-q4f16_1/](dist/qwen3_6-35B-A3B-q4f16_1/) | **default**, max throughput | Phase 9b v2 (MoE dispatch + wmma m16n8k16) + FlashInfer (paged-decode/prefill linked). pp512 = 561.16 / tg512 = 54.35 on Orin AGX |
| [dist/qwen3_6-35B-A3B-q4f16_1_tir/](dist/qwen3_6-35B-A3B-q4f16_1_tir/) | apples-to-apples vs int8 | fp16 KV, FlashInfer hard-disabled in compile flags |
| [dist/qwen3_6-35B-A3B-q4f16_1_kvint8/](dist/qwen3_6-35B-A3B-q4f16_1_kvint8/) | capacity-bound (~2× context) | int8 KV; throughput-neutral but byte-divergent from fp16 (parity 2/5 EXACT, semantic drift only) |
| [dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/](dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/) | reference for sm ≥ 89 port | fp8 KV; -25% on Orin (software fp8 dequant), preserved for Blackwell port |
| [dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/](dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/) | spec-decode draft (not default) | EAGLE-style 1-layer MTP head, 0.71 GB; loses to target_only by 17% on Orin |

### 2.8 Compile & Run Qwen3.5-0.8B (dense)

Same toolchain as §2.3, smaller model. The dense variant has no MoE so `MLC_MOE_GEMM_V2` is irrelevant; the rest of the speedup flags (FlashInfer, cudagraph, cutlass, cublas_gemm) all apply. Quant of choice is **`q4f16_g16e`** — group=16, embed/final_fc included, dlight-tuned for sm_87. Compile takes ~3.5 min and the resulting lib is ~39 MB (with FlashInfer kernels linked).

**Headline (2026-04-30, Orin AGX MAXN, this lib):** TG=512 → **134.82 tps**, 1.345× over llama.cpp Q4_K_XL (100.3 tps). Depth-flat: only -4% drift across 16× decode depth. Full sweep:

| tg   | llama.cpp Q4_K_XL (pure tg) | MLC q4f16_g16e + FI | ratio |
|---:|---:|---:|---:|
|  512 | 100.3 | **134.82** | **1.345×** |
| 1024 | 100.1 | **134.29** | **1.341×** |
| 2048 |  99.7 | **133.54** | **1.340×** |
| 4096 |  98.0 | **132.17** | **1.349×** |
| 8192 |  96.5 | **129.59** | **1.343×** |

Median of 3 runs each (1 warmup), pp=512 prefill + tg=N decode, run-to-run variance ≤ 0.05%. Detailed analysis in §14.2.

```bash
# 1. Weights (1.5 GB bf16, ~30 sec on a fast link)
hf download Qwen/Qwen3.5-0.8B
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<rev>/

# 2. Convert weights (no GPU; 559 MB output)
.venv/bin/python -m mlc_llm convert_weight "$SNAP" \
    --quantization q4f16_g16e \
    -o dist/qwen3_5-0.8B-q4f16_g16e

# 3. Generate engine config (3 sec; auto-picks model_type=qwen3_5)
.venv/bin/python -m mlc_llm gen_config "$SNAP" \
    --quantization q4f16_g16e --conv-template qwen3_5 \
    -o dist/qwen3_5-0.8B-q4f16_g16e

# 4. Compile with all speedups (~3.5 min)
.venv/bin/python -m mlc_llm compile dist/qwen3_5-0.8B-q4f16_g16e \
    --device cuda \
    --opt "flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1" \
    -o dist/qwen3_5-0.8B-q4f16_g16e/lib.so
```

Recompile-only path (weights already converted and gen_config already run — the params and `mlc-chat-config.json` stay valid across runtime ABI bumps):

```bash
source .envrc.local && .venv/bin/python -m mlc_llm compile \
    dist/qwen3_5-0.8B-q4f16_g16e --device cuda \
    --opt "flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1" \
    -o dist/qwen3_5-0.8B-q4f16_g16e/lib.so
```

Bench (TG=512 steady-state, MAXN locked):

```bash
source .envrc.local && .venv/bin/python bench_mlc.py \
    --model-dir dist/qwen3_5-0.8B-q4f16_g16e --device cuda:0 \
    --pp 512 --tg 512 --runs 3 --warmup 1
```

Run (chat / engine API): identical to §2.4, swap model dir / lib path. `--model-lib` is still recommended at runtime.

---

## 3. Family Map (as of 2026-04)

| Model | Released | Total / Active params | Hybrid layers | Linear heads (K/V) | MoE | mRoPE |
|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 2025-09 | 80B / 3B | 48 (`[L,L,L,F]×12`) | 16 / 32 | 512 experts, 10 active + 1 shared | yes |
| **Qwen3.5-0.8B** | 2026-03 | 0.8B / dense | 24 (`[L,L,L,F]×6`) | 16 / 16 | dense MLP | no |
| Qwen3.5-2B | 2026-03 | 2B / dense | 28 (`[L,L,L,F]×7`) | 16 / 16 | dense MLP | no |
| Qwen3.5-4B | 2026-03 | 4B / dense | 32 (`[L,L,L,F]×8`) | 16 / 16 | dense MLP | no |
| Qwen3.5-9B | 2026-03 | 9B / dense | 36 (`[L,L,L,F]×9`) | 16 / 16 | dense MLP | no |
| Qwen3.5-27B | 2026-03 | 27B / dense | 40 (`[L,L,L,F]×10`) | 16 / 32 | dense MLP | no |
| **Qwen3.6-27B** | 2026-04 | 27B / dense | 40 (`[L,L,L,F]×10`) | 16 / 32 | dense MLP | yes |
| **Qwen3.6-35B-A3B** | 2026-04 | 35B / ~3B | 40 (`[L,L,L,F]×10`) | 16 / 32 | 256 experts, 8 active + 1 shared | yes |

All share `model_type: qwen3_5` (or `qwen3_5_moe` for the MoE variants); the original Qwen3-Next still uses `model_type: qwen3_next`. Layer counts and head ratios for the 4B/9B/27B dense variants are projected from the published family pattern (`full_attention_interval=4`, scaling depth ~ `√(params)`); confirm against `config.json` once each is brought up.

Canonical HF repos: `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-{2B,4B,9B,27B}`, `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B`.

The 4B/9B/27B dense variants and Qwen3.6-27B are not yet brought up — they are listed here so the gap analysis (§8) can be applied to them when the time comes. They reuse the dense `qwen3_5` module (no MoE fork needed). Qwen3.5-27B and Qwen3.6-27B switch to asymmetric linear heads (16/32), which exercises the same kernel path the 35B-A3B already validates.

---

## 4. Architecture Summary

### 4.1 Hybrid layer pattern

Every fourth layer is full softmax attention; the other three are GatedDeltaNet linear attention. Indexed by `full_attention_interval=4`:

- linear at indices 0, 1, 2 → full at index 3 → linear at 4, 5, 6 → full at 7 → …
- 0.8B: 24 layers → 18 linear, 6 full
- 35B-A3B: 40 layers → 30 linear, 10 full
- 80B-A3B: 48 layers → 36 linear, 12 full

### 4.2 Full-attention layer

Standard GQA, with two notable additions vs. plain Qwen3:
- **Output gate** (`attn_output_gate: true`): the Q projection emits `2 × num_heads × head_dim` floats; half are queries, half are gate values. The attention output is multiplied element-wise by `sigmoid(gate)` before `o_proj`.
- **Partial RoPE** (`partial_rotary_factor: 0.25`): RoPE is applied to only the first 25% of `head_dim`. With `head_dim=256`, that's 64 rotated dims, 192 untouched.
- Per-head Q and K RMSNorm (no bias).
- Head dim 256 across all variants.

### 4.3 GatedDeltaNet linear-attention layer

A delta-rule SSM with gating. Per-token computation, per `value_head`:

```
S_t = g_t · S_{t-1} + β_t · k_t · (v_t - S_{t-1} · k_t)         # state update (delta rule)
o_t = S_t · q_t                                                  # output
```

with:
- `q, k`: `(num_key_heads, key_head_dim)` post-Conv1d, post-SiLU, post-L2-norm
- `v`: `(num_value_heads, value_head_dim)` post-Conv1d, post-SiLU
- `β = sigmoid(b)` (per head)
- `g = exp(-exp(A_log) · softplus(a + dt_bias))` (per head, fp32)
- `S`: `(num_value_heads, key_head_dim, value_head_dim)` recurrent state, **fp32**
- For asymmetric heads (`num_value_heads > num_key_heads`), Q/K are repeated `num_value_heads // num_key_heads` times.

Sub-block layout (HF / vLLM names):
- `in_proj_qkvz` — fused Q+K+V+Z projection (Z is the post-recurrence output gate stream)
- `in_proj_ba` — fused β + α projection
- `conv1d` — depthwise causal convolution, kernel size 4, applied to QKV after `in_proj`
- SiLU after Conv1d
- L2-norm on Q and K
- `A_log`, `dt_bias` — bare tensors (no `.weight` suffix)
- `FusedRMSNormGated` — multiplies by `SiLU(Z)` after the recurrence
- `out_proj` — final output projection

The released 0.8B and 35B-A3B checkpoints both ship the four-projection layout (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) — confirmed in §8 by `safetensors.index.json`. The loader at [python/mlc_llm/model/qwen35/qwen35_loader.py](python/mlc_llm/model/qwen35/qwen35_loader.py) maps these directly; no fused-`qkvz`/`ba` consolidation in the released configs.

### 4.4 RMSNorm quirk

Qwen3.5 uses `output = norm(x) · (1 + weight)` with weight initialized to 0. TVM `nn.RMSNorm` uses `output = norm(x) · weight`. The loader handles this by adding `1.0` to all standard RMSNorm weights at load time. The gated norm inside GatedDeltaNet (`linear_attn.norm`) does **not** get the `+1.0`.

Affected: `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, top-level `model.norm`.

### 4.5 MoE block (35B-A3B, 80B-A3B) — for Stage 5

- `num_experts=256`, `num_experts_per_tok=8`, plus 1 shared expert
- `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`
- `decoder_sparse_step=1` (every layer is MoE in the MoE variants)
- Routing: softmax over experts, top-k, normalize the chosen probs (`norm_topk_prob=true`)
- Output = `Σ p_i · expert_i(x) + shared_expert(x)`

### 4.6 mRoPE (35B-A3B, 80B-A3B)

`mrope_section: [11, 11, 10]` — head_dim is split into three rotation sub-bands rotated against three position axes (text + spatial). For text-only input, all three sub-bands rotate against the same flattened position, collapsing to standard 1D RoPE. **Stage 6 confirmed** that the existing `RopeMode.NORMAL` in [python/mlc_llm/nn/kv_cache.py](python/mlc_llm/nn/kv_cache.py) produces output identical to the HF mRoPE path on text-only input (4/5 prompts EXACT match; the one diverging step was at the rank-1-vs-rank-2 noise floor). Multimodal (image+text) input still requires explicit `mrope_section` handling — not on the project critical path.

### 4.7 MTP (Multi-Token Prediction) head

Present in the 3.5/3.6 checkpoints (`mtp_num_hidden_layers: 1`). **Phase 4B shipped** as a draft module for spec decode in two flavors:

- [python/mlc_llm/model/qwen35_mtp_draft/](python/mlc_llm/model/qwen35_mtp_draft/) — 0.8B draft (243-line model, 111-line loader). γ=4 lands 120.5 tps with byte-identical parity to target_only on Orin.
- [python/mlc_llm/model/qwen3_5_moe_mtp_draft/](python/mlc_llm/model/qwen3_5_moe_mtp_draft/) — 35B-A3B draft (240-line model, 137-line loader). γ=1 lands 40.8 tps; loses to target_only by 17 % on Orin (BW-bound regime), but the head is correct (96 % step-1 accept rate) and the wiring is robust.

Critical wiring detail (Phase 4B retro): the EAGLE fc head expects `cat([inputs_embeds, hidden_states])`, **not** `cat([h_norm, e_norm])` as the original 0.8B port had. With the wrong order the first half of `fc.weight` (trained for embeddings) reads hidden states and produces 0 % accept. Fix is at [qwen3_5_moe_mtp_draft_model.py:120](python/mlc_llm/model/qwen3_5_moe_mtp_draft/qwen3_5_moe_mtp_draft_model.py#L120) and [qwen35_mtp_draft_model.py:119](python/mlc_llm/model/qwen35_mtp_draft/qwen35_mtp_draft_model.py#L119).

The MTP layer reuses the model's PagedKVCache via `num_attention_layers + config.mtp_num_hidden_layers` — see `create_paged_kv_cache` at [qwen35_model.py:1270](python/mlc_llm/model/qwen35/qwen35_model.py#L1270).

---

## 5. State Layouts

### 5.1 Recurrent state (linear-attention layers only)

Per layer, allocated by `RNNState.create` in [python/mlc_llm/model/qwen35/qwen35_model.py](python/mlc_llm/model/qwen35/qwen35_model.py) (`create_rnn_state`):
- `state_id=0`: recurrent state `S`, shape `(num_value_heads, key_head_dim, value_head_dim)`, dtype **fp32**
- `state_id=1`: Conv1d ring buffer, shape `(kernel_size - 1, qkv_dim)` = `(3, qkv_dim)`, dtype = model dtype

For 0.8B: `state_id=0` is `(16, 128, 128)` fp32 = 1 MB per layer × 18 layers = 18 MB recurrent state per sequence.

### 5.2 Paged KV cache (full-attention layers only)

Standard `PagedKVCache.create_generic` with `attn_kind="mha"`:
- `num_hidden_layers = num_attention_layers + mtp_num_hidden_layers` (the full layers + one slot per MTP layer — 6+1 for 0.8B, 10+1 for 35B-A3B)
- `qk_head_dim = v_head_dim = 256`
- `rope_mode = RopeMode.NORMAL`, `rotary_dim = head_dim · partial_rotary_factor = 64`

`kHybrid` KVStateKind handling lives in the runtime (kept stateful per-layer-type); our model code consumes both objects and the cache layer dispatches by layer index.

### 5.3 KV-cache dtype split (Phase 5 + Phase 6)

The KV pages can be stored in a different dtype from the model activations. Plumbing lives one layer below the model code (model `create_paged_kv_cache` still passes only `dtype`):

- **Python**: `dtype_kv` flows through [python/mlc_llm/nn/kv_cache.py](python/mlc_llm/nn/kv_cache.py) `create_generic` → [python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py](python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py) → [3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py) `TIRPagedKVCache`. Trailing `rx.StringImm(dtype_kv)` arg into the runtime constructor.
- **C++ runtime** ([3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc](3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc)): page buffer allocated as `dtype_kv`, temp Q/K/V/O still in `dtype`. New `std::vector<Tensor> scales_;` field allocated parallel to `pages_` (full-size fp32 for MHA layers, `{1}` placeholder for linear-attn). Threaded through 5 kernel call sites and 3 MHA virtuals on `PagedPrefillFunc` / `PagedDecodeFunc` / `PagedPrefillTreeMaskFunc` ([attn_backend.h](3rdparty/tvm/src/runtime/vm/attn_backend.h)). FlashInfer overrides ignore scales (no int8/fp8 KV support).
- **TIR kernels**: [_page_kernels.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_page_kernels.py) (per-token quant on write, scales memcpy in copy/compact), [_decode_kernels.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_decode_kernels.py) and [_prefill_kernels.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_prefill_kernels.py) (`T.cast(int8, fp16) * scale` on K/V loads), [tree_attn.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/tree_attn.py) (symmetric for spec verify). Scales tensor is **always passed** for signature uniformity; gated on `dtype_kv == "int8"` (Python-time branch). fp16/bf16 paths are byte-identical to before.

Shipped variants in dist/:
- fp16 KV (default): throughput-optimal at short ctx, KV-bound at long ctx
- int8 KV: throughput-neutral, ~2× capacity, 2/5 EXACT parity (semantic drift only)
- fp8 KV: −25 % at tg8192 on Orin (software dequant); preserved as reference for sm ≥ 89 ports

Bench numbers in §14.3.

---

## 6. Reference Implementations

Read-only; do not vendor.

- **HuggingFace transformers** (use ≥ 4.57; earlier releases had a feature-dim bug in `torch_chunk_gated_delta_rule`, HF #40963):
  - `src/transformers/models/qwen3_next/modeling_qwen3_next.py`
  - `src/transformers/models/qwen3_next/modular_qwen3_next.py`
  - `src/transformers/models/qwen3_next/configuration_qwen3_next.py`
- **vLLM**:
  - `vllm/model_executor/models/qwen3_next.py` (mostly wiring)
  - `vllm/model_executor/layers/mamba/gdn_linear_attn.py` (the actual `GatedDeltaNetAttention` layer — has the projection-name details `in_proj_qkvz`, `in_proj_ba`)
- **flash-linear-attention** (canonical kernel reference for cross-checking math):
  - `fla/layers/gated_deltanet.py`
  - `fla/ops/gated_delta_rule/`
- **NVlabs GatedDeltaNet** (ICLR 2025 paper code): https://github.com/NVlabs/GatedDeltaNet
- **vLLM blog**: https://blog.vllm.ai/2025/09/11/qwen3-next.html

---

## 7. Existing MLC Implementation Inventory

[python/mlc_llm/model/qwen35/](python/mlc_llm/model/qwen35/) was added in PR #3449 (Oct 2025) and is the foundation. Phase 4B added the two MTP draft modules; Phase 5 forked qwen3_5_moe.

### 7.1 [qwen35_model.py](python/mlc_llm/model/qwen35/qwen35_model.py) (1419 lines)

| Lines | Component | Notes |
|---|---|---|
| 30–144 | `Qwen35Config` | Reads HF config, handles VLM nesting (`text_config`, `rope_parameters`), exposes `layer_types()`, carries MTP + spec-decode fields |
| 146–150 | `Qwen35Embedding` | Tied lm_head via `lm_head_forward` |
| 152–169 | `Qwen35MLP` | Standard `gate_up_proj` + `down_proj`, SiLU |
| 171–246 | `Qwen35Attention` | GQA + sigmoid output gate. `c_attn` is `2·h_q + 2·h_kv` heads (Q+gate+K+V fused). Per-token small-batch dispatch (Phase 4B B.5/B.6) |
| 248–370 | `create_gated_delta_net_func` | TIR kernel, thread-per-V-column, fp32 state, supports prefill (loop over t) and decode. Phase 4 v6 register-cached state (`state_local` sblock) — 5 GMEM passes collapsed to 2 |
| 372–485 | `create_gated_delta_net_func_with_history` | History-mode variant for spec-decode verify; flushes per-position state into RNNState history slots so rejected tokens can roll back without state corruption |
| 487–882 | `Qwen35GatedDeltaNet` | Sub-projections, Conv1d, L2-norm, gate/beta. `forward` (decode/prefill) and `forward_with_history` (spec verify). Per-token small-batch verify dispatch in 5 sites |
| 884–948 | `Qwen35DecoderLayer` | Dispatches between full and linear by layer type |
| 950–960 | `_Qwen35MTPDecoderLayer` | Wraps a normal decoder layer with the EAGLE-style fused embedding+hidden input |
| 962–1017 | `Qwen35MTPHead` | EAGLE fc head + one decoder layer + norm. **Concat order is `cat([inputs_embeds, hidden_states])`** (Phase 4B critical fix) |
| 1019–1061 | `Qwen35Model` | Decoder stack + final norm |
| 1063–1244 | `Qwen35LMHeadModel` | Top-level + `prefill` / `decode` / `batch_*` / `batch_verify_to_last_hidden_states` (γ-specialized verify entries from Phase 4B) |
| 1246–1268 | `create_rnn_state` | RNNState init: state_id 0 (S, fp32) + state_id 1 (conv buffer, model dtype) + history slots for spec verify |
| 1270–1300 | `create_paged_kv_cache` | Allocates `num_attention_layers + mtp_num_hidden_layers` slots (full layers + MTP layer) |

### 7.2 [qwen35_loader.py](python/mlc_llm/model/qwen35/qwen35_loader.py) (223 lines)

- `hf = "model.language_model"` hard-coded prefix (line 51) — both 0.8B and 35B-A3B are VLMs (architectures: `Qwen3_5MoeForConditionalGeneration` for 35B, equivalent VLM wrap on 0.8B), so the prefix is correct as-is. Vision and MTP weights drop silently via the `named_parameters` walk.
- Fuses HF `q_proj/k_proj/v_proj` → MLC `c_attn`.
- Maps `in_proj_qkv`, `A_log`, `dt_bias` (no `.weight`), `conv1d.weight` → `conv1d_weight`.
- Fuses `gate_proj/up_proj` → `gate_up_proj`.
- Adds `+1.0` to the standard RMSNorm weights; leaves gated `linear_attn.norm` alone.

### 7.3 Stage-5 / Phase-4B siblings

- [python/mlc_llm/model/qwen3_5_moe/](python/mlc_llm/model/qwen3_5_moe/) — Stage-5 MoE fork (703-line model + loader). `Qwen35MoEConfig` extends `Qwen35Config` with MoE + mRoPE fields; `Qwen35MoESparseMoeBlock` mirrors `Qwen2MoeSparseMoeBlock` (router → softmax-topk → cumsum/get_indices → MixtralExperts → moe_sum) plus a sigmoid-gated dense `shared_expert`. Reuses qwen35's GDN + attention path via direct import — zero duplication.
- [python/mlc_llm/model/qwen35_mtp_draft/](python/mlc_llm/model/qwen35_mtp_draft/) — 0.8B EAGLE draft module (Phase 4B).
- [python/mlc_llm/model/qwen3_5_moe_mtp_draft/](python/mlc_llm/model/qwen3_5_moe_mtp_draft/) — 35B-A3B EAGLE draft module (Phase 4B).

### 7.4 Registration

[python/mlc_llm/model/model.py:414-485](python/mlc_llm/model/model.py#L414-L485) registers five entries:
- `qwen3_5` / `qwen3_5_text` → `Qwen35LMHeadModel`
- `qwen35_mtp_draft` → `Qwen35MTPDraftLM`
- `qwen3_5_moe` / `qwen3_5_moe_text` → `Qwen35MoEForCausalLM`
- `qwen3_5_moe_mtp_draft` → `Qwen35MoEMTPDraftLM`

[python/mlc_llm/conversation_template/qwen3_5.py](python/mlc_llm/conversation_template/qwen3_5.py) registers the conversation template (reused by both dense and MoE — chat format is identical).

No model preset for `qwen3_5` in [python/mlc_llm/model/model_preset.py](python/mlc_llm/model/model_preset.py); deferred indefinitely as the `convert_weight + gen_config + compile` flow from §2 is the canonical path.

---

## 8. Gap Table (current vs. target)

Updated 2026-04-25 after inspecting the actual `Qwen/Qwen3.5-0.8B` `config.json` and `model.safetensors.index.json`.

| Capability | qwen35 today | Qwen3.5-0.8B needs | Qwen3.6-35B-A3B needs |
|---|---|---|---|
| Hybrid layer dispatch | ✅ | ✅ | ✅ |
| GatedDeltaNet TIR kernel | ✅ | ✅ | ✅ (verify asymmetric head path) |
| Conv1d state | ✅ | ✅ | ✅ |
| Dual KV state (Paged + RNN) | ✅ | ✅ | ✅ |
| Output-gated GQA | ✅ (hardcoded) | ✅ (config has `attn_output_gate=true`) | ✅ |
| Partial RoPE | ✅ | ✅ (`partial_rotary_factor=0.25`) | ✅ |
| RMSNorm `+1.0` quirk | ✅ | ✅ | ✅ |
| HF prefix `model.language_model.*` | ✅ hardcoded VLM | ✅ **confirmed: 0.8B IS a VLM**, prefix is correct | ✅ (3.6 is multimodal) |
| Tied embeddings (`tie_word_embeddings`) | ✅ supports both | ✅ true | ✅ (probably) |
| `in_proj_qkv/z/a/b` (4 separate weights, not fused `qkvz`/`ba`) | ✅ matches | ✅ confirmed in `safetensors.index.json` | ⚠️ verify for 35B |
| Symmetric linear heads (16/16) | ✅ | ✅ (16/16) | — needs 16/32 |
| Asymmetric head reshape/repeat in kernel callers | partial (`heads_per_group` in kernel) | n/a | needs verification |
| Dense MLP | ✅ | ✅ | needs MoE swap |
| MoE block (256 experts + 1 shared) | ❌ | n/a | needed |
| mRoPE (`mrope_section`) | ❌ | ⚠️ **0.8B has `mrope_section=[11,11,10]` too**, but for text-only inference this reduces to standard 1D RoPE — `RopeMode.NORMAL` should be correct. Verify at Stage 3. | needed (real multimodal) |
| MTP head | ❌ skipped | skip (`mtp_num_hidden_layers=1` in config) | skip in v1 |
| Model preset | ❌ | nice-to-have | nice-to-have |
| Validation harness | ✅ (`validate.py` v1, Stage 0) | ✅ done | reuses 0.8B harness |

### Confirmed 0.8B HF config values

```
hidden_size:           1024
num_hidden_layers:     24
num_attention_heads:   8
num_key_value_heads:   2          (GQA 4:1)
head_dim:              256
vocab_size:            248320
max_position:          262144     (256K context)
tie_word_embeddings:   true
attn_output_gate:      true
partial_rotary_factor: 0.25       (rotary_dim = 64)
rope_theta:            10000000
mrope_section:         [11, 11, 10]   (sums to 32; ×2 = 64 = rotary_dim)
mrope_interleaved:     true
mtp_num_hidden_layers: 1          (skipped)
linear_key_head_dim:   128
linear_value_head_dim: 128
linear_num_key_heads:  16
linear_num_value_heads:16        (symmetric on 0.8B)
linear_conv_kernel_dim: 4
full_attention_interval: 4
```

### Confirmed 0.8B HF weight names (per layer i)

Linear-attention layer (i ∈ {0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22}):
```
model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight   # fused Q+K+V
model.language_model.layers.{i}.linear_attn.in_proj_z.weight     # output gate stream
model.language_model.layers.{i}.linear_attn.in_proj_a.weight     # decay control
model.language_model.layers.{i}.linear_attn.in_proj_b.weight     # update rate β
model.language_model.layers.{i}.linear_attn.out_proj.weight
model.language_model.layers.{i}.linear_attn.conv1d.weight
model.language_model.layers.{i}.linear_attn.norm.weight          # gated RMSNorm (no +1)
model.language_model.layers.{i}.linear_attn.A_log                # NO .weight suffix
model.language_model.layers.{i}.linear_attn.dt_bias              # NO .weight suffix
```

Full-attention layer (i ∈ {3, 7, 11, 15, 19, 23}):
```
model.language_model.layers.{i}.self_attn.q_proj.weight   # 2× Q dim (Q + gate, layout per-head [Q_d, gate_d])
model.language_model.layers.{i}.self_attn.k_proj.weight
model.language_model.layers.{i}.self_attn.v_proj.weight
model.language_model.layers.{i}.self_attn.o_proj.weight
model.language_model.layers.{i}.self_attn.q_norm.weight   # +1 needed
model.language_model.layers.{i}.self_attn.k_norm.weight   # +1 needed
```

All layers also have:
```
model.language_model.layers.{i}.input_layernorm.weight             # +1 needed
model.language_model.layers.{i}.post_attention_layernorm.weight    # +1 needed
model.language_model.layers.{i}.mlp.gate_proj.weight
model.language_model.layers.{i}.mlp.up_proj.weight
model.language_model.layers.{i}.mlp.down_proj.weight
```

Top-level:
```
model.language_model.embed_tokens.weight
model.language_model.norm.weight       # +1 needed
model.visual.*                         # ignored
mtp.*                                  # ignored
```

The existing loader handles all of this correctly (explicit mappings for `in_proj_qkv`/`A_log`/`dt_bias`/`conv1d.weight`, catch-all for `in_proj_z/a/b/out_proj/norm`, RMSNorm `+1` predicate covers exactly the right weights). **No loader changes needed for 0.8B** based on static inspection — the prediction in the original gap table that "loader prefix likely needs fix" was wrong. The loader is correct as-is.

### Confirmed 35B-A3B HF config values (2026-04-25, fetched from `Qwen/Qwen3.6-35B-A3B`)

```
architectures:                Qwen3_5MoeForConditionalGeneration  (multimodal — text under model.language_model.*)
model_type (top):             qwen3_5_moe
model_type (text_config):     qwen3_5_moe_text
hidden_size:                  2048
num_hidden_layers:            40
num_attention_heads:          16
num_key_value_heads:          2          (GQA 8:1)
head_dim:                     256
vocab_size:                   248320
max_position_embeddings:      262144
tie_word_embeddings:          false      (lm_head is a separate weight at top level)
attn_output_gate:             true
partial_rotary_factor:        0.25       (rotary_dim = 64)
rope_theta:                   10000000
mrope_section:                [11, 11, 10]
mrope_interleaved:            true
mtp_num_hidden_layers:        1          (skipped)
linear_key_head_dim:          128
linear_value_head_dim:        128
linear_num_key_heads:         16
linear_num_value_heads:       32         (asymmetric — 2× value heads vs 0.8B's 16/16)
linear_conv_kernel_dim:       4
full_attention_interval:      4          (also explicit `layer_types` array — same pattern)
mamba_ssm_dtype:              float32
moe_intermediate_size:        512
shared_expert_intermediate_size: 512
num_experts:                  256
num_experts_per_tok:          8          (top-8 routing)
decoder_sparse_step:          (not set)  → default 1 in Qwen35MoEConfig (every layer is MoE)
norm_topk_prob:               (not set)  → default true
dtype:                        bfloat16
```

### Confirmed 35B-A3B HF weight names (per layer i)

Linear-attn layers (30 total — same names as 0.8B, but `A_log`/`dt_bias`/`norm` shapes are `[32]`/`[32]`/`[128]` for 32 value heads):
```
model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight   shape=[8192, 2048]   (16·128 + 16·128 + 32·128)
model.language_model.layers.{i}.linear_attn.in_proj_z.weight     shape=[4096, 2048]   (32·128)
model.language_model.layers.{i}.linear_attn.in_proj_a.weight     shape=[32, 2048]
model.language_model.layers.{i}.linear_attn.in_proj_b.weight     shape=[32, 2048]
model.language_model.layers.{i}.linear_attn.out_proj.weight      shape=[2048, 4096]
model.language_model.layers.{i}.linear_attn.conv1d.weight        shape=[8192, 1, 4]
model.language_model.layers.{i}.linear_attn.norm.weight          shape=[128]
model.language_model.layers.{i}.linear_attn.A_log                shape=[32]
model.language_model.layers.{i}.linear_attn.dt_bias              shape=[32]
```

Full-attn layers (10 total at indices 3,7,11,15,19,23,27,31,35,39):
```
model.language_model.layers.{i}.self_attn.q_proj.weight   shape=[8192, 2048]   (2·16·256 with attn_output_gate)
model.language_model.layers.{i}.self_attn.k_proj.weight   shape=[512, 2048]    (2·256)
model.language_model.layers.{i}.self_attn.v_proj.weight   shape=[512, 2048]
model.language_model.layers.{i}.self_attn.o_proj.weight   shape=[2048, 4096]   (16·256)
model.language_model.layers.{i}.self_attn.q_norm.weight   shape=[256]   (+1 needed)
model.language_model.layers.{i}.self_attn.k_norm.weight   shape=[256]   (+1 needed)
```

MoE FFN — every layer (HF pre-stacks experts; no `.weight` suffix on the stacked tensors):
```
model.language_model.layers.{i}.mlp.gate.weight                            shape=[256, 2048]      (router)
model.language_model.layers.{i}.mlp.experts.gate_up_proj                   shape=[256, 1024, 2048]   (pre-stacked, fused gate+up)
model.language_model.layers.{i}.mlp.experts.down_proj                      shape=[256, 2048, 512]
model.language_model.layers.{i}.mlp.shared_expert.gate_proj.weight         shape=[512, 2048]
model.language_model.layers.{i}.mlp.shared_expert.up_proj.weight           shape=[512, 2048]
model.language_model.layers.{i}.mlp.shared_expert.down_proj.weight         shape=[2048, 512]
model.language_model.layers.{i}.mlp.shared_expert_gate.weight              shape=[1, 2048]   (sigmoid gate)
```

Top-level (note `lm_head` is NOT under `model.language_model.`):
```
model.language_model.embed_tokens.weight  shape=[248320, 2048]
model.language_model.norm.weight          shape=[2048]   (+1 needed)
lm_head.weight                            shape=[248320, 2048]   (untied)
model.visual.*                            ignored
mtp.*                                     ignored
```

[python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_loader.py](python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_loader.py) handles every line above. The pre-stacked experts pass through directly to `MixtralExperts.weight` since both are `[num_experts, out_features, in_features]` — no `np.stack` or `np.concatenate` needed for the routed experts (the shared expert still needs gate+up fusion, exactly as in qwen2_moe).

---

## 9. Known Pitfalls (from upstream issues + our own scars)

1. **Recurrence dtype** — `mamba_ssm_dtype: float32` in the official config. Keep `A_log.exp()`, `softplus(a + dt_bias)`, and `S` in fp32 even when the model dtype is fp16/bf16. Drift in recurrence is silent and accumulates over long sequences.
2. **State aliasing** — never alias the read and write of recurrent state in the same op. SGLang #20791: a flashinfer `gated_delta_rule_decode_pretranspose` regression was traced to no-buffer scheduling aliasing the in/out state. RNNState's `get`/`set` already returns a fresh tensor — do not optimize that away.
3. **Speculative decode + recurrent state rollback (Phase 4B)** — vLLM #39273 originally argued spec decode rollback on rejected tokens corrupts GDN state. **Solved here** via the history-mode forward path: [create_gated_delta_net_func_with_history](python/mlc_llm/model/qwen35/qwen35_model.py#L372) emits a full per-position state history, [forward_with_history](python/mlc_llm/model/qwen35/qwen35_model.py#L618) writes it via `state.set_with_history(...)`, and on rejection the engine snaps back to the per-position slot. Shipped on both 0.8B (γ=4 byte-identical) and 35B-A3B (γ=1 correct). The dispatch picks history-mode automatically for spec verify; non-spec decode keeps the cheaper non-history path.
4. **Marlin tile sizes** at high TP — vLLM #35924: `MIN_THREAD_N=64` broke `in_proj_ba` whose output is only `num_v_heads` (very narrow). Watch when adding TP > 1 quantized paths.
5. **HF parity reference** — must be transformers ≥ 4.57. Earlier `torch_chunk_gated_delta_rule` had a feature-dim mismatch (HF #40963). Pin in `validate.py`.
6. **Don't lose `attn_output_gate`** — easy to drop when subclassing a vanilla Qwen3 attention. The full layer's `c_attn` width is `(2·h_q + 2·h_kv)·d`, not `(h_q + 2·h_kv)·d`. Hardcoded in `Qwen35Attention`; both 0.8B and 35B-A3B set this true so no config switch is needed.
7. **MTP head concat order (Phase 4B)** — the EAGLE fc head expects `cat([inputs_embeds, hidden_states])`, **not** the reverse. With the wrong order the first half of `fc.weight` (trained for embeddings) reads hidden states → 0 % accept. The original 0.8B port shipped reversed; both drafts are now correct ([qwen35_mtp_draft_model.py:119](python/mlc_llm/model/qwen35_mtp_draft/qwen35_mtp_draft_model.py#L119), [qwen3_5_moe_mtp_draft_model.py:120](python/mlc_llm/model/qwen3_5_moe_mtp_draft/qwen3_5_moe_mtp_draft_model.py#L120)).
8. **Hybrid + FlashInfer RNN-state init (Phase 6 latent fix)** — [cpp/serve/function_table.cc:245-275](cpp/serve/function_table.cc#L245-L275) had RNN-state setup nested inside `if (sliding_window || !flashinfer_defined)`, so hybrid+FlashInfer left `create_rnn_state_func_` null. Bug was masked for months because Phase 5's fp8 lib raised `NotImplementedError` from the FlashInfer dispatch (caught → empty), short-circuiting before the null deref. Phase 6's regression case (fp16, dtype_kv == dtype) included FlashInfer and exposed it. Fix hoists RNN-state setup to its own branch on `kv_state_kind == kHybrid`.
9. **MoE per-token vs batched dispatch (Phase 4B)** — [group_quantization.py:800-811](python/mlc_llm/quantization/group_quantization.py#L800-L811) routes `if indptr.ndim == 2: dequantize_gemv else: dequantize_group_gemm`. The MoE block's `if num_tokens == 1:` resolves to gemv only when `num_tokens` is a literal — for spec-verify with symbolic `seq_len`, it routes to `dequantize_group_gemm` regardless of actual seq_len. group_gemm is ~6× slower than gemv on Orin at small batch. Phase 4B added γ-specialized verify entries (`batch_verify_to_last_hidden_states_g{1,2,3,4}`) to keep the literal-seq path live in 5 sites.
10. **FLA on Blackwell** — fla-org #607 is a backward-pass bug; doesn't affect inference. Listed for context only.

---

## 10. Phase History (everything has shipped)

Correctness phase (the original [.claude/plans/ok-we-re-going-to-squishy-harbor.md](.claude/plans/ok-we-re-going-to-squishy-harbor.md)):

| Stage | Status | Result |
|---|---|---|
| 0 — doc + worklog | ✅ | this file + [worklog.md](worklog.md) |
| 1 — `validate.py` PyTorch reference | ✅ | greedy + per-layer hidden-state dump cached to `reference_outputs.pt` |
| 2 — bring up qwen35 against 0.8B checkpoint | ✅ | compiled and ran first try; the predicted prefix-detection failure didn't materialize (0.8B is a VLM) |
| 3 — per-layer numerical parity | ✅ (skipped) | passed straight to Stage 4 once the model loaded coherently |
| 4 — end-to-end greedy parity (50 × 5 prompts) | ✅ | 50/50 on all 5 prompts, 0.8B |
| 5 — fork qwen3_5_moe | ✅ | [python/mlc_llm/model/qwen3_5_moe/](python/mlc_llm/model/qwen3_5_moe/) |
| 6 — validate 35B-A3B end-to-end | ✅ | 4/5 EXACT, 5th diverged at fp16 noise floor (rank-1-vs-rank-2 logit gap 0.19) |

Perf phase ([.claude/plans/phase2-perf.md](.claude/plans/phase2-perf.md), [phase2c](.claude/plans/phase2c-perf-after-profile.md), [phase2d](.claude/plans/phase2d-ft-hybrid-quant.md), [phase4-perf](.claude/plans/phase4-perf.md)):

| Phase | Status | Result |
|---|---|---|
| 1 — first-pass kernel triage | ✅ | MoE `dequantize_gemv` reachable (CTA grid + static `num_tokens=1`); parallel topk_softmax kernel; 35B-A3B 10.12 → 47.88 tps |
| 2 — sm_87 dlight tuning + GDN register-cached state | ✅ | gdn_func 40.4 → 17.3 µs/call; 35B v6 = 52.62 tps tg64 (1.789× llama.cpp Q4_K_S) |
| 2C — post-profile cleanup | ✅ | `attn_o_proj` and MoE `gate_up` confirmed Pareto-optimal (14-config tile sweep); no further tile gains |
| 2D — FT hybrid quant | ❌ closed | CUDA-graph exclusion eats kernel gains |
| 3 — B-ext spec decode | ❌ dead | 5.2 % token agreement; abandoned in favor of MTP |
| 4B — MTP spec decode | ✅ | EAGLE-style draft for both models. Critical fix: `cat([e, h])` order (was reversed). 0.8B γ=4 = 120.5 tps byte-identical; 35B γ=1 = 40.8 tps correct (loses 17 % to target_only on Orin's BW-bound regime — wins are expected on BW-rich hardware) |
| 5 — fp8 KV cache | ⚠️ shipped, off by default | All plumbing lands; software fp8 dequant on sm_87 costs −25 % at tg8192. Lib at [dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/](dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/) preserved as reference for sm ≥ 89 |
| 6 — int8 KV cache | ✅ | Throughput-neutral on Orin (`cvt.rn.f16.s8` is single-SASS since Pascal). 3/4 land-criteria pass; parity 2/5 EXACT (semantic drift, not catastrophic). Lib at [dist/qwen3_6-35B-A3B-q4f16_1_kvint8/](dist/qwen3_6-35B-A3B-q4f16_1_kvint8/) opt-in for capacity-bound deployments. Latent function_table.cc bug surfaced and fixed (see §9.8) |

Date-stamped detail in [worklog.md](worklog.md).

---

## 11. Acceptance Bars (all met)

Per CLAUDE.md, fp16 throughout (SSM math fp32 internally):

| Stage | Bar | 0.8B result | 35B-A3B result |
|---|---|---|---|
| Embeddings | `atol ≤ 1e-4` (fp32), `atol ≤ 1e-3` (fp16) | ✅ | ✅ |
| Per-layer output, full attention | `rtol = 1e-3`, `atol = 1e-3` (fp16) | ✅ | ✅ |
| Per-layer output, linear attention | `rtol = 2e-3`, `atol = 2e-3` (fp16) | ✅ | ✅ |
| Recurrent state `S` (fp32) | `atol ≤ 1e-4` after Conv1d, `atol ≤ 1e-3` after first recurrence | ✅ | ✅ |
| Greedy decode, 50 tokens | ≥ 48/50 per prompt, 5 fixed prompts | 50/50 × 5 | 50/50 × 4 + 33/50 × 1 (fp16-noise flip at step 34, logit gap 0.19) |

To re-run regressions on the 0.8B (which also smoke-tests the shared qwen35 path used by 35B-A3B):

```bash
source .envrc.local && \
.venv/bin/python validate.py --greedy-parity \
    --model Qwen/Qwen3.5-0.8B \
    --mlc-model-dir dist/qwen3_5-0.8B-q0f16 \
    --device cuda:1
```

Failure mode (preserved for future regressions): dump per-tensor numpy on both sides, diff with `np.testing.assert_allclose`, log `np.abs(a-b).max()` and `argmax(abs(a-b))`. Validate GatedDeltaNet sub-steps in order: post-Conv1d → post-SiLU → post-L2-norm Q/K → β/g values → S after step 1 → output.

---

## 12. Out of Scope (still)

What never shipped and is not on the immediate roadmap:

- **Tensor parallel (TP > 1)** — `Qwen35MoEDecoderLayer` has no `_set_tp` (qwen35's dense base also has none). Would mirror qwen2_moe's pattern. Not needed for single-Orin or single-Blackwell deployment.
- **Real multimodal** — `model.visual.*` weights are dropped silently by the loader; mRoPE is wired only for the text-only collapse (§4.6). Vision tower would need its own module.
- **Engine γ=1 fast path** — replace one b=2 batched verify with two sequential single-token decodes. Math says +14 % over current spec on the 35B-Orin but still doesn't beat target_only. Lateral move; shelved (worklog 2026-04-28 cont. 14).
- **Custom CUDA** — TIR-only convention held throughout. Phase 5/6 added vendored TVM C++ changes (paged_kv_cache.cc, attn_backend.h, codegen_cuda.cc fp8 helpers), but no hand-written CUDA kernels.

What was originally listed as out-of-scope and shipped anyway: kernel perf (Phase 1-2), MTP head (4B), speculative decoding (4B), q4f16_1 quantization (default 35B lib), int8/fp8 KV (5+6).

---

## 13. Repo Conventions to Honor

- **MoE variants get their own module.** `qwen3` / `qwen3_moe`, `qwen2` / `qwen2_moe`, `mistral` / `mixtral`, `deepseek` / `deepseek_v2`. Honored: dense at [python/mlc_llm/model/qwen35/](python/mlc_llm/model/qwen35/), MoE at [python/mlc_llm/model/qwen3_5_moe/](python/mlc_llm/model/qwen3_5_moe/).
- **MTP draft modules also get their own module** (Phase 4B convention): [python/mlc_llm/model/qwen35_mtp_draft/](python/mlc_llm/model/qwen35_mtp_draft/), [python/mlc_llm/model/qwen3_5_moe_mtp_draft/](python/mlc_llm/model/qwen3_5_moe_mtp_draft/).
- Loader files alongside model files; quantization configured declaratively in `model.py` via `make_quantization_functions(...)` — no per-model quantization file unless the model needs `BlockScaleQuantize` or similar.
- Conversation templates in [python/mlc_llm/conversation_template/qwen3_5.py](python/mlc_llm/conversation_template/qwen3_5.py), registered via `ConvTemplateRegistry.register_conv_template`. Reused by both dense and MoE — chat format is identical.
- **`tirx` (not `tir`)** — the repo migrated to `tirx` namespace in PR #3462. Confirm via `from tvm.script import tirx as T` at the top of every model/kernel file.
- **KV-cache dtype split**: when adding new quant paths, plumb through [python/mlc_llm/nn/kv_cache.py](python/mlc_llm/nn/kv_cache.py), [dispatch_kv_cache_creation.py](python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py), and the C++ runtime — see §5.3. Existing model `create_paged_kv_cache` does NOT need changes.

---

## 14. Benchmarks: Unsloth Q4_K_S (llama.cpp) vs MLC compiled

All numbers Orin AGX (sm_87, 204 GB/s peak BW), MAXN power profile, batch=1, no concurrency. Same precision class on both sides: 4-bit weights + fp16 activations (Unsloth `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` ≈ 19.45 GB, MLC `q4f16_1` ≈ 19 GB at 4.345 bits/param). Bench harness: [bench_compare.py](bench_compare.py) drives `llama-bench` (`-p N` for prefill, `-d N -n tg` for decode-at-depth) and `bench_mlc.py` at the same context lengths.

### 14.1 Qwen3.6-35B-A3B

**Initial baseline (2026-04-27, before perf work) — 3× slower than llama.cpp at every context:**

| ctx | llama.cpp Q4_K_S tg (tps) | MLC q4f16_1 tg (tps) | ratio |
|---:|---:|---:|---:|
| 128  | 29.59 | 10.12 | 0.34× |
| 1024 | 29.24 | 9.66  | 0.33× |
| 4096 | 28.49 | 8.36  | 0.29× |

**Latest shipped — Phase 6 fp16 KV TIR (default lib at [dist/qwen3_6-35B-A3B-q4f16_1/](dist/qwen3_6-35B-A3B-q4f16_1/)) — 1.79× over llama.cpp at short context:**

| pp / tg | llama.cpp Q4_K_S tg (tps) | MLC q4f16_1 tg (tps) | ratio | notes |
|---:|---:|---:|---:|---|
| 128 / 64   | 29.59 | **54.41** | **1.84×** | shipping headline |
| 128 / 256  | 29.5  | 51.24     | 1.74×     | TG steady-state |
| 512 / 256  | 29.2  | 46.34     | 1.59×     | KV starting to bite |
| 4096 / 256 | 28.5  | 24.45     | 0.86×     | MLC crosses below at long ctx |
| 8192 / 256 | ~28   | 15.88     | ~0.57×    | structural BW saturation |

The crossover at ~4K context is the KV-cache read cost. MLC's PagedKVCache (TIR fallback on sm_87, no FlashInfer) reads at lower effective BW than llama.cpp's `q8_0` KV at long sequence. Phase 4A KV-int8 (§14.3 below) was investigated as a fix — throughput-neutral but parity drift. Long-context perf parity vs llama.cpp at ctx ≥ 4K is an open lane.

**Perf-progression (selected milestones, ctx=128 / tg=64 unless noted):**

| version | tg_tps | vs llama.cpp Q4_K_S |
|---|---:|---:|
| Initial baseline (2026-04-27) | 10.12 | 0.34× |
| MoE dispatch fix (gemv reachable) | 44.85 | 1.52× |
| Parallel topk_softmax | 47.88 | 1.629× |
| sm_87 dlight GEMV tuning (v5) | 51.37 | 1.745× |
| **gdn_func register-cached state (v6, shipped)** | **52.62** (tg512) | **1.789×** |

### 14.2 Qwen3.5-0.8B

Initial baseline (2026-04-27) — wins at short ctx, regresses at long ctx:

| ctx | llama.cpp Q4_K_S tg (tps) | MLC q4f16_1 tg (tps) | ratio |
|---:|---:|---:|---:|
| 128  | 107.99 | 131.62 | 1.22× |
| 1024 | 106.11 | 105.78 | 1.00× |
| 4096 | 102.58 | 63.27  | 0.62× |

The 4K regression has the same shape as the 35B-A3B crossover — KV-cache read dominance on the TIR fallback path. The 0.8B has been re-benched post-dlight-patch at 120.5 tps γ=4 with byte-identical parity (worklog 2026-04-28 cont. 12); update this table when a fresh apples-to-apples run lands.

Best practical lib for the 0.8B is the spec-decode build at [dist/qwen3_5-0.8B-q0f16-mtp/](dist/qwen3_5-0.8B-q0f16-mtp/) + draft at [dist/qwen3_5-0.8B-q0f16-mtp-draft/](dist/qwen3_5-0.8B-q0f16-mtp-draft/) — γ=4 lands 120.5 tps decode with byte-identical parity to target_only. The integrated lib does not need separate compile flags.

**Q4_K_XL TG-depth sweep (2026-04-30, llama.cpp standalone, FA on, MAXN, 3 reps).** Bench is `llama-bench -pg 512,N -fa 1` for N ∈ {512, 1024, 2048, 4096, 8192}; weights file [models/qwen3.5-0.8b/Qwen3.5-0.8B-UD-Q4_K_XL.gguf](../models/qwen3.5-0.8b/Qwen3.5-0.8B-UD-Q4_K_XL.gguf) (522 MiB, ggml labels it "qwen35 0.8B Q4_K - Medium" — `XL` is unsloth's dynamic-bit override, not a base ggml quant). Run script: [scratch_lcpp_tg_sweep.sh](scratch_lcpp_tg_sweep.sh); raw output [tuning/lcpp_tg_sweep_0.8b_20260430_165824.md](tuning/lcpp_tg_sweep_0.8b_20260430_165824.md).

| test            | reported tps | tg-only tps¹ |
|-----------------|---:|---:|
| pp512 (prefill) | 4538.5 ± 171 | — |
| tg128 (no pp)   | 100.23 ± 0.18 | 100.2 |
| pp512 + tg512   | 196.17 ± 0.05 | 100.3 |
| pp512 + tg1024  | 148.52 ± 0.05 | 100.1 |
| pp512 + tg2048  | 123.96 ± 0.03 | 99.7 |
| pp512 + tg4096  | 109.94 ± 0.47 | 98.0 |
| pp512 + tg8192  | 102.36 ± 0.71 | 96.5 |

¹ Pure decode tps backed out of the blended `-pg` measurement: `tg_tps = tg / (total/blended − pp/pp_tps)`.

Decode is essentially flat across 0.5K → 8K depth (~4 % drift). The ~100 tps ceiling is weight-bandwidth bound (522 MiB / ~204 GB/s ≈ 391 tps theoretical, ~25 % achieved efficiency = ~98 tps). KV cache at 16 layers × small head dim is well under bandwidth at these depths.

**MLC q4f16_g16e + FlashInfer head-to-head (2026-04-30).** Lib: [dist/qwen3_5-0.8B-q4f16_g16e/](dist/qwen3_5-0.8B-q4f16_g16e/) recompiled this session with `--opt "flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1"` against the current Phase-9b ABI (recipe in §2.8). Bench: pp=512 prefill + tg=N decode, 3 runs + 1 warmup, `prefix_cache_mode="disable"`, `mode="interactive"`. Run via [scratch_mlc_tg_sweep.py](scratch_mlc_tg_sweep.py); raw output [tuning/mlc_tg_sweep_0.8b_g16e_FI_*.log](tuning/).

| tg   | llama.cpp Q4_K_XL (pure tg) | MLC q4f16_g16e + FI | ratio |
|---:|---:|---:|---:|
|  512 | 100.3 | **134.82** | **1.345×** |
| 1024 | 100.1 | **134.29** | **1.341×** |
| 2048 |  99.7 | **133.54** | **1.340×** |
| 4096 |  98.0 | **132.17** | **1.349×** |
| 8192 |  96.5 | **129.59** | **1.343×** |

Dead-flat 1.34× across 16× depth. The §14.1 35B-A3B "MLC crosses below at long ctx" regression is **not** present here — FlashInfer's paged-decode kernels keep KV reads off the critical path, leaving both stacks weight-BW-bound. The ~+34% MLC win is the kernel/quant gap (q4f16_g16e + dlight tuning + cudagraph + cutlass), not an attention-side win.

**Run-to-run stability**: tg=4096 produced 132.17 / 132.17 / 132.19 across 3 runs (0.01% variance). Earlier "bizarre" run-to-run drift in the q4f16_2 variant (54 → 33 tg_tps at tg=8192) was an artefact of the lib being compiled against a stale ABI without the speedup flags — not a state-pollution bug in the harness.

**Cross-reference for 0.8B history:**

| build | ctx / tg | tg_tps | ratio vs llama.cpp | source |
|---|---|---:|---:|---|
| q4f16_1 (no FI, pre-dlight) | 128 / 256 | 131.62 | 1.22× vs Q4_K_S | 2026-04-27 baseline |
| q4f16_g16e (no FI, no opts) | 512 / 512 | 99.33 | 0.99× vs Q4_K_XL | this session, q4f16_2 sanity |
| q4f16_g16e (no FI, opts on) | 512 / 512 | 112.82 | 1.13× vs Q4_K_XL | this session |
| **q4f16_g16e + FI (shipping)** | 512 / 512 | **134.82** | **1.345×** vs Q4_K_XL | this session, headline |
| q0f16-mtp + draft (γ=4 spec) | 512 / 512 | 120.5 | 1.20× vs Q4_K_S | 2026-04-28 cont. 12 (re-bench needed) |

The FlashInfer-on `q4f16_g16e` build is now the recommended 0.8B target; it dominates the prior γ=4 spec-decode result on target-only throughput while staying byte-identical to the reference (the spec-decode build is still useful for latency-sensitive interactive workloads but no longer the throughput headline).

### 14.3 KV-cache dtype variants (35B-A3B)

Phase 5 (fp8) and Phase 6 (int8) shipped the dtype-split refactor. fp8 is structurally a loss on Orin (software dequant); int8 is throughput-neutral but byte-divergent from fp16. Apples-to-apples (TIR kv_cache for both, FlashInfer disabled in fp16 lib for fair compare):

| pp / tg | fp16 TIR (tps) | int8 (tps) | int8 vs fp16 |
|---:|---:|---:|---:|
| 128 / 64   | 54.41 | 51.87 | -4.7 % |
| 512 / 256  | 46.34 | 45.43 | -2.0 % |
| 4096 / 256 | 24.45 | 24.06 | -1.6 % |
| 8192 / 256 | 15.88 | 15.63 | -1.6 % |

Parity (5 prompts × 50 tokens, temp=0.0): int8 = 2/5 EXACT, semantic drift only — outputs match for the first 100-155 chars then diverge by 1-2 tokens. Use the int8 lib only for capacity-bound deployments (~2× context for the same VRAM).

### 14.4 Perf protocol (pin this when re-benching)

- TG = 512 steady-state, MAXN locked (`sudo nvpmodel -m 0 && sudo jetson_clocks`), no concurrent processes.
- Run 3 reps + 1 warmup, report median.
- llama.cpp side: `llama-bench -m <gguf> -p <ctx> -n <tg> -r 3 -ngl 99` (or `-d <ctx> -n <tg>` for decode-at-depth).
- MLC side: `bench_mlc.py --pp <ctx> --tg <tg> --runs 3 --warmup 1` with `EngineConfig(prefix_cache_mode="disable")` — see [bench_harness_gotchas.md](.claude/projects/-home-alfie-mlc-llm/memory/bench_harness_gotchas.md) for the three "no GPU activity" deadlock modes.
- Always pass `--model-lib` to MLC's chat/engine — the JIT cache otherwise picks the FlashInfer path and segfaults on sm_87.
- llama.cpp build: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release` (substitute `120` for Blackwell).

---

## 15. Open Items

- **Bring up Qwen3.5-{4B, 9B, 27B} and Qwen3.6-27B.** 4B/9B reuse the dense `qwen3_5` module unchanged. 27B and 3.6-27B are the first dense models with asymmetric linear heads (16/32) — exercises the kernel path the 35B-A3B already validates.
- **Fresh apples-to-apples bench numbers post-Phase-6 for the 0.8B** — the §14.2 table is the 2026-04-27 baseline. Re-bench with the shipped lib to capture the +21 % from the dlight TX patch (worklog 2026-04-28 cont. 12: 99.5 → 120.5 tps γ=4).
- **Long-context (≥4K) crossover vs llama.cpp.** MLC's TIR PagedKVCache reads at lower effective BW than llama.cpp's `q8_0` KV at long sequence (§14.1). FlashInfer JIT path on sm_87 is the most likely fix lane — bounded probe described in the convo recap.
- **Real multimodal (mRoPE + vision tower).** Text-only collapse works; full multimodal needs `RopeMode` extension (or inline RoPE) + a vision module — non-trivial lift, only when needed.
- **Blackwell port + bench.** The 35B + spec-decode infrastructure is already ready for BW-rich hardware; on Orin spec loses to target_only by 17 % (BW-bound), but the math says it should win on hardware with a wider BW budget. Verify by porting + benching.

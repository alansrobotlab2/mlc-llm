# Qwen3-Next / Qwen3.5 / Qwen3.6 in MLC-LLM — Technical Reference

This is the living technical document for bringing up the Qwen3-Next family of hybrid (GatedDeltaNet + GQA) models in MLC-LLM. Living = update as decisions land or assumptions change. Date-stamped progress goes in [`worklog.md`](./worklog.md).

---

## 1. Goal & Scope

End goal: **Qwen3.6-35B-A3B** (35B-param hybrid MoE, ~3B activated) running end-to-end on MLC-LLM with greedy-decode parity vs. HuggingFace transformers.

De-risk path: validate **Qwen3.5-0.8B** first. It is the smallest member of the family, dense (no MoE), with symmetric linear-attention heads and standard RoPE. Anything that breaks here is in the GatedDeltaNet / hybrid stack, isolated from MoE complexity.

Strict ordering — nothing on 35B until 0.8B passes the parity bar in §10.

---

## 2. Family Map (as of 2026-04)

| Model | Released | Total / Active params | Hybrid layers | Linear heads (K/V) | MoE | mRoPE |
|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 2025-09 | 80B / 3B | 48 (`[L,L,L,F]×12`) | 16 / 32 | 512 experts, 10 active + 1 shared | yes |
| **Qwen3.5-0.8B** | 2026-03 | 0.8B / dense | 24 (`[L,L,L,F]×6`) | 16 / 16 | dense MLP | no |
| Qwen3.5-2B / 4B / 9B | 2026-03 | dense | similar | symmetric | dense MLP | no |
| **Qwen3.6-35B-A3B** | 2026-04 | 35B / ~3B | 40 (`[L,L,L,F]×10`) | 16 / 32 | 256 experts, 8 active + 1 shared | yes |

All share `model_type: qwen3_5` (or `qwen3_5_moe` for the MoE variants); the original Qwen3-Next still uses `model_type: qwen3_next`.

Canonical HF repos: `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.6-35B-A3B`.

---

## 3. Architecture Summary

### 3.1 Hybrid layer pattern

Every fourth layer is full softmax attention; the other three are GatedDeltaNet linear attention. Indexed by `full_attention_interval=4`:

- linear at indices 0, 1, 2 → full at index 3 → linear at 4, 5, 6 → full at 7 → …
- 0.8B: 24 layers → 18 linear, 6 full
- 35B-A3B: 40 layers → 30 linear, 10 full
- 80B-A3B: 48 layers → 36 linear, 12 full

### 3.2 Full-attention layer

Standard GQA, with two notable additions vs. plain Qwen3:
- **Output gate** (`attn_output_gate: true`): the Q projection emits `2 × num_heads × head_dim` floats; half are queries, half are gate values. The attention output is multiplied element-wise by `sigmoid(gate)` before `o_proj`.
- **Partial RoPE** (`partial_rotary_factor: 0.25`): RoPE is applied to only the first 25% of `head_dim`. With `head_dim=256`, that's 64 rotated dims, 192 untouched.
- Per-head Q and K RMSNorm (no bias).
- Head dim 256 across all variants.

### 3.3 GatedDeltaNet linear-attention layer

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

Note the existing MLC code splits the projection differently (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) — the loader at [python/mlc_llm/model/qwen35/qwen35_loader.py](python/mlc_llm/model/qwen35/qwen35_loader.py) maps these to the HF fused names. Verify this still works for the actual 0.8B checkpoint; HF layout may have consolidated to `in_proj_qkvz` / `in_proj_ba` in the released config.

### 3.4 RMSNorm quirk

Qwen3.5 uses `output = norm(x) · (1 + weight)` with weight initialized to 0. TVM `nn.RMSNorm` uses `output = norm(x) · weight`. The loader handles this by adding `1.0` to all standard RMSNorm weights at load time. The gated norm inside GatedDeltaNet (`linear_attn.norm`) does **not** get the `+1.0`.

Affected: `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, top-level `model.norm`.

### 3.5 MoE block (35B-A3B, 80B-A3B) — for Stage 5

- `num_experts=256`, `num_experts_per_tok=8`, plus 1 shared expert
- `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`
- `decoder_sparse_step=1` (every layer is MoE in the MoE variants)
- Routing: softmax over experts, top-k, normalize the chosen probs (`norm_topk_prob=true`)
- Output = `Σ p_i · expert_i(x) + shared_expert(x)`

### 3.6 mRoPE (35B-A3B, 80B-A3B)

`mrope_section: [11, 11, 10]` — head_dim is split into three rotation sub-bands rotated against three position axes (text + spatial). For text-only input, behaves as standard RoPE on a flattened position. Will need explicit handling in MLC because the existing `RopeMode.NORMAL` in [python/mlc_llm/nn/kv_cache.py](python/mlc_llm/nn/kv_cache.py) does not honor `mrope_section`.

### 3.7 MTP (Multi-Token Prediction) head

Present in the 3.5/3.6 checkpoints (`mtp_num_hidden_layers: 1`). Skipped in v1 — neither the existing qwen35 loader nor our validation harness needs it. Tracked here so we don't lose it.

---

## 4. State Layouts

### 4.1 Recurrent state (linear-attention layers only)

Per layer, allocated by `RNNState.create` in [python/mlc_llm/model/qwen35/qwen35_model.py](python/mlc_llm/model/qwen35/qwen35_model.py) (`create_rnn_state`):
- `state_id=0`: recurrent state `S`, shape `(num_value_heads, key_head_dim, value_head_dim)`, dtype **fp32**
- `state_id=1`: Conv1d ring buffer, shape `(kernel_size - 1, qkv_dim)` = `(3, qkv_dim)`, dtype = model dtype

For 0.8B: `state_id=0` is `(16, 128, 128)` fp32 = 1 MB per layer × 18 layers = 18 MB recurrent state per sequence.

### 4.2 Paged KV cache (full-attention layers only)

Standard `PagedKVCache.create_generic` with `attn_kind="mha"`:
- `num_hidden_layers = num_attention_layers` (only the full layers — 6 for 0.8B)
- `qk_head_dim = v_head_dim = 256`
- `rope_mode = RopeMode.NORMAL`, `rotary_dim = head_dim · partial_rotary_factor = 64`

`kHybrid` KVStateKind handling lives in the runtime (kept stateful per-layer-type); our model code consumes both objects and the cache layer dispatches by layer index.

---

## 5. Reference Implementations

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

## 6. Existing MLC Implementation Inventory

[python/mlc_llm/model/qwen35/](python/mlc_llm/model/qwen35/) was added in PR #3449 (Oct 2025) and is the foundation we build on.

### 6.1 [qwen35_model.py](python/mlc_llm/model/qwen35/qwen35_model.py) (868 lines)

| Lines | Component | Notes |
|---|---|---|
| 28–129 | `Qwen35Config` | Reads HF config, handles VLM nesting (`text_config`, `rope_parameters`), exposes `layer_types()` |
| 138–142 | `Qwen35Embedding` | Tied lm_head via `lm_head_forward` |
| 144–154 | `Qwen35MLP` | Standard `gate_up_proj` + `down_proj`, SiLU |
| 157–211 | `Qwen35Attention` | GQA + sigmoid output gate. `c_attn` is `2·h_q + 2·h_kv` heads (Q+gate+K+V fused) |
| 219–369 | `create_gated_delta_net_func` | TIR kernel, thread-per-V-column, fp32 state, supports prefill (loop over t) and decode |
| 377–598 | `Qwen35GatedDeltaNet` | Sub-projections, Conv1d, L2-norm, gate/beta computation, calls TIR kernel, RNNState read/write, gated output norm + `out_proj` |
| 605–645 | `Qwen35DecoderLayer` | Dispatches between full and linear by layer type |
| 758–780 | `create_rnn_state` | RNNState init: state_id 0 (S, fp32) + state_id 1 (conv buffer, model dtype) |
| 789–809 | `create_paged_kv_cache` | Only allocates for the full-attention layers |

### 6.2 [qwen35_loader.py](python/mlc_llm/model/qwen35/qwen35_loader.py) (177 lines)

- `hf = "model.language_model"` hard-coded prefix (line 51) — VLM layout. **This will likely break on the text-only 0.8B checkpoint** which uses plain `model.*`. First fix candidate.
- Fuses HF `q_proj/k_proj/v_proj` → MLC `c_attn`.
- Maps `in_proj_qkv`, `A_log`, `dt_bias` (no `.weight`), `conv1d.weight` → `conv1d_weight`.
- Fuses `gate_proj/up_proj` → `gate_up_proj`.
- Adds `+1.0` to the standard RMSNorm weights; leaves gated norm alone.
- Drops vision and MTP weights silently.

### 6.3 Registration

[python/mlc_llm/model/model.py:408-431](python/mlc_llm/model/model.py#L408-L431) registers `qwen3_5` and `qwen3_5_text`. No conv_template override; defaults to the model name lookup.

[python/mlc_llm/conversation_template/qwen3_5.py](python/mlc_llm/conversation_template/qwen3_5.py) registers the conversation template.

No model preset for `qwen3_5` in [python/mlc_llm/model/model_preset.py](python/mlc_llm/model/model_preset.py) yet — we'll add one for 0.8B.

---

## 7. Gap Table (current vs. target)

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

## 8. Known Pitfalls (from upstream issues)

1. **Recurrence dtype** — `mamba_ssm_dtype: float32` in the official config. Keep `A_log.exp()`, `softplus(a + dt_bias)`, and `S` in fp32 even when the model dtype is fp16/bf16. Drift in recurrence is silent and accumulates over long sequences.
2. **State aliasing** — never alias the read and write of recurrent state in the same op. SGLang #20791: a flashinfer `gated_delta_rule_decode_pretranspose` regression was traced to no-buffer scheduling aliasing the in/out state. RNNState's `get`/`set` already returns a fresh tensor — do not optimize that away.
3. **Speculative decode** — vLLM #39273: spec decode rollback on rejected tokens corrupts the recurrent state. Defer spec decode for v1.
4. **Marlin tile sizes** at high TP — vLLM #35924: `MIN_THREAD_N=64` broke `in_proj_ba` whose output is only `num_v_heads` (very narrow). Watch when adding TP > 1 quantized paths.
5. **HF parity reference** — must be transformers ≥ 4.57. Earlier `torch_chunk_gated_delta_rule` had a feature-dim mismatch (HF #40963). Pin in `validate.py`.
6. **Don't lose `attn_output_gate`** — easy to drop when subclassing a vanilla Qwen3 attention. The full layer's `c_attn` width is `(2·h_q + 2·h_kv)·d`, not `(h_q + 2·h_kv)·d`. Confirm post-Stage-2.
7. **FLA on Blackwell** — fla-org #607 is a backward-pass bug; doesn't affect inference. Listed for context only.

---

## 9. Implementation Plan (mirror of stages in [the plan](.claude/plans/ok-we-re-going-to-squishy-harbor.md))

- **Stage 0** — write this doc + `worklog.md`. ✅ in progress.
- **Stage 1** — `validate.py` PyTorch reference harness. Greedy generate + per-layer hidden-state dump on a fixed prompt. Cache to `reference_outputs.pt`.
- **Stage 2** — bring up qwen35 against the actual 0.8B checkpoint. Most likely first failure: HF prefix detection. Acceptance: model compiles and runs without exception.
- **Stage 3** — per-layer numerical parity. Tolerances per §10. Add intermediate hooks on GatedDeltaNet sub-steps when something diverges.
- **Stage 4** — end-to-end greedy parity (50 tokens × 5 prompts). Acceptance per §10.
- **Stage 5** — fork into `qwen3_5_moe` for the MoE variants (repo convention; mirrors `qwen3`/`qwen3_moe`).
- **Stage 6** — validate 35B-A3B end-to-end with the same harness.

---

## 10. Acceptance Bars

Per CLAUDE.md, fp16 throughout (SSM math fp32 internally):

| Stage | Bar |
|---|---|
| Embeddings | `atol ≤ 1e-4` (fp32 path), `atol ≤ 1e-3` (fp16) |
| Per-layer output, full attention | `rtol = 1e-3`, `atol = 1e-3` (fp16) |
| Per-layer output, linear attention | `rtol = 2e-3`, `atol = 2e-3` (fp16, recurrence accumulates) |
| Recurrent state `S` (fp32 in both) | `atol ≤ 1e-4` after Conv1d, `atol ≤ 1e-3` after first recurrence step |
| Greedy decode, 50 tokens | ≥ 48/50 identical per prompt, on 5 fixed prompts |

Failure mode: when a layer fails, dump per-tensor numpy on both sides, diff with `np.testing.assert_allclose`, log `np.abs(a-b).max()` and `argmax(abs(a-b))`. Validate GatedDeltaNet sub-steps in order: post-Conv1d → post-SiLU → post-L2-norm Q/K → β/g values → S after step 1 → output.

---

## 11. Out of Scope (v1)

- Performance work (kernels, fusions)
- Custom CUDA
- MTP head
- Speculative decoding (vLLM #39273 makes this risky on hybrid GDN)
- Quantization beyond `q0f16`/`q0bf16` until Stage 4 passes for 0.8B

---

## 12. Repo Conventions to Honor

- **MoE variants get their own module.** `qwen3` / `qwen3_moe`, `qwen2` / `qwen2_moe`, `mistral` / `mixtral`, `deepseek` / `deepseek_v2`. Stage 5 → `python/mlc_llm/model/qwen3_5_moe/`.
- Loader files alongside model files; quantization configured declaratively in `model.py` via `make_quantization_functions(...)` — no per-model quantization file unless the model needs `BlockScaleQuantize` or similar.
- Conversation templates in `python/mlc_llm/conversation_template/<name>.py`, registered via `ConvTemplateRegistry.register_conv_template`.
- Tirx (not tir) — the repo migrated to `tirx` namespace in PR #3462; the qwen35 kernel already uses it.

---

## 13. Open Items

- Confirm the actual `in_proj_*` naming on the released 0.8B checkpoint (`in_proj_qkv` + separate `z/a/b` as in MLC, or fused `in_proj_qkvz` + `in_proj_ba` as in vLLM). May require loader changes if HF has consolidated.
- Confirm `tie_word_embeddings` for 0.8B; the existing `Qwen35Embedding.lm_head_forward` assumes tied.
- Decide whether to add a `qwen3_5_0.8b` preset to `model_preset.py` after we know the exact config values.
- Decide mRoPE strategy for Stage 5: extend `RopeMode` enum and the C++ kv_cache path, or apply RoPE inline in Python before `attention_with_fused_qkv`.

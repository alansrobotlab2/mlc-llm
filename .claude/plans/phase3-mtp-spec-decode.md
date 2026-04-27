# Phase 3 — Self-speculative decoding via MTP head

**Date opened:** 2026-04-27
**Predecessor:** [phase2c-perf-after-profile.md](phase2c-perf-after-profile.md). T1 (asym q4 g=32) shipped at commits `bc61c785` + `6661e464` for +1.8% headline (1.295× llama.cpp tg128). Kernel-only path is at the AGX ALU ceiling.
**Target:** ≥1.5× headline on 0.8B via spec decode alone, multiplicative with the existing q4 stack. Stretch: 1.8×.
**Why MTP and not Lookahead/external draft:** Qwen3.5-0.8B was trained with `mtp_num_hidden_layers=1`. Recovering that head is the cheapest draft-free path, beats Lookahead's GDN-incompatibility, and beats external-draft's accept-rate ceiling.

## Stages

### Stage 1 — Recover MTP weights at convert time

The 0.8B `model.safetensors.index.json` includes `mtp.*` entries that are currently dropped by [qwen35_loader.py](../../python/mlc_llm/model/qwen35/qwen35_loader.py). Audit the loader's prefix-skip list, find where `mtp.*` is filtered. Extend the loader to keep one MTP layer's weights when `mtp_num_hidden_layers ≥ 1` in config.

Acceptance: `convert_weight` reports the MTP weight names alongside the rest. Param count goes up by one decoder layer's worth.

### Stage 2 — Add MTP module to qwen35_model.py

The MTP layer is structurally a single decoder block (RMSNorm + attention + MLP) plus a projection back to vocab logits. Reference: vLLM's `qwen3_next.py` `Qwen3NextMTPLayer` and HF transformers `Qwen3_5MTP`. Implement as a separate `nn.Module` that takes `(hidden_state, prev_token_embed)` and returns next-token logits.

Acceptance: model with MTP loads + compiles. q0f16 build size goes up modestly.

### Stage 3 — Wire MTP as draft model in MLC's EAGLE pipeline

MLC has an EAGLE-style draft+verify implementation in [cpp/serve/engine_actions/eagle_*.cc](../../cpp/serve/engine_actions/). It expects a separate "draft" model handle. For self-speculation, we register the target model with an MTP-decorated forward that emits both the verify-time logits and the next-step MTP draft.

Two implementation paths:
- **(a)** Compile two artifacts — target with MTP off (verify path), target with MTP on (draft path) — and load both. Cleaner separation, doubles params on disk and in VRAM.
- **(b)** Single artifact with both paths exposed as separate Relax functions. Reuses parameters, requires more compile-time work.

Path (a) is the faster way to get end-to-end working; (b) is the optimization. Start with (a), revisit if VRAM is tight.

Acceptance: `MLCEngine` runs with `speculative_mode="eagle"`, draft proposes γ tokens, target verifies, accept rate is non-zero.

#### 2026-04-27 path decision: (a), trimmed variant

Going with **path (a) trimmed**: the draft artifact is a *small standalone model* (`embed_tokens + pre-fc norms + fc + 1 MTP decoder layer + final norm`), not "target with MTP on." Reasoning, after reading the EAGLE C++ in detail:

- The EAGLE pipeline ([eagle_batch_draft.cc](../../cpp/serve/engine_actions/eagle_batch_draft.cc), [eagle_batch_verify.cc](../../cpp/serve/engine_actions/eagle_batch_verify.cc), [eagle_new_request_prefill.cc](../../cpp/serve/engine_actions/eagle_new_request_prefill.cc)) drives the draft via `TokenEmbed`, `FuseEmbedHidden`, `BatchPrefillToLastHidden`, `BatchDecodeToLastHidden`, and `GetLogits`-or-fallback-to-target-via-`CanGetLogits()=false`. **It has no concept of "calling MTP on the target artifact"** — the draft must be a separate artifact with EAGLE-named Relax functions.
- MLC ships the [eagle_model.py](../../python/mlc_llm/model/eagle/eagle_model.py) template — single-decoder-layer model with no `lm_head`, exposes `embed`, `fuse_embed_hidden_states`, `batch_prefill_to_last_hidden_states`, `batch_decode_to_last_hidden_states`, `create_paged_kv_cache`. We can fork that template directly for our MTP draft.
- VRAM cost at q0f16: `embed_tokens` ≈ 510 MB (vocab 248k × hidden 1024 × 2 B) + 1 MTP layer ≈ 30 MB. Acceptable on 16 GB AGX. q4 quant later → ~135 MB total. The embed dup is unavoidable without C++ surgery to redirect `models_[1]->TokenEmbed` to `models_[0]->TokenEmbed`; deferred as opt.
- The integrated target-with-MTP artifact built last session (`dist/qwen3_5-0.8B-q0f16-mtp/`) is *not* used in EAGLE mode — the MTP weights inside it are ignored. The draft artifact carries the only used MTP weights. For the EAGLE bench config we'll pair it with the *no-MTP* target artifact (`dist/qwen3_5-0.8B-q0f16/`) to avoid wasting 30 MB on a dead MTP layer.
- The **rnn_state question is moot** with this split: the draft model has no GDN, so its `BatchDecodeToLastHidden` only takes `(hidden_states, paged_kv_cache)`. The target's GDN state advances only on verify (target invocation), which is the correct semantics. No reconciliation needed in `mtp_decode` because `mtp_decode` won't be wired through EAGLE — the EAGLE-named draft functions are.

Concrete steps for Stage 3:
1. Create `python/mlc_llm/model/qwen35_mtp_draft/qwen35_mtp_draft_model.py` — fork [eagle_model.py](../../python/mlc_llm/model/eagle/eagle_model.py), swap LlamaAttention/LlamaFFN for `Qwen35Attention`/`Qwen35MLP`, add the pre-fc norms (Qwen3.5 MTP layout: `pre_fc_norm_embedding(input_embed)` and `pre_fc_norm_hidden(hidden_states)` then concat in `[h_norm, e_norm]` order then `fc`), final `norm` after the decoder layer.
2. Create `python/mlc_llm/model/qwen35_mtp_draft/qwen35_mtp_draft_loader.py` — pulls `embed_tokens.weight` (from `model.language_model.embed_tokens.weight`) and `mtp.*` (top-level). Reuses the c_attn / gate_up_proj fusion logic and the `+1.0` RMSNorm offset from [qwen35_loader.py](../../python/mlc_llm/model/qwen35/qwen35_loader.py).
3. Register `qwen3_5_mtp_draft` in [model.py](../../python/mlc_llm/model/model.py).
4. Convert + compile draft artifact: `dist/qwen3_5-0.8B-q0f16-mtp-draft/`.
5. `MLCEngine(model="dist/qwen3_5-0.8B-q0f16", additional_models=["dist/qwen3_5-0.8B-q0f16-mtp-draft"], speculative_mode="eagle", spec_draft_length=4)`. Validate accept rate > 0 on a smoke prompt.

### Stage 4 — Bench + parity on 0.8B

- Greedy verify is bit-exact equivalent to non-speculative greedy. **Parity must be 50/50 on every prompt that the non-spec build matches.** Any divergence is a bug, not noise.
- Sweep γ ∈ {2, 3, 4, 5, 6} to find optimal. Theory: optimal γ is where `T_draft·γ + T_verify(γ) ≈ γ · T_target_decode · α` (α = accept rate).
- Bench tg128, tg512 on Orin AGX. Compare against q4f16_g32_asym non-spec baseline (132.20 / 125.98 tps).

Acceptance: ≥1.5× over non-spec baseline = ≥198 tps tg128. Greedy parity 100%.

### Stage 5 — Transfer to 35B-A3B

**First: verify 35B checkpoint has MTP weights.** Quick check:
```bash
python -c "
import json
idx = json.load(open(SNAP+'/model.safetensors.index.json'))
print([k for k in idx['weight_map'] if 'mtp' in k.lower()][:10])
"
```
If empty, this stage is a research project (need to find or train an MTP head for the 35B). If non-empty, proceed.

Apply Stages 1-3 verbatim to 35B-A3B target. **Expect at least one bug** in the MoE-under-batched-verify path — speculated tokens go through expert routing in a batched forward that the non-spec path doesn't exercise. Likely surface area:
- [cpp/serve/engine_actions/eagle_batch_verify.cc](../../cpp/serve/engine_actions/eagle_batch_verify.cc) for batching
- [python/mlc_llm/model/qwen35/qwen35_moe_model.py](../../python/mlc_llm/model/qwen35/qwen35_moe_model.py) for routing under batch

Recalibrate γ on 35B — accept rate may differ.

Acceptance: ≥1.8× over non-spec 35B baseline. Greedy parity 100%.

## What transfers from 0.8B → 35B

| component | transfers? | notes |
|---|---|---|
| Loader change for MTP weights | yes | Same prefix list |
| MTP module code | mostly | Bigger dims but same architecture |
| EAGLE pipeline wiring | yes | Same C++ paths |
| γ value | **no** | Recalibrate |
| MoE batched-verify routing | **N/A** on 0.8B | New surface area on 35B |

## Stop conditions

- **0.8B doesn't reach 1.5×**: revisit γ, then revisit accept-rate (is the MTP head distribution actually close to target's? Could need a fine-tune pass).
- **35B doesn't have MTP weights**: pivot to external draft (Qwen3.5-0.8B as draft for 35B). Same EAGLE pipeline; loses self-speculative parity guarantees on accept rate but pipeline-wise it's identical.
- **MoE+spec produces wrong outputs on 35B**: bisect by disabling speculation per layer, find which expert config breaks under batched verify.

## Risks

- **MTP head was dropped because it didn't help training/inference upstream.** Possible but unlikely — Qwen team published the checkpoint with `mtp_num_hidden_layers=1` set, suggesting they intended it for inference use.
- **EAGLE pipeline assumptions may break with the GDN target.** The pipeline was designed for attention-only models; the recurrent state in GDN means the draft and verify paths must agree on where the state is at every step. Worth a careful read of [cpp/serve/model.cc](../../cpp/serve/model.cc) before building.
- **Single decoder layer may not be enough draft capacity.** If accept rate is <50%, headline gain is limited regardless of γ. Fallback: add MTP layers post-hoc via a short fine-tune (out of scope but documented as the escape valve).

## Open questions

1. Does the 35B checkpoint ship with MTP weights? (verify in Stage 5, no work before then)
2. Does MLC's EAGLE pipeline currently support hybrid GDN+attention models? (read [eagle_batch_draft.cc](../../cpp/serve/engine_actions/eagle_batch_draft.cc) before implementing)
3. Path (a) vs (b) for the dual artifacts — make the call early after reading the EAGLE C++.

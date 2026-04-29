# Phase 4 — Next perf avenues for Qwen3.6-35B-A3B on Orin AGX

**Date opened:** 2026-04-28  
**Last triaged:** 2026-04-28 (cont. 8) — **4B re-opened: 64% accept rate at γ=1 after concat-order fix.**
**Predecessor:** [phase2d-ft-hybrid-quant.md](phase2d-ft-hybrid-quant.md) (closed, −3.8%), B-ext spec decode (dead, 5.2% token agreement).  
**Current best:** v6 at **52.62 tps tg512 / 174 tps tg64** (1.789× llama.cpp Q4_K_S).  
**Lib:** `dist/qwen3_6-35B-A3B-q4f16_1/`

## Status (post-cont. 8 wiring fix)

| Phase | Status | Next |
|---|---|---|
| **4A** KV cache int8 | **DEFERRED** — ~1 wk TVM kernel work | Revisit only if 4B doesn't deliver and we want another +5–15% |
| **4B** MTP self-spec | **GO — wiring fix landed, B.3 next** | The original 0.8B port had `cat([h_norm, e_norm])` (reversed). vLLM uses `cat([embeds, hidden])`. After fixing all 3 Python sites, 35B γ=1 lands **64% accept rate**. State drift on rejected tokens still corrupts trajectory past ~20 tokens (verify uses non-history forward). **B.3:** port `forward_with_history` from qwen35_model.py to qwen3_5_moe_model.py and wire `batch_verify_to_last_hidden_states` to use it; recompile target; sweep γ ∈ {1,2,3,4} for clean accept-rate measurement and tps. |
| **4C** GDN chunk-scan | Open, lower EV than B.3 | Profile first if pursued |
| **4D** Meta-schedule | **DEAD** — regressed 1.78× | Closed |

**Phase 4 recommendation: B.3.** The Phase 3 / cont. 8 conclusion that "the MTP head is a training auxiliary" was based on contaminated probes (same `cat([h, e])` bug in [scripts/mtp_head_pytorch_check.py:122](../../scripts/mtp_head_pytorch_check.py#L122)). With the head proven usable, the path to a real speedup is: history mode → clean trajectories → measurable wall-clock gain. Original plan target was tg512 ≥ 79 tps (≥ 1.5× v6); at 64% step-1 accept and ~2 average accept_len, the math is in range.

## What landed in cont. 8

- **New module** `python/mlc_llm/model/qwen3_5_moe_mtp_draft/` — 35B variant of the MTP draft (parameterized hidden, MoE block instead of dense MLP).
- **EAGLE-compat methods on the MoE target** — `*_to_last_hidden_states` × 5 + `get_logits` + matching spec entries in [qwen3_5_moe_model.py](../../python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py). Required for spec decode on the 35B.
- **`_infer_kv_state_kind` fix** in [interface/compile.py](../../python/mlc_llm/interface/compile.py) — the new draft model_type was falling through to `hybrid` (segfault in `CreateKVCache`); now correctly maps to `kv_cache`.
- **The concat-order fix in 3 sites:** new 35B draft model, original 0.8B draft model (was buggy), and the integrated `Qwen35MTPHead.forward` in qwen35_model.py:966 (also buggy). All 3 changed from `cat([h_norm, e_norm])` → `cat([e_norm, h_norm])`.
- **Recompiled libs:** `dist/qwen3_6-35B-A3B-q4f16_1/lib.so` (with EAGLE methods; backup at `lib_v6_pre_eagle.so.bak`), `dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/lib.so` (new artifact, with corrected concat).
- **Stale and need recompile before re-benching Phase 3:** `dist/qwen3_5-0.8B-q0f16-mtp-draft/lib.so` and `dist/qwen3_5-0.8B-q0f16-mtp/lib.so`. Phase 3's "0% accept rate" was almost certainly the same bug.

## Summary of remaining avenues (original, kept for context)

Ranked by expected return / effort:

| Phase | Approach | Expected gain | Effort | Risk |
|---|---|---|---|---|
| **4A** | KV cache int8 quantization | +5–15% tg512 | ~~1–2 sessions~~ ~1 wk (revised) | Low |
| **4B** | MTP self-speculative (GDN rollback unblock) | +50–100% (1.5–2×) if accept ≥60% | 3–6 sessions | High |
| **4C** | GDN chunk-scan kernel (FLA-style) | +5–15% (scan-heavy steps) | 2–4 sessions | Medium |
| **4D** | Meta-schedule sweep on hot kernels | +2–5% | 1–2 sessions | Low — **CLOSED, regressed** |

~~Do phases in order.~~ **Revised order:** 4B is now the only viable lane. 4A deferred. 4D dead. 4C only if 4B lands.

---

## Phase 4A — KV cache int8 quantization (DEFERRED)

> **2026-04-28 update:** A.1 audit revealed there is no `kv_cache_dtype` plumbing in MLC. TVM's `PagedKVCache` takes a single dtype that propagates everywhere. Real cost is ~1 wk of TVM kernel work (modify ~6 TIR kernels for dequant-on-read + quant-on-write, scale storage layout in paged blocks). On hold pending 4B. Original plan kept below for reference if this is ever revived.

### Background

The 35B-A3B decode is memory-bandwidth-bound. Every decode step reads weight matrices (already at int4, ~18.6 GB) and KV cache (currently fp16). On Orin at 204 GB/s:

- Weight read per step ≈ 18.6 GB × (seq_budget / max_seq) — amortized over active sequences.
- KV cache read per step ≈ `2 × num_layers × kv_heads × head_dim × seqlen × 2 bytes` per token generated.

At tg512 (long context), the KV cache dominates. Halving it from fp16 → int8 reduces total memory traffic and could yield **+5–15% on tg512**, less on tg64.

MLC has a `kv_cache_dtype` option in `gen_config`. No recompile required if the runtime already supports int8 KV; if not, a small dlight schedule change may be needed.

### Stages

#### A.1 — Check existing support

```bash
grep -r "kv_cache_dtype\|int8_kv\|quantize_kv" python/mlc_llm/ cpp/serve/ | grep -v ".pyc"
```

Look for whether `kv_cache_dtype="int8"` is wired end-to-end (gen_config → PagedKVCache → attention kernel). If it is, skip to A.2. If not, scope the delta.

#### A.2 — Gen config with int8 KV

```bash
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<rev>/
python -m mlc_llm gen_config "$SNAP" \
  --quantization q4f16_1 \
  --conv-template qwen3_5 \
  --kv-cache-dtype int8 \
  -o dist/qwen3_6-35B-A3B-q4f16_1_kv8/
```

Copy existing `lib.so` (no recompile needed if kv dtype is a runtime param). If a recompile is needed, add `--device cuda -o ...kv8/lib.so`.

#### A.3 — Bench

Same bench harness as v6:
```bash
python bench.py --model dist/qwen3_6-35B-A3B-q4f16_1_kv8/ --tg 512 --tg 64 --pp 128
```

**Acceptance gate:** tg512 tps ≥ 55.0 (+4.5% over v6 52.62). If < v6, close 4A.

**Also validate:** greedy parity — 50/50 on the 5 canonical prompts at 50 tokens. int8 KV introduces quantization noise; if parity drops below 48/50, investigate (int4 KV might be too aggressive; check if int8 is stable).

---

## Phase 4B — MTP self-speculative decode (GO)

> **2026-04-28 cont. 8 update:** B.2 landed end-to-end on the 35B with a **64% step-1 accept rate at γ=1** after the concat-order fix described in [Status](#status-post-cont-8-wiring-fix) above. Initial 0% reading was a wiring bug carried over from the original 0.8B port and the contaminated PyTorch probe at scripts/mtp_head_pytorch_check.py. vLLM's qwen3_5_mtp.py confirms the correct convention is `cat([embeds, hidden])`, not `cat([hidden, embeds])`.
>
> **Smoke results (γ ∈ {1, 2, 3, 4}, 35B, completions API + ignore_eos)**
>
> | γ | accept_count (per step) | step1 acc | step2 acc | step3 acc | step4 acc | avg accept_len | decode tps |
> |---:|---|---:|---:|---:|---:|---:|---:|
> | 1 | [39, 25] | **64%** | — | — | — | 1.64 | 22.8 |
> | 2 | [29, 19, 15] | 66% | 79% | — | — | 2.17 | 22.1 |
> | 3 | [33, 15, 10, 7] | 45% | 67% | 70% | — | 1.97 | 17.2 |
> | 4 | [9, 3, 3, 3, 3] | 33% | 100% | 100% | 100% | 2.33 | 15.7 |
>
> Decode tps is currently *below* target_only baseline (45 tps tg32 warmup-dominated) because of state drift: `batch_verify_to_last_hidden_states` uses regular forward (no GDN history), so rejected tokens corrupt the GDN state. Trajectories degrade into repetition loops within ~20 tokens — which actually inflates late-trajectory accept rates (draft and target both predict the same repeating token) but breaks the text. **B.3 is what unlocks the wall-clock win.**

### B.3 — Port `forward_with_history` to qwen3_5_moe_model (next session)

The 0.8B already has this in `qwen35_model.py`:
- `Qwen35GatedDeltaNet.forward_with_history` ([qwen35_model.py:606](../../python/mlc_llm/model/qwen35/qwen35_model.py#L606)) — TIR kernel that scatters per-position GDN state into history slots so subsequent `PopN` can roll back to the accepted prefix bit-exactly.
- `Qwen35DecoderLayer.forward_with_history` ([line 890](../../python/mlc_llm/model/qwen35/qwen35_model.py#L890)) — wraps the GDN call.
- `Qwen35Model.forward_with_history` ([line 1015](../../python/mlc_llm/model/qwen35/qwen35_model.py#L1015)) — chains layers.
- `Qwen35LMHeadModel._forward_to_last_hidden_with_history` + `batch_verify_to_last_hidden_states` ([line 1101 / 1174](../../python/mlc_llm/model/qwen35/qwen35_model.py#L1101)) — the entry point.

**Steps:**
1. Mirror these into `qwen3_5_moe_model.py`. The MoE layer's structure differs only at the MLP — both call `Qwen35GatedDeltaNet` (already has `forward_with_history`), so the work is just plumbing through `Qwen35MoEDecoderLayer.forward_with_history` and `Qwen35MoEModel.forward_with_history`.
2. Update `Qwen35MoEForCausalLM.batch_verify_to_last_hidden_states` to call `_forward_to_last_hidden_with_history`.
3. Engine side: ensure `set_use_history_mode(True)` is called before `BeginForward` on the verify (already wired in `cpp/serve/engine_actions/eagle_batch_verify.cc` per Phase 3 cont. 4).
4. Recompile target lib. Backup current at `lib_v6_eagle_no_history.so.bak`.

**Acceptance:** spec output text matches target_only byte-identically on a long prompt (≥ 100 tokens). Then sweep γ ∈ {1, 2, 3, 4}: report decode tps and accept rate. Pick the best γ and commit to gen_config.

### B.4 — Final bench (after B.3)

```bash
for gamma in 1 2 3 4; do
    python bench.py --model dist/qwen3_6-35B-A3B-q4f16_1/ \
        --draft dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/ \
        --spec-draft-length $gamma \
        --tg 512 --tg 64
done
```

**Acceptance gate:** tg512 ≥ 79 tps (≥ 1.5× v6 baseline 52.62). At 64% step-1 accept × ~1.64 accept_len the math says this is reachable. If we miss the gate, fall back to whichever γ gives the best wall-clock (might still be net positive at γ=1 even at 22 tps if the warmup amortizes).

### B.5 — Recompile 0.8B drafts and re-validate Phase 3

The 0.8B `dist/qwen3_5-0.8B-q0f16-mtp-draft/lib.so` and `dist/qwen3_5-0.8B-q0f16-mtp/lib.so` are stale (built before the concat-order fix). Recompile and re-run `scripts/spec_smoke.py`. The "Phase 3 dead at 0% accept rate" verdict from the 2026-04-28 entry needs to be re-evaluated under the corrected wiring.

> Original plan (B.1, B.2 stages) kept below for reference.

### Background

Phase 3 (MTP spec decode, [phase3-mtp-spec-decode.md](phase3-mtp-spec-decode.md)) reached **Stage 3 on the 0.8B**: engine runs, EAGLE pipeline fires, accept rate is non-zero. **Stage 4 is blocked** by a hard constraint: TVM's `RNNState` does not support rollback after multi-token append.

What this means: when the EAGLE verify step rejects draft tokens, it must restore the GDN recurrent state to the pre-draft snapshot. TVM's current `RNNState` primitive is append-only — there is no `checkpoint()` / `restore()` op. The state after a multi-token draft cannot be walked back.

The 35B-A3B has the same constraint at larger scale. Solving this once on the 0.8B unblocks both models.

### The rollback problem, precisely

In EAGLE verify:
1. Draft model proposes γ tokens, running GDN state forward γ steps in draft artifact.
2. Target model verifies all γ tokens in a single batched forward. **Target GDN state advances by the number of accepted tokens.**
3. On partial accept (k < γ tokens accepted), the target GDN state must be at position `prefix + k`, not `prefix + γ`.
4. On full reject (k = 0), target GDN state must stay at `prefix`.

Currently there is no mechanism to achieve step 3/4 because TVM's GDN state is a live NDArray that was already advanced to `prefix + γ` during the batched verify forward.

### Option A: State checkpoint/restore in TVM TIR (recommended)

Add two new TIR primitives (or use existing NDArray copy semantics) to snapshot and restore the GDN recurrent state around the verify call:

1. Before verify: `snapshot = rnn_state.copy()` — shallow copy of the NDArray (GPU memcpy, ~O(batch × hidden²) floats for 35B that's small vs. KV cache).
2. After verify: if not all accepted, `rnn_state = snapshot[:, :k]`-semantics, or more precisely, re-run target forward up to `prefix + k` from the snapshot.

The cleanest implementation: expose `BatchDecodeToLastHiddenWithStateCheckpoint` and `RestoreStateFromCheckpoint` in [cpp/serve/model.cc](../../cpp/serve/model.cc), which are called from [eagle_batch_verify.cc](../../cpp/serve/engine_actions/eagle_batch_verify.cc) at the accept/reject boundary.

**Code touch points:**
- `cpp/serve/model.cc` — add `CheckpointRNNState()` / `RestoreRNNState()` methods
- `cpp/serve/engine_actions/eagle_batch_verify.cc` — call checkpoint before verify, restore on partial/full reject
- `python/mlc_llm/model/qwen3_next/qwen3_next_model.py` — expose the GDN state as a checkpointable object (may already be via `nn.RNNState` attrs)

**Acceptance for this sub-step:** EAGLE verify with partial-reject reproduces the same hidden state as non-speculative target on the same prefix. Validate by running non-spec and spec in lockstep on a long prompt, checking GDN state tensors match after each accepted token.

### Option B: Single-token draft (γ=1) — avoids rollback entirely

With γ=1, the verify step either accepts or rejects the single drafted token. On accept, the target GDN state was already advanced correctly by the verify forward. On reject, **the verify forward ran on the right token anyway** (the target corrects and also generates the right next token). No rollback needed.

γ=1 spec decode has a theoretical speedup ceiling of `1/(1−α)` where α is accept rate. If accept rate is 70%, that's 3.3× the throughput of one target decode per accept. But the wall-clock gain depends on draft model cost:

```
tps_spec ≈ tps_target × (α × γ + 1) / (t_draft × γ + t_verify) × t_target
```

With γ=1 and 0.8B draft at ~490 tps vs 35B target at ~53 tps, draft cost is ~10% of target cost. Even at γ=1 with 60% accept rate: rough speedup ~1.4×.

**Use γ=1 as the first working milestone.** It bypasses the rollback blocker, validates the pipeline end-to-end on the 35B, and gives a real (if sub-optimal) speedup.

### Option C: Reorder verify to avoid state advance on rejected tokens

Instead of running the full batched verify forward (which advances GDN state by γ), run verify token-by-token and stop at the first reject. State only advances as far as accepted tokens go. No rollback needed; no checkpoint needed.

Cost: lose the batching benefit — γ=4 becomes 4 serial target steps instead of 1 batched. At low accept rates this degrades to worse than non-spec. Only useful if γ is small (2-3) and accept rate is high.

Not recommended as the primary path, but useful to validate the accept rate before implementing checkpointing.

### Stages

#### B.1 — Validate 35B has MTP weights

```bash
python -c "
import json
SNAP='~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<rev>'
idx = json.load(open(f'{SNAP}/model.safetensors.index.json'))
print([k for k in idx['weight_map'] if 'mtp' in k.lower()][:10])
"
```

If empty → MTP weights absent → Phase 4B is a training project (out of scope). Close 4B, proceed to 4C.  
If non-empty → continue.

#### B.2 — Get γ=1 working on 35B (Option B path, no rollback)

Reuse the MTP draft artifact infrastructure from Phase 3 (stages 1–3 done for 0.8B). Adapt for 35B-A3B:

1. Audit `qwen35_mtp_draft_model.py` / `qwen35_mtp_draft_loader.py` for anything 0.8B-specific (hidden dim, num heads, etc). Parameterize from config.
2. Convert + compile 35B MTP draft artifact: `dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/`.
3. Smoke: `MLCEngine(model=35B, additional_models=[35B-mtp-draft], speculative_mode="eagle", spec_draft_length=1)`.
4. Measure accept rate and tps on canonical prompts.

**Acceptance:** accept rate > 30%, tps improvement over non-spec baseline measurable.

#### B.3 — Implement state checkpoint/restore (Option A, unblocks γ>1)

Implement `CheckpointRNNState` / `RestoreRNNState` in `cpp/serve/model.cc`. Wire into `eagle_batch_verify.cc`. Validate state correctness (lockstep comparison vs non-spec). Then sweep γ ∈ {2, 3, 4, 5, 6}.

**Acceptance:** greedy parity 50/50 on all canonical prompts at γ=4. tps ≥ 1.5× baseline (≥79 tps tg512).

#### B.4 — Optimal γ sweep + final bench

```bash
for gamma in 2 3 4 5 6; do
    python bench.py --model dist/qwen3_6-35B-A3B-q4f16_1/ \
        --draft dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/ \
        --spec-draft-length $gamma \
        --tg 512 --tg 64
done
```

Commit best γ to `gen_config.json`.

---

## Phase 4C — GDN chunk-scan kernel

### Background

The GDN (GatedDeltaNet) recurrent scan is the most compute-unique op in the model — it's a linear-attention chunk-scan that doesn't map cleanly to standard GEMM or element-wise primitives. Currently runs via generic dlight TIR.

The flash-linear-attention (FLA) library (`../flash-linear-attention/`) has hand-written CUDA kernels for exactly this scan, including Triton and raw CUDA variants. Reference: `../flash-linear-attention/fla/ops/gated_delta_rule/`.

Integrating a faster scan kernel could speed up the GDN-specific layers (roughly half the 40 layers in 35B-A3B are GDN-heavy). Estimated **+5–15%** on decode throughput.

**Do this after 4B (or in parallel with 4A):** if MTP spec decode is running, the scan kernel speedup applies to both draft and target forwards, compounding the benefit.

### Stages

#### C.1 — Profile: what fraction of decode time is GDN scan?

```bash
nsys profile --stats=true python bench_single_step.py --model dist/qwen3_6-35B-A3B-q4f16_1/ --steps 20
# Or use the existing bench_moe_kernel.py to isolate scan timing
```

If GDN scan < 10% of per-step time → expected gain from C is marginal, deprioritize.  
If GDN scan ≥ 15% → worth implementing.

#### C.2 — Benchmark FLA scan vs current TIR scan

Write a standalone benchmark comparing:
- Current TIR dlight kernel (extract from existing compiled lib)
- FLA Triton kernel via `fla.ops.gated_delta_rule.chunk_gated_delta_rule`

Shapes to bench: the 35B-A3B's GDN hidden dims (look up from config, likely hidden=2048, heads=32, head_dim=64 or similar). Sweep chunk sizes {32, 64, 128}.

**Gate:** FLA kernel must be ≥15% faster than TIR before investing in integration.

#### C.3 — Integration path (if gate passes)

Option A: Triton-based (portable but requires Triton on Orin — check sm_87 support).  
Option B: CUDA C++ extern, registered as TVM packed function (same pattern as FT, but without CUDA graph issue since it's a single scan op not ~200 GEMM dispatches).

Wrap in TIR shell to stay CUDA-graph-capturable. Wire into `qwen3_next_model.py`'s GDN forward.

---

## Phase 4D — Meta-schedule tuning (DEAD)

> **2026-04-28 update:** 500-trial evolutionary+xgb canary on `attn_o_proj` (the highest-headroom kernel) produced **59.3 µs vs dlight 33.3 µs = 1.78× regression**. Plateaued by trial 192. dlight's hand-written `dl.gpu.GEMV()` schedule for int4-dequant-fused low-batch GEMV is too specialized to beat with the generic `post-order-apply` space generator. The other three target kernels share the same schedule — same wall. Phase 4D closed. Original plan kept below for reference.

### Background

dlight's default schedules are well-tuned for common GEMM shapes but not specifically optimized for sm_87 Orin or for the asymmetric shapes in the 35B-A3B (e.g., lm_head at 248k×2048). A targeted meta-schedule sweep may find better tile/thread configs.

**Target kernels** (by cost, descending from bench_moe_kernel.py baseline):
1. `lm_head`: 1523 µs, 248064×2048 — largest single kernel
2. `gdn_in_proj_qkv`: 580 µs, 8192×2048
3. `attn_o_proj`: 237 µs, 2048×4096
4. `gdn_in_proj_z`: 251 µs, 4096×2048

### Stages

#### D.1 — Extract hot kernels as standalone bench targets

Use existing `bench_moe_kernel.py` infrastructure. Confirm timing is stable (std < 2%).

#### D.2 — Meta-schedule sweep

```python
# Per-kernel meta-schedule search, example for lm_head:
from tvm import meta_schedule as ms
ms.tune_tir(mod=extracted_lm_head_mod, target="cuda -arch=sm_87",
            work_dir="tuning/lm_head", max_trials_global=1000)
```

Keep default dlight as fallback; only apply tuned schedule if speedup ≥ 5% on the kernel.

#### D.3 — Integrate best schedules

Apply via `ApplyHistoryBest` in the compile pipeline. Rebuild lib, full bench.

**Acceptance:** tg512 ≥ 54.5 tps (+3.6% over v6). Lower bar than other phases because this is incremental.

---

## Stop conditions (whole-phase)

- **4A:** tg512 < v6 after int8 KV → close. Investigate whether int8 KV introduces numerical instability (check accept rate / parity).
- **4B:** 35B has no MTP weights → close. Or: γ=1 accept rate < 20% after B.2 → spec overhead exceeds benefit even at best γ, close.
- **4C:** FLA scan kernel ≥15% faster gate fails → close.
- **4D:** No single kernel ≥5% faster → close.

---

## Status updates (2026-04-28)

### 4A — REVISED (not as scoped)
**Investigation revealed:** MLC's TVM has no `kv_cache_dtype` plumbing. `PagedKVCache.create_generic` takes a single `dtype` that propagates to every attention kernel ([python/mlc_llm/nn/kv_cache.py:32](../../python/mlc_llm/nn/kv_cache.py#L32)). No `int8`/`fp8`/`e4m3` matches anywhere in TVM kv_cache or MLC `cpp/serve/`. Flashinfer underneath has separate `dtype_q/dtype_kv/dtype_o` but is called with all three equal — and on Orin (sm_87) flashinfer isn't used anyway, the TIR path is.

**Effort to actually deliver int8 KV:** thread `dtype_kv` through `PagedKVCache.create_generic`, modify ~6 TIR kernels (`_attention_prefill`, `_attention_decode`, `_kv_cache_transpose_append`, `_copy_single_page`, `_compact_kv_copy`, `_kv_cache_debug_get_kv`) for dequant-on-read + quant-on-write with per-token/per-head scales, decide scale storage layout in paged blocks. ~1 week of TVM kernel work, not "1–2 sessions." On hold pending 4B.

### 4B.1 — PASS
35B snapshot `995ad96` ships **19 MTP weight keys**: 1 MTP layer + EAGLE-style fc head (`mtp.fc.weight`, `mtp.pre_fc_norm_{embedding,hidden}.weight`, `mtp.norm.weight`, plus `mtp.layers.0.{self_attn, mlp.experts/shared_expert, *_layernorm}`). Architecture mirrors the 0.8B's MTP head except MoE replaces dense MLP. Phase 4B is a real path — not a training project.

### 4D — DEAD (canary regressed)
- **Bench reproduced:** dlight per-call timings on the four target kernels are stable to <0.2% std. Real numbers (vs the plan's stale baselines): lm_head 1717 µs, gdn_in_proj_qkv 63 µs, attn_o_proj 35 µs, gdn_in_proj_z 34 µs. End-to-end physical ceiling for 4D is **+9.5%** (Amdahl: these four are 31.7% of decode time at v6, all ~66–82% of peak BW).
- **Canary:** 500-trial evolutionary+xgb sweep on `attn_o_proj` (highest-headroom kernel). Result: **59.3 µs tuned vs 33.3 µs dlight = 1.78× regression**, plateaued by trial 192. Tooling (`scratch_ms_smoke.py`, `tune_kernel.py`) all green; the regression is real.
- **Why:** dlight's `dl.gpu.GEMV()` schedule is hand-specialized for low-batch GEMV-with-int4-dequant at exactly these shapes. Meta-schedule's general space generator explores matmul-style tilings and wmma paths (we saw a `TensorIntrin 'wmma_fill_16x16x16_f16' is not registered` error on DB reload, confirming wmma candidates were generated) — wmma with B=1 wastes compute because MMA needs M ≥ 16.
- **Not chasing the other three kernels:** they're all GEMV+int4-dequant hitting the same `dl.gpu.GEMV()` schedule and would hit the same wall.

### Recommended next phase
4B is now the only viable lane. Order: **B.1 ✓ → B.2 (port MTP draft loader from 0.8B to 35B, γ=1 first to bypass RNNState rollback) → measure accept rate → B.3 (state checkpoint/restore) only if γ=1 lands a real speedup.**

## Dependency map

```
4A (KV int8)    — standalone, start immediately
4D (meta-sched) — standalone, start immediately in parallel with anything
4B (MTP spec)   — requires B.1 to confirm MTP weights; independent from 4A/4C
4C (GDN scan)   — independent from 4A/4B; gains compound with 4B if both land
```

~~Recommended order for a single engineer~~ **(superseded)**: post-triage, the only live lane is **4B.2 → 4B.3 (Option A or B) → 4B.4** as the recommended order. 4A deferred, 4C dependent, 4D dead.

## Reference numbers

| Config | tg512 tps | tg64 tps | vs llama.cpp | Status |
|---|---:|---:|---:|---|
| v6 (current best) | 52.62 | ~174 | 1.789× | shipped |
| llama.cpp Q4_K_S | 29.40 | — | 1.000× | external bar |
| 4A target | ≥55.0 | ~180 | ~1.87× | deferred |
| 4B target (γ=1, α=60%) | ~74 | ~240 | ~2.5× | next milestone |
| 4B target (γ=4, α=60%) | ≥79 | ~250 | ~2.7× | post-rollback |
| 4D | — | — | — | dead (regressed 1.78×) |

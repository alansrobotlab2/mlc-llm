# Phase 4 — Next perf avenues for Qwen3.6-35B-A3B on Orin AGX

**Date opened:** 2026-04-28  
**Predecessor:** [phase2d-ft-hybrid-quant.md](phase2d-ft-hybrid-quant.md) (closed, −3.8%), B-ext spec decode (dead, 5.2% token agreement).  
**Current best:** v6 at **52.62 tps tg512 / 174 tps tg64** (1.789× llama.cpp Q4_K_S).  
**Lib:** `dist/qwen3_6-35B-A3B-q4f16_1/`

## Summary of remaining avenues

Ranked by expected return / effort:

| Phase | Approach | Expected gain | Effort | Risk |
|---|---|---|---|---|
| **4A** | KV cache int8 quantization | +5–15% tg512 | 1–2 sessions | Low |
| **4B** | MTP self-speculative (GDN rollback unblock) | +50–100% (1.5–2×) if accept ≥60% | 3–6 sessions | High |
| **4C** | GDN chunk-scan kernel (FLA-style) | +5–15% (scan-heavy steps) | 2–4 sessions | Medium |
| **4D** | Meta-schedule sweep on hot kernels | +2–5% | 1–2 sessions | Low |

**Do phases in order.** 4A is standalone; 4B requires GDN rollback work that also unblocks clean transfer of 4C. 4D can run in parallel with anything.

---

## Phase 4A — KV cache int8 quantization

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

## Phase 4B — MTP self-speculative decode (unblock GDN rollback)

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

## Phase 4D — Meta-schedule tuning (parallelizable)

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

## Dependency map

```
4A (KV int8)    — standalone, start immediately
4D (meta-sched) — standalone, start immediately in parallel with anything
4B (MTP spec)   — requires B.1 to confirm MTP weights; independent from 4A/4C
4C (GDN scan)   — independent from 4A/4B; gains compound with 4B if both land
```

Recommended order for a single engineer: **4A → 4B.1+4B.2 → 4B.3 → 4C (if profiling shows payoff) → 4D (background)**

## Reference numbers

| Config | tg512 tps | tg64 tps | vs llama.cpp |
|---|---:|---:|---:|
| v6 (current best) | 52.62 | ~174 | 1.789× |
| llama.cpp Q4_K_S | 29.40 | — | 1.000× |
| 4A target | ≥55.0 | ~180 | ~1.87× |
| 4B target (γ=4, α=60%) | ≥79 | ~250 | ~2.7× |
| 4A+4B combined | ≥82 | ~260 | ~2.8× |

# Phase 9 — Close the prefill gap on Qwen3.6-35B-A3B (Orin AGX)

**Date opened:** 2026-04-29
**Predecessor context:** Phases 4–7 closed out single-request decode work (ship at 1.79× llama.cpp on tg). Phase 8 ([phase8-hybrid-prefix-cache.md](phase8-hybrid-prefix-cache.md)) covers cross-request TTFT amortization. **Phase 9 is the orthogonal lever: make the prefill itself faster within a single request.**

**Gap (measured 2026-04-29 sweep):**

| metric | MLC q4f16_1 | llama.cpp Q4_K_S | MLC × |
|---|---:|---:|---:|
| **pp512** | **206.83 tps** | **583.35 tps** | **0.355× (2.82× slower)** |
| tg512 | 54.45 | 29.19 | 1.866× |
| tg8192 | (pending) | 28.48 | — |

**Goal:** halve the prefill gap. Target **≥ 350 tps pp512 (1.7×)** which is still behind llama.cpp but credible. Stretch: **≥ 450 tps (2.18×)** which would put MLC ahead on weighted prefill+decode for typical 512/512 chat patterns.

The headline framing this enables: "MLC ships 1.8× decode AND parity-class prefill on 35B-A3B". Today the deck is "1.8× decode, 3× slower prefill" — defensible but uncomfortable.

---

## Why this is the right phase 9

Three reasons the prefill gap warrants its own phase rather than sitting in the 0.8B perf backlog:

1. **TTFT is user-visible** in serving deployments. Even with Phase 8 prefix cache, novel prompts still pay full prefill. At 583 tps llama.cpp's 512-token TTFT is 0.88 s; MLC is 2.48 s. That's the difference between "instant" and "noticeable lag" in a chat UI.
2. **Prefill is GEMM-bound, decode is BW-bound**. They're different optimization regimes. The decode-phase tile/scheduling work (Phases 4A, 4B, dlight TX patch, Option D gdn_func register-cache) doesn't help prefill at all. **The 4× gap noted in the prior worklog (cont. 13 / 2026-04-27) was never directly attacked** — it's been a known TODO since the first 35B compile.
3. **Same bottleneck likely on Phase 8.** Phase 8's prefix cache amortizes shared prefill, but cache-miss requests still go through this slow path. Closing the prefill gap compounds Phase 8's gain.

---

## What's known about the slow path

From the prior worklog (cont. 13, 2026-04-27 night) and ad-hoc nsys runs across the perf phases:

**Suspect 1 — MoE expert dispatch on 35B.**
Active params/token = 3B but throughput is ⅓ of llama.cpp's. The MoE block on 35B has **256 experts × top-8 routing**. For prefill at seq_len=512 with batch=1, that's `512 × 8 = 4096 expert activations` per layer. Routing path:
- `topk_softmax_kernel` → 8-expert selection per token
- `cumsum / get_indices` → token-to-expert grouping
- `dequantize_group_gemm` → grouped GEMM across ragged token batches
- `scatter_output` → un-shuffle back

Suspect points:
- `dequantize_group_gemm` uses tile sizes optimized for batch=1 decode, not seq=512 ragged batching
- No CUDA-graph capture for the MoE block (PackedFunc dispatch boundaries inside the dispatcher)
- Scatter-gather is probably too granular per token

**Suspect 2 — full-attn KV path during prefill.**
10 of 40 layers are GQA full-attn. At seq_len=512 prefill, those produce 512×512 attention scores. Worklog cont. 13 noted: "the 0.8B regression from 130 → 63 tps as ctx grows 128 → 4096 is purely the 6 full-attn layers — GDN's recurrent state cost is depth-invariant."

For prefill the symptom is different — not depth-scaling, but raw attention compute. **Update 2026-04-30:** Phase 6 shipped a function_table.cc fix (RNN-state init hoisted out of the FlashInfer branch) that made FlashInfer compile-compatible with hybrid models — meaning the runtime crash from Phase 4 is no longer the gate. Whether FlashInfer is *currently linked* depends on the compile flags: the pre-Phase-8 lib was built with `flashinfer=1` and had 122 FlashInfer symbols; today's Phase 8 rebuild used `flashinfer=0` so the current `lib.so` is FlashInfer-off. Either way the runtime gap is small at seq=512 — `attn_paged` is 2.96% of total kernel time per Stage 9.1, so even a 4× FlashInfer speedup would buy <2% wall-clock. Lever C as originally framed is moot; FlashInfer toggling is a deployment knob, not a Phase 9 lever.

**Suspect 3 — dequant matmul tile selection.**
Today's nsys-traced profile shows `fused_dequantize_fused_NT_matmul9_cast4_kernel` at 1.75 ms/decode. At prefill seq=512 the same kernel runs at higher batch — tile selection matters more, and the existing dlight schedules were tuned for decode-batch-1.

---

## Stages and land criteria

### Stage 9.1 — Profile the prefill (1 session)

Before any tuning, get a clean nsys trace of a 512-token prefill on the 35B with `--cuda-graph-trace=node`. Bucket by kernel category:
- MoE block (`topk_*`, `dequantize_group_gemm*`, `scatter_output`, `moe_dequantize_gemv*`)
- Attention (`batch_prefill_paged_kv_kernel`, related)
- Linear projections (`fused_dequantize*_NT_matmul*`)
- GDN (`gdn_func_kernel`, `depthwise_conv1d*`, `rnn_state_*`)

Compare to the decode breakdown from Phase 7 follow-ups (today's session). The buckets that grew disproportionately are the targets.

**Deliverable:** [scratch_nsys_prefill_only.py](../../scratch_nsys_prefill_only.py) (analogous to [scratch_nsys_target_only.py](../../scratch_nsys_target_only.py) but configured to capture only the prefill, not subsequent decode). Bucketed table.

**Land criterion:** quantitative answer to "where is the 2.82× gap actually coming from"? Pick top 1-2 buckets that account for ≥ 70 % of the gap.

### Stage 9.2 — Address the dominant bottleneck (2-3 sessions)

The plan branches based on Stage 9.1's findings. Three pre-staged hypotheses with rough effort estimates:

**(A) Tile/schedule retuning for the prefill batch shape.** The runtime dispatcher picks a dlight schedule per prim_func by shape; meta_schedule is what finds better picks. Honest lever: profile what schedule is selected for `dequantize_group_gemm`, `fused_dequantize*_NT_matmul*` (linear projections), and the FlashInfer prefill kernel at the seq=512 batch shape, then run [tune_kernel.py](../../tune_kernel.py)'s meta_schedule recipe (already validated on `attn_o_proj` 2026-04-28 — see memory entry [meta_schedule API quirks](../../.claude/projects/-home-alfie-mlc-llm/memory/ms_tune_tir_quirks.md)) on the dominant kernels. Continuous with the [tuning/attn_o_proj_500/](../../tuning/attn_o_proj_500/) work; not greenfield. Effort: 1-2 sessions.

**(B) ~~MoE expert dispatch fusion~~ — RETIRED (2026-04-30).** Stage 9.1 measured `topk_router` bucket at **0.22 ms / 0.01% of prefill**. The topk_softmax → cumsum → get_indices → moe_sum stack is essentially free. There is no Stage 9.2 lever in dispatch; the cost is entirely in the GEMM. (Worklog 2026-04-30 entry has the data.)

**(C) ~~FlashInfer ABI fix on sm_87~~ — RETIRED (2026-04-30).** Lever C's premise was that FlashInfer was disabled on Orin from Phase 4. Phase 6 shipped the fix at [function_table.cc:245-267](../../cpp/serve/function_table.cc#L245-L267) (RNN-state init hoisted out of the FlashInfer branch), so FlashInfer is now compile-compatible with hybrid models. Whether it is actually linked depends on the most recent compile flags: the pre-Phase-8 lib was built with `flashinfer=1` and had 122 FlashInfer symbols + the `BatchPrefillWithPagedKV*` runtime path; today's Phase-8-rebuilt lib uses `flashinfer=0` and falls back to TIR. Either way the prefill bucket showed `attn_paged` at 2.96 % of total kernel time, so the FlashInfer toggle is sub-2 % wall regardless. Lever C as originally framed is moot — the *fix* landed in Phase 6 — and FlashInfer toggling is a deployment-flag choice, not a Phase 9 lever.

We pick whichever Stage 9.1 surfaces. **Stage 9.1 outcome (2026-04-30):** lever A only — 77.79% of prefill kernel time is in two MoE group-GEMM kernels (`dequantize_group_gemm[1]_kernel`, decode-tuned at CTA_COUNT=64 with an admitted-but-mismeasured prefill regression). Concrete sub-lever: shape-based dispatch via [low_batch_specialization.py](../../python/mlc_llm/compiler_pass/low_batch_specialization.py) generalized to MoE GEMM, emitting a prefill-tuned variant (CTA_COUNT≈1024, BLK_M∈{16,32}) selected at the call site.

**Stage 9.2 outcome (2026-04-30):** the cheap one-constant levers were tried; net **+2.9 % pp512** (202 → 207.9), gate not met. Decode unchanged across all variants (the `if num_tokens == 1: dequantize_gemv` static shortcut at [qwen3_5_moe_model.py:137-138](../../python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py#L137-L138) keeps b=1 decode entirely off `dequantize_group_gemm`). What was tried:
- **CTA_COUNT 64 → 1024** (recovers the original commit's ~3% prefill regression). Net +2.9% pp. Shipped.
- **BLK_M 8 → 16, BLK_K 32 → 64** (bigger tiles for better per-tile arithmetic intensity). Net **−0.5 %** vs baseline; reverted. Bigger tiles increase per-CTA shared-mem footprint (~10 KB → ~26 KB), dropping per-SM block count from ~9 → ~3 on Orin sm_87. Lower occupancy outweighs the per-tile efficiency gain.

The hand-tuned schedule at BLK_M=8/BLK_K=32 sits near a local optimum where SM occupancy is high. Both directions out lose. Stage 9.1's projection of "2× MoE GEMM speedup from CTA bump" was wrong because Orin's 16 SMs cap concurrent blocks at ~32-64 regardless of grid size; CTA_COUNT only mattered when fewer than ~64 tiles existed (i.e. small workloads, no longer hit since the static decode shortcut).

**Land criterion (lever A) — original:** pp512 ≥ 290 tps (1.4× over current 207). **Status: not met by single-constant tile tuning.** Reachability requires either:
- **Tensor-core MMA rewrite** of `dequantize_group_gemm`'s inner block (sm_87 supports `m16n8k16` fp16 MMA). Major change; potentially 3-5× kernel speedup if executed cleanly.
- **Meta-schedule** on an unscheduled variant. Existing scaffolding at [tune_kernel.py](../../tune_kernel.py) + [tuning/attn_o_proj_500/](../../tuning/attn_o_proj_500/) is single-shape dense GEMV — needs extension for the persistent-loop group GEMM with int4 weights. ~50 min of search per shape (~1000 trials × ~3 s).

Either is a fresh 1-2 session investment.

### Stage 9.3 — Stack additional levers as needed (1-2 sessions)

If 9.2 lands one lever and the next bucket has visible room, repeat. Common stack:
- Dequant + matmul fusion (similar to MoE: fuse the dequant into the GEMM rather than two-pass)
- Cudagraph capture extension to cover the MoE dispatch path (compounds with any of 9.2's wins)
- Tile retuning for the linear projections in prefill batch shape

**Land criterion:** pp512 ≥ 350 tps (1.7× current). Gate for shipping.

### Stage 9.4 — Regression bench + final number (0.5 session)

Re-run the full sweep harness ([scratch_mlc_tg_sweep.py](../../scratch_mlc_tg_sweep.py) + llama-bench equivalent). Confirm decode tps unchanged within ±2 %.

**Land criterion:** sign-off table with tg + pp at all 5 context depths, both stacks, no decode regression.

---

## Risk register

1. ~~**FlashInfer-on-Orin ABI work (lever C) might not be tractable.**~~ Lever C retired 2026-04-30 (FlashInfer is already running on the headline lib post-Phase 6).
2. **Tile retuning regressions decode.** dlight schedules dispatch by shape, so retuning the prefill batch shape *should* be orthogonal to the decode-batch-1 shape. But meta_schedule's tuned schedule replaces the dlight default for that prim_func across all shapes unless we keep the dlight schedule registered alongside. Mitigation: a "prefill-shape-only" entry point if needed (analogous to Phase 4B's per-token-small-batch verify dispatch).
3. **The bottleneck might be in attention (FlashInfer kernel) rather than MoE/linear.** Stage 9.1 might surface `BatchPrefillWithPagedKVCacheRun` as the dominant cost. In that case the levers are different — FlashInfer kernel tuning is harder to attack from MLC's side (vendored kernel; tuning happens via the `BeginForward` plan-cache or upstream FlashInfer changes). If this is what 9.1 finds, the lever set shrinks.
4. **Memory pressure during prefill at long context.** seq² fp16 attention activation at seq=8192 = `8192² × 2 = 128 MiB / layer × 10 = 1.28 GiB`. Orin still fits in headroom, but the existing `prefill_chunk_size=512` already segments this into chunks. At the headline pp=512 there's no chunking overhead; only at longer contexts. Worth confirming the chunk size isn't tuned suboptimally for the seq=512 case.

---

## Out of scope

- Decode tps (saturated, separate phase line).
- Cross-request prefix cache (Phase 8).
- Quantization changes (q3 / mxfp4 weights — Phases 5/6/7 closed; if revisited, separate phase).
- 0.8B prefill (similar gap but different deployment priority).

## Land-criteria summary

| Gate | Metric | Bar | Status (2026-04-30) |
|---|---|---|---|
| 1 (profile) | Stage 9.1 bucketed prefill trace | Top 1-2 buckets identified, ≥ 70 % of gap explained | ✅ **MET** — 77.8 % in two `dequantize_group_gemm[1]_kernel` |
| 2 (single lever) | pp512 tps after 9.2 | ≥ 290 (1.4× current 207) | ❌ **NOT MET** — got 207.9 tps (+2.9 %) from CTA_COUNT bump |
| 3 (stacked) | pp512 tps after 9.3 | ≥ 350 (1.7× current) | n/a — 9.3 not attempted |
| 4 (no regression) | tg512 tps | within ±2 % of 54.45 | n/a — 54.45 baseline was on FlashInfer-on lib; current lib is at 44.99 (FlashInfer-off, separate question). v1 v.s. baseline is +0.1 %, no regression |
| 5 (stretch) | pp512 tps if 9.3 over-delivers | ≥ 450 (2.18×, parity-class) | n/a |

**Phase 9 verdict (2026-04-30):** Stage 9.1 closed cleanly. Stage 9.2 shipped a +2.9 % pp partial; gate-2 not met by single-constant tile tuning. The path to gate-2 is tensor-core MMA or meta-schedule — both 1-2 session investments. Phase 9 is **closed as partial** until that work is scoped.

**Phase 9b is the open follow-up:** [phase9b-tir-mma-group-gemm.md](phase9b-tir-mma-group-gemm.md). Stage 1 spike (2026-04-30) validated that dlight's `MatmulFP16Tensorization` does emit wmma intrinsics on Orin sm_87, but production integration needs a kernel rewrite (the persistent-loop wrapper + int4-unpack indices are incompatible with dlight's matmul recognizer). Stage 2 deferred to a fresh session committed to option α (drop persistent loop, grid launch).

If Phase 9 reopens, start with the Phase 9b plan's "Session resume notes" header. The worklog 2026-04-30 Stage 9.2 + Stage 9b Stage 1 entries have the full bench tables and the kernel-tuning ceiling analysis.

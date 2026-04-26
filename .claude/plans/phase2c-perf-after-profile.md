# Phase 2C — Profile-Driven Perf Plan (post-baseline, target 2.5× llama.cpp)

**Date:** 2026-04-25 EOD (post-handoff session)
**Baseline:** Qwen3.5-0.8B q4f16_g16e on Orin AGX MAXN, jetson_clocks locked. tg128=109.51 tps, tg512=115.13 tps. Equivalent llama.cpp Q4_K_S = 102 tps tg128. Currently 1.07× llama.cpp.
**Target:** 250+ tps tg128 = ~2.5× llama.cpp. Implies wall-clock budget per token ≤ 4.0 ms (currently 9.17 ms). Roofline for 449 MB params at 204 GB/s LPDDR5 = 2.2 ms; theoretical ceiling ~370 tps.

## What the profile actually says

Captured: `profile_qwen35_decode.nsys-rep` (5.5 MB), `profile_kern_sum.csv`. Run via [profile_decode.py](../../profile_decode.py): 1 warmup pass + 1 profiled pass × 64-token decode each, total 126 forward passes.

**Per-fwd kernel time = 7.95 ms** (sum of all GPU kernel time / 126 fwd). Per-token wall-clock 9.17 ms (109 tps) → ~1.2 ms engine glue overhead.

| Category | % kernel time | ms/fwd | Effective BW | Interpretation |
|---|---|---|---|---|
| **MLP dequant+matmul** | 25.3 | 2.01 | 65 GB/s (32% peak) | gate_up + down × 24 layers, 132 MB read |
| **lm_head** | 21.6 | 1.72 | 75 GB/s (37% peak) | 1024×248320, 127 MB read |
| **GDN dequant+matmul** | 18.0 | 1.43 | 80 GB/s (39% peak) | in_proj_qkv/z/a/b/out × 18 layers |
| **GDN state I/O** | 14.6 | 1.16 | — | rnn_state_get/set: paged ↔ contiguous |
| **GDN recurrence kernel** | **4.3** | **0.34** | — | gdn_func — **NOT the bottleneck** |
| Norms | 3.5 | 0.28 | — | RMSNorm + add+norm fusions |
| Attn dequant+matmul | 3.2 | 0.25 | — | c_attn + o_proj × 6 full-attn layers |
| GDN conv1d | 3.0 | 0.24 | — | causal depthwise + state update |
| Full-attn paged-KV kernel | 2.4 | 0.19 | — | TIR fallback (FlashInfer disabled) |
| Other (sampling, embed, cast) | 4.1 | 0.33 | — | |

**The handoff's premise was wrong.** It identified `gdn_func_kernel` as the top-EV target (12.5% block occupancy). But `gdn_func_kernel` is **only 4.3% of kernel time**. Even a 4× rewrite saves 3.2% wall-clock. Pursuing it as Tier-1 was a misread.

**The real top-EV is dequant+matmul kernel quality.** Combined, they are 68.1% of kernel time, all running at 30–40% of LPDDR5 peak. Closing that to 80% peak across the board would save ~2.5 ms = +35% headline tps alone.

## Path to 250 tps — honest ladder

| Stage | Action | Effect | Est. ms saved | tps after | × llama.cpp |
|---|---|---|---|---|---|
| 0 | baseline q4f16_g16e | — | — | 109 | 1.07 |
| 1 | **Try q4f16_ft at g=16** (patch FT preprocessor's `assert group_size in [None,64,128]`) | Replaces all dequant+matmul with CUTLASS fused. Was +7% on broken model — closer to peak BW per kernel. | 0.7–1.0 | 130 | 1.27 |
| 2 | **GDN state I/O fold** (collapse rnn_state_get → gdn_func → rnn_state_set into a single kernel that reads/writes paged state directly) | Eliminate 1.16 ms of state-shuffling. | 0.7–0.9 | 145 | 1.42 |
| 3 | **lm_head TIR rewrite** (large matmul with unusual aspect ratio 1024×248320 — 37% peak BW today; needs different tile/split-K) | Bring lm_head to 80% peak. | 0.8–1.0 | 165 | 1.62 |
| 4 | **Engine glue audit** (cut per-step overhead from 1.2 ms → 0.4 ms; KV page mgmt, NVTX, FFI, request bookkeeping) | | 0.6–0.8 | 185 | 1.81 |
| 5 | **FlashInfer FFI fix** for full-attn (currently TIR fallback on 6/24 layers) | | 0.1–0.2 | 190 | 1.86 |
| 6 | **Speculative decoding** (draft model + parallel verify; 0.8B is small enough to self-distill or use a 0.4B draft. Realistic 1.5–2× with reasonable accept rate.) | The lever for 2.5×. | 1.5–2.5× tax-on | 280–380 | 2.7–3.7 |

**Without spec decode (T1–T5): ~1.85× llama.cpp = 190 tps.** That's the realistic kernel-only ceiling at the engineering scope of "profile-driven, validated each step". To reach 2.5×, **T6 speculative decoding is the load-bearing piece.**

## Approach for T1 (FT at g=16)

`python/mlc_llm/quantization/ft_quantization.py` has an assertion blocking `group_size` ∉ {None, 64, 128}. Loosen it to allow 16, 32. Then check the CUTLASS preprocessor (`cutlass_preprocessors.cc:254`) — last session this asserted on row/col-byte alignment (now fixed for `out%64 or in%64`). Need to verify g=16 doesn't break the preprocessor's group-byte arithmetic.

**Acceptance:** q4f16_ft_g16 build compiles, smoke test shows correct output ("Paris..."), tg128 ≥ 120 tps. Re-run greedy parity (≥48/50 on 0.8B regression set).

## Approach for T2 (GDN state I/O fold)

Current pattern in [qwen35_model.py:Qwen35GatedDeltaNet.forward](../../python/mlc_llm/model/qwen35/qwen35_model.py):
```
state = paged_rnn_state.get(layer_id)            # rnn_state_get_0/1 kernels
out, new_state = gdn_func(q, k, v, gate, beta, state)  # gdn_func_kernel
paged_rnn_state.set(layer_id, new_state)         # rnn_state_set_0/1 kernels
```
Three kernels, each transferring the 64-KB-per-head state through HBM. Fold into a single kernel that reads from paged buffer at start, runs the recurrence in registers/shared mem, writes back at end. This is bigger than `gdn_func_kernel`'s scope — it lives in the RNN-state PrimFunc layer of TVM/Relax.

**Implementation paths:**
- (a) Inline rewrite: hand-write a TIR func that takes paged buffer + offset, does the get-compute-set in one launch.
- (b) Modify the gdn_func TIR to operate directly on paged state buffer (with the page index lookup baked in).

(a) is cleaner; (b) is more invasive but reuses existing scheduler.

**Acceptance:** total tg128 ≥ 145 tps. Greedy parity preserved.

## Approach for T3 (lm_head)

The lm_head is one matmul: `[1024] @ [1024, 248320]^T` with int4 weights. Why is it 37% peak?
- Aspect ratio: K=1024 (small), N=248320 (huge). Standard tiling may underutilize SMs.
- Split-N parallelism across SMs: at 16 SMs × 16 threads/warp × N=248320 → 968 N tiles per SM. But each lm_head call is short (1.7 ms) and the kernel may be launching too few blocks for full occupancy.

**Investigate first** with `ncu --kernel-name 'fused_dequantize_NT_matmul7'` to get exact occupancy, memory throughput, stall reasons. Then decide between TIR schedule rewrite vs CUTLASS-FT vs custom kernel.

**Acceptance:** lm_head kernel single-instance ≤ 0.9 ms (currently 1.69 ms). Headline tg128 ≥ 165 tps.

## Approach for T4 (engine glue)

Profile inserts NVTX ranges around `BatchDecode` / `BatchSampleTokensWithProbAfterTopP`. The 1.2 ms gap is invisible to current NVTX. Options:
- Add finer NVTX ranges in `cpp/serve/engine_actions/batch_decode.cc`, `action_commons.cc:443`
- Look for unnecessary StreamSyncs (gpu_sampler.cc:691 is one — see if it can be deferred)
- KV cache page allocation per token (currently page_size=16; a token usually doesn't span pages)

## Approach for T6 (spec decode)

MLC has the EAGLE / draft-batch pipeline (`eagle_batch_draft.cc` exists). Two options:
- **Self-speculative**: use the same model as both draft and target with shorter context window; only viable if there's a way to run it in different states efficiently.
- **External draft**: use a smaller pretrained model. Qwen2.5-0.5B might be a fit; needs to share tokenizer with Qwen3.5-0.8B (it does).

**Acceptance:** end-to-end tg128 ≥ 250 tps with greedy=true and ≥48/50 parity preserved (since speculative decoding with greedy verify is by construction equivalent to non-speculative greedy).

## Stop conditions / re-bench cadence

- After EACH tier item lands: re-run `bench_mlc.py --pp 128 --tg 512 --runs 3` AND `validate.py --greedy-parity` 0.8B regression. Append to worklog.md.
- If a tier item delivers <50% of estimated savings: profile again, the model has shifted.
- If two consecutive tiers each <5% headline: revisit the plan.

## Risks / open questions

- **q4f16_ft at g=16 might still fail** (CUTLASS group-byte alignment on int4). If so, T1 may regress; fall back to writing a custom TIR fused-dequant-matmul.
- **GDN state fold may break correctness** if the paging machinery relies on the get/set call pattern for sequence boundary handling. Validate via 0.8B parity early.
- **Spec decode quality**: external 0.5B draft model will have lower accept rate than self-distilled. Reasonable accept rate (~70%) gives ~1.5× speedup; lower (~40%) may give only 1.1×.
- **engine glue may be lock-bound** rather than compute-bound — in which case the audit needs to find specific syncs, not just shave kernel time.

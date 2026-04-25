# Plan: Phase 2 — Decode Perf for Qwen3.6-35B-A3B on MLC-LLM

## Context

Correctness phase (Stages 0–6 in [worklog.md](../../worklog.md)) shipped 2026-04-25. Both the dense Qwen3.5-0.8B path and the MoE Qwen3.6-35B-A3B path produce token-level parity with HF transformers fp16 reference (50/50 on 0.8B; 4/5 prompts at 50/50 + 1/5 at 33/50 with a single fp16-noise-floor flip on 35B).

The opening Phase-2 benchmark — MLC q4f16_1 vs `unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S` through llama.cpp on Blackwell sm_120, single-batch — produced this:

| Stack | Decode tg128 (tps) | Prefill pp512 (tps) |
|---|---|---|
| llama.cpp (Q4_K_S, fp16 act) | **207.72 ± 1.84** | 7322.27 ± 95.81 |
| MLC (q4f16_1) | 85.45 (median) | 40,819 (**measurement broken**) |

**llama.cpp wins decode 2.43×** at the metric that drives chat UX. The MLC prefill number is unreliable — `engine._generate()` yields the first delta before GPU prefill is actually finished, so the 40k tps figure reflects asynchrony in the streaming path, not throughput. Re-instrument before trusting any prefill comparison.

This plan exists to close the decode gap and beat llama.cpp by ≥50%.

## Goal

**Single load-bearing target:** decode `tg128` tokens/sec for 35B-A3B, q4-class, single batch, Blackwell sm_120, no concurrency.

- **Current:** 85.5 tps
- **Bar to beat (parity):** 207.7 tps (llama.cpp Q4_K_S)
- **Goal (≥50% over llama.cpp):** **≥311 tps**, i.e. **3.6× the current MLC number**

A 3.6× improvement is large but not implausible — the decode path is memory-bound and the suspected dominant bottleneck (no fused dequant+matmul) directly doubles HBM traffic per weight. Closing that one item alone could halve the gap.

## Workflow: 0.8B dev loop, 35B acceptance gate

The previous phase used 35B as the primary target. Phase 2 inverts that:

- **Dev loop = Qwen3.5-0.8B q4f16_1.** Each compile is ~30 sec, each bench cycle is ~1–2 min, fits on the 5090 (cuda:1) with Blackwell free for parallel work. ~2× faster iteration than 35B.
- **Acceptance gate = Qwen3.6-35B-A3B q4f16_1.** Every Tier-1/2 fix that passes the 0.8B loop must be re-benched on 35B before being declared a win. The 0.8B numbers are directional, not load-bearing.

**Why 0.8B is valid for most items**: fused q4 dequant+matmul, FlashInfer sm_120 cache, CUDA-graph capture, and GDN TIR kernel tuning all hit the same code paths in both models. **Why 35B is still mandatory**: MoE-specific paths (`MixtralExperts` dispatch, routing softmax-topk, expert dispatch overhead) only fire on 35B and can't be measured on 0.8B at all.

**Cardinal risk**: 0.8B's per-step kernel-launch overhead is over-represented (less GPU work per step, same launch count), so a CUDA-graph-capture win measured there will look bigger than it is on 35B. Halve any 0.8B graph-capture gain in your head before getting excited.

## Setup (one-time, ~15 min)

Before Tier-1 work begins, the dev loop needs its own llama.cpp baseline:

1. **Compile MLC q4f16_1 build of 0.8B:**
   ```
   SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<rev>/
   .venv/bin/python -m mlc_llm convert_weight "$SNAP" --quantization q4f16_1 -o dist/qwen3_5-0.8B-q4f16_1
   .venv/bin/python -m mlc_llm gen_config   "$SNAP" --quantization q4f16_1 --conv-template qwen3_5 -o dist/qwen3_5-0.8B-q4f16_1
   .venv/bin/python -m mlc_llm compile dist/qwen3_5-0.8B-q4f16_1 --device cuda -o dist/qwen3_5-0.8B-q4f16_1/lib.so
   ```
2. **Download `unsloth/Qwen3.5-0.8B-GGUF` Q4_K_S** (~500 MB) → `dist/gguf/`.
3. **Bench both stacks at matched conditions** — same `pp` and `tg` values used for the 35B run (pp=512, tg=128, runs=5). Save logs as `bench_llamacpp_0.8B.log` and `bench_mlc_q4_0.8B.log`. These are the regression baseline for every subsequent fix.

## Phase 2A — Measurement (Tier 1, gating)

**Nothing in Tier 2 is allowed to start until Tier 1 lands.** Optimizing without a clean profile is guessing.

### A.1 — Fix the bench harness's prefill measurement

The current `bench_mlc.py` derives prefill tps from `time-to-first-delta` from `engine._generate()`. This races GPU completion. Replace with one of:

- **Option (a):** non-streaming `engine.completions.create(...)` returns a `usage` object; check whether MLC populates `usage.prefill_tokens_per_sec` / `usage.decode_tokens_per_sec` or equivalent.
- **Option (b):** read `engine.metrics()` (if exposed) before and after the request to derive timings from cumulative counters.
- **Option (c):** instrument inside the engine — wrap a single `prefill` + `decode` call in `cudaEventRecord` and read elapsed-time directly. Most effort, most accurate.

Pick (a) first; if the protocol doesn't expose timings, fall back to (b), then (c). Acceptance: harness reports a `prefill_tps` whose 1/value is greater than a single decode step time. A prefill that "completes" faster than one decode step means measurement is still wrong.

### A.2 — Profile a decode step on 0.8B

Run `nsys profile` (or `ncu` for kernel-level detail) on a single decode step against `dist/qwen3_5-0.8B-q4f16_1/`. Goal: produce a flame graph or rank-ordered kernel time breakdown so we know which slice of the decode step is the fattest pipe.

Categories to bin time into:

- Routed-expert GEMM / `MixtralExperts` path (35B only — 0.8B is dense)
- Dense MLP `gate_up_proj` / `down_proj` GEMM (0.8B's analog)
- Dequant ops on the q4 weights (suspect #1)
- Full-attention QKV + paged KV access (FlashInfer fallback path — suspect #2)
- GatedDeltaNet TIR kernel (suspect #3)
- Per-step launch overhead / runtime / scheduler

Output: a one-page profile summary in `worklog.md` ranking the top-5 cumulative-time kernels and their share of decode wall-clock. **This profile dictates the order of Tier 2.** Whatever ranks #1 gets fixed first, regardless of where it sits on the suspected list below.

### A.3 — Same profile on 35B for cross-check

Once 0.8B profiling is clean, run the same profile on 35B's decode. If the top kernels rank the same way, the 0.8B dev loop is validated. If 35B's #1 is `MixtralExperts` (which doesn't exist in 0.8B), the workflow needs adjustment — probably a quick dedicated pass on routing/dispatch before resuming the 0.8B loop for the rest.

## Phase 2B — Known hot-spots (Tier 2, ranked by suspected impact)

This ranking is informed by the bench result + hardware reasoning, but the profile from A.2 supersedes it. If A.2 says #3 is actually #1, do #3 first.

### B.1 — Fused dequant + matmul for q4f16_1

**Suspected #1 bottleneck.** llama.cpp's `mul_mat_q` kernels read Q4_K weights once and fuse dequant into the matmul tile. MLC's q4f16_1 path runs `dequant_to_fp16` → `matmul_fp16xfp16`, doubling HBM bandwidth per weight read. On a memory-bound MoE/MLP decode this directly explains a ~2× gap.

Two paths to test:

- **B.1a — Switch to a quantization scheme that already has fused kernels in MLC.** Candidates: `q4f16_ft` (fast type), AWQ, GPTQ. Each has a different dequant layout; verify that one of them is fused at runtime in MLC's current TVM code. Rebuild + bench before writing any kernel.
- **B.1b — Write a fused TIR kernel for q4f16_1.** The q4f16_1 layout is well-documented in `python/mlc_llm/quantization/group_quantization.py`. The TIR primitive `T.dequantize_fused_matmul(...)` may already exist; if not, write it as a tiled load that decompresses Q4 into shared memory and runs the matmul in one kernel.

Acceptance: 0.8B decode tps improves by ≥50% (memory-bound roofline). Re-bench on 35B.

### B.2 — FlashInfer sm_120 prebuilt

The compile log warns it can't open `~/.cache/flashinfer/0.6.9/120f/cached_ops/...` because the prebuilt binary cache lacks sm_120. MLC falls back to a TIR paged KV cache, slower on every full-attention layer (10/40 on 35B, 6/24 on 0.8B).

Build FlashInfer from source against sm_120, populate the cache, recompile MLC's lib.so. Verify by checking the compile log no longer warns and decode tps moves.

Acceptance: 0.8B decode tps improves by ≥10–20% (only some of the layers benefit, and these are not the dominant cost for the linear-heavy hybrid). Re-bench on 35B.

### B.3 — CUDA graph capture for decode

Verify whether MLC enables CUDA graph capture on the decode-step graph. If not, enable it. Each decode step is a fixed-shape graph (single token) which is the canonical case where graph capture pays off.

Check: search MLC source for `cudaGraph*` API calls or `tvm.target.cuda(graph=True)` / similar. Modify the engine config or compile flag to enable capture.

Acceptance: 0.8B decode tps improves measurably (probably 1.2–1.5×, larger relative gain than on 35B because 0.8B has less GPU work per step and so more relative launch overhead). Re-bench on 35B; halve the 0.8B gain mentally before getting excited.

## Phase 2C — Speculative (Tier 3)

Only touch these if Tier 2 doesn't close the gap.

### C.1 — GatedDeltaNet TIR kernel re-tune

The existing kernel at `python/mlc_llm/model/qwen35/qwen35_model.py::create_gated_delta_net_func` was written for correctness, not perf, and never tuned for sm_120. Affects 18/24 layers on 0.8B and 30/40 on 35B. Profile-driven tuning targets: split-K, tensor-core matmul fragments where math allows, vectorized loads, shared-memory tiling.

This is the highest-effort item — budget multiple days. Don't start unless A.2's profile actually identifies GDN as a top-3 kernel.

### C.2 — Engine knob audit

Walk `mlc-chat-config.json` and engine config: `prefill_chunk_size`, `tensor_parallel_shards`, `kv_cache_page_size`, `attention_sink_size`. Check none are leaving obvious perf on the table.

Cheap to do, low expected return.

### C.3 — Spec decode

MLC supports speculative decoding (`batch_verify` is in `mlc_llm compile`'s temp-buffer estimate). Run a speculator (e.g. a smaller dense model) against the 35B-A3B target. Effective tps for repetitive workloads can be 1.5–2× over greedy on memory-bound paths.

Adds complexity (need a calibrated draft model) but doesn't require touching the perf-critical kernels. Would be worth investigating if Tier 2 stalls.

## Stop conditions

Phase 2 ends when one of these is true:

- **Win:** 35B decode tps ≥ 311 (≥50% over llama.cpp Q4_K_S baseline).
- **Diminishing returns:** three consecutive Tier-2/3 items each yielded <10% improvement on 0.8B and the headline 35B number is plateaued. Document the final state and call it.
- **Architectural ceiling:** profile shows >80% of decode time is in HBM read of weights (already memory-bound roofline) and the q4 quant ratio is fixed. Document the roofline and stop.

## Re-bench cadence

After any landing fix, run **the full 35B bench** (same conditions as the baseline above) and append the result to `worklog.md`. Don't stack multiple un-validated optimizations — each one needs a clean A/B against the prior baseline so we know which item delivered the gain.

## Risks and known unknowns

- **The 0.8B dev loop assumption.** All Tier-2 items are claimed to "hit the same code paths." Some don't — `MixtralExperts` dispatch and softmax-topk routing only exist on 35B. If profiling reveals expert dispatch is itself a top-3 kernel on 35B, those fixes need the slower iteration loop directly.
- **Quant scheme switch (B.1a) may regress correctness.** A scheme like AWQ requires a calibration pass; q4f16_1 doesn't. If we switch quant, re-run the 0.8B greedy parity (≥48/50 bar) and the 35B greedy parity before claiming a perf win — a 3× decode tps improvement that breaks token parity is worthless.
- **FlashInfer sm_120 build can be tricky.** May need patches if upstream doesn't carry sm_120 yet. Budget more if (B.2) appears to take >2 hours.
- **The user said "drag race" expecting MLC to win 50%.** This plan now targets that explicitly. If we land at "35% faster than llama.cpp," that's still a substantial win but it doesn't meet the original ask. Be honest in the worklog about what shipped vs. what was promised.

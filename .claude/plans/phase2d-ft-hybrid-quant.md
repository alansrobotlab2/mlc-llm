# Phase 2D — Hybrid q4f16_ft Quantization for 35B-A3B

**Date:** 2026-04-28
**Baseline:** v6 lib at 52.62 tps tg64 / 163.42 pp_tps (1.789× llama.cpp Q4_K_S). gdn_func register-cached, MoE tile-tuning exhausted.
**Target:** 55-56 tps tg64 (+5-7%) by routing dense q4 GEMVs through CUTLASS FpAIntB instead of dlight, while keeping MoE experts on dlight q4f16_1.
**Risk:** medium-to-high. The CUTLASS group_size constraint (g=64 minimum) may degrade quantization quality on the 35B in ways that look fluent but are factually wrong (the 0.8B q4f16_ft test exhibited "capital of the world" failure mode).

## Why this might work

The v6 nsys profile shows ~5.5 ms/tok in dense q4 GEMVs:

| kernel | shape | per-call µs | calls/tok | per-tok ms |
|---|---|---:|---:|---:|
| gdn_in_proj_qkv | K=2048 N=8192 | 38.9 | 36 | **1.40** |
| lm_head | K=2048 N=248064 | 1712 | 1 | **1.71** |
| attn_o_proj / GDN out_proj | K=4096 N=2048 | 14.3 | 47 | **0.67** |
| gdn_in_proj_z + silu/multiply | K=2048 N=4096 | 18.9 | 35 | **0.66** |
| shared_expert_gate_up | K=2048 N=1024 | ~12 | 47 | ~0.56 |
| shared_expert_down | K=512 N=2048 | ~10 | 47 | ~0.47 |
| **total dense** | | | | **~5.5** |

Per the prior 0.8B q4f16_ft test (worklog 2026-04-25), CUTLASS FpAIntB was **+7.1%** faster than dlight on a comparable q4 dense matmul (110.55 → 118.56 tps). At the 35B-A3B's 5.5 ms/tok dense budget, a 10-20% kernel-level FT speedup translates to **0.5–1.1 ms/tok = +1.5–3 tps** on the headline.

The 0.8B FT investigation also confirmed:
- ✅ CUTLASS FpAIntB lib `libfpA_intB_gemm.so` is built with **sm_87 cubins** (verified via `cuobjdump --list-elf`).
- ✅ The mlc_llm `fastertransformer.gemm_fp16_int_bias` extern is wired up at [op/ft_gemm.py:73](../../python/mlc_llm/op/ft_gemm.py#L73).
- ✅ Bias + activation (silu/gelu/relu) fusion is supported in the FT extern signature — could fuse `silu(gate_up)` into one kernel for shared expert.
- ✅ The CUTLASS FpAIntB has an `m ≤ 4` fast path (line 81 of fpA_intB_gemm_impl.h) that handles the B=1 decode case explicitly.

## Why this might NOT work (gates to clear before commit)

### Gate 1 — kernel-level speedup at sm_87 (microbench)

CUTLASS is tuned primarily for desktop Ampere (A100, 4090). Orin's small SM count (16) and lower clocks may eat into the win. The 0.8B's +7.1% test was at smaller shapes; at 35B's larger K and N the dispatch may pick different tile configs. **Pre-flight: microbench FT vs current dlight at all 6 production dense shapes; if FT is not ≥10% faster on at least one major contributor, skip.**

### Gate 2 — quantization quality at g=64

CUTLASS's `FineGrainedScaleZeroIterator` hard-bakes `group_size / 64` into row-offset arithmetic at [fine_grained_scale_zero_iterator.h:159](../../3rdparty/tvm/3rdparty/cutlass_fpA_intB_gemm/cutlass_extensions/include/cutlass_extensions/transform/threadblock/fine_grained_scale_zero_iterator.h#L159). Means our options are **g=64 or g=K** (per-channel) — no g=32 without 1-2 days of CUTLASS surgery.

The 0.8B test at g=64 (`q4f16_ft_g64`) produced "100 / 101 / C" garbage. The 35B has 40× more params and more redundancy; it *may* survive coarser quantization, but this is speculative. **Pre-flight: after compiling, run `scripts/coherence_smoke.py --max-tokens 80` on 3-5 prompts and compare to the v6 q4f16_1 baseline. If outputs are factually wrong, the run is dead.**

### Gate 3 — MoE expert quantization compatibility

`FTQuantize.visit_module` at [ft_quantization.py:108](../../python/mlc_llm/quantization/ft_quantization.py#L108) only mutates `nn.Linear` and `nn.Embedding`. `MixtralExperts` (the 256 routed experts) is NOT touched — it would stay at fp16, ballooning the model to ~77 GB and OOM'ing Orin. **Mitigation: extend FTQuantize to fall back to GroupQuantize for MixtralExperts (the same pattern already used for size-misaligned Linears at lines 156-158).** Code change is small (10-20 lines).

## Plan

### T1 — Microbench pre-flight (1 hour, go/no-go gate)

1. Add an FT-path harness alongside the existing dlight harness in [bench_moe_kernel.py](../../bench_moe_kernel.py). Call `faster_transformer_dequantize_gemm` directly via the extern, build a one-call Relax module, lower with `BLASDispatch(target)` enabled (path is at [compiler_pass/pipeline.py:126](../../python/mlc_llm/compiler_pass/pipeline.py#L126) — currently disabled for q4 because `cublas_gemm` is hard-disabled at [interface/compiler_flags.py:103-113](../../python/mlc_llm/interface/compiler_flags.py#L103-L113), but the FT path may go through a different dispatch).
2. Time at all 6 production shapes:
   - GDN in_proj_qkv: K=2048, N=8192, B=1
   - GDN in_proj_z + silu/multiply: K=2048, N=4096, B=1
   - GDN out_proj / attn o_proj: K=4096, N=2048, B=1
   - shared_expert gate_up: K=2048, N=1024, B=1
   - shared_expert down: K=512, N=2048, B=1
   - lm_head: K=2048, N=248064, B=1
3. Use g=64 for FT (production constraint).
4. Compare median µs against v6 baseline ([baseline_kernels_v6.json](../../baseline_kernels_v6.json)).

**Decision rule:**
- If FT is ≥10% faster on ≥3 shapes including ≥1 big (lm_head or in_proj_qkv) → proceed to T2.
- Otherwise → close phase, stick with v6.

### T2 — Hybrid quantization wiring (2-3 hours)

1. Edit [ft_quantization.py:108](../../python/mlc_llm/quantization/ft_quantization.py#L108) `visit_module` to add a branch:
   ```python
   if isinstance(node, MixtralExperts):
       group_quantize = self.config.fallback_group_quantize()
       self.quant_map.map_func[weight_name] = group_quantize.quantize_weight
       return GroupQuantizeMixtralExperts.from_mixtral_experts(node, group_quantize)
   ```
   Pattern is the same as the existing GroupQuantize fallback at lines 156-158. Need to set `quant_map.param_map` for `gate_up_proj.weight` → `q_weight + q_scale` etc.
2. Verify `q4f16_ft_g64` config at [quantization.py:194](../../python/mlc_llm/quantization/quantization.py#L194) has the correct group_size on the GroupQuantize fallback (currently hardcoded g=32 in `FTQuantize.fallback_group_quantize`, see [ft_quantization.py:42](../../python/mlc_llm/quantization/ft_quantization.py#L42)).
3. Run `convert_weight` for 35B-A3B with `--quantization q4f16_ft_g64`. Should produce a model dir comparable in size to v6's q4f16_1 (~18 GB params).

### T3 — Compile, bench, coherence (1 hour)

1. `mlc_llm gen_config` + `mlc_llm compile` with the standard Orin opt string: `flashinfer=0;cublas_gemm=1;cudagraph=1;cutlass=1`.
2. Run [bench_mlc.py](../../bench_mlc.py) with `--baseline baseline_35B_q4f16_1_v6_gdn.json`. **Acceptance: tg_tps ≥ 54.5 (i.e. ≥+3.5% over v6).**
3. Run [scripts/coherence_smoke.py](../../scripts/coherence_smoke.py). **Acceptance: greedy outputs are factually plausible on the 3-prompt suite. Compare against the v6 baseline outputs.**

### T4 — Profile & decide on go-further (30 min)

If T3 lands a meaningful gain:
1. nsys profile, look at remaining ms/tok distribution.
2. Check whether dense kernels are now at near-peak BW or whether more headroom exists (e.g., epilogue-fused FT with silu+multiply for shared expert).
3. Update worklog with v7 baseline. Decide if there's a v8.

## Stop conditions

- **T1 gate fails** (FT not ≥10% faster on ≥3 shapes): close phase, mark in worklog as "FT path unprofitable on Orin sm_87 at production shapes." Pivot to B-ext spec decode (Phase 3).
- **T3 coherence fails** (factually-wrong outputs at q4f16_ft_g64): close phase, mark "g=64 quant too coarse for GDN-MoE 35B." Pivot to B-ext spec decode.
- **T3 perf gate fails** (<+3.5% e2e, even though kernels are faster): means engine glue / cuda graph already hides most of the gain. Close, pivot.

## Quick-context for the next session

**Where we left off** (commit `c5b76e70`, 2026-04-28):
- v6 lib at 52.62 tps tg64, 1.789× llama.cpp.
- gdn_func register-cached state landed (commit `94f2d18b`, +1.25 tps).
- MoE matmul tile-tuning ruled out empirically (sweep at v6, no config beats current by >0.2%).
- Tokenizers verified identical between Qwen3.5-0.8B and Qwen3.6-35B-A3B (B-ext spec decode is viable from a vocab perspective).

**First moves in the next session**:
1. Read this plan + the worklog's last 3 entries (start at line 7 for the 2026-04-28 cont. entries).
2. Decide between this phase (Phase 2D, FT hybrid) and Phase 3 (B-ext spec decode). Both are viable; this phase is shorter (1 day if it works, half a day to know if it doesn't).
3. If proceeding with 2D: start with T1 microbench. The pre-flight is cheap and decisive.

**Compile recipe (Orin)** — pinned because flashinfer=on segfaults the C++ engine:
```bash
.venv/bin/python -m mlc_llm compile dist/qwen3_6-35B-A3B-q4f16_ft_g64 --device cuda \
  --opt "flashinfer=0;cublas_gemm=1;cudagraph=1;cutlass=1" \
  -o dist/qwen3_6-35B-A3B-q4f16_ft_g64/lib.so
```

**E2E bench recipe**:
```bash
source .envrc.local && .venv/bin/python bench_mlc.py \
    --model-dir dist/qwen3_6-35B-A3B-q4f16_ft_g64 --device cuda:0 \
    --pp 128 --tg 64 --runs 3 --warmup 1 \
    --baseline baseline_35B_q4f16_1_v6_gdn.json \
    --json-out baseline_35B_q4f16_ft_v7.json
```

**Coherence smoke**:
```bash
source .envrc.local && .venv/bin/python scripts/coherence_smoke.py \
    --model-dir dist/qwen3_6-35B-A3B-q4f16_ft_g64 --max-tokens 80
```

## Why not start with Phase 3 (B-ext spec decode) instead

Phase 3 (external-draft spec decode using Qwen3.5-0.8B as draft for the 35B-A3B target) is the only path to +20 tps. But:
- Multi-day, ~3-5 sessions.
- Two independent failure modes: (a) MLC engine doesn't have the right spec-decode plumbing for two-model setups (existing path is EAGLE-shaped), and (b) accept rate may be mediocre because Qwen3.5 and Qwen3.6 are different generations.
- A token-level agreement preflight (run both models greedy on the same prompts, count match-rate) takes a couple hours and is a cheap go/no-go before the engine work.

Phase 2D is shorter and the failure modes are clearer (kernel-microbench gate + coherence gate). Either it lands ~+3 tps or it doesn't, in <1 day. Worth doing first.

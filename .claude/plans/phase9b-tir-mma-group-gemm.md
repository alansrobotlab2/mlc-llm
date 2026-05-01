# Phase 9b — TIR tensor-core MMA for `dequantize_group_gemm` (Orin sm_87)

> ## 🔖 Session resume notes (read first)
>
> **Status as of 2026-04-30 (cont. session 4):** **SHIPPED.** Stage 2d landed. pp512 = **523.67 tps** (2.52× over Stage 9.2 baseline 207.95), tg512 = 44.86 (parity, -0.04%). Past every gate including the 450-tps "parity to llama.cpp" stretch. Lib at [dist/qwen3_6-35B-A3B-q4f16_1/lib_phase9b_v2.so](../../dist/qwen3_6-35B-A3B-q4f16_1/lib_phase9b_v2.so) (env-var opt-in: `MLC_MOE_GEMM_V2=1` at compile time). Default lib.so still on v1 baseline pending decision to flip default. Worklog session entry has the full bench table + debug notes.
>
> **Status as of 2026-04-30 (cont. session 3):** Stages 1, 2a, 2c LANDED standalone. **Hand-tensorize works** with parity. Production lib **unchanged**; integration is Stage 2d (next session).
>
> **Stage 2 standalone bench (Orin AGX, sm_87):**
>
> | stage | shape | t/iter | TFLOPS | parity | notes |
> |---|---|---:|---:|---|---|
> | 1 V1.5 (dlight Matmul + manual dequant sched) | M=4096 N=1024 K=2048 | 1.013 ms | 17.0 | ✓ | upper bound; uses 4 MB W_fp16 global temp |
> | 2a hand-tensorize (single expert, no persistent loop) | same | 3.7 ms | 4.6 | ✓ | per-tile dequant in shared, no global temp |
> | 2c hand-tensorize (multi-expert, lookup-table dispatch) | 1920×1024×2048, 4 experts | 1.8 ms | 4.6 | ✓ | precomputed `tile_to_e/m/n` tables; 4×-9× over scalar production |
>
> Gap from 2a to 1 V1.5 (4.6 vs 17 TFLOPS): missing **software pipeline** + **double buffering**. Both require the W cooperative-fetch block to be a simple BufferStore — our dequant breaks that constraint. Closing the gap is Stage 2b: split W into a per-tile dequant producer + a simple-copy cooperative fetch. Deferred — 4.6 TFLOPS already 4×-9× over production scalar.
>
> **Stage 2d (next session) — production integration:**
>
> The path that worked in scratch (`scratch_phase9b_grouped.py`) uses **precomputed dispatch tables** instead of an in-kernel persistent loop. Steps:
>
> 1. **New helper op** `compute_moe_dispatch_tables(indptr) -> (tile_to_e, tile_to_m, tile_to_n)` in [moe_matmul.py](../../python/mlc_llm/op/moe_matmul.py). Single small TIR prim_func, BW-bound. Output sized to `upper_bound = (ceildiv(B, BLK_M) + Ne) * tiles_per_n`. Produces 3 int32 arrays.
> 2. **New prim_func** `dequantize_group_gemm_v2(x, w, scale, tile_to_e, tile_to_m, tile_to_n) -> O`. Adapted from `scratch_phase9b_grouped.py:make_prim_func`. Drops the persistent loop. Each block: read `(e, m_offset, n_offset) = (tile_to_e[bx], tile_to_m[bx], tile_to_n[bx])`, run hand-tensorized matmul. Schedule from `scratch_phase9b_grouped.py:schedule_hand_tensorize`.
> 3. **Wire at [qwen3_5_moe_model.py:137](../../python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py#L137)** — replace `dequantize_group_gemm(...)` call with `compute_moe_dispatch_tables(indptr)` then `dequantize_group_gemm_v2(...)`. Both calls happen for prefill only — the b=1 decode shortcut (`if num_tokens == 1: dequantize_gemv`) stays unchanged, decode is unaffected.
> 4. **BLK_M change**: production has BLK_M=8 (mma m=16 doesn't fit). Bump to BLK_M=16 in v2.
> 5. **Bounds checking**: scratch proto assumes per-expert tokens divisible by BLK_M; production must `if_then_else` the X read and the O write at row boundaries. Easy to add (production already has the pattern at [moe_matmul.py:721-725](../../python/mlc_llm/op/moe_matmul.py#L721-L725) and [:743-744](../../python/mlc_llm/op/moe_matmul.py#L743-L744)).
> 6. **Recompile** [dist/qwen3_6-35B-A3B-q4f16_1/lib.so](../../dist/qwen3_6-35B-A3B-q4f16_1/lib.so) (~20-30 min); smoke test 10-token completion; bench pp512.
>
> **Estimated wall:** Stage 2d = 1 session (hand-port + glue + compile + smoke). Bench is 0.5 session.
>
> **Working artifacts:**
> - [scratch_phase9b_handtensorize.py](../../scratch_phase9b_handtensorize.py) — Stage 2a single-expert prototype (4.6 TFLOPS).
> - [scratch_phase9b_grouped.py](../../scratch_phase9b_grouped.py) — Stage 2c multi-expert prototype with lookup-table dispatch (4.58 TFLOPS, parity PASS at 1920 tokens × 4 experts).
> - `/tmp/phase9b_handtensorize.txt`, `/tmp/phase9b_grouped.txt` — scheduled-body dumps for inspection.
>
> **Stage 1 working artifacts:**
> - [scratch_phase9b_mma_proto.py](../../scratch_phase9b_mma_proto.py) — pure fp16 matmul, dlight tensorize ✓
> - [scratch_phase9b_dequant_proto.py](../../scratch_phase9b_dequant_proto.py) — V1, V1.5, V3, V2 variants. **V1.5 is the path forward**: two blocks at root + dlight Matmul + manual thread-bind on the leftover dequant block. Builds, runs, parity passes.
> - [scratch_phase9b_build_test.py](../../scratch_phase9b_build_test.py) — minimal regression test for the `from_expr + global_symbol` build pattern.
> - `/tmp/phase9b_v1_body.txt`, `/tmp/phase9b_v3_body.txt` — dlight-scheduled bodies for inspection.
>
> **Build incantation (was the spike's blocker):**
> ```python
> f = my_prim_func.with_attr("global_symbol", "main")
> mod = tvm.IRModule.from_expr(f)               # not IRModule({"main": prim_func})
> with target:
>     mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
> rt = tvm.tirx.build(mod["main"], target=target)
> ```
> The prior spike used `tvm.IRModule({"main": ...})` which doesn't attach a global_symbol; codegen then can't find buffer params (`Find undefined Variable X`). Fix: `IRModule.from_expr(prim_func.with_attr("global_symbol", "main"))`.
>
> **Validated facts (don't re-spike):**
> - dlight's `MatmulFP16Tensorization` ([3rdparty/tvm/python/tvm/s_tir/dlight/gpu/matmul.py:490](../../3rdparty/tvm/python/tvm/s_tir/dlight/gpu/matmul.py#L490)) emits wmma m16n8k16 fp16 MMA + Ampere software pipeline annotations on sm_87.
> - `get_wmma_intrin_group()` at [3rdparty/tvm/python/tvm/s_tir/tensor_intrin/cuda.py:1377](../../3rdparty/tvm/python/tvm/s_tir/tensor_intrin/cuda.py#L1377) is the API for option β hand-tensorize.
> - dlight's `auto_inline_producers` does NOT inline a producer block at prim_func root into the matmul's b_g2s — confirmed via V1 (the dequant block is left at root, untouched); the leftover block must be manually scheduled or dlight will leave it as non-thread-bound (illegal at GPU codegen).
> - Inlining the int4-unpack expression *directly* into the matmul block (V3) **disqualifies the matmul from `MatmulFP16Tensorization`** (no wmma in scheduled body, scalar fallback at 0.11 TFLOPS) — confirms the recognizer rejects `vk // NUM_ELEM_PER_STORAGE` divides on the reduction axis.
> - `compute_inline()` of the dequant block (V2) fails with "block ... is an output block" — TIR Schedule classifies the block as output because `W_fp16` is alloc_buffer at root scope. To make it inlinable would require allocating `W_fp16` inside an enclosing block.
> - The recognizer's three constraints are captured in memory entry [dlight_matmul_recognizer.md](../../.claude/projects/-home-alfie-mlc-llm/memory/dlight_matmul_recognizer.md) (auto-loaded).
>
> **Stage 1 bench summary (M=4096, N=1024, K=2048, single expert):**
>
> | variant | time | TFLOPS | wmma? | notes |
> |---|---:|---:|---|---|
> | pure fp16 matmul + dlight Matmul | 1.013 ms | 17.0 | ✓ | upper bound (no dequant) |
> | V1.5: two-block + dlight Matmul + manual dequant sched | 1.016 ms | 16.9 | ✓ | **selected for Stage 2**; dequant is essentially free |
> | V3: dequant inline in matmul block + dlight default | 163.5 ms | 0.11 | ✗ | falls to Reduction()/Fallback() — proves int4-divide blocks tensorization |
>
> **Production state to preserve:**
> - [python/mlc_llm/op/moe_matmul.py](../../python/mlc_llm/op/moe_matmul.py) has uncommitted `TX, TY, CTA_COUNT = 8, 32, 1024` change (was 64). Source-vs-lib are coherent. Don't revert source without rebuilding lib, and vice versa.
> - Backup libs: `lib_pre_phase9.so.bak` (CTA=64 baseline), `lib_phase9_cta1024.so` (= current production), `lib_phase9_v2_blkm16_blkk64.so` (regressed v2). All in `dist/qwen3_6-35B-A3B-q4f16_1/`.
>
> ---

**Date opened:** 2026-04-30
**Predecessor:** [phase9-prefill-throughput.md](phase9-prefill-throughput.md). Stage 9.1 identified the kernel; Stage 9.2 confirmed the tile-constant ceiling. Phase 9b is the structural rewrite that breaks the ceiling.

**Goal:** rewrite `dequantize_group_gemm` ([python/mlc_llm/op/moe_matmul.py:600-770](../../python/mlc_llm/op/moe_matmul.py#L600-L770)) so the inner GEMM uses sm_87's `m16n8k16` fp16 MMA via TIR's wmma intrinsics, keeping int4 dequant into a shared-memory fp16 W tile and the existing persistent-loop expert dispatch around it. **Why TIR rather than Triton:** stays inside MLC's compile pipeline, future quant variants (int3/mxfp4 on sm_89+) can extend the pattern, and the TVM `wmma_*` intrinsic family + dlight `MatmulFP16Tensorization` are already battle-tested for fp16×fp16 → fp32 — the missing piece is just the dequant fusion.

**Headline projection.** Stage 9.1 measured the kernel at ~80 GFLOPS sustained (~0.3 % of fp16 scalar peak / 0.04 % of TC peak). A well-formed wmma path on this shape typically lands ~30-50 % of TC peak. Conservative 5× kernel speedup → pp512 wall 2475 → ~1085 ms = **~470 tps pp512**, well past Stage 9.2's 290 tps gate and inside Phase 9's stretch goal of 450 tps (parity-class to llama.cpp). 10× would put MLC at parity-or-ahead.

---

## What's already in tree

Audit (2026-04-30) found:

- **`get_wmma_intrin_group`** at [3rdparty/tvm/python/tvm/s_tir/tensor_intrin/cuda.py:1377-1439](../../3rdparty/tvm/python/tvm/s_tir/tensor_intrin/cuda.py#L1377-L1439). Returns named intrinsics for `wmma_load_*`, `wmma_sync_*`, `wmma_fill_*`, `wmma_store_*` at the m16n8k16 shape. Supports fp16×fp16 → fp32 (and → fp16) with `trans_b` flag. **This is the API we tensorize against.**
- **`MatmulFP16Tensorization`** at [3rdparty/tvm/python/tvm/s_tir/dlight/gpu/matmul.py:490-704](../../3rdparty/tvm/python/tvm/s_tir/dlight/gpu/matmul.py#L490-L704). dlight schedule rule that takes an unscheduled fp16 matmul prim_func, applies `cache_read("wmma.matrix_a"|"wmma.matrix_b")`, `cache_write("wmma.accumulator")`, and `sch.tensorize` against `get_wmma_intrin_group(...)`. **This is the model we'll adapt.**
- **No int4-dequant + MMA composition exists.** `group_gemm` (fp16 unquant, line 385) is also scalar; `dequantize_group_gemm` (line 600) is what we're touching. Neither uses MMA today.

The dequantization step doesn't need to change — it already lands fp16 weights into a shared-memory `W_tile` ([moe_matmul.py:719-725](../../python/mlc_llm/op/moe_matmul.py#L719-L725)). The wmma fragment load reads from `W_tile`, so dequant is upstream of the tensorized inner block. No fp16 weight materialization to global memory is needed.

---

## Approach — two-track exploration

### Track A: dlight-direct (spike first, ~30 min)

Strip the manual schedule from `dequantize_group_gemm`'s inner block, hand the unscheduled body to `dl.gpu.MatmulFP16Tensorization()`, see if dlight can recognize the matmul pattern inside our persistent-loop wrapper.

**Risk:** dlight's `get_reduction_blocks` may reject the prim_func because of the outer `sblock("CTA")` nesting + indptr-driven expert dispatch. If it bails, abort and switch to Track B.

**Upside if it works:** Stage 1 is a ~10-line change (strip schedule + apply dlight). Tile sizes, software pipeline, fragment caches all handled by dlight's existing logic.

### Track B: hand-tensorize (fallback / production path)

Manually emit the cache_read/cache_write + tensorize calls inside our existing schedule:

```python
def _schedule():
    sch = s_tir.Schedule(_func)
    main_block = sch.get_sblock("compute")

    # Restructure compute block to be 16x16 micro-tile shaped:
    # split BLK_M=16 -> [None, 16], BLK_N -> [None, 16], BLK_K -> [None, 16]
    # blockize the (i_inner, j_inner, k_inner) micro-block

    # Cache fp16 W_tile shared -> wmma.matrix_b fragment:
    A_frag = sch.cache_read(blockize_inner, 0, "wmma.matrix_a")
    B_frag = sch.cache_read(blockize_inner, 1, "wmma.matrix_b")
    C_frag = sch.cache_write(blockize_inner, 0, "wmma.accumulator")

    # Tensorize via wmma intrinsic group:
    intrins = get_wmma_intrin_group(load_scope="shared", store_scope="shared",
                                    in_dtype="float16", out_dtype="float32",
                                    trans_b=True)
    sch.tensorize(loop_a, intrins["load_a"])
    sch.tensorize(loop_b, intrins["load_b"])
    sch.tensorize(loop_compute, intrins["compute"])
    sch.tensorize(loop_init, intrins["init"])
    sch.tensorize(loop_store, intrins["store"])
```

Tile constraints for MMA shape:
- **BLK_M ≥ 16** (one MMA m-tile = 16). Drop in-flight Phase 9.2 BLK_M=8 → 16.
- **BLK_N divisible by 16.** Keep 128 (8 MMA n-tiles per BLK_N) or trim to 64 (4 MMA n-tiles).
- **BLK_K divisible by 16.** Keep 32 (2 MMA k-tiles per BLK_K) or 64 (4 MMA k-tiles).
- **Padding for non-aligned cases.** N=1024 / BLK_N=128 = 8 ✓ aligned. M = batch_size_var, may need pad via `pad_einsum`.

**Risk:** more code, more places to bug. ~1.5-2 sessions if Track A bails.

---

## Stages

### Stage 1 — Standalone MMA prototype on a single shape (1 session)

Write a small standalone TIR kernel that does **int4-dequant + MMA matmul** at one fixed shape (e.g., gate_up: M=4096, N=1024, K=2048). No persistent-loop wrapper, no expert dispatch. Goal: prove the dequant + wmma composition compiles, runs, and produces correct numerics.

Tasks:
- New file: [scratch_phase9b_mma_proto.py](../../scratch_phase9b_mma_proto.py)
- Build the prim_func body: dequantize int4 weight to shared fp16 W_tile, then matmul with wmma intrins (or via dlight if Track A works).
- Reference: scalar `dequantize_group_gemm` body, single-expert call.
- Validation: `np.allclose(mma_output, scalar_output, rtol=1e-3, atol=1e-3)`.
- Bench: time both kernels on the same shape via `tvm.cuda.kernel_time` or wall-clock loop. Land ≥ 3× speedup vs scalar baseline.

**Land criterion:** standalone kernel compiles on Orin sm_87, parity within fp16 noise vs scalar, and bench shows ≥ 3× speedup.

### Stage 2 — Integration into `dequantize_group_gemm` (1-2 sessions)

**Stage 1 outcome (2026-04-30 cont.):** V1.5 pattern (two blocks at root + dlight Matmul + manual dequant schedule) lands the standalone case at 16.9 TFLOPS. Stage 2 is "lift this into the production grouped GEMM kernel".

The structural choice for Stage 2 is **two-prim_func + Relax-level fusion** vs **single prim_func with persistent-loop drop**. Both paths preserve the V1.5 result; they differ in how the per-tile dequant fuses.

**Path 2A — Split into two TIR prim_funcs at the Relax level (recommended).**
- New `dequant_per_expert(w[Ne,N,num_storage], scale[Ne,N,num_group], indptr) -> W_fp16[B_max, N]` — BW-bound, ~0.5 ms for 8 active experts × 4 MB writes.
- Keep `group_gemm` (the existing fp16 unquant version at [moe_matmul.py:385](../../python/mlc_llm/op/moe_matmul.py#L385)) and let dlight tensorize it.
- Glue at the Relax level: `R.call_tir(dequant_per_expert, ...)` then `R.call_tir(group_gemm, ...)`.
- Memory: pre-allocate one 32 MB `W_fp16_scratch` buffer reused across all MoE layers (40 of them) — single allocation, ~0 host overhead.
- Risk: `group_gemm` ([moe_matmul.py:385](../../python/mlc_llm/op/moe_matmul.py#L385)) is also hand-scheduled with the persistent-loop pattern — same blockers as the dequant version. Need to add an unscheduled fp16 group_gemm variant + let dlight schedule it.

**Path 2B — Single prim_func, drop persistent loop, grid launch + indptr scan (option α from prior spike).**
- Rewrite [moe_matmul.py:600-770](../../python/mlc_llm/op/moe_matmul.py#L600-L770) to launch a flat grid of `upper_bound_total_tiles = (ceildiv(B, BLK_M) + Ne) * tiles_per_row` blocks. Each block scans indptr to find its `(expert_e, m_offset, n_offset)` (a few hundred cycles, amortized over ~10k MMA cycles per tile).
- Inside the block: dequant a `BLK_N × BLK_K` slice of W into shared per ko-iter (same pattern as V1.5's W cooperative fetch, but per tile, not whole-W), then do the MMA matmul.
- Hand-tensorize the inner block via `get_wmma_intrin_group(...)` + `sch.tensorize`. dlight's recognizer won't see through the `sblock("CTA")` wrapper, so we don't go through `dl.gpu.Matmul()` — instead we manually emit the cache_read/cache_write/tensorize calls following the patterns dlight uses internally.
- Risk: more code, more bug surface. The persistent-loop drop alone (without tensorize) was already a 1-2 session investment.

**Recommendation: try 2A first.** It rides the path Relax already exercises for non-MoE linear projections (`FuseDequantizeMatmulEwise` + dlight pipeline) — least pipeline-divergence, lowest bug surface. If 2A's group_gemm path can't be made dlight-friendly, fall back to 2B.

**Tasks (Path 2A):**
1. Add a standalone `dequant_moe_weights(w, scale, indptr) -> W_fp16` prim_func to [moe_matmul.py](../../python/mlc_llm/op/moe_matmul.py). Single hand-schedule (cooperative fetch, BW-bound, no MMA needed). Output is `(top_k * num_tokens, N)` — only the active experts get materialized.
2. Add an unscheduled fp16 `group_gemm_unscheduled(x, W_fp16, indptr) -> O` variant. Keep the persistent loop OUT, let dlight tensorize the inner matmul.
3. Wire both at the Qwen3.5 MoE layer: `qwen3_5_moe_model.py` calls `dequant_moe_weights` then `group_gemm_unscheduled`. Pre-allocate `W_fp16_scratch` (32 MB) at the engine level if it's not already covered by Relax's memory planner.
4. Recompile [dist/qwen3_6-35B-A3B-q4f16_1/lib.so](../../dist/qwen3_6-35B-A3B-q4f16_1/lib.so).
5. Smoke test: 10-token completion via [scratch_phase7_followup_smoke.py](../../scratch_phase7_followup_smoke.py).

**Land criterion:** full 35B compile clean, engine loads, smoke output coherent.

### Stage 3 — Bench + decode regression check (0.5 session)

Run the same sweep used for Stage 9.2: pp=512, tg=[512, 4096] × 3 runs.

Tasks:
- [scratch_mlc_tg_sweep.py](../../scratch_mlc_tg_sweep.py) on the rebuilt lib.
- Compare to Stage 9.2 baselines (CTA=64: pp=202; CTA=1024: pp=207.9).
- Confirm tg512 within ±2 % of 44.99 (decode goes through `dequantize_gemv` so should be untouched, but verify).

**Land criterion:** pp512 ≥ 290 tps (Stage 9.2 gate-2), tg unchanged, no smoke regression.

### Stage 4 — Tile-parameter sweep + ship (0.5 session)

If Stage 3 lands but doesn't hit gate-3 (≥ 350 tps), sweep MMA tile params:
- BLK_M ∈ {16, 32, 48}
- BLK_N ∈ {64, 128, 256}
- BLK_K ∈ {32, 64}
- num_warps (TY) ∈ {2, 4, 8}

Each variant: ~15-min compile + 3-min bench. ~6 viable (BLK_M, BLK_N, BLK_K) combos that fit in shared mem on Orin. Land the best.

**Land criterion:** pp512 ≥ 350 tps (gate-3) or document why the ceiling is lower.

---

## Risk register

1. **dlight matmul recognizer rejects the persistent-loop body.** Track A bails; Track B is the path. Adds ~1 session.
2. **MMA fp32 accumulator + persistent-loop O_tile interaction.** Today's kernel stores fp32 O_tile and writes through scalar store. wmma.accumulator is also fp32 but lives in register fragment — the cache_write back to shared/global needs `accumulator_shared_to_global` pattern from dlight. May need extra copy steps that hurt the win.
3. **Software pipeline overhead at small BLK_K.** Dlight's `software_pipeline_*` annotations on sm > 75 add a 2-stage prefetch that helps with K=many but may overhead at K=32 (only 2 ko iters per tile). Could leave wins on the table or regress on small contexts. Mitigation: profile both with and without pipeline annotations.
4. **Bank conflicts on `W_tile` after dequant.** The dequant writes int4 → fp16 in a non-MMA-friendly layout. wmma fragments expect specific shared-mem layouts (16×16 row-major or 16×16 with `storage_align(factor=16, offset=8)`). Mitigation: apply `storage_align` on `W_tile` like dlight does.
5. **Padding when batch_size is symbolic and non-MMA-aligned.** The persistent loop has `T.if_then_else(m_offset + i < row[1], ..., zero)` for boundary handling. wmma fragments don't compose with element-wise predication — need to either pad `row[1]` to multiples of 16 or zero-fill out-of-range elements at the dequant step. dlight uses `pad_einsum`; we'll need an analogous mechanism.
6. **Decode regression.** Decode at b=1 doesn't hit `dequantize_group_gemm` (shortcut at [qwen3_5_moe_model.py:137](../../python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py#L137)), so the change is decode-safe by construction. Spec-verify γ > 4 (rare) is the only path that still sees small workloads through this kernel; absolute time is trivial.
7. **ABI / TVM version drift.** This vendored TVM has the wmma intrinsics (audit confirms). Low risk.

---

## Out of scope

- The non-quantized `group_gemm` ([line 385](../../python/mlc_llm/op/moe_matmul.py#L385)). Same kernel, used for fp16 weights only. Once the dequant version is MMA, the fp16 version is 1-line port (skip the dequant step).
- `dequantize_gemv` (b=1 GEMV). Decode-only; bandwidth-bound; tensorization wouldn't help.
- The `attn_paged` / FlashInfer ragged-prefill path. 2.96 % of prefill cost.
- 0.8B port. Same kernel applies, but 0.8B is dense — no MoE GEMM. Out of scope until 0.8B perf becomes a bottleneck.

---

## Land-criteria summary

| Gate | Metric | Bar | Status |
|---|---|---|---|
| 1 (proto compiles) | Stage 1 prim_func | int4-dequant + wmma compiles + parity vs scalar | ✅ **MET** 2026-04-30 cont. session 2 — V1.5 pattern (two blocks at root + manual dequant sched + dlight Matmul tensorize). Build path required `IRModule.from_expr(f.with_attr("global_symbol", "main"))` instead of `IRModule({"main": f})`. |
| 2 (proto wins) | Stage 1 wall-clock | ≥ 3× faster than scalar at same shape | ✅ **MET** — V1.5 at 1.016 ms / 16.9 TFLOPS vs V3 dlight-default scalar at 163.5 ms / 0.11 TFLOPS = 160× at this shape. |
| 2a (hand-tensorize works) | Stage 2a hand-tensorize on production-shape body | parity + speedup vs scalar | ✅ **MET** 2026-04-30 cont. session 3 — 3.7 ms / 4.6 TFLOPS standalone single-expert; parity PASS. Below V1.5 because we skip software pipeline + double-buffer (the dequant breaks the manifest_shared_memory_local_stage constraint); 4.6 TFLOPS still ~9× the production scalar (~0.5 TFLOPS estimated from Stage 9.1's ~80 GFLOPS sustained). |
| 2c (multi-expert proto) | grouped GEMM with multiple experts | parity + perf parity vs single-expert | ✅ **MET** — 1.76 ms / 4.58 TFLOPS, parity PASS at 1920 tokens × 4 experts. Per-tile lookup table dispatch (precomputed `tile_to_e/m/n`) is essentially free vs single-expert. |
| 3 (production integration) | Stage 2d compile + smoke | full 35B compile + 10-token coherent text | ✅ **MET** 2026-04-30 cont. session 4 — `MLC_MOE_GEMM_V2=1 mlc_llm compile dist/qwen3_6-35B-A3B-q4f16_1` → lib_phase9b_v2.so. Smoke: same 10-token output as baseline (`'Thinking Process:\n\n1.  **Identify'`). Engine load 49 s (warm cache). |
| 4 (prefill win) | pp512 tps | ≥ 290 (Stage 9.2 gate-2 bar) | ✅ **MET** — **523.67 tps** (1.81× past gate). |
| 5 (no regression) | tg512 tps | within ±2 % of 44.99 | ✅ **MET** — 44.86 tps (-0.04 % vs 44.88 baseline). |
| 6 (stretch) | pp512 tps | ≥ 450 (Phase 9 stretch / parity to llama.cpp) | ✅ **MET** — **523.67 tps** (1.16× past stretch). |

### Spike verdict (2026-04-30)

**Technical viability confirmed: dlight's `MatmulFP16Tensorization` does emit `T.tvm_mma_sync` + `T.tvm_load_matrix_sync` on Orin sm_87 for fp16 matmul, including the software-pipeline annotations.** The barrier to Stage 2 integration is structural, not capability:

1. dlight's matmul recognizer expects standard for-loop reduction; the production `dequantize_group_gemm` has a `while T.tvm_thread_invariant(...)` persistent-CTA loop scanning indptr.
2. dlight's index-map analysis bails on `vk // NUM_ELEM_PER_STORAGE` (int4 unpack); needs the dequant in a separate producer block writing fp16 W.
3. `auto_inline_producers` only inlines blocks within the matmul's own scope; a producer at prim_func root stays unfused.

The shipped MLC pipeline avoids these via Relax-level `FuseDequantizeMatmulEwise` + `FuseTIR` (which fuses two TIR funcs at the Relax level, after which dlight sees a single fused matmul). The MoE GEMM doesn't go through this path because it's hand-scheduled in `python/mlc_llm/op/moe_matmul.py`.

### Next-session decision

| option | effort | scope | recommended? |
|---|---|---|---|
| α — drop persistent loop, grid launch + tile lookup | 1-2 sessions | bigger but clean kernel rewrite | ✅ recommended |
| β — hand-tensorize inner block (no dlight) | 1.5-2 sessions | surgical kernel-only change | ok fallback |
| γ — Triton w4a16 group GEMM | 1 session | adds Triton runtime dep | only if speed-to-ship matters more than consistency |

**Recommendation:** option α. The persistent-loop pattern was Hopper-tuned (saturates 128 SMs); on Orin's 16 SMs Stage 9.2 measured CTA_COUNT 64 vs 1024 within 3 %, so the persistent loop saves nothing. Removing it simplifies the schedule, lets dlight tensorize cleanly, and the win generalizes to int3 / mxfp4 / fp8 quant variants future-shippped on sm_89+.

If gate 4 (≥ 290 tps) lands but gate 6 (≥ 450 tps) doesn't after option α, ship as Phase 9b partial. Gate 6 is the "win the deck" outcome and may need a follow-up Stage 4 tile sweep on the rewritten kernel.

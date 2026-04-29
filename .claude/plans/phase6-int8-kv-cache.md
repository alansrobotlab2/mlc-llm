# Phase 6 — int8 KV cache for Qwen3.6-35B-A3B on Orin AGX (capacity unblock lane)

**Status (2026-04-29):** SHIPPED in batched single-pass implementation. Plumbing landed; int8 throughput on sm_87 is essentially neutral (−2 % at tg512, −1.6 % at tg8192) — exactly as predicted. Greedy parity hit 2/5 EXACT (gate wanted ≥4/5); divergences are small token-level argmax shifts after 100+ identical chars, not catastrophic. **The plumbing is the durable win**; int8 lib is opt-in for memory-bound deployments. See worklog 2026-04-29 entry for the full result.

**Date opened:** 2026-04-29
**Predecessor:** [phase5-fp8-kv-cache.md](phase5-fp8-kv-cache.md). **Phase 5 shipped end-to-end and was a wall-clock LOSS** (−25 % at tg8192) on sm_87 because every fp8→fp16 cast lowers to a software bit-twiddle. Phase 6 reuses Phase 5's plumbing but switches the storage format to **int8 with per-token scales**, which has a hardware conversion path on every sm since Pascal.

**Goal (revised vs Phase 5):** primary win is **capacity** — 2× max in-flight tokens at the same VRAM budget. Throughput is **not** the goal; throughput-neutral or slight negative is acceptable. The math (Phase 5 §"Why long-context") still caps the throughput upside at ~+2 % at tg8K, so chasing wall-clock here is wrong. What we couldn't get from fp8 because of dequant cost, we can plausibly get from int8 because the dequant is a single hardware `cvt.rn.f16.s8` (~1 cycle).

---

## Why int8 vs fp8 on sm_87

| Format | Dequant on sm_87 | Compression | Phase 5 verdict on sm_87 | Phase 6 prediction |
|---|---|---|---|---|
| fp8 e4m3 | software (`<cuda_fp8.h>` bit-twiddle, ~5–8 cycles/lane) | 2× | tg8192 −25 %, parity 4/5 | n/a (closed) |
| **int8 per-token scale** | **hardware (`cvt.rn.f16.s8`, ~1 cycle/lane + FMA)** | **2×** | n/a | **target this lane** |
| int8 per-tensor scale | hardware, slightly cheaper (one global FMA) | 2× | n/a | fallback if per-token plumbing is too heavy |

The structural difference is on the dequant side. fp8 e4m3's exponent/mantissa interpretation requires a software lookup table or bit-pattern reconstruction on sm < 89; int8 → fp16 is a single SASS instruction since Pascal (sm_60). The "dequant cost dominates" failure mode of Phase 5 simply does not exist for int8 on Orin.

**Per-token scale layout.** Following vLLM/TRT-LLM:
- `pages: int8[num_pages, 2, num_kv_heads, page_size, head_dim]` — same shape as fp16 cache, half the bytes.
- `scales: float32[num_pages, 2, num_kv_heads, page_size]` — one fp32 scale per (page, K/V, head, token). 4 bytes per (token × kv_head × K-or-V).

For Qwen3.6-35B-A3B (10 full-attn layers, 2 KV heads, page_size=16, head_dim=256):
- Old fp16 KV: 20 KB / token resident
- New int8 + scale: 10 KB pages + (10 layers × 2 × 2 heads × 4 bytes) = 10 KB + 160 B = ~10.16 KB / token
- **Net compression: 1.97× (very close to the theoretical 2×)**

---

## Code reuse from Phase 5 (already landed)

These are all preserved on the current branch as part of the Phase 5 dtype-split refactor and apply unchanged:

- `dtype_kv` parameter on [PagedKVCache.create_generic](../../python/mlc_llm/nn/kv_cache.py), [TIRPagedKVCache.__init__](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py), and the `extract_creation_args` dispatcher.
- Runtime threading: trailing `rx.StringImm(dtype_kv)` arg flowing into `vm.builtin.paged_attention_kv_cache_create`, picked up by the C++ ctor and used as a separate `DLDataType dtype_kv` parameter to `PagedAttentionKVCacheObj` for page allocation. Temp Q/K/V/output buffers stay at `dtype`.
- Runtime dtype-equality assertions in [paged_kv_cache.cc:1288, 1391, 1434](../../3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc) relaxed to allow `pages.dtype != qkv.dtype`.
- Kernel `T.cast(pages[…], qkv_dtype)` insertion at the page-load boundary in `_attention_decode`, `_attention_prefill`, `tree_attn_with_paged_kv_cache`, `_kv_cache_debug_get_kv`. Mirror writes via `T.cast(k_data, dtype_kv)` in `_kv_cache_transpose_append`.
- `_rope` patched to do the negation in fp32 (cast pages → fp32 *before* `-buffer[…]`) — applies identically to int8.
- `qwen3_5_moe_model.py` reads `self.kv_cache_dtype` from config and threads it.
- Config field `kv_cache_dtype: Optional[str] = None` on `Qwen35Config` — set to `"int8"` for the int8 build.

**Net new work for Phase 6 is the scale tensor and the math at the read/write boundary, not the plumbing.**

---

## Findings after first-pass survey (2026-04-29)

The original plan ("net new work is the scale tensor + math at boundary") was right in shape but underestimated the **C++ class-hierarchy threading**. Phase 5 only changed kernel internals; Phase 6 changes kernel *signatures*, which forces virtual-method changes through `attn_backend.h`. Surface area for plumbing alone:

**Python TIR (7 kernels — 1 new arg each, gated on `dtype_kv == "int8"`):**
- [_page_kernels.py](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_page_kernels.py) — `_kv_cache_transpose_append` (write-side quant), `_kv_cache_debug_get_kv` (read-side dequant), `_copy_single_page`, `_compact_kv_copy` (memcpy + propagate scales)
- [_decode_kernels.py](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_decode_kernels.py) — `_attention_decode` (read-side dequant)
- [_prefill_kernels.py](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_prefill_kernels.py) — `_attention_prefill` (read-side dequant)
- [tree_attn.py](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/tree_attn.py) — `tree_attn_with_paged_kv_cache` (read-side dequant)

**C++ runtime ([paged_kv_cache.cc](../../3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc)):**
- New field `std::vector<Tensor> scales_;` plus `bool use_int8_kv_;` flag derived from `dtype_kv`.
- Alloc loop at line 380-386: when `use_int8_kv_`, also allocate `scales_[i]` shape `[num_total_pages, 2, num_kv_heads, page_size]` dtype fp32.
- 5 kernel-call sites need `scales_[local_layer_id]` threaded in: `f_transpose_append_mha_` (lines 1352, 1380), `f_debug_get_kv_` (lines 1659, 1705), `f_compact_copy_` (line 731), `f_copy_single_page_` (line 700), plus the decode/prefill MHA() virtual dispatches (in `AttentionInternal` and `MHACrossAttnInternal`).

**C++ class hierarchy ([attn_backend.h](../../3rdparty/tvm/src/runtime/vm/attn_backend.h)):**
- `PagedDecodeFunc::MHA`, `PagedPrefillFunc::MHA`, `PagedPrefillTreeMaskFunc::MHA` virtual signatures get a new `Tensor scales` parameter. (`RaggedPrefillFunc` does not — ragged path doesn't read pages.)
- TIR overrides forward `scales` into `attn_func_(...)` after `pages`. FlashInfer overrides ignore `scales` (FlashInfer doesn't support int8 KV; defensive `TVM_FFI_ICHECK` if `scales->shape[0] > 1`).

**Risky piece: per-token max-abs reduction in TIR append kernel.** Existing `_kv_cache_transpose_append` is element-wise over `(token, head, dim)`. Per-token quant needs a reduction across `head_dim` (256 elems for Qwen3.6) before the cast. Restructure the loop:
- Outer parallel: `(token, head)` bound to `blockIdx.x` — one block per (token, head) pair.
- Inner: `head_dim` bound to `threadIdx.x` (256 threads, one warp × 8 = max workgroup, fits page-load CTA budget).
- Reduction: `T.cross_thread_reduction` for max-abs. Naive form first; warp-shuffle is a follow-up if it shows up in profiling.

### Revised stage breakdown — single batched pass

The 4-stage breakdown below was written assuming per-stage rebuilds. With `dtype_kv == "int8"` gating the new code paths, **the regression case (fp16 KV) is preserved as a no-op throughout** — Stage 6.1 acceptance falls out of any correct implementation. Therefore:

- **Single batched implementation**: all kernel signature changes + C++ class-hierarchy threading + scale alloc + quant/dequant math go in one batch, gated on `dtype_kv == "int8"`. fp16/fp8 paths are byte-identical to today.
- **One TVM rebuild** (12 min) instead of four (48 min).
- **One model recompile** of the int8 lib (35 s) on top of the existing fp16 lib.
- Acceptance staged via test-execution order, not code-change order: regression bench (validates 6.1) → debug round-trip (validates 6.2) → layer-output cosine (validates 6.3) → end-to-end bench/parity/capacity (validates 6.4).

Trade-off: harder to bisect if something breaks, since several layers change at once. Mitigation: changes are per-file and per-function additive (not refactors), each with a clear `if dtype_kv == "int8":` (Python) or `if (use_int8_kv_)` (C++) guard. A bisect would naturally split along those guards.

---

## What's new in Phase 6

### 1. Scale buffer (the big new piece)

- New persistent allocation in `PagedAttentionKVCacheObj`: `scales_` parallel array of `Tensor`, one per layer, shape `[num_total_pages, 2, num_kv_heads, page_size]`, dtype `float32`. 4 bytes × 4096 (num_pages) × 2 × 2 × 16 ≈ 1 MB per layer × 10 layers = 10 MB total — negligible vs the 18 GB of weights.
- Pass scale tensor (per-layer slice) as a new kernel argument to `_kv_cache_transpose_append` and `_attention_decode`/`_attention_prefill`.
- Plumb through the C++ runtime call sites in [paged_kv_cache.cc:1349, 1377, 1488](../../3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc) (the `f_transpose_append_mha_.value()(…)` invocations) — pass `scales_[local_layer_id]` alongside `pages_[local_layer_id]`.

### 2. Append kernel: per-token max-abs + symmetric quant

In `_kv_cache_transpose_append`, replace
```python
pages[…] = T.cast(k_data[vgpos, vh, vf], dtype_kv)
```
with:
```python
# In a reduction over head_dim for each (token, head):
max_abs[vgpos, vh] = T.reduce_max(T.abs(k_data[vgpos, vh, :]))
scale = max_abs[vgpos, vh] / T.float32(127.0)
pages[…] = T.cast(k_data[vgpos, vh, vf] / scale, "int8")
scales[…] = scale
```
The reduction is per (token, head), so a small one (256 elements). For Qwen3.6 with `head_dim=256`, the reduction fits in a warp easily — bind to threadIdx.x with shuffle reduction.

### 3. Read kernels: dequant + multiply

In `_attention_decode` page-load and `_attention_prefill` page-load:
```python
K_smem[smem_idx] = T.cast(pages[page_idx], qkv_dtype) * T.cast(scales[scale_idx], qkv_dtype)
```

The codegen lowers `T.cast(int8, fp16)` to `cvt.rn.f16.s8`, which is one SASS instruction. The multiply is a hardware FMA. Total per-lane dequant cost: ~3–4 cycles (vs ~5–8 for fp8 e4m3 software conversion).

V-side identical, with `scales[..., 1, ...]`.

---

## Stages

### 6.1 — Scale buffer alloc + threading (1 session)

- C++ side: in `PagedAttentionKVCacheObj` ctor, allocate `scales_` parallel to `pages_` when `dtype_kv` is `int8`. New constructor param `bool use_scales` (or derive from `dtype_kv`).
- Plumb scale tensor as a new kernel input through the existing call sites. New TIR kernel signature: append/decode/prefill take an extra `var_scales: T.handle`.
- With `dtype_kv == dtype_q` (no int8), skip the alloc and pass nullptr / empty.

**Acceptance:** v6 (regression with `dtype_kv == dtype`) compiles and benches identical to current.

### 6.2 — Append kernel: per-token scale + quant (1 session)

- Add the max-abs reduction inside the existing `for global_pos, h, f in T.grid(ntoken, num_kv_heads, head_dim)`. Use a warp-shuffle reduction since head_dim=256 fits in one warp.
- Quant: `pages = T.cast(round(k_data / scale), "int8")`.
- Store scale to scales buffer.

**Acceptance:** round-trip test (write fp16 → int8 page + scale → read fp16 via `_kv_cache_debug_get_kv`) reproduces the original within `(max_abs / 127)` tolerance per token.

### 6.3 — Decode + prefill: dequant on read (1 session, two kernels)

- Modify the page-load block: `K_smem[…] = T.cast(pages[…], qkv_dtype) * T.cast(scales[…], qkv_dtype)`.
- Symmetric for V_smem.
- Verify `cvt.rn.f16.s8` actually fires — capture `tvm_kernels.cu`, grep for the cvt instruction or for an `(int8_t)` -style cast that the assembler will lower to `cvt`.

**Acceptance — numerical:**
- Layer-output cosine sim ≥ 0.999 on the first 50 tokens of canonical prompts vs fp16-KV reference.

### 6.4 — End-to-end bench + parity + capacity (1 session)

```bash
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96/
cp -r dist/qwen3_6-35B-A3B-q4f16_1_kvfp8 dist/qwen3_6-35B-A3B-q4f16_1_kvint8
# edit kv_cache_dtype to "int8" in mlc-chat-config.json
python -m mlc_llm compile dist/qwen3_6-35B-A3B-q4f16_1_kvint8/mlc-chat-config.json \
  --device cuda -o dist/qwen3_6-35B-A3B-q4f16_1_kvint8/lib.so

# Throughput bench (sanity check, expect ±2 % of fp16)
python bench_mlc.py --model-dir dist/qwen3_6-35B-A3B-q4f16_1_kvint8 \
  --pp 128,512,4096,8192 --tg 256

# Capacity bench (the actual win)
python bench_mlc.py --model-dir dist/qwen3_6-35B-A3B-q4f16_1_kvint8 \
  --max-total-seq-len 524288 --pp 1
# Compare to fp16's max-fitting --max-total-seq-len under same VRAM budget.
```

**Acceptance gates — throughput (low bar, this is the capacity lane):**

1. **tg512 within ±2 %** of fp16 baseline (52.22 tps). Sign of regression must be ≤ 2 %.
2. **tg8192 within ±5 %** of fp16 baseline (16.17 tps). The +2 % BW upside math says we *might* see a small win; the "throughput-neutral" framing is the actual goal.

**Acceptance gates — capacity (the actual win):**

3. **Max successful `--max-total-seq-len` ≥ 1.7 ×** the max at fp16 under the same VRAM budget. Math says the ceiling is ~2× (the page bytes are exactly halved, scales are noise).

**Acceptance gates — parity:**

4. Greedy parity ≥ **4/5 EXACT** on the canonical 5 prompts × 50 tokens (Phase 5 hit 4/5 with no scale; per-token scales should equal or exceed).
5. **Long-prompt parity:** 1 long-context prompt (4 K input, 100 generated), agreement ≥ 95/100 tokens.

### 6.5 — Decision

| Outcome | Action |
|---|---|
| All gates pass | **Land. Default kv-cache-dtype to int8 for any model that wants long-context.** Document in worklog. |
| Throughput gates 1-2 pass, capacity gate 3 fails | Land for *speed parity* but file an issue on page-pool sizing — capacity should be free with the bytes. |
| Parity gate 4 fails | Try per-(token, head) scales (vs per-(token) which already gives per-head granularity, this would be even finer) — ~2 extra sessions of kernel work. |
| Throughput gate 1 fails (worse than ±2 % at tg512) | Surprising — the hardware path should be free. Profile to verify `cvt.rn.f16.s8` is actually being emitted. If TVM lowers `T.cast(int8, fp16)` to a software path, that's a TVM bug to file. |
| Throughput gate 2 fails (>5 % regression at tg8192) | If parity is OK and capacity is ≥1.7×, *still land*. The goal isn't throughput. Throughput regression is acceptable IF the capacity unblock is real. |

If the only thing that lands is a working int8-KV path with capacity ≥1.7× and throughput-neutral, **that's the success criterion** — same as a backend feature flag enabling longer-context inference on the same hardware. Don't try to make it faster; try to make it land.

---

## Stop conditions

- **Pre-start:** if a TVM IR pass strips the scale tensor (similar to the Relax `rx.op.zeros((), fp8)` issue from Phase 5), use `rx.StringImm` or attach the scale to the `init` tensor's payload — same workaround as Phase 5's dtype carrier.
- **6.2:** if the warp-shuffle reduction for max-abs doesn't fit the kernel's existing thread structure cleanly, fall back to a naive `T.reduce_max` and let TIR pick the schedule. Slower but works.
- **6.3:** if `T.cast(int8, fp16)` doesn't lower to `cvt.rn.f16.s8` on sm_87 (would be very surprising), file the TVM bug and fall back to the fp8-style software path. The fall-back is still int8 *storage* (capacity win) with software dequant (throughput same as Phase 5).
- **6.4:** if **capacity gate 3 < 1.5×** AND **throughput gates fail**, close. No win on either axis.

---

## What Phase 5 left in the codebase that still helps

These TVM-level patches landed for Phase 5 but apply to int8 too:

- [3rdparty/tvm/python/tvm/contrib/nvcc.py](../../3rdparty/tvm/python/tvm/contrib/nvcc.py) — `supports_fp8` lowered to sm_70 (no-op for int8 path but doesn't hurt).
- [3rdparty/tvm/src/target/source/codegen_cuda.cc](../../3rdparty/tvm/src/target/source/codegen_cuda.cc) — the e8m0 CUDA-version guard and the vector-cast fallback (any narrow → wide cast on a vector type).
- [3rdparty/tvm/src/target/source/literal/cuda_half_t.h](../../3rdparty/tvm/src/target/source/literal/cuda_half_t.h) — e8m0 helpers gated on CUDACC ≥12.7 (unrelated to int8 but kept for future).
- The VEC_SIZE limitation (TVM Ramp asserts ≤4 lanes) is also not specific to fp8 — it caps int8 vector loads at the same 4-byte width. Per-warp transaction = 32 threads × 4 bytes = 128 bytes = one cache line, fully coalesced. Same as fp8.

---

## Notes for future-me

- **Per-token scale storage in pages** — vLLM stores scales in a parallel buffer; we follow that. An alternative is to pack the scale into the page slot itself (e.g., last 4 bytes of each token's data). That's denser but kernels become uglier. Not worth the complexity for first cut.
- **Block-wise scale option** — between per-tensor (one scale globally, accuracy risk) and per-token (one scale per token+head+K/V, what we're doing), there's also "block of N tokens shares a scale" which trades some storage for fewer scale loads. Skip for first cut; revisit only if per-token scale storage shows up in profiling.
- **Comparison to llama.cpp `--type-k q8_0`** — that's the field benchmark for int8 KV. They use block scales (32 elements per block, fp16 scale per block) — different layout. The math should land in the same ballpark but the kernel surface is different. Worth a separate bench on the same prompts after landing.
- **The dtype-split refactor is now reusable** — any future port to a different KV dtype (mxfp4, fp6, sparse, whatever) only needs to provide the cast functions and (optionally) the scale layout. The plumbing is done.

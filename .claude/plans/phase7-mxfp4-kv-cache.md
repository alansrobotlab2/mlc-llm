# Phase 7 — mxfp4 KV cache for Qwen3.6-35B-A3B on Orin AGX (capacity + long-context throughput)

**Date opened:** 2026-04-29
**Predecessor:** [phase6-int8-kv-cache.md](phase6-int8-kv-cache.md). **Phase 6 shipped int8 plumbing throughput-neutral** (−1.6 % at tg8192, capacity 2× — not measured but by construction). The dtype-split refactor + parallel scale tensor + per-token quant are all in place and reusable. Phase 7 is a small extension: switch the storage format from **int8 with per-token scales** to **mxfp4 (4-bit float) with per-block scales** for **2× more capacity (4× total vs fp16)** and **a real long-context throughput win at 64K+** where KV reads finally dominate weights.

**Goal:** primary win is **capacity — 4× max in-flight tokens at fixed VRAM** (vs Phase 6 int8's 2×). Secondary win is **throughput at long context** where KV BW dominates: theoretical 30 % faster at 128K, 50 %+ at 256K on Orin (vs fp16). Parity will be looser than int8 (4-bit vs 8-bit) and we accept that — same fallback as Phase 6 (land for capacity-bound use cases, opt-in).

---

## Why mxfp4 vs Phase 6 int8

| Format | Bytes/elem (KV) | Block-scale overhead | Effective bits | Capacity vs fp16 | Predicted Δ@128K |
|---|---|---|---|---|---|
| fp16 (baseline) | 2.0 | — | 16 | 1× | 0 % |
| int8 + per-token fp32 scale (Phase 6) | 1.0 | 0.0156 (4 B / 256 elems) | ~8.13 | 2× | ~−2 % |
| **mxfp4 + per-block scale** (block=32) | **0.5** | **0.0078 (1 B / 32 elems)** if E8M0, or **0.125** if fp32 | **~4.13 / ~4.25** | **~3.9× / ~3.8×** | **+10 %** (BW math) |

The arithmetic for the predicted speedup at long context, with weight reads ≈ 2 GB and seq-len N at decode step:
- KV BW per step (10 layers × 2 K/V × 2 kv_heads × 256 head_dim): 20 KB/token × N
- fp16: at N=128K → 2.5 GB KV → BW total 4.5 GB/step → ~50 ms → 20 tps (rough)
- int8: KV halves → 1.25 GB → BW total 3.25 GB → ~30 % faster
- mxfp4: KV quarters → 0.625 GB → BW total 2.6 GB → ~45 % faster vs fp16

Phase 6's measurement at 8K showed near-zero throughput delta because KV at 8K is only ~4 % of step BW. The win compounds with context length.

### Hardware path on sm_87

mxfp4 has zero native hardware support on Orin (sm_120+ only via Blackwell's MX MMA). But unlike fp8 — where the conversion is a software bit-twiddle inside `<cuda_fp8.h>` (5–8 cycles/lane) — mxfp4 unpack is **arithmetic, not magic**:
1. Extract 4 bits via `(byte >> shift) & 0xF` — 1 cycle.
2. Reinterpret as E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit. The 16 possible values are: `[0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0]`. Lookup table or a small piecewise-linear poly.
3. Multiply by per-block fp32 scale (or shift by E8M0 scale).

Total: 3-5 cycles per element on sm_87, vs fp8's 5-8 in a software lookup. Net dequant cost is comparable to int8 (~1-3 cycles), making this a viable Orin-friendly format unlike fp8.

---

## Code reuse from Phase 6 (already landed)

Everything that landed for int8 is reusable as-is:

- **Scales tensor plumbing** through C++ runtime ([paged_kv_cache.cc](../../3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc) `scales_` field + 5 kernel call-site threadings).
- **Kernel signature uniformity** — every page-touching TIR kernel (`_kv_cache_transpose_append`, `_kv_cache_debug_get_kv`, `_attention_decode`, `_attention_prefill`, `tree_attn_with_paged_kv_cache`, `_copy_single_page`, `_compact_kv_copy`) accepts the scales handle.
- **C++ class hierarchy** ([attn_backend.h](../../3rdparty/tvm/src/runtime/vm/attn_backend.h) MHA virtuals on PagedDecodeFunc/PagedPrefillFunc/PagedPrefillTreeMaskFunc).
- **function_table.cc fix** for hybrid + FlashInfer.
- **Python config field** `kv_cache_dtype` on `Qwen35Config` and `Qwen35MoEConfig`, threaded through `dtype_kv` to `create_generic`.
- **Dispatch logic** in [dispatch_kv_cache_creation.py](../../python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py) — strips `dtype_kv` for FlashInfer cache (FlashInfer only supports the same dtype for q/k/v).

**Net new work for Phase 7 is the storage format change (int8 → packed-u4), a different scale layout (per-token → per-block), and the unpack-to-fp16 math at the read boundary.**

---

## Storage layout

### Pages — packed-u4 (two fp4 values per int8 byte)

```
pages: int8[num_pages, 2, num_kv_heads, page_size, head_dim/2]
```

Note the trailing `head_dim/2` — each int8 byte holds two consecutive E2M1 values. For head_dim=256 (Qwen3.6), that's 128 bytes per (token, K-or-V, head). vs Phase 6's int8: 256 bytes per the same. Halved.

**Why packed-u4 in int8 storage:** TVM doesn't have a native `uint4` dtype on the GPU side. Storage as int8 with manual nibble packing is the standard workaround (also what vLLM/TRT-LLM do for fp4 KV). Read-side: `nibble_lo = (byte) & 0xF; nibble_hi = (byte >> 4) & 0xF`. Write-side: `byte = (hi << 4) | (lo & 0xF)`.

### Scales — per-block (block_size=32 along head_dim)

```
scales: float32[num_pages, 2, num_kv_heads, page_size, head_dim/32]
```

For head_dim=256, that's 8 fp32 scales per (token, K-or-V, head) — 32 bytes. vs Phase 6's int8: 4 bytes per (token, K-or-V, head). 8× larger scale tensor, but the page tensor is half size, so total bytes per token still drop by ~2× vs int8.

**Per-(token, head, block) granularity** is finer than int8's per-(token, head). Reasoning: 4 bits gives ~16 levels of resolution, so a tighter scale per fewer elements is worth the bookkeeping cost. Block=32 matches the OCP MX spec and is the standard choice in vLLM/TRT-LLM.

**Optional E8M0 scale (true MX format)**: 1 byte per 32 elements instead of 4. Adds another ~3× capacity savings on the scale tensor, but kernels need bit-shift dequant. **Defer**: stage 7.5 follow-up if the fp32-scale path is otherwise good.

### fp4 (E2M1) value table

| nibble | sign | exp | mant | value |
|---:|---:|---:|---:|---:|
| 0000 | 0 | 00 | 0 | 0 |
| 0001 | 0 | 00 | 1 | 0.5 |
| 0010 | 0 | 01 | 0 | 1.0 |
| 0011 | 0 | 01 | 1 | 1.5 |
| 0100 | 0 | 10 | 0 | 2.0 |
| 0101 | 0 | 10 | 1 | 3.0 |
| 0110 | 0 | 11 | 0 | 4.0 |
| 0111 | 0 | 11 | 1 | 6.0 |
| 1xxx | 1 | (mirror) | (mirror) | -value |

Quant: `quant = round(x / scale * 1.5)` then map to nearest fp4 grid (the LUT is irregular: gaps at 2.5, 3.5, 4.5, 5.0 are dropped). Standard implementation: scale by max-fp4 (=6.0), clamp, then look up.

Dequant in TIR: do the lookup with a small inline LUT, then multiply by scale. Or use a polynomial approximation. The OCP MX paper's reference impl is a 16-entry table.

---

## Stages

### 7.1 — Storage layout change (Python TIR + C++ alloc) (~1 session)

**Python TIR kernel signatures** — same as Phase 6 (scales handle present), but:
- `pages` matched as `int8` shape `(num_pages, 2, num_kv_heads, page_size, head_dim/2)`. Note trailing `/2`.
- `scales` matched as `float32` shape `(num_pages, 2, num_kv_heads, page_size, head_dim/32)`. Per-block.
- Add a guard `assert head_dim % 32 == 0` and `assert head_dim % 2 == 0` (both true for Qwen3.6 head_dim=256).

**C++ runtime** ([paged_kv_cache.cc](../../3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc)):
- Modify the alloc loop. When `dtype_kv == "mxfp4"`:
  - `pages_[i]` shape `{num_total_pages, 2, num_kv_heads, page_size, head_dim/2}` int8
  - `scales_[i]` shape `{num_total_pages, 2, num_kv_heads, page_size, head_dim/32}` fp32
- Phase 6's int8 path keeps its existing per-(token, head) scale layout. Use `dtype_kv == "mxfp4"` to switch.

**Acceptance:** `dtype_kv="float16"` and `dtype_kv="int8"` paths regression-test identically to Phase 6. Compile a stub mxfp4 lib (kernels still use the old int8 code paths, just to test the alloc layout) — engine loads, reports correct memory.

### 7.2 — Append kernel: per-block max-abs + 4-bit pack (~1 session)

In `_kv_cache_transpose_append` (mxfp4 path):
1. Per (token, head, K-or-V): split head_dim into 8 blocks of 32 elements.
2. Per block: compute `max_abs`, derive `scale = max_abs / 6.0` (max E2M1 representable = 6).
3. Per element: `q_fp4 = quantize_to_e2m1(x / scale)` — find nearest fp4 grid value, encode as 4-bit.
4. Pack two consecutive fp4 nibbles into one int8 byte: `byte = (hi << 4) | (lo & 0xF)`.
5. Write `scales[..., block_idx]` and the packed `pages[..., dim_idx/2]`.

**Thread layout:** 1 block per (token, kv_head) — same as Phase 6. Body iterates 8 blocks of 32 elements each, serial.

**Reference for `quantize_to_e2m1`** (Python pseudocode):
```python
def fp4_quant(x):
    sign = x < 0
    x = abs(x)
    if x < 0.25:    nibble = 0   # 0
    elif x < 0.75:  nibble = 1   # 0.5
    elif x < 1.25:  nibble = 2   # 1
    elif x < 1.75:  nibble = 3   # 1.5
    elif x < 2.5:   nibble = 4   # 2
    elif x < 3.5:   nibble = 5   # 3
    elif x < 5.0:   nibble = 6   # 4
    else:           nibble = 7   # 6
    return (sign << 3) | nibble
```

In TIR this becomes a chain of `T.if_then_else`s or a small LUT. Both cost roughly the same on sm_87 — pick whichever the autoscheduler likes better.

**Acceptance:** standalone round-trip test (`scratch_phase7_round_trip.py`, modeled on Phase 6's). Write fp16 → mxfp4 → read fp16 reproduces the input within `max_abs/8` per element (4-bit precision bound, conservatively).

### 7.3 — Read kernels: unpack + multiply by block scale (~1 session)

In `_attention_decode`, `_attention_prefill`, `tree_attn_with_paged_kv_cache`, `_kv_cache_debug_get_kv`:

```python
# In K_smem load (per element vec_idx):
byte = pages[page, 0, head, slot, vec_idx // 2]
nibble = T.if_then_else(vec_idx % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
val_fp16 = fp4_dequant_lut(nibble)              # E2M1 → fp16
block_idx = vec_idx // 32
K_smem[...] = val_fp16 * scales[page, 0, head, slot, block_idx]
```

**`fp4_dequant_lut`** — a 16-entry table. In TIR, express as nested `T.if_then_else` or `T.const` array. The LUT may inline at compile time and become a switch in PTX. Inspect the generated CUDA to verify (capture `tvm_kernels.cu` for one of the prefill kernels).

**Vectorization caveat:** Phase 6 had `VEC_SIZE=4` (4 elements per thread per iteration). For mxfp4, two consecutive elements share an int8 byte, so vec_size is naturally 2 (or multiples of 2). Reading 4 elements = 2 bytes per thread, aligned. Should be fine. Verify the TIR Ramp asserts don't trip (Phase 5 found `lanes <= 4`).

**Acceptance:**
- Layer-output cosine sim ≥ 0.995 on the first 50 tokens vs fp16 reference (looser than Phase 6's 0.999 — 4-bit drift is real).
- Standalone round-trip test passes within fp4 quant noise.

### 7.4 — End-to-end bench + parity + capacity (~1 session)

```bash
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96/
cp -r dist/qwen3_6-35B-A3B-q4f16_1 dist/qwen3_6-35B-A3B-q4f16_1_kvmxfp4
sed -i 's/"kv_cache_dtype": null/"kv_cache_dtype": "mxfp4"/' dist/qwen3_6-35B-A3B-q4f16_1_kvmxfp4/mlc-chat-config.json
PYTHONPATH=3rdparty/tvm/python:python LD_LIBRARY_PATH=3rdparty/tvm/build:build:$LD_LIBRARY_PATH \
  .venv/bin/python -m mlc_llm compile dist/qwen3_6-35B-A3B-q4f16_1_kvmxfp4/mlc-chat-config.json \
  --device cuda -o dist/qwen3_6-35B-A3B-q4f16_1_kvmxfp4/lib.so

# Throughput (the long-context lane is where the win is)
python -u bench_mlc.py --model-dir dist/qwen3_6-35B-A3B-q4f16_1_kvmxfp4 \
  --pp 128,512,4096,8192,32768,65536,131072 --tg 256 --runs 1 --warmup 1

# Capacity (the actual primary goal)
python -u bench_mlc.py --model-dir dist/qwen3_6-35B-A3B-q4f16_1_kvmxfp4 \
  --pp 1 --max-total-seq-len 524288  # 4× current default
# Compare to fp16 max-fitting --max-total-seq-len under same VRAM budget.

# Parity (5 prompts × 50 tokens, fp16-TIR vs mxfp4)
python -u scratch_phase7_parity.py --ref-dir dist/qwen3_6-35B-A3B-q4f16_1_tir --cmp-dir dist/qwen3_6-35B-A3B-q4f16_1_kvmxfp4
```

**Acceptance gates — throughput (capacity is the primary, throughput is bonus):**

| Gate | Bar |
|---|---|
| 1. tg512 | within ±5 % of fp16 (Phase 6 int8 hit −2 %; mxfp4 should be slightly worse but within range) |
| 2. tg32K | **≥ +5 %** over fp16 (this is where KV reads start to matter and the BW saving compounds) |
| 3. tg128K | **≥ +20 %** over fp16 (math says ~+30 %; allow margin for unpack overhead) |

**Acceptance gates — capacity:**

| Gate | Bar |
|---|---|
| 4. Max successful `--max-total-seq-len` ≥ **3.5×** the fp16 max (math says 4×, allow margin for scale tensor overhead) |

**Acceptance gates — parity:**

| Gate | Bar |
|---|---|
| 5. Greedy parity ≥ **2/5 EXACT** on canonical 5×50 (looser than Phase 6's 4/5 because 4-bit drift is real; we still expect 100+ char identical prefixes per Phase 6 pattern) |
| 6. Long-prompt parity: 1 long-context prompt (4 K input × 100 tokens), agreement ≥ **80/100 tokens** (looser than Phase 6's 95/100) |

### 7.5 — Decision

| Outcome | Action |
|---|---|
| All gates pass | **Land. Mark mxfp4 as the recommended kv-cache-dtype for any model needing 64K+ context.** |
| Throughput gate 3 (tg128K) fails but 2 (tg32K) passes + capacity ≥3.5× | Land for capacity, document the throughput cliff at very long context as Orin BW limit. |
| Throughput gates 1-3 all fail | Likely the unpack cost dominates. Try the E8M0 scale (smaller scale tensor → less BW). Try larger block size (block=64 or 128) to reduce scale tensor BW. |
| Capacity gate 4 fails (< 3×) | Investigate: scale tensor size? Page allocator overhead? Should be a math problem, not a tuning one. |
| Parity gate 5 fails (0/5 EXACT) | Try block=16 (tighter granularity, more accurate but more bytes). If still 0/5, the E2M1 grid is too coarse for KV — accept and document, OR try fp4-E1M2 (1.5x more usable values in the dynamic range that matters for K/V). |

If we land mxfp4 with capacity ≥3.5× and throughput-neutral at short context but +10-30 % at long context, **that is a legitimate win on Orin** — the first KV-quant scheme that actually moves throughput in our favor on this hardware. It also enables ports to other BW-bound platforms.

---

## Stop conditions

- **Pre-start:** verify TVM TIR can express a 16-entry LUT cleanly. If `T.const_array` or similar isn't there, the LUT becomes a chain of `T.if_then_else` — works but uglier. Worst case: write a small TIR helper macro.
- **7.2:** if the per-block max-abs reduction creates compile-time issues (TIRx auto-scheduler may not pattern-match the inner block-of-32 reduction), fall back to per-token max-abs (lose 1-2 bits of precision but simpler kernel).
- **7.3:** if `T.cast(int8 packed-u4, fp16)` emits a slow software path (highly unlikely — should just be byte ops + LUT), capture `tvm_kernels.cu` and verify the unpack is `& 0xF` / `>> 4`. If TVM lowers it weirdly, hand-write the dequant via `T.shift_right` / `T.bitwise_and`.
- **7.4:** if **capacity gate 4 < 2.5×** AND **throughput at 128K is no better than int8**, close. The cost of a more complex kernel isn't paying off vs the int8 we already have.

---

## What Phase 6 left in the codebase that still helps

These are the durable gains from Phase 6 that this phase reuses unchanged:

- [3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc](../../3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc) — `scales_` field, alloc loop, threading through 5 kernel call sites.
- [3rdparty/tvm/src/runtime/vm/attn_backend.h](../../3rdparty/tvm/src/runtime/vm/attn_backend.h) — `Tensor scales` parameter in PagedDecodeFunc/PagedPrefillFunc/PagedPrefillTreeMaskFunc MHA() virtuals.
- [cpp/serve/function_table.cc](../../cpp/serve/function_table.cc) — hybrid+FlashInfer RNN-state fix (latent bug pre-Phase-6, fixed there, applies forever).
- All 7 TIR kernels in [3rdparty/tvm/python/tvm/relax/frontend/nn/llm/](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/) accept `scales_handle` uniformly. mxfp4 just changes what's INSIDE the kernels, not their signatures.
- [python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py:541](../../python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py#L541) and [python/mlc_llm/model/qwen35/qwen35_model.py:65](../../python/mlc_llm/model/qwen35/qwen35_model.py#L65) `kv_cache_dtype` config field, threaded to `create_generic`. Just add `"mxfp4"` to the recognized dtype string.

**Net new work scope:** ~3 kernel functions need the unpack/pack math (`_kv_cache_transpose_append`, `_kv_cache_debug_get_kv`, `_attention_decode`, `_attention_prefill`, `tree_attn_with_paged_kv_cache`, `_copy_single_page`, `_compact_kv_copy`) — each gains ~10-30 lines of body changes. Total ~150-250 LoC across all of TIR. C++ runtime: ~10 LoC change to alloc loop only. **Substantially less work than Phase 6.**

---

## Notes for future-me (and for the fresh-session pickup)

- **Start by reading Phase 6's worklog entry (2026-04-29 in worklog.md).** That documents which abstractions exist, what gotchas are landmined, and the function_table.cc fix that unblocked hybrid models with FlashInfer. The plumbing it describes is what Phase 7 builds on.
- **The compile cycle is 12 min for TVM C++ + 4 min for the relax/CUDA model.** Batch all kernel changes into one TVM rebuild. Use `python -u` for bench scripts to avoid stdout buffering eating output on timeout.
- **For TIR kernels, Python-time `if dtype_kv == "mxfp4":` branching at function-definition time gives one clean prim_func body per dtype.** Same pattern as Phase 6.
- **For the bench on Orin, long context is where the win is.** Don't waste time on tg64/tg512 — the math says no win there. Bench tg32K, tg64K, tg128K.
- **Capacity test is the primary acceptance.** Walk `--max-total-seq-len` until OOM, compare to fp16's max-fitting limit. This is what was deferred in Phase 6 and should be the first thing measured here.
- **Parity will be worse than int8.** That's expected. Set the gate accordingly; don't tune the kernel toward parity if it costs throughput.
- **E8M0 scale** (1 byte per block) is a follow-up. Stage 7.5 if everything else lands. It saves ~3 % more capacity and doesn't help throughput much (per-block scale already small).
- **Block size choice (32 vs 16 vs 64):** OCP standard is 32. vLLM uses 32. TRT-LLM offers configurable. Stick with 32 unless a tuning issue surfaces.
- **The Phase 6 int8 lib stays.** Phase 7 mxfp4 is an additional opt-in dtype, not a replacement. int8 is still the right answer when you want 8-bit precision (e.g., higher-fidelity workloads where the 2/5 parity from int8 was the issue, not capacity).

## Pickup checklist

When this is opened in a fresh session, the very first things to do:

1. `git status` and `git log -3 --oneline` — confirm Phase 6 is committed (the function_table.cc fix in particular).
2. Read the 2026-04-29 worklog entry for Phase 6, plus the current state of [phase6-int8-kv-cache.md](phase6-int8-kv-cache.md).
3. Verify Phase 6 regression bench still passes with the current build: `python -u bench_mlc.py --model-dir dist/qwen3_6-35B-A3B-q4f16_1_kvint8 --pp 128 --tg 64`.
4. Sketch the fp4 LUT in a small TIR scratch file before touching the real kernels — the LUT is the only piece without a Phase 6 analog.
5. Stage 7.1 first: change just the alloc shapes, keep the kernel bodies identical to Phase 6's int8 path, verify the engine loads. THEN start changing kernel bodies.

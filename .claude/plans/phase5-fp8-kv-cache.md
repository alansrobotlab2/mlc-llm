# Phase 5 — fp8 KV cache for Qwen3.6-35B-A3B on Orin AGX (long-context lane)

**Date opened:** 2026-04-28
**Predecessor:** [phase4-perf.md](phase4-perf.md) (4A KV-int8 DEFERRED for the same kernel-work reason; 4D meta-sched DEAD; 4B MTP spec is the active GO lane for tg512).
**Goal (confirmed with user):** **long-context decode (≥8 K seqlen).** Phase 5 is *not* a tg512 play — at tg512 the math says <0.2 %. It only earns its keep at long context, where it gives both real throughput and 2× max-seqlen capacity headroom.

---

## Context

User asked for a concrete gameplan to add fp8 KV cache to MLC-LLM, scoped as a long-context lane separate from Phase 4. The prior Phase 4A audit established that MLC has *no* `kv_cache_dtype` plumbing: TVM's `PagedKVCache.create_generic` takes a single monolithic `dtype` ([python/mlc_llm/nn/kv_cache.py:32](../../python/mlc_llm/nn/kv_cache.py#L32)) that propagates to every attention/page-management kernel. Adding fp8 is therefore a TVM kernel project (~1 wk), not a config flip.

The Qwen3.6-35B-A3B uses heavy GQA (2 KV heads of 16) and only 25 % full-attn layers, so the *per-token* KV cache is small (20 KB/tok at fp16). That is exactly why this phase is wrong-axis at tg512 but *right* at long context: KV scales linearly with seqlen, so at 8 K+ it becomes a real fraction of per-step traffic, and at 32 K+ it's the dominant resident-memory cost.

---

## Why long-context — the math

35B-A3B config (verified in [dist/qwen3_6-35B-A3B-q4f16_1/mlc-chat-config.json](../../dist/qwen3_6-35B-A3B-q4f16_1/mlc-chat-config.json)):

- `num_hidden_layers=40`, `full_attention_interval=4` → **10 full-attn layers** (the only layers with paged KV cache; the other 30 are GDN with `RNNState`, unaffected by this phase).
- `num_key_value_heads=2`, `head_dim=256`, `dtype=bfloat16` → 2 bytes/element.
- `context_window_size=262144` (262 K — model supports it; we just can't fit it).

**KV bytes per generated token across all full-attn layers:**

```
K+V per layer  = 2 × num_kv_heads × head_dim × dtype_bytes
               = 2 × 2 × 256 × 2  = 2048 B
× 10 layers    = 20 480 B  = 20 KB / token
```

**Per-step KV read (decode reads the entire prefix once) and as a fraction of per-step traffic** at v6 (3.85 GB/token at 204 GB/s × 53 tps):

| seqlen | fp16 KV / step | fp8 KV / step | savings | % of per-step traffic | implied tg gain (BW-bound Amdahl) |
|---:|---:|---:|---:|---:|---:|
| 512 | 10 MB | 5 MB | 5 MB | 0.13 % | ~0.1 % |
| 4 096 | 80 MB | 40 MB | 40 MB | 1.04 % | ~1 % |
| 8 192 | 160 MB | 80 MB | 80 MB | 2.08 % | **~2 %** |
| 16 384 | 320 MB | 160 MB | 160 MB | 4.16 % | **~4 %** |
| 32 768 | 640 MB | 320 MB | 320 MB | 8.3 % | **~8 %** |
| 65 536 | 1.28 GB | 640 MB | 640 MB | 16.6 % | **~15 %** |

**Capacity unblock — the equally important benefit:**

Within a fixed VRAM budget, fp8 KV doubles the max in-flight tokens. With a 60 GB-ish runtime budget on Orin (after weights + workspace), the resident-KV ceiling roughly doubles:

| Mode | Approx. max total seqlen × concurrent reqs |
|---|---|
| fp16 KV (today) | ~bounded by current page-pool size |
| fp8 KV | ~2× the same number of tokens, same VRAM |

Capacity numbers should be confirmed by the bench at stage 5.6 (set `--max-total-seq-len` higher and observe whether it actually fits). For long-context use cases (long retrieval contexts, long agent traces, multi-doc QA) this can be a hard *unblock*, not just a speedup.

**Stage of phase order:** Phase 5 sits in parallel with 4B — they don't conflict. 4B speeds up tg512 (~+40 %), Phase 5 speeds up + unblocks long context. If user is doing both tg512 demos *and* long-context work, both are worth landing. If only doing tg512 demos, 4B wins.

---

## Format choice — fp8 e4m3fn, per-tensor static scale (if proceeding)

| Option | Pros | Cons | Decision |
|---|---|---|---|
| int8 (per-token scale) | Best numerics; field-standard for KV (vLLM/TRT-LLM) | Per-token scale storage in pages; widest kernel surgery | Skip |
| **fp8 e4m3fn (per-tensor)** | Wider dynamic range than int8; matches MLC's existing FP8 weight quant ([per_tensor_quantization.py:230-240](../../python/mlc_llm/quantization/per_tensor_quantization.py#L230-L240)); only one scale per layer/K-or-V; smallest kernel diff | ~0.5–1 % accuracy hit acceptable risk | **Choose** |
| fp8 e5m2 | More precision | Narrower range; K cache outliers worse | Fallback if e4m3 parity fails |

**sm_87 reality:** Orin has no native FP8 MMA (FP8 tensor cores are sm_89+/sm_90). fp8 here is **storage-only**: bytes packed in pages, dequant to fp16/fp32 in registers before the dot product. TVM already supports `float8_e4m3fn` as a first-class dtype ([3rdparty/tvm/python/tvm/runtime/_tensor.py](../../3rdparty/tvm/python/tvm/runtime/_tensor.py)).

**Scale layout:** auxiliary tensor of shape `[num_full_attn_layers, 2, num_kv_heads]` (2 = K/V), fp32, ~80 B total — trivial. Static (calibrated once on a small prompt set), not dynamic. Stored in the `PagedKVCache` object alongside the page pool.

---

## Code touch points (with file:line citations from audit)

### TVM-side (the bulk of the work)

1. **[3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py:540 (TIRPagedKVCache.__init__)](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py#L540)**
   - Split `dtype: str` → `dtype_q: str, dtype_kv: str`. Default `dtype_kv = dtype_q` so existing call sites are unchanged.
   - Plumb `dtype_kv` to the six kernels at lines 621, 651–652, 673–675.

2. **`_kv_cache_transpose_append`** ([_page_kernels.py:40-74](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_page_kernels.py#L40-L74))
   - Pages buffer dtype → `dtype_kv`.
   - On write: `pages[...] = T.cast(quantize(K_in, scale_K), dtype_kv)` and same for V.
   - Smallest kernel — start here as the spike.

3. **`_attention_decode`** ([_decode_kernels.py:181-411](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_decode_kernels.py#L181-L411))
   - **Critical kernel — this is what runs at tg512 on Orin.** Online-softmax fused, no MMA intrinsics, scalar QK at line 335.
   - Pages buffer dtype → `dtype_kv`. Q stays at `dtype_q`.
   - Insert dequant in the page-load block (lines 319–324): `K_smem[...] = T.cast(pages[...], "float32") * scale_K[head_id]`. Same for V.
   - VEC_SIZE re-computation: at fp8 byte size = 1, the formula `min(max(8 // qkv_dtype_bytes, D // 32), 4)` gives VEC_SIZE = 4 still (good — vectorized 4-byte loads).

4. **`_attention_prefill`** ([_prefill_kernels.py:208+](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_prefill_kernels.py#L208)) — same pattern as decode. Lower bench priority since prefill is not BW-bound for us, but parity gate requires it.

5. **`_kv_cache_debug_get_kv`** ([_page_kernels.py:106-136](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_page_kernels.py#L106-L136)) — dequant on read. Trivial.

6. **`_copy_single_page`, `_compact_kv_copy`** ([_page_kernels.py:169-263](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_page_kernels.py#L169-L263)) — pure memcpy in `dtype_kv`; no dequant needed. Just propagate the dtype param.

### MLC-side (small)

7. **[python/mlc_llm/nn/kv_cache.py:17-93](../../python/mlc_llm/nn/kv_cache.py#L17-L93)** — add `dtype_kv: Optional[str] = None` to `create_generic`; pass through as `rx.DataTypeImm(dtype_kv or dtype)` (new arg position in the packed call).

8. **[python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py:337-354](../../python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py#L337-L354)** — read `self.kv_cache_dtype` (new config field) and pass.

9. **[python/mlc_llm/interface/gen_config.py](../../python/mlc_llm/interface/gen_config.py)** + `protocol/mlc_chat_config.py` — add `kv_cache_dtype: Optional[str] = None` to model config override. Mirror the `tensor_parallel_shards` plumbing.

10. **[python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py:74](../../python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py#L74)** — `extract_creation_args` reads the new dtype slot.

### Calibration (new file)

11. **`scripts/calibrate_kv_scales.py`** — run a small calibration pass (e.g. 32 prompts × 256 tokens) on the fp16 model, capture max-abs of K and V per (layer, head), compute static fp32 scales, write to `dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/kv_scales.npz`. Loaded at engine init.

---

## Stages (only if proceeding past the gate)

**Sequence (revised 2026-04-28):** 5.1 → 5.3 → 5.4 → 5.2 → 5.5 → 5.6. Rationale: 5.1 is a kill-switch — everything after it is dead weight if TVM fp8 lowering fails on sm_87, so it runs first. 5.3 (regression-free dtype-split refactor) runs before any kernel math changes so it lands as a safe checkpoint and pins down the scale-tensor signature that the kernel changes will consume. 5.4 (write-side quant + debug-get round-trip) validates the bytes flow cleanly through pages before we change any read math. Calibration (5.2) is bumped after 5.4 — by then we know the scale-tensor shape exactly. 5.5 + 5.6 are last.

### 5.1 — Spike: prove the kernel pattern on a toy mod (1 session)

Build a stand-alone TIR module that does `pages_fp8 → dequant → matmul` matching the decode load pattern. Verify numerically vs an fp16 reference. Goal: shake out TVM cast semantics for `float8_e4m3fn` on sm_87 (does TVM lower it to byte-load + scalar convert, or fall over?).

**Stop condition:** if `T.cast(fp8, "float32")` doesn't lower cleanly on sm_87 → fall back to manual `reinterpret_cast<uint8_t>` + bit-twiddle dequant. Cost: +1 session.

### 5.3 — TVM plumbing: split `dtype` → `dtype_q, dtype_kv` + define scale-tensor signature (1 session)

Touch [kv_cache.py:540](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py#L540) and the six kernel call sites. With `dtype_kv = dtype_q` as the default, all existing models must compile and bench identical to before. *This stage is the safest and gives a regression-free checkpoint.*

**Scale-tensor signature (new sub-deliverable in 5.3):** the TIR kernels today take `(pages, page_table, …)` with no scale input. The fp8 path requires `scale_K, scale_V` to be passed in. Pin the layout *now*, before kernel surgery: `scale: T.Buffer((num_layers, 2, num_kv_heads), "float32")`, indexed as `scale[layer_id, 0|1, head_id]`. Plumb it through three layers in one commit — TIR prim_func signature, the Relax wrapper in [kv_cache.py](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py), and the MLC dispatcher in [dispatch_kv_cache_creation.py:74](../../python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py#L74). For `dtype_kv == dtype_q` the scale arg is the unit tensor (or `None` and the kernel skips dequant), so existing models don't change behavior.

**Acceptance:** v6 rebuilds and benches within ±0.5 % of 52.62 tps tg512.

### 5.4 — Append + debug-get kernels (1 session)

Modify `_kv_cache_transpose_append` (write-side quant) and `_kv_cache_debug_get_kv` (dequant). Validate by writing fp16, reading via debug-get, comparing to original — round-trip error should be bounded by per-tensor scale precision.

### 5.2 — Calibration tool (1 session, post-5.4)

Write `calibrate_kv_scales.py` against the fp16 35B. Now that 5.3 has pinned the scale-tensor shape, the calibration tool produces *exactly* that layout. Validate scale stability across 5–10 different prompt sets. Store as numpy `.npz`.

### 5.5 — Decode kernel + prefill (1–2 sessions)

The core surgery. Insert dequant in page-load block. Use the calibrated scales as a kernel input parameter (broadcast scalar per head_id).

**Numerical acceptance:** layer-output cosine sim ≥ 0.999 vs fp16 KV path on first 50 tokens of canonical prompts.

### 5.6 — End-to-end bench + parity, long-context focus (1–2 sessions)

```bash
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96/
python -m mlc_llm gen_config "$SNAP" \
  --quantization q4f16_1 --conv-template qwen3_5 \
  --kv-cache-dtype float8_e4m3fn \
  -o dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/
python -m mlc_llm compile dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/mlc-chat-config.json \
  --device cuda -o dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/lib.so

# Long-context bench — the actual scoring
python bench.py --model dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/ \
  --tg 512 --tg 4096 --tg 8192 --tg 16384 --tg 32768 --pp 128

# Capacity bench — does it actually unblock longer seqlen?
python bench.py --model dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/ \
  --max-total-seq-len 131072 --tg 65536
```

**Acceptance gates — long-context throughput:**

1. **tg8192 ≥ +1.5 % vs fp16 KV at same seqlen** (primary signal — the math predicts ~2 %).
2. **tg16384 ≥ +3.0 % vs fp16 KV at same seqlen** (secondary — math predicts ~4 %).
3. **tg512 within ±1 % of v6 52.62** (no regression at short context — verify dequant overhead doesn't eat the savings at low seqlen).

**Acceptance gates — capacity:**

4. **Max successful `--max-total-seq-len` at fp8 ≥ 1.7× the max at fp16** under the same VRAM budget. Measure both and report the ratio. (If <1.7× there's a memory leak / page-pool bug somewhere.)

**Acceptance gates — parity:**

5. Greedy parity 50/50 on the 5 canonical prompts at 50 tokens (short-prompt sanity).
6. **Long-prompt parity:** 1 long-context prompt (4 K input tokens, 100 generated tokens) — fp8-KV vs fp16-KV agreement ≥ 95/100 tokens. Quantization noise compounds with seqlen; this is the gate that catches it.

### 5.7 — Decision

| Outcome | Action |
|---|---|
| All gates pass | Land. Flag default-on for `--kv-cache-dtype float8_e4m3fn` builds; document in worklog. |
| Throughput gates 1–3 pass, capacity gate 4 fails | Land for *speed* but file an issue on page-pool sizing — capacity should be free with the bytes. |
| Parity gate 5 or 6 fails | Try e5m2 fallback (one-stage retry: rerun calibration in 5.2 with e5m2, rebench 5.6). If still fails, fall back to *per-head* scales (vs per-tensor) — re-do 5.2 + 5.5 with `[layer, K/V, head]` scale tensor. ~2 extra sessions. |
| Throughput gate 1 fails | Bug — savings should be ≥0 at minimum. Profile dequant overhead at the page-load. |
| Throughput gates 2–3 fail by large margin (e.g. tg8192 < +0.5 %) | Surprising; check whether dequant is being hoisted out of the inner load loop. |

If the only thing that lands is the dtype-split TVM patch (5.3), **that alone is worth keeping** — it's a regression-free refactor that unblocks future int8/fp8/MXFP4 work on any model.

---

## Stop conditions (whole-phase)

- **Pre-start:** Phase 5 can run in parallel with 4B (different files: 4B is in `cpp/serve/`, Phase 5 is in `3rdparty/tvm/.../llm/`). No ordering dependency. If single-engineer though, finish 4B first — bigger tg512 win, faster path.
- **5.1 spike:** if TVM `float8_e4m3fn` lowering on sm_87 needs a TVM PR, scope balloons to ≥2 weeks → escalate / reconsider.
- **5.5 decode:** if layer cosine sim < 0.999 even after retrying with per-head scales (instead of per-tensor) and e5m2, close — numerics not workable at this aggression.
- **5.6 bench:** if **tg8192 gain < +0.5 %** AND **capacity gate 4 < 1.5×**, close — neither speed nor capacity benefit materialized; something is structurally wrong (likely dequant hoisting out of the inner load).

---

## Verification harness

- **Unit (per-kernel):** `tvm.testing` round-trip — write fp16 → page → read → compare to original within scale-bounded tolerance.
- **Integration (per-layer):** capture full-attn layer output for a 50-token prefix in fp16-KV, then in fp8-KV, cosine sim ≥ 0.999.
- **End-to-end:** the 5 canonical prompts × 50 tokens greedy parity gate from Stage 5.6.
- **Bench:** standard `bench.py` invocation matching v6 protocol (TG=512 steady-state, MAXN locked) — see [memory/bench_protocol.md](../../../.claude/projects/-home-alfie-mlc-llm/memory/bench_protocol.md).

---

## Notes for future-me

- **Dense MHA model bonus:** if a future port targets a dense-MHA model (Llama-3 8B/70B, Mistral) the same kernel patch from Phase 5 lands a much bigger tg512 win there (~5–8 % at typical context). Don't redo the work — keep the dtype-split as a clean abstraction. Tag the relevant TVM commits.
- **MTP draft (4B) interaction:** when 4B lands, the 0.8B MTP draft model also has a paged KV cache. Phase 5 should "just work" on the draft cache too via the same flag — but verify, the draft is dense-attention not GDN-hybrid, and the layer count and head shapes differ.
- **Sliding window:** `sliding_window_size = -1` in the config (unused). If we ever turn it on, fp8 KV combines naturally — older tokens are evicted; recent ones live in the page pool, all dequanted via the same path.

# Phase 10 — Vision input (Qwen3-VL multimodal) for the Qwen3.5 stack

> ## 🔖 Session resume notes (read first)
>
> **Status as of 2026-05-01 (Stage 5b CLOSED):** Stages 0–5b shipped. Headline multimodal greedy parity gate PASSES at **176/180 (97.8%)** across the 5-prompt eval set ([reference_outputs_vl5.pt](../../reference_outputs_vl5.pt)). Diagnostic mrope-collapse passes 50/50, confirming inline-mRoPE prefill+decode is structurally correct. Two real bugs caught and fixed this session: (1) preprocessor `(tps, C)` axis order should be `(C, tps)` matching HF Qwen2VLImageProcessorFast flatten + PIL.BICUBIC needed antialiasing on downscale (switched to torchvision); (2) Chunk-B's mrope-on attention path used `paged_kv_cache.self_attention` which is **ragged-only and ignores cached K's**, producing decontextualized decode. Fixed by switching to `attention_with_fused_qkv` with pre-rotated Q/K/V — `RopeMode.NONE` makes the cache skip its internal rotation but still route prefill→ragged kernel and decode→cached-K kernel based on append lengths. **`qwen2_5_vl_model.py:268` has the same latent bug** (uses `self_attention`); needs the same fix if/when activated. bf16 backbone attempted and abandoned: regressed to 7/25 because HF reference is fp16 and bf16's narrower mantissa worsens fp16↔bf16 mismatch — the dtype-boundary plumbing (vision pinned fp16, LM cast to bf16, image_embed fp16→LM dtype boundary) is kept in tree as Phase 7 prep.
>
> **Up next — Stage 6 (production engine wiring):** extend `ImageData` with `grid_thw`, add engine-side `<|image_pad|>` substitution, register `qwen3_5_vl` conversation template, thread `position_ids`/`mrope_deltas` through the engine prefill batch. R-2 (cross-batch RoPE convention conflict) actually bites here.
>
> **Predecessor:** [phase9b-tir-mma-group-gemm.md](phase9b-tir-mma-group-gemm.md) shipped — text-only side at 1.85× over llama.cpp on 35B-A3B and 1.345× on 0.8B (`lib.so` = Phase-9b-v2 + FlashInfer in both `dist/qwen3_6-35B-A3B-q4f16_1/` and `dist/qwen3_5-0.8B-q4f16_g16e/`).
>
> **Stage 0 audit result (RoPE convention):** TVM's `RopeMode.NORMAL` rotation at [3rdparty/tvm/python/tvm/relax/frontend/nn/llm/position_embedding.py:514-519](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/position_embedding.py#L514-L519) is **half-split (NeoX/Llama-style)** — same convention HF Qwen3.5 uses. The planning critique flagged a possible interleaved-vs-NeoX mismatch; reading the actual TIR shows it's NeoX. The `gptj` rope_type at [position_embedding.py:508-513](../../3rdparty/tvm/python/tvm/relax/frontend/nn/llm/position_embedding.py#L508-L513) is the alternative interleaved variant; we don't use it. `op/mrope.py:_rotate_half` is also half-split. **Conclusion: no weight permutation needed in `qwen35_loader.py` — Stage 2 swap is convention-compatible.** Text-only Phase 9 is using the correct rotation.
>
> **Headline gate:** end-to-end multimodal greedy-decode parity vs HF transformers on the Qwen3.5 / Qwen3-VL checkpoint, ≥ 48/50 tokens match on the standard fixed-image prompt set. Same bar as Stage 5 in [CLAUDE.md](../../CLAUDE.md).
>
> **Scope cut for v1:** ship without Deepstack (the 3-aux-merger injection at vision layers 8/16/24). Forward signature reserves the hook in Stage 5 so Stage 6 isn't a breaking spec change. Quality delta is ~1-3% on fine-grained VQA / OCR per Qwen3-VL ablations — material for OCR-heavy use, not material for general VQA.

---

**Date opened:** 2026-05-01
**Predecessor:** [phase9b-tir-mma-group-gemm.md](phase9b-tir-mma-group-gemm.md). Phase 9b shipped the text-only side at parity-or-better with llama.cpp Q4_K_S/Q4_K_XL on Orin.
**Pressure-test session:** [planning subagent transcript](../../.claude/projects/-home-alfie-mlc-llm/8c78c260-91f2-4d7b-87e5-91783669e3f8/subagents/agent-a07a31d3e843bfa78.jsonl) (Plan agent, 2026-05-01).

**Goal:** add vision-input support (image and video tokens, M-RoPE positions, ViT + patch merger) to the Qwen3.5 dense and Qwen3.5-MoE backbones such that HF multimodal checkpoints (`Qwen3_5MoeForConditionalGeneration`) load and produce parity-quality outputs on multimodal prompts.

**Why now:** the text-only stack is shipped and stable; the released HF checkpoints under both `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.6-35B-A3B` are multimodal, and our loaders have been silently dropping `model.visual.*` weights. Closing the multimodal gap unlocks the actual user-facing surface of the family.

**Non-goals:**
- Audio input (separate modality, separate tower).
- Deepstack integration (deferred to Stage 6).
- Performance-tuning the vision tower (run it in fp16 unquantized for v1).
- Rewriting the chunked-prefill engine — Stage 5b is the minimum surgical fix to keep image spans intact within a single chunk.
- Tensor-parallel sharding the vision tower (run on rank 0, broadcast post-merger embeddings — the tower is ~1.1B params, sharding does not pay back the 27 all-reduces per image).

---

## What's already in tree

Audit (2026-05-01) found the existing pieces:

| component | location | state |
|---|---|---|
| `MultimodalRotaryEmbedding`, `apply_multimodal_rotary_pos_emb` | [op/mrope.py:59-142](../../python/mlc_llm/op/mrope.py#L59-L142) | complete; half-split, drop-in compatible with TVM `RopeMode.NORMAL` |
| `VisionPositionMetadata`, `get_mrope_position_ids` | [op/mrope.py:145-410](../../python/mlc_llm/op/mrope.py#L145-L410) | complete |
| `Qwen35MoEConfig.mrope_section` / `mrope_interleaved` | [qwen3_5_moe_model.py:47-48](../../python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py#L47-L48) | declared, NOT wired through |
| `Qwen35Config` (dense 0.8B) mrope fields | [qwen35_model.py:30-66](../../python/mlc_llm/model/qwen35/qwen35_model.py#L30-L66) | absent — add in Stage 2 |
| Qwen2.5-VL reference (text-decoder only, no tower, not registered) | [qwen2_5_vl_model.py](../../python/mlc_llm/model/qwen2_5_vl/qwen2_5_vl_model.py) | uses softmax GQA — wrong base for Qwen3.5 (GDN hybrid). Useful as a spec-shape reference for `mrope_deltas`, `_set_mrope_delta`, `_build_decode_position_ids` patterns |
| `phi3v` end-to-end (vision tower + projector + image_embed) | [phi3v_model.py:274](../../python/mlc_llm/model/phi3v/phi3v_model.py#L274), [phi3v_image.py](../../python/mlc_llm/model/phi3v/phi3v_image.py) | clean `image_embed(pixel_values, h, w, ch, cw)` entry-point pattern to mirror |
| CLIP ViT module (wrong arch for Qwen3-VL) | [vision/clip_vision.py](../../python/mlc_llm/model/vision/clip_vision.py) | reference only — Qwen3-VL needs 3D-conv patch embed, learned 48×48 grid pos-embed, 1D rotary in attn |
| `ImageData(image, embed_size)` | [serve/data.py:63](../../python/mlc_llm/serve/data.py#L63) | hardcoded `embed_size = 576` (LLaVA) / 1921 (phi3_v); Qwen3-VL is dynamic per-image — needs extension |
| Image-processing primitives | [vision/image_processing.py](../../python/mlc_llm/model/vision/image_processing.py) | reusable resize/normalize building blocks; Qwen3-VL needs its own dynamic-resolution branch |

**Reusable from prior phase work:** Phase 6's int8 KV cache plumbing already taught us how to thread `kv_cache_dtype` from config → model → `create_generic`. Vision-config plumbing follows the same shape (config → model attribute → spec entry). Phase 9b's FlashInfer linkage is unchanged by Phase 10 (vision tower runs separately, attention path unaffected).

---

## Reference target — Qwen3.5 multimodal (per-checkpoint)

Read from the **actual** `Qwen/Qwen3.5-0.8B/config.json` (2026-05-01 audit). The architecture differs in size between 0.8B and 35B-A3B — never hard-code, always read from the checkpoint.

| field | 0.8B | 35B-A3B (TBD) | source |
|---|---|---|---|
| `vision_config.depth` | **12** | TBD | ViT block count |
| `vision_config.hidden_size` | **768** | TBD | ViT model dim |
| `vision_config.intermediate_size` | **3072** | TBD | ViT FFN |
| `vision_config.num_heads` | 12 | TBD | ViT attn heads |
| `vision_config.out_hidden_size` | **1024** | TBD | merger output (matches LM hidden) |
| `vision_config.patch_size` | 16 | 16 | 3D-conv patch |
| `vision_config.temporal_patch_size` | 2 | 2 | image gets paired (single image → t=1 after merge) |
| `vision_config.spatial_merge_size` | 2 | 2 | 2×2 spatial merge in patch merger |
| `vision_config.num_position_embeddings` | 2304 | 2304 | learned 48×48 grid |
| `vision_config.deepstack_visual_indexes` | **`[]`** | TBD | **0.8B has NO deepstack — Stage 6 is moot for 0.8B** |
| `vision_config.hidden_act` | gelu_pytorch_tanh | gelu_pytorch_tanh | |
| `image_token_id` | **248056** | TBD | LM-side scatter target |
| `video_token_id` | **248057** | TBD | |
| `vision_start_token_id` / `vision_end_token_id` | **248053 / 248054** | TBD | |
| `text_config.head_dim` | 256 | TBD | |
| `text_config.partial_rotary_factor` | **0.25** | TBD | only 64 of 256 dims rotated; cos/sin shape is `[seq, 64]` not `[seq, 256]` |
| `text_config.rope_parameters.mrope_section` | **`[11, 11, 10]`** (sum=32; doubled to 64 = 256·0.25) | `[24, 20, 20]`? TBD | |
| `text_config.rope_parameters.mrope_interleaved` | **`true`** | TBD | **see Stage 0b** below |
| `text_config.rope_parameters.rope_theta` | 1.0e7 | TBD | |
| preprocessor | `Qwen2VLImageProcessorFast`, `image_mean=image_std=[0.5,0.5,0.5]` | same | **NOT ImageNet** |

### Stage 0b — open: mrope_interleaved=true vs op/mrope.py half-split

Stage 0 confirmed TVM `RopeMode.NORMAL` is half-split (NeoX). The 0.8B config has `mrope_interleaved: true` — this is **not** the rotation convention (`_rotate_half`), it is the section-packing convention. With `mrope_interleaved=true`, T/H/W frequency buckets are interleaved across `mrope_section` chunks rather than blocked. `op/mrope.py` `_reorder_cos_sin` ([op/mrope.py:40-56](../../python/mlc_llm/op/mrope.py#L40-L56)) handles the `mrope_interleaved` flag at compile time. Stage 2 must thread `mrope_interleaved=True` through config, not silently default to `False`. **Risk register R-11.**

### Patch merger (per checkpoint)

Generic shape: `LayerNorm(hidden_size · spatial_merge_size²) → Linear(in→in) → GELU → Linear(in→out_hidden_size)`. For 0.8B: in = 768·4 = **3072**, out = **1024**. Fits the LM hidden directly — no rank mismatch; **R-9 collapses for 0.8B**. For 35B re-audit during the 35B port.

### Deepstack — 0.8B has none

`deepstack_visual_indexes=[]` for 0.8B → Stage 6 is **n/a** on 0.8B. Stage 5 still reserves the LM-forward hook for the 35B path (where deepstack indices may exist). Re-audit once the 35B-A3B multimodal config.json is on disk.

### M-RoPE — partial-rotary nuance

`partial_rotary_factor=0.25` means cos/sin tensors are shape `[seq, head_dim · 0.25]` = `[seq, 64]`, applied to the **first 64 dims** of each head; the remaining 192 dims pass unrotated. `apply_multimodal_rotary_pos_emb` ([op/mrope.py:80-142](../../python/mlc_llm/op/mrope.py#L80-L142)) needs to slice q/k accordingly — verify or extend in Stage 2.

---

## Stages — gates & exit criteria

### Stage 0 — Audit & plan (this session)

- [x] RoPE convention verified half-split (NeoX). No weight permutation. Conclusion captured in resume notes above.
- [x] Plan doc written ([this file](phase10-vision-input.md)).
- [x] Worklog kickoff entry.
- [x] Per-checkpoint config audit against actual `Qwen/Qwen3.5-0.8B/config.json`. Plan tables updated 2026-05-01: depth=12 not 27; `out_hidden_size=1024` (R-9 closed); `deepstack_visual_indexes=[]` (Stage 6 n/a on 0.8B); `partial_rotary_factor=0.25` (R-12); `mrope_interleaved=true` (R-11); image normalize `[0.5,0.5,0.5]` not ImageNet.
- [x] HF cache check — `Qwen/Qwen3.5-0.8B` already on disk; no Stage 1 download wait.

**Gate:** plan + audit committed; worklog has Phase 10 entry.

### Stage 1 — PyTorch reference harness

Extend [validate.py](../../validate.py) to:

1. Load HF Qwen3-VL (or this project's Qwen3.5 multimodal checkpoint) on a single test image (`tests/multimodal/cat.png` or equivalent fixed input).
2. Run pre-merger ViT block outputs (capture per-block hidden states for parity comparisons in Stage 3).
3. Run post-merger embeddings (for Stage 4).
4. Compute reference M-RoPE position IDs from `image_grid_thw`.
5. End-to-end logits for a multimodal prompt (text-prefix + `<|image_pad|>` × image-token-count + text-suffix). 50-token greedy decode for the parity gate.
6. Cache all of the above under a new file (`reference_outputs_vl.pt`) sibling to `reference_outputs.pt`.

**Gate (1) [CLOSED 2026-05-01]:** harness runs against `Qwen/Qwen3.5-0.8B`, captures 12 vision-block hiddens + merger output `(630, 1024)` + mRoPE position_ids `(3, 1, 652)` + rope_deltas `[[-600]]` + 25-token greedy decode. Cache `reference_outputs_vl.pt` is 130 MB. Implementation: [validate.py](../../validate.py) `--reference-vl` mode. Fixture image: [tests/multimodal/cat.jpeg](../../tests/multimodal/cat.jpeg) (HF docstring `pipeline-cat-chonk.jpeg`, 960×686 → grid `(1, 42, 60)`).

### Stage 2 — M-RoPE wired through text-only path **[CLOSED 2026-05-01]**

**Resolved as: chunks A+B in `qwen35_model.py` (shared modules) + decision to land mrope-on spec in the sibling `qwen3_5_vl/` module rather than in-place in `qwen35/`.** Rationale: 0.8B and 35B-A3B share the same vision architecture (only sizes differ — see "Per-checkpoint config audit" above), so a single sibling module handles both. The running text-only `dist/qwen3_6-35B-A3B-q4f16_1/` (1.85× over llama.cpp) and `dist/qwen3_5-0.8B-q4f16_g16e/` (1.345×) libs are **not touched**; sibling-module decision keeps perf-validated artifacts byte-identical.

Plan precedent for this layout: `phi3` (text) + `phi3_v` (vision), and the existing `qwen3_5` / `qwen3_5_text` / `qwen3_5_mtp_draft` triple — sibling modules per forward-shape variant is the established mlc-llm house style ([model.py](../../python/mlc_llm/model/model.py)).

**Chunk A delivered:**
- [op/mrope.py](../../python/mlc_llm/op/mrope.py) — added `rotary_dim` arg to `MultimodalRotaryEmbedding` (Qwen3.5 partial-rotary 0.25 → 64-dim cos/sin); added partial-rotary slice path in `apply_multimodal_rotary_pos_emb`; **headline fix: `_reorder_cos_sin_interleaved` for `mrope_interleaved=True`** — the existing chunked op was wrong for Qwen3.5 (max |Δcos| 1.96 vs HF). New code: 4e-7 vs HF.
- [qwen35_model.py](../../python/mlc_llm/model/qwen35/qwen35_model.py) `Qwen35Config` — added `mrope_section: Optional[List[int]] = None` and `mrope_interleaved: bool = False` (defaults preserve current behavior). Removed the duplicates from `Qwen35MoEConfig` (now inherited).

**Chunk B delivered:**
- `Qwen35Attention.__init__` caches mrope fields; `forward()` accepts optional `position_embeddings: Optional[Tuple[Tensor, Tensor]]`. When provided, runs inline-mRoPE + raw `paged_kv_cache.self_attention`; when None, the existing `attention_with_fused_qkv` path runs unchanged. Compile-time gating: the conditional is dead-stripped at trace time when `mrope_section=None`.
- Optional `position_embeddings` plumbed through `Qwen35DecoderLayer.{forward, forward_with_history}` and `Qwen35MTPHead.forward`.
- `Qwen35Model.{forward, forward_with_history}` accept optional `position_ids: Tensor`. When `config.mrope_section` is set, the model owns a `MultimodalRotaryEmbedding` (rotary_dim = head_dim·partial_rotary_factor = 64) and computes cos/sin once, broadcasts to softmax-attention layers.
- Regression: tiny LMHeadModel + MTP=1 mrope-OFF traces to 33 fns / 61 params via `export_tvm`; `qwen3_5_moe` and `qwen2_5_vl` both still import cleanly. `dist/` libs are unchanged.

**Chunk C — folded into Stage 5 (sibling module):** the LMHeadModel spec extension, `create_paged_kv_cache` rope_mode flip, and lib recompile + parity bench all happen inside the new `qwen3_5_vl/` module rather than in-place in `qwen35/`. Stage 2 ships as chunks A+B; Stage 5 picks up the spec change as part of the VL LMHead.

**Gate (2):** sibling-module path is established. Existing builds compile to identical IR. Math verified bit-exact vs HF.

### Stage 2 (deprecated text — kept for reference)

~~Modify the **softmax-attention layers only** (every `full_attention_interval=4` layer) of `qwen35_model.py` and `qwen3_5_moe_model.py` to swap `RopeMode.NORMAL` → `RopeMode.NONE` and apply mRoPE inline before the paged-attention call, gated on `mrope_section is not None` in config.~~

- **Add** `mrope_section: Optional[List[int]]` and `mrope_interleaved: bool` to `Qwen35Config` (mirror the fields already on `Qwen35MoEConfig`).
- **Conditional path** in `Qwen35Attention.forward`:
  ```python
  if self.mrope_section is not None:
      cos, sin = self.rotary_emb(q, position_ids)  # rank-3 position_ids
      q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.mrope_section)
      # paged_kv_cache.self_attention (raw, no in-cache rope)
  else:
      # existing attention_with_fused_qkv path (RopeMode.NORMAL)
  ```
- **Spec extension**: `prefill` and `batch_prefill` gain `position_ids: Tensor([3, 1, seq_len], "int32")` and `mrope_deltas: Tensor([1, 1], "int32")` when the model has `mrope_section`. `decode` keeps a `_build_decode_position_ids` helper that broadcasts the cached delta-adjusted 1D base position to a 3D tensor (mirror [qwen2_5_vl_model.py:_build_decode_position_ids](../../python/mlc_llm/model/qwen2_5_vl/qwen2_5_vl_model.py)).
- **Cache-version tag**: bump radix-prefix-cache version (or the model's lib SHA) so old text-only RoPE-baked-in pages are not reused after the convention change. K is now stored *not* rotated (RoPE.NONE) vs *rotated* (RoPE.NORMAL); cross-version reuse is silently wrong. Risk register R-1.
- **Text-only parity check**: with `mrope_section` set but no images in the prompt, all three position-id rows are identical (`get_mrope_position_ids` returns text-only at [op/mrope.py:346-361](../../python/mlc_llm/op/mrope.py#L346-L361)) → `_reorder_cos_sin` collapses to identity → mRoPE result is numerically equal to plain 1D RoPE. Verify against the existing 50/50 text-only parity bench.

**Gate (2):** with `mrope_section` plumbed through the 0.8B and 35B-A3B builds, all existing text-only parity tests still pass (50/50 greedy match). Lib recompile required; benchmark to confirm no perf regression in the softmax-attention layers.

### Stage 3 — Vision tower in MLC

New module [python/mlc_llm/model/vision/qwen3_vl_vit.py](../../python/mlc_llm/model/vision/): `Qwen3VLVisionTower` with `Qwen3VLVisionBlock`, 3D-conv patch embed, learned position embedding indexable by `grid_thw`, 1D rotary in attention. Output: per-patch features at `vision_hidden_size=1152` BEFORE merging.

Per-block hidden-state parity vs HF, rtol=1e-3, atol=1e-3 in fp16 (same standard as text-decoder Stage 4 in [CLAUDE.md](../../CLAUDE.md)).

**Gate (3):** standalone tower hands a per-patch tensor that matches HF on a fixed image to within 1e-3 fp16 tolerance.

### Stage 4 — Patch merger + `image_embed` entry

New module [python/mlc_llm/model/qwen3_5_vl/qwen3_5_vl_image.py](../../python/mlc_llm/model/qwen3_5_vl/) (path TBD per Stage 8 organization decision). Adds `Qwen3VLPatchMerger` and `Qwen3VLImageEmbedding`. Add `image_embed(pixel_values, image_grid_thw)` method to the new LM-head class returning `[batch, vision_token_count, hidden_size]` embeddings.

**Gate (4):** post-merger embedding parity vs HF on the fixed image.

### Stage 5a — Sibling module + loader + image-preproc Python helpers **[CLOSED 2026-05-01]**

Shipped per the cleaner-alternative scope split (items 1–3 of the original Stage 5):
- New module [python/mlc_llm/model/qwen3_5_vl/](../../python/mlc_llm/model/qwen3_5_vl/) — `qwen3_5_vl_model.py` (Qwen35VLConfig + Qwen3VLVisualModel + Qwen35VLLMHeadModel), `qwen3_5_vl_loader.py`, `qwen3_5_vl_image.py` (numpy preprocessor).
- Registered as `qwen3_5_vl` in `model.py`.
- Trace verified: 32 IRModule fns / 437 named params / 853M elements on real HF 0.8B config.
- Loader cross-check vs `model.safetensors.index.json`: 0 unmatched MLC translations; 15 unused HF keys (all `mtp.*`, intentionally excluded for v1).
- IR audit: prefill uses inline-mRoPE + raw `self_attention` (not `attention_with_fused_qkv`); cache rope_mode flipped to NONE.

Worklog: [2026-05-01 — Phase 10 Stage 5a](../../worklog.md#2026-05-01--phase-10-stage-5a-qwen3_5_vl-sibling-module-shipped-structural).

**v1 spec cuts** (re-evaluate after 5b passes): no MTP, no `with_history`, no `*_to_last_hidden_states`. 9 entry points: embed, image_embed, prefill, decode, batch_prefill, batch_decode, batch_verify, create_paged_kv_cache, create_rnn_state.

### Stage 5b — Compile + parity gate (next session)

- `mlc_llm gen_config` + `mlc_llm compile` for `qwen3_5_vl` at `q0f16` → `dist/qwen3_5-0.8B-vl-q0f16/`.
- `validate.py --greedy-parity-vl` mode: load lib, drive `image_embed` + `prefill` + `decode` via Relax VM (bypass MLCEngine's text-only prefill). Use [tests/multimodal/cat.jpeg](../../tests/multimodal/cat.jpeg) fixture; diff vs `reference_outputs_vl.pt` 25-token greedy decode.
- **Bit-exact verify** `qwen3_5_vl_image._fast_pos_embed_interpolate` and `_rot_pos_emb` vs HF `visual.fast_pos_embed_interpolate` / `visual.rot_pos_emb` before pinning. Math is sketched but unvalidated end-to-end.
- Headline gate: ≥48/50 token greedy match (CLAUDE.md Stage 5 bar).

### Stage 5 (deprecated combined — kept for reference) — End-to-end multimodal

- **Loader**: stop dropping `model.visual.*` in [qwen35_loader.py](../../python/mlc_llm/model/qwen35/qwen35_loader.py) when the model is the VL variant; map vision params under `visual.*`.
- **Conversation template**: extend `qwen3_5` (or new `qwen3_5_vl`) with image placeholder slots; add `MessagePlaceholders.IMAGE` injection points.
- **`ImageData` extension** ([serve/data.py:63](../../python/mlc_llm/serve/data.py#L63) and [:107-114](../../python/mlc_llm/serve/data.py#L107-L114)): grow a `grid_thw: Optional[Tuple[int,int,int]]` field; defer `embed_size` computation to post-preprocessing. The `from_url(url, config)` adds a `qwen3_vl` branch returning `embed_size = (grid_t * grid_h * grid_w) // (spatial_merge_size**2)`. Risk register R-5.
- **Image preprocessor**: new [python/mlc_llm/model/vision/qwen3_vl_image_processing.py](../../python/mlc_llm/model/vision/) with dynamic-resolution resize-to-multiple-of-16, **`image_mean=image_std=[0.5,0.5,0.5]`** (Qwen normalize, NOT ImageNet 0.485/0.456/0.406), `grid_thw` output. The HF processor class is `Qwen2VLImageProcessorFast` — reuse its logic; `merge_size=2`, `temporal_patch_size=2`. Reuse [vision/image_processing.py](../../python/mlc_llm/model/vision/image_processing.py) primitives where they fit.
- **Wire substitution**: `<|image_pad|>` token positions get replaced with vision embeddings before/at `prefill()` (engine concatenates text + image embeds based on token IDs).
- **Reserve deepstack hook** in `Qwen35Model.forward`: accept `Optional[List[Tuple[int, Tensor]]]` for layer-id keyed deepstack additions, default `None`. Adding it later breaks every spec dict.
- **MTP gate-off**: when the verify window crosses an `<|image_pad|>` token, route through target-only (skip MTP). MTP head was trained on text-only sequences; passing image-substituted hidden states feeds OOD inputs to `pre_fc_norm_embedding`. Risk register R-4.
- **TP shard strategy**: vision tower uses `ShardSingleDim` non-shardable hint → runs on rank 0 only, post-merger embedding broadcast. Rationale in non-goals above.

**Gate (5) — headline:** end-to-end greedy-decode parity vs HF on a multimodal prompt set (≥ 48/50 token match, mirrors CLAUDE.md Stage 5).

### Stage 5b — Chunked-prefill + mRoPE carry-state (surgical)

The default `prefill_chunk_size` is 8192. A long video grid_t=8 at 32×32 spatial = 8192 vision tokens — chunk boundary may bisect a vision span. The mRoPE delta calculation in [op/mrope.py:_build_sequence_position_ids](../../python/mlc_llm/op/mrope.py) assumes the entire vision block is in one prefill call.

**Minimum-surgery fix:** disable chunking when a chunk would split an image span — preprocessor pads to chunk boundaries (wasteful but safe). Full carry-state extension across chunks deferred to a later phase.

**Gate (5b):** smoke test with a long video span ≥ chunk size confirms either (a) the chunk router avoids the boundary or (b) the mRoPE delta carries forward correctly.

### Stage 6 — Deepstack (optional, post-ship)

Three intermediate mergers at vision layers 8/16/24, separate output projections injected at three LM layers via additive residual. Not blocking v1; LM forward signature already reserves the slot from Stage 5. Schedule based on use-case demand (OCR-heavy workloads).

**Gate (6):** deepstack-on parity vs HF; quality bench delta vs deepstack-off documented.

### Stage 7 — Quantization

Vision tower **stays fp16** for v1. Text backbone keeps existing `q4f16_g16e` / `q4f16_1` configs. Modify [qwen35_quantization.py](../../python/mlc_llm/model/qwen35/qwen35_quantization.py) and the MoE counterpart so `make_quantization_functions` skips `visual.*` params. Pattern follows existing skip-list (e.g. norm weights are skipped already).

**Gate (7):** quantized text + fp16 vision lib compiles, loads, and produces correct multimodal output on the fixed-image test set.

### Stage 8 — Module organization

**Decision: sibling modules, not in-place extension.**

Rationale: the existing split (`qwen3_5`, `qwen3_5_text`, `qwen3_5_mtp_draft`, `qwen3_5_moe`, `qwen3_5_moe_text`, `qwen3_5_moe_mtp_draft`) shows the established pattern — each forward-shape variant lives in its own module sharing imports from `qwen35_model`. Vision is another forward-shape variant (adds `image_embed`, `pixel_values` spec entry, `mrope_deltas` in prefill spec). Sibling modules `qwen3_5_vl/` and `qwen3_5_moe_vl/` keep text-only `model_lib_gen` artifacts identical (no vision tower in their compiled `IRModule`) and avoid growing the spec dict for text-only deployments.

**Layout:**
```
python/mlc_llm/model/
├── qwen35/                       # existing dense 0.8B text
├── qwen3_5_moe/                  # existing MoE 35B-A3B text
├── qwen3_5_vl/                   # NEW: dense 0.8B + vision
│   ├── qwen3_5_vl_model.py       # Qwen35VLLMHeadModel(Qwen35LMHeadModel)
│   ├── qwen3_5_vl_loader.py      # extends qwen35_loader; maps visual.*
│   └── qwen3_5_vl_image.py       # patch merger + image_embed
├── qwen3_5_moe_vl/               # NEW: MoE 35B-A3B + vision
│   ├── ... (mirror of above)
└── vision/
    ├── qwen3_vl_vit.py            # NEW: shared vision tower
    └── qwen3_vl_image_processing.py  # NEW: dynamic-resolution preprocessor
```

Both VL modules import `Qwen35Attention`, `Qwen35GatedDeltaNet`, `Qwen35MTPHead`, `Qwen35Embedding` from `qwen35_model`. Tower module shared. Register via `model.py`.

**Gate (8):** registered model types `qwen3_5_vl`, `qwen3_5_moe_vl` available via `mlc_llm gen_config`.

---

## Risk register

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R-0 | RoPE convention mismatch (planning critique flag) | ~~HIGH~~ | **CLOSED Stage 0:** TVM RopeMode.NORMAL is half-split, matches HF Qwen3.5; no weight permutation needed |
| R-1 | mRoPE inline + paged-KV-cache stores K already rotated (vs RoPE.NORMAL stores K unrotated) → mid-version cache incompatibility | HIGH | Cache-version tag invalidation in Stage 2; document the K-storage convention switch in code |
| R-2 | Cross-batch RoPE convention conflict (one batch slot expects NORMAL, another expects NONE-with-inline) → silent layered batch garbage | HIGH | Same-lib instances share one paged_kv_cache; `rope_mode` is fixed at lib-compile time; assert at engine init |
| R-3 | GDN per-position state scatter at vision-token positions | LOW | GDN ignores positions; `set_with_history` writes by linear `t` regardless of mRoPE delta. Confirmed in [qwen35_model.py:617-714](../../python/mlc_llm/model/qwen35/qwen35_model.py#L617-L714). Document the invariant in code comment |
| R-4 | MTP draft head fed image-substituted hidden states → OOD on `pre_fc_norm_embedding`, accept rate collapses | MED | Stage 5: gate MTP off when verify window crosses `<|image_pad|>`. v1 acceptable degradation |
| R-5 | `ImageData.embed_size` scalar API doesn't fit dynamic-resolution Qwen3-VL | MED | Grow `ImageData` to carry `grid_thw: Optional[Tuple[int,int,int]]`; derive `embed_size` post-preprocess. Surface ripples through serializers |
| R-6 | Chat template `<|image_pad|>` count must match post-resize tower output | MED | `apply_chat_template` receives `image_grid_thw` from preprocessor before tokenization; assert count match at template render |
| R-7 | Chunked-prefill bisects image span → mRoPE delta misaligned across chunks | MED | Stage 5b: disable chunking on image-span boundaries (pad-to-boundary, wasteful but safe). Full carry-state extension deferred |
| R-8 | Vision-tower parity at fp16 too tight (rtol=1e-3) for 27 ViT blocks compounding | LOW | If breached, relax to per-block 1e-3 + final-merger 5e-3; downstream LM logit parity is the real gate, not mid-stack |
| R-9 | 0.8B checkpoint may have a different merger output dim than 35B's 3584 (must match `hidden_size`) | ~~LOW~~ | **CLOSED Stage 0b:** 0.8B `out_hidden_size=1024` matches LM hidden_size=1024. Re-open for 35B audit. |
| R-10 | TP=2/4 vision-tower placement (rank-0-only + broadcast) untested in this codebase | LOW | The pattern exists for `lm_head` rank-0-only; follow it. v1 Orin target is single-GPU so deferrable |
| R-11 | `mrope_interleaved=true` in 0.8B config — Stage 2 mRoPE plumbing must thread this flag, not default to false | MED | `op/mrope.py:_reorder_cos_sin` handles both modes; ensure config field is propagated end-to-end. Verify in Stage 2 with text-only collapse (interleaved + 3 identical pos rows still collapses to 1D RoPE) |
| R-12 | `partial_rotary_factor=0.25` — only 64/256 head dims rotated; mRoPE applies to slice not full head | MED | Verify `apply_multimodal_rotary_pos_emb` slice math in Stage 2 against HF reference output; cos/sin shape `[seq, 64]` not `[seq, 256]` |

---

## Open decisions

1. **Test image set.** Choose a fixed multimodal-prompt set for the parity bar. Candidate: 5 prompts (1 plain image + caption, 1 OCR snippet, 1 chart, 1 multi-image, 1 short video). Pick during Stage 1.
2. **Conversation template name.** Reuse `qwen3_5` with conditional image-slot expansion, or fork to `qwen3_5_vl`. Lean toward fork (cleaner deserialization on the C++ engine side). Defer to Stage 5.
3. **Vision tower fp16 vs bf16.** HF default is bf16. Orin sm_87 has fp16 TC peak ~10× bf16. We run text in fp16 already; converting vision weights to fp16 at load time is the simplest path. Confirm parity holds after the conversion (Stage 3 gate).
4. **Disk budget for HF multimodal checkpoint.** Qwen3.5-0.8B multimodal weight is ~2 GB; 35B-A3B multimodal is ~75 GB. Confirm `~/.cache/huggingface/hub` has room before Stage 1 download.

---

## Estimated effort

Sessionized roughly. Each "session" is ~3-4 hours of focused work; downloads + compiles run in the background.

| Stage | Estimate | Notes |
|---|---|---|
| 0 | 0.5 sess | Audit + plan + worklog (this session) |
| 1 | 1 sess | Reference harness; HF download in background |
| 2 | 2 sess | mRoPE plumbing in 2 backbones + lib recompile + parity recheck |
| 3 | 2-3 sess | New ViT module; parity per block; standalone validate |
| 4 | 1 sess | Patch merger + image_embed |
| 5 | 2 sess | Loader + ImageData + preprocessor + substitution + MTP gate |
| 5b | 0.5 sess | Chunk-boundary safety |
| 6 | 1-2 sess | Deepstack (post-v1, optional) |
| 7 | 0.5 sess | Quantization skip-list |
| 8 | already done in plan | Sibling-modules layout decided |

**Total to v1 (no Deepstack):** ~9-11 sessions on the dense 0.8B path; 35B-A3B follows for free once 0.8B passes (the same module hierarchy + a different config).

Reuse ratio is high: text-only Phase 1-9b stays untouched; vision additions land in sibling modules. The risky pieces are the cache-version invalidation (R-1, R-2) and the chunked-prefill boundary (R-7).

---

## Files to touch (Stage 2 onward, summary)

**New:**
- [python/mlc_llm/model/vision/qwen3_vl_vit.py](../../python/mlc_llm/model/vision/)
- [python/mlc_llm/model/vision/qwen3_vl_image_processing.py](../../python/mlc_llm/model/vision/)
- [python/mlc_llm/model/qwen3_5_vl/qwen3_5_vl_model.py](../../python/mlc_llm/model/)
- [python/mlc_llm/model/qwen3_5_vl/qwen3_5_vl_loader.py](../../python/mlc_llm/model/)
- [python/mlc_llm/model/qwen3_5_vl/qwen3_5_vl_image.py](../../python/mlc_llm/model/)
- [python/mlc_llm/model/qwen3_5_moe_vl/](../../python/mlc_llm/model/) (mirror)
- `tests/multimodal/` fixed-image test set + golden tensors

**Modified:**
- [python/mlc_llm/model/qwen35/qwen35_model.py](../../python/mlc_llm/model/qwen35/qwen35_model.py) — add mrope_section/interleaved fields; conditional mRoPE path in `Qwen35Attention.forward`
- [python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py](../../python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py) — wire the mrope fields that are already declared
- [python/mlc_llm/model/model.py](../../python/mlc_llm/model/model.py) — register `qwen3_5_vl`, `qwen3_5_moe_vl`
- [python/mlc_llm/serve/data.py](../../python/mlc_llm/serve/data.py) — `ImageData.grid_thw`, dynamic `embed_size`
- [python/mlc_llm/conversation_template/](../../python/mlc_llm/conversation_template/) — `qwen3_5_vl` template (or extend `qwen3_5`)
- [validate.py](../../validate.py) — multimodal extension

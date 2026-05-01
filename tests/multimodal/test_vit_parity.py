#!/usr/bin/env python3
"""
Phase 10 Stage 3 parity gate — numpy clone of the MLC vision tower vs HF.

Loads HF ``Qwen/Qwen3.5-0.8B`` to capture (a) the visual.* weights and (b) the
deterministic preprocessing helpers (``fast_pos_embed_interpolate``,
``rot_pos_emb``). Runs a numpy implementation that mirrors
``python/mlc_llm/model/vision/qwen3_vl_vit.py`` op-for-op, then diffs each
per-block output against the cached HF ``vision_block_outputs`` from
``reference_outputs_vl.pt``.

If the numpy clone matches HF to fp16 tolerance, the MLC tower is highly
likely correct: the Relax ops in the MLC version are 1:1 with the numpy ones
here. A separate Stage 5 step (real lib compile + run) catches TVM codegen
bugs.

Usage:
    .venv/bin/python tests/multimodal/test_vit_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "reference_outputs_vl.pt"
TOL_RTOL = 1e-3
TOL_ATOL = 5e-3  # ViT blocks compound; relax slightly from the LM 1e-3 bar


# ─────────────────────────────────────────────────────────────────────────────
# Numpy clone of qwen3_vl_vit.py (op-for-op).
# ─────────────────────────────────────────────────────────────────────────────


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    # x: (..., d)
    x32 = x.astype(np.float32)
    mean = x32.mean(axis=-1, keepdims=True)
    var = x32.var(axis=-1, keepdims=True)
    out = (x32 - mean) / np.sqrt(var + eps)
    out = out * weight.astype(np.float32) + bias.astype(np.float32)
    return out.astype(x.dtype)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _patch_embed(pixel_values: np.ndarray, weights: dict, cfg: dict) -> np.ndarray:
    """Conv3D collapses (T, H, W) patches → one token each."""
    n = pixel_values.shape[0]
    in_ch = cfg["in_channels"]
    tps = cfg["temporal_patch_size"]
    ps = cfg["patch_size"]
    embed = cfg["hidden_size"]
    # (n, in_ch, tps, ps, ps)
    x = pixel_values.reshape(n, in_ch, tps, ps, ps).astype(np.float32)
    # Conv3D weight: (embed, in_ch, tps, ps, ps). Stride==kernel==input-spatial → output (n, embed, 1, 1, 1).
    # Equivalent to flatten + matmul.
    w = weights["proj.weight"].astype(np.float32)  # (embed, in_ch, tps, ps, ps)
    b = weights["proj.bias"].astype(np.float32)    # (embed,)
    w_flat = w.reshape(embed, -1)                  # (embed, in_ch*tps*ps*ps)
    x_flat = x.reshape(n, -1)                      # (n, in_ch*tps*ps*ps)
    return (x_flat @ w_flat.T + b).astype(np.float16)


def _attention(
    hidden_states: np.ndarray,
    rotary_cos: np.ndarray,
    rotary_sin: np.ndarray,
    weights: dict,
    cfg: dict,
) -> np.ndarray:
    """Self-attention without causal mask, full 1D rotary on q/k.

    Mirrors HF's eager + apply_rotary_pos_emb_vision: rotation and the
    softmax-attention math run in fp32; only the input/output projections
    use the model dtype. fp16 rounding accumulates badly across 12 blocks
    if rotation runs in fp16 (saw max diff 0.25 at block 5).
    """
    s, dim = hidden_states.shape
    h = cfg["num_heads"]
    d = dim // h
    qkv_w = weights["qkv.weight"]   # (dim*3, dim)
    qkv_b = weights["qkv.bias"]
    proj_w = weights["proj.weight"]
    proj_b = weights["proj.bias"]

    qkv = hidden_states @ qkv_w.T + qkv_b                   # (s, dim*3) — fp16 dtype follows hidden_states
    qkv = qkv.reshape(s, 3, h, d).transpose(1, 2, 0, 3)     # (3, h, s, d)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # HF apply_rotary_pos_emb_vision: cast to fp32 for rotation, back to original dtype
    q32 = q.astype(np.float32)
    k32 = k.astype(np.float32)
    cos = rotary_cos[None, ...].astype(np.float32)          # (1, s, d)
    sin = rotary_sin[None, ...].astype(np.float32)
    q32 = q32 * cos + _rotate_half(q32) * sin
    k32 = k32 * cos + _rotate_half(k32) * sin
    q = q32.astype(hidden_states.dtype)
    k = k32.astype(hidden_states.dtype)

    # Eager attention in fp32 throughout (matches HF eager_attention_forward).
    # Earlier experiment confirmed: casting attn-weights back to fp16 before V
    # matmul makes parity DRAMATICALLY worse — HF keeps fp32 through to V.
    scale = d ** -0.5
    attn = np.einsum("hsd,htd->hst", q.astype(np.float32), k.astype(np.float32)) * scale
    attn = attn - attn.max(axis=-1, keepdims=True)
    attn = np.exp(attn)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    out = np.einsum("hst,htd->hsd", attn, v.astype(np.float32)).astype(hidden_states.dtype)
    out = out.transpose(1, 0, 2).reshape(s, dim)
    return out @ proj_w.T + proj_b


def _mlp(hidden_states: np.ndarray, weights: dict) -> np.ndarray:
    fc1_w = weights["linear_fc1.weight"]
    fc1_b = weights["linear_fc1.bias"]
    fc2_w = weights["linear_fc2.weight"]
    fc2_b = weights["linear_fc2.bias"]
    x = hidden_states @ fc1_w.T + fc1_b
    x = _gelu_tanh(x.astype(np.float32)).astype(hidden_states.dtype)
    return x @ fc2_w.T + fc2_b


def _block(
    hidden_states: np.ndarray,
    rotary_cos: np.ndarray,
    rotary_sin: np.ndarray,
    weights: dict,
    cfg: dict,
) -> np.ndarray:
    eps = cfg["layer_norm_eps"]
    attn_in = _layer_norm(hidden_states, weights["norm1.weight"], weights["norm1.bias"], eps)
    hidden_states = hidden_states + _attention(attn_in, rotary_cos, rotary_sin, _sub(weights, "attn"), cfg)
    mlp_in = _layer_norm(hidden_states, weights["norm2.weight"], weights["norm2.bias"], eps)
    hidden_states = hidden_states + _mlp(mlp_in, _sub(weights, "mlp"))
    return hidden_states


def _sub(weights: dict, prefix: str) -> dict:
    pfx = prefix + "."
    return {k[len(pfx):]: v for k, v in weights.items() if k.startswith(pfx)}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    if not CACHE.exists():
        print(f"[parity] No cache at {CACHE}. Run validate.py --reference-vl first.")
        return 1
    cache = torch.load(CACHE, weights_only=False)
    pixel_values = cache["pixel_values"] if "pixel_values" in cache else None
    image_grid_thw = cache["image_grid_thw"]
    vision_block_outputs = cache["vision_block_outputs"]
    n_blocks = len(vision_block_outputs)
    print(f"[parity] cache loaded: image_grid_thw={image_grid_thw.tolist()}, {n_blocks} block tensors")
    print(f"[parity] block 0 ref shape: {vision_block_outputs[0].shape}, dtype: {vision_block_outputs[0].dtype}")

    # Cache stored only the SHAPE of pixel_values, not the values. Need to redo the
    # processing to get the actual tensor. Cheap — just one image preprocessor pass.
    print("[parity] re-running HF processor to recover pixel_values + visual weights ...")
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    model_id = cache["model_id"]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    image = Image.open(cache["image_path"]).convert("RGB")
    inputs = processor(text=["dummy"], images=[image], return_tensors="pt")
    pixel_values_torch = inputs["pixel_values"]
    grid_thw_torch = inputs["image_grid_thw"]

    # Load model on CPU just to grab visual weights + run the deterministic preproc.
    print("[parity] loading HF model (CPU; visual only) ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    visual = model.model.visual
    head_dim = visual.config.hidden_size // visual.config.num_heads

    # Compute pos_embeds + rotary via HF helpers
    with torch.no_grad():
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw_torch)              # (N, hidden)
        rotary_pos_emb = visual.rot_pos_emb(grid_thw_torch)                         # (N, head_dim)
        emb = torch.cat([rotary_pos_emb, rotary_pos_emb], dim=-1)
        rotary_cos = emb.cos().to(torch.float16).numpy()                            # (N, head_dim)
        rotary_sin = emb.sin().to(torch.float16).numpy()
        pos_embeds_np = pos_embeds.to(torch.float16).numpy()
        pixel_values_np = pixel_values_torch.to(torch.float16).numpy()

    print(f"[parity] preproc shapes: pixel_values={pixel_values_np.shape} pos_embeds={pos_embeds_np.shape} cos={rotary_cos.shape}")

    # Extract visual weights into a flat dict — fp16 already (we cast model.half).
    visual_weights = {k: v.detach().cpu().numpy() for k, v in visual.state_dict().items()}
    print(f"[parity] visual state_dict keys: {len(visual_weights)} tensors")

    # Build cfg
    cfg = {
        "in_channels": visual.config.in_channels,
        "temporal_patch_size": visual.config.temporal_patch_size,
        "patch_size": visual.config.patch_size,
        "hidden_size": visual.config.hidden_size,
        "num_heads": visual.config.num_heads,
        "layer_norm_eps": 1e-6,
    }

    # Run numpy clone, comparing per-block.
    print("\n[parity] === per-block diff vs HF cache ===")
    h = _patch_embed(pixel_values_np, _sub(visual_weights, "patch_embed"), cfg)
    h = h + pos_embeds_np
    print(f"[parity] post-patch-embed+pos_embed: shape={h.shape} dtype={h.dtype}")

    # Per-block magnitudes grow rapidly (block 0: |x|≤3, block 5: |x|≤93,
    # block 11: |x|≤2464). Use scale-aware tolerance: 1% of the per-block
    # abs_max plus a 1e-3 floor for blocks with small magnitudes.
    import time
    all_pass = True
    for i in range(n_blocks):
        t0 = time.time()
        block_w = _sub(visual_weights, f"blocks.{i}")
        h = _block(h, rotary_cos, rotary_sin, block_w, cfg)
        elapsed = time.time() - t0
        if not np.isfinite(h).all():
            print(f"  block {i:2d}  [FAIL]  produced non-finite values "
                  f"(nan={np.isnan(h).sum()}, inf={np.isinf(h).sum()})  in {elapsed:.1f}s",
                  flush=True)
            all_pass = False
            continue
        ref = vision_block_outputs[i].squeeze().astype(np.float32)
        ours = h.astype(np.float32)
        diff = np.abs(ours - ref)
        ref_scale = max(np.abs(ref).max(), 1.0)
        rel = diff.max() / ref_scale
        ok = rel <= 1e-2 and diff.mean() <= 5e-3
        status = "PASS" if ok else "FAIL"
        line = (
            f"  block {i:2d}  [{status}]  "
            f"max={diff.max():.3e}  mean={diff.mean():.3e}  rel={rel:.2e}  "
            f"|ref|max={ref_scale:.2e}  ({elapsed:.1f}s)"
        )
        print(line, flush=True)
        sys.stdout.flush()
        if not ok:
            all_pass = False
    print(flush=True)
    print("[parity] per-block OVERALL:", "PASS" if all_pass else "FAIL", flush=True)

    # === Merger parity ===
    # The merger is what the LM actually consumes — the real downstream gate.
    # Even if block 11 drifts at register tokens (compounded fp16 rounding),
    # the merger LayerNorm + projections may renormalize that away.
    print(flush=True)
    print("[parity] === merger output diff vs HF cache ===", flush=True)
    merger_w = _sub(visual_weights, "merger")
    spatial_merge_size = visual.config.spatial_merge_size
    merge_sq = spatial_merge_size ** 2
    merger_in_dim = cfg["hidden_size"] * merge_sq
    out_hidden = visual.config.out_hidden_size
    eps = cfg["layer_norm_eps"]

    # Pre-shuffle norm (use_postshuffle_norm=False): norm over (N, hidden_size)
    h_normed = _layer_norm(h, merger_w["norm.weight"], merger_w["norm.bias"], eps)
    # Reshape (N, hidden) → (N // merge², hidden · merge²)
    h_grouped = h_normed.reshape(-1, merger_in_dim)
    fc1_w = merger_w["linear_fc1.weight"]
    fc1_b = merger_w["linear_fc1.bias"]
    fc2_w = merger_w["linear_fc2.weight"]
    fc2_b = merger_w["linear_fc2.bias"]
    h_fc1 = h_grouped @ fc1_w.T + fc1_b
    h_act = torch.nn.functional.gelu(torch.from_numpy(h_fc1.astype(np.float32))).numpy().astype(h_fc1.dtype)
    h_out = h_act @ fc2_w.T + fc2_b

    ref_merger = cache["merger_output"].astype(np.float32)
    diff = np.abs(h_out.astype(np.float32) - ref_merger)
    ref_scale = max(np.abs(ref_merger).max(), 1.0)
    rel = diff.max() / ref_scale
    merger_ok = rel <= 1e-2 and diff.mean() <= 5e-3
    print(
        f"  merger    [{'PASS' if merger_ok else 'FAIL'}]  "
        f"max={diff.max():.3e}  mean={diff.mean():.3e}  rel={rel:.2e}  "
        f"|ref|max={ref_scale:.2e}  shape={h_out.shape}",
        flush=True,
    )

    overall = all_pass and merger_ok
    print(flush=True)
    print("[parity] OVERALL:", "PASS" if overall else "FAIL", flush=True)
    sys.stdout.flush()
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

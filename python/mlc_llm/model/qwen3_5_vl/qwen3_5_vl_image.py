"""Qwen3.5-VL image preprocessor (Python-side, pure numpy).

Mirrors the deterministic prep that ``Qwen2VLImageProcessorFast`` performs and
the ``visual.fast_pos_embed_interpolate`` / ``visual.rot_pos_emb`` helpers in
``transformers.models.qwen3_5.modeling_qwen3_5``. Outputs the four tensors the
MLC ``image_embed`` spec entry expects:

  * ``pixel_values``        — shape ``(N, C·T·P·P)`` flattened patches
  * ``pos_embeds``          — shape ``(N, vision_hidden)`` interpolated learned pos embed
  * ``rotary_cos`` / ``sin``— shape ``(N, vision_head_dim)`` per-token vision rotary

plus ``image_grid_thw = (T, H, W)`` for the engine-side substitution path.

Stage 5a scope: math is staged here so the Stage 5b parity bench can call it
without pulling in TVM. The interpolation / rotary derivations were validated
end-to-end against HF in ``tests/multimodal/test_vit_parity.py`` (rel ≤ 1e-2
through 12 blocks, merger output PASS).
"""

from __future__ import annotations

import dataclasses
from typing import Tuple  # noqa: UP035

import numpy as np


# Qwen2VL image normalization (NOT ImageNet)
QWEN_IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
QWEN_IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


@dataclasses.dataclass
class Qwen3VLPreprocOutput:
    """Container for the four ``image_embed`` inputs + grid metadata."""

    pixel_values: np.ndarray  # (N, C*T*P*P) fp16
    pos_embeds: np.ndarray  # (N, hidden) fp16
    rotary_cos: np.ndarray  # (N, head_dim) fp16
    rotary_sin: np.ndarray  # (N, head_dim) fp16
    image_grid_thw: Tuple[int, int, int]  # noqa: UP006

    @property
    def num_patches(self) -> int:
        return self.pixel_values.shape[0]

    def num_image_tokens(self, spatial_merge_size: int) -> int:
        """Number of post-merger LM tokens this image contributes."""
        return self.num_patches // (spatial_merge_size ** 2)


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:  # noqa: UP006
    """Round to multiples of ``factor`` while preserving aspect ratio and total pixel budget.

    Mirrors ``transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize``.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio must be < 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = ((height * width) / max_pixels) ** 0.5
        h_bar = max(factor, int(height / beta // factor) * factor)
        w_bar = max(factor, int(width / beta // factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        h_bar = int(height * beta // factor + 1) * factor
        w_bar = int(width * beta // factor + 1) * factor
    return h_bar, w_bar


def preprocess_image(
    rgb_image: np.ndarray,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    spatial_merge_size: int = 2,
    vision_hidden_size: int = 768,
    vision_num_heads: int = 12,
    num_position_embeddings: int = 2304,
    pos_embed_weight: np.ndarray = None,  # (num_position_embeddings, vision_hidden_size)
    rope_theta: float = 10000.0,
) -> Qwen3VLPreprocOutput:
    """Single-image preprocessor. Defers learned-pos-embed lookup to the caller
    (pos_embed_weight comes from the loaded HF visual.pos_embed.weight).

    Args:
        rgb_image: ``(H, W, 3)`` uint8 array.
        pos_embed_weight: ``(num_position_embeddings, vision_hidden_size)`` fp16/fp32
            learned pos embed grid. The vision config defaults to 48×48 = 2304.
    Returns:
        ``Qwen3VLPreprocOutput`` with all four tensors populated as fp16.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {rgb_image.shape}")
    h0, w0 = rgb_image.shape[:2]
    factor = patch_size * spatial_merge_size  # 32 for 0.8B
    h, w = smart_resize(h0, w0, factor)

    # Match HF Qwen2VLImageProcessorFast: torchvision BICUBIC + antialias=True
    # on a (3, H, W) uint8 tensor. PIL.Image.resize(BICUBIC) does NOT antialias
    # on downscale and produces ~0.3 max-abs deviation post-normalize on the cat
    # fixture; torchvision matches HF bit-exactly.
    import torch as _torch
    import torchvision.transforms.functional as _tvF

    img_u8 = np.transpose(rgb_image, (2, 0, 1))  # (3, h0, w0) uint8
    img_t = _torch.from_numpy(np.ascontiguousarray(img_u8))
    img_t = _tvF.resize(
        img_t, [h, w],
        interpolation=_tvF.InterpolationMode.BICUBIC,
        antialias=True,
    ).to(_torch.float32)
    img = img_t.numpy() / 255.0  # (3, h, w) fp32
    # normalize per-channel
    mean_chw = QWEN_IMAGE_MEAN.reshape(3, 1, 1)
    std_chw = QWEN_IMAGE_STD.reshape(3, 1, 1)
    img = (img - mean_chw) / std_chw  # (3, h, w)

    # Single image → temporal_patch_size copies stacked
    img_t = np.stack([img] * temporal_patch_size, axis=0)  # (T, 3, h, w)
    grid_t = 1
    grid_h = h // patch_size
    grid_w = w // patch_size

    # Reshape + permute to match HF Qwen2VLImageProcessorFast flatten:
    # (grid_t, tps, C, gh/m, m, p, gw/m, m, p) → (grid_t, gh/m, gw/m, m, m, C, tps, p, p)
    # The (C, tps) order matters: HF places channel BEFORE temporal_patch in the
    # innermost tuple, so each flattened patch is laid out (C, tps, p, p) and
    # the conv3d weight (loaded as (embed, C, tps, p, p)) reads the elements
    # in the correct order. Reversing this gives a 0.3 max-abs deviation in
    # pixel_values vs HF.
    patches = img_t.reshape(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )
    # source axes: (grid_t=0, tps=1, C=2, gh/m=3, m_h=4, p_h=5, gw/m=6, m_w=7, p_w=8)
    patches = np.transpose(patches, (0, 3, 6, 4, 7, 2, 1, 5, 8))
    n = grid_t * grid_h * grid_w
    pixel_values = patches.reshape(n, 3 * temporal_patch_size * patch_size * patch_size).astype(
        np.float16
    )

    # ── pos_embed bilinear interpolation ───────────────────────────────────
    pos_embeds = _fast_pos_embed_interpolate(
        grid_thw=(grid_t, grid_h, grid_w),
        spatial_merge_size=spatial_merge_size,
        pos_embed_weight=pos_embed_weight,
        num_position_embeddings=num_position_embeddings,
    )

    # ── 1D rotary cos/sin per token ────────────────────────────────────────
    rotary_cos, rotary_sin = _rot_pos_emb(
        grid_thw=(grid_t, grid_h, grid_w),
        spatial_merge_size=spatial_merge_size,
        vision_hidden_size=vision_hidden_size,
        vision_num_heads=vision_num_heads,
        rope_theta=rope_theta,
    )

    return Qwen3VLPreprocOutput(
        pixel_values=pixel_values,
        pos_embeds=pos_embeds.astype(np.float16),
        rotary_cos=rotary_cos.astype(np.float16),
        rotary_sin=rotary_sin.astype(np.float16),
        image_grid_thw=(grid_t, grid_h, grid_w),
    )


def _fast_pos_embed_interpolate(
    grid_thw: Tuple[int, int, int],  # noqa: UP006
    spatial_merge_size: int,
    pos_embed_weight: np.ndarray,
    num_position_embeddings: int,
) -> np.ndarray:
    """Bilinear-interpolate the learned ``num_pos_embed × num_pos_embed`` grid
    onto the image's per-patch grid. Mirrors HF
    ``Qwen3_5VisionModel.fast_pos_embed_interpolate``.
    """
    if pos_embed_weight is None:
        raise ValueError("pos_embed_weight must be provided (loaded from HF visual.pos_embed.weight)")
    grid_t, grid_h, grid_w = grid_thw
    num_grid_per_side = int(num_position_embeddings ** 0.5)  # 48 for 0.8B
    # Per-token row/col coords on the learned grid
    h_idxs = np.linspace(0, num_grid_per_side - 1, grid_h, dtype=np.float32)
    w_idxs = np.linspace(0, num_grid_per_side - 1, grid_w, dtype=np.float32)

    h0 = np.floor(h_idxs).astype(np.int64)
    h1 = np.minimum(h0 + 1, num_grid_per_side - 1)
    w0 = np.floor(w_idxs).astype(np.int64)
    w1 = np.minimum(w0 + 1, num_grid_per_side - 1)
    dh = (h_idxs - h0).reshape(grid_h, 1)
    dw = (w_idxs - w0).reshape(1, grid_w)

    pe = pos_embed_weight.reshape(num_grid_per_side, num_grid_per_side, -1).astype(np.float32)
    # bilinear blend per-(row, col)
    out = (
        pe[h0[:, None], w0[None, :]] * (1 - dh)[:, :, None] * (1 - dw)[:, :, None]
        + pe[h0[:, None], w1[None, :]] * (1 - dh)[:, :, None] * dw[:, :, None]
        + pe[h1[:, None], w0[None, :]] * dh[:, :, None] * (1 - dw)[:, :, None]
        + pe[h1[:, None], w1[None, :]] * dh[:, :, None] * dw[:, :, None]
    )  # (grid_h, grid_w, hidden)
    # Repeat over temporal patches and apply the merger's row-major reorder.
    out = np.tile(out[None, ...], (grid_t, 1, 1, 1))  # (T, H, W, hidden)
    # Apply spatial-merge reordering: groups of spatial_merge_size × spatial_merge_size
    out = out.reshape(
        grid_t,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        -1,
    )
    out = np.transpose(out, (0, 1, 3, 2, 4, 5))
    out = out.reshape(grid_t * grid_h * grid_w, -1)
    return out


def _rot_pos_emb(
    grid_thw: Tuple[int, int, int],  # noqa: UP006
    spatial_merge_size: int,
    vision_hidden_size: int,
    vision_num_heads: int,
    rope_theta: float,
) -> Tuple[np.ndarray, np.ndarray]:  # noqa: UP006
    """Returns ``(cos, sin)`` of shape ``(N, head_dim)``.

    Mirrors HF ``Qwen3_5VisionModel.rot_pos_emb`` + the ``cos = cat([rope, rope], -1)``
    that the attention path applies before ``emb.cos()``. Algorithm:

    1. ``VisionRotaryEmbedding(dim=head_dim/2)`` builds ``inv_freq`` of len ``head_dim/4``.
    2. ``freqs[g, j] = g * inv_freq[j]``  (shape ``(max_grid, head_dim/4)``).
    3. Per-token ``pos_ids = (h_coord, w_coord)`` after spatial-merge reorder.
    4. ``rope = freqs[pos_ids].flatten(1)``  → shape ``(N, head_dim/2)``.
    5. ``emb = cat([rope, rope], -1)``      → shape ``(N, head_dim)``.
    6. ``cos = emb.cos(); sin = emb.sin()``.

    **Stage 5b TODO:** verify bit-exact vs ``visual.rot_pos_emb`` on the cat
    fixture before pinning.
    """
    grid_t, grid_h, grid_w = grid_thw
    head_dim = vision_hidden_size // vision_num_heads  # 64 for 0.8B
    half = head_dim // 2  # 32 — VisionRotaryEmbedding(dim=half)
    inv_freq_len = half // 2  # 16 — len of inv_freq
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, half, 2, dtype=np.float32) / float(half))
    )
    assert inv_freq.shape == (inv_freq_len,)

    # Per-position (h, w) coordinates with spatial-merge reorder.
    h_pos = np.arange(grid_h).reshape(grid_h, 1).repeat(grid_w, axis=1)
    w_pos = np.arange(grid_w).reshape(1, grid_w).repeat(grid_h, axis=0)
    h_pos = (
        h_pos.reshape(
            grid_h // spatial_merge_size, spatial_merge_size,
            grid_w // spatial_merge_size, spatial_merge_size,
        )
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    w_pos = (
        w_pos.reshape(
            grid_h // spatial_merge_size, spatial_merge_size,
            grid_w // spatial_merge_size, spatial_merge_size,
        )
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    # pos_ids: (N_per_t, 2) → (N_per_t, 2, inv_freq_len) after gather.
    pos_ids = np.stack([h_pos, w_pos], axis=-1)  # (N_per_t, 2)
    # Build freqs over [0, max_grid) for arbitrary index.
    max_grid = max(grid_h, grid_w)
    freqs = np.outer(np.arange(max_grid, dtype=np.float32), inv_freq)  # (max_grid, inv_freq_len)
    rope = freqs[pos_ids]  # (N_per_t, 2, inv_freq_len)
    rope = rope.reshape(pos_ids.shape[0], 2 * inv_freq_len)  # (N_per_t, head_dim/2)
    rope = np.tile(rope, (grid_t, 1))  # (N_per_t * grid_t, head_dim/2)
    emb = np.concatenate([rope, rope], axis=-1)  # (N, head_dim)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return cos, sin

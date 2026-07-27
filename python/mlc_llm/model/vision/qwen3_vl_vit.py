"""
Qwen3-VL vision tower for MLC-LLM (Phase 10 Stage 3).

Ports `Qwen3_5VisionModel` from `transformers.models.qwen3_5.modeling_qwen3_5`
(modeling_qwen3_5.py:995-1183) for inference under TVM Relax. Same architecture
across 0.8B (depth=12, hidden=768) and 35B-A3B (depth=27, hidden=1152) — only
sized differently. ``Qwen3VLVisionTower`` is config-driven and handles both.

Forward signature is split into two pieces:
  * ``Qwen3VLVisionTower.embed_patches(pixel_values, pos_embeds)`` — runs the 3D
    patch embedding and adds the (precomputed, bilinearly interpolated from the
    learned 48x48 grid) position embedding. Caller supplies ``pos_embeds``
    because ``fast_pos_embed_interpolate`` depends on ``image_grid_thw`` and is
    cleaner to compute externally in Python where indices/weights are static.
  * ``Qwen3VLVisionTower.forward(hidden_states, position_embeddings)`` — runs
    the 12 (or 27) ``Qwen3VLVisionBlock`` blocks. ``position_embeddings`` is
    a ``(cos, sin)`` tuple of shape ``(seq_len, head_dim)`` — full-rotary 1D
    rotary, per-token 2D row/col positions looked up from a static
    ``Qwen3VLVisionRotaryEmbedding`` freq table.

The merger lives in a separate module (``qwen3_vl_image.py``, Stage 4).

Shapes for the 0.8B + cat fixture (image_grid_thw=(1,42,60)):
  pixel_values              (2520, 1536)         # 1536 = 3·2·16·16
  patch_embed output         (2520, 768)
  + pos_embeds                 (2520, 768)
  per-block output           (2520, 768)         # × 12
  merger output (Stage 4)    (630, 1024)          # 2520 / spatial_merge=2² = 630
"""

from __future__ import annotations

import dataclasses
import os
from typing import List, Optional, Tuple  # noqa: UP035

import numpy as np
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.op import vit_flash_attn


@dataclasses.dataclass
class Qwen3VLVisionConfig:
    """Configuration of the Qwen3-VL vision tower.

    Sourced from the HF ``vision_config`` block of the Qwen3.5 model config.
    See ``.claude/plans/phase10-vision-input.md`` for the per-checkpoint table.
    """

    depth: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_heads: int = 12
    out_hidden_size: int = 1024  # patch-merger output dim; matches LM hidden
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304  # learned 48x48 grid
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    rope_theta: float = 10000.0  # 1D vision rotary, NOT the LM theta
    dtype: str = "float16"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def _rotate_half(x: Tensor) -> Tensor:
    """``[x1, x2] -> [-x2, x1]`` along the last axis (NeoX/half-split)."""
    x1, x2 = op.split(x, 2, axis=-1)
    return op.concat([op.negative(x2), x1], dim=-1)


class Qwen3VLVisionPatchEmbed(nn.Module):
    """3D-conv patch embedding.

    Input ``(N, in_channels · temporal_patch_size · patch_size · patch_size)``
    pre-flattened by the Qwen2VL processor. Internally reshapes to ``NCDHW``
    and runs a single ``Conv3D`` with ``stride == kernel == [tps, ps, ps]``,
    so each conv output is exactly one patch token.
    """

    def __init__(self, config: Qwen3VLVisionConfig):
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        self.proj = nn.Conv3D(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=[config.temporal_patch_size, config.patch_size, config.patch_size],
            stride=[config.temporal_patch_size, config.patch_size, config.patch_size],
            bias=True,
            dtype=config.dtype,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        n = pixel_values.shape[0]
        x = op.reshape(
            pixel_values,
            (n, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size),
        )
        x = self.proj(x)  # (n, embed_dim, 1, 1, 1)
        return op.reshape(x, (n, self.embed_dim))


class Qwen3VLVisionMLP(nn.Module):
    """Two-layer FFN with ``gelu_pytorch_tanh`` activation."""

    def __init__(self, config: Qwen3VLVisionConfig):
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_fc1(x)
        x = op.gelu(x, approximate="tanh")
        return self.linear_fc2(x)


class Qwen3VLVisionAttention(nn.Module):
    """Multi-head self-attention without causal mask, full 1D rotary on q/k.

    Single-image path: cu_seqlens = [0, seq_len], so the attention is a plain
    ``softmax(QK^T · scale) V`` over all patches. (HF supports multi-image
    batching via cu_seqlens; for v1 we only handle one image at a time, which
    is what the parity bench exercises.)
    """

    def __init__(self, config: Qwen3VLVisionConfig):
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, hidden_states: Tensor, position_embeddings: Tuple[Tensor, Tensor]) -> Tensor:  # noqa: UP006
        seq_len = hidden_states.shape[0]
        in_dtype = hidden_states.dtype
        # qkv: (s, dim*3) -> (s, 3, h, d) -> (3, h, s, d)
        qkv = self.qkv(hidden_states)
        qkv = op.reshape(qkv, (seq_len, 3, self.num_heads, self.head_dim))
        qkv = op.permute_dims(qkv, axes=[1, 2, 0, 3])
        q, k, v = op.split(qkv, 3, axis=0)
        q = op.squeeze(q, 0)  # (h, s, d)
        k = op.squeeze(k, 0)
        v = op.squeeze(v, 0)

        # HF apply_rotary_pos_emb_vision + eager_attention_forward run in fp32
        # throughout (rotation, qk-matmul, softmax, av-matmul). Doing the same
        # in fp16 collapses tower parity (max diff 2.03 / rel 39% on the cat
        # fixture). Cast q/k/v to fp32, run attention math, cast result back.
        q32 = op.astype(q, "float32")
        k32 = op.astype(k, "float32")
        v32 = op.astype(v, "float32")
        cos, sin = position_embeddings
        cos32 = op.astype(op.unsqueeze(cos, dim=0), "float32")  # (1, s, d)
        sin32 = op.astype(op.unsqueeze(sin, dim=0), "float32")
        q32 = op.add(op.multiply(q32, cos32), op.multiply(_rotate_half(q32), sin32))
        k32 = op.add(op.multiply(k32, cos32), op.multiply(_rotate_half(k32), sin32))

        if vit_flash_attn.enabled(self.head_dim):
            # Item 0r. One kernel replaces QK^T + softmax + P@V *and* the transpose
            # below, and never materializes the (h, s, s) fp32 score matrix — 305 MB at
            # the cat fixture's 2520 patches, which is the whole reason those three
            # kernels cost 14.71 ms/layer against a 3.67 ms compute floor (§20.9).
            #
            # `MLC_QWEN35_VL_PRESCALE_Q` is inert on this path: the scale is applied to
            # `q` as it is read into shared, which is the prescaled form by construction.
            attn_out = vit_flash_attn.flash_attention(q32, k32, v32, self.scaling)
            attn_out = op.astype(attn_out, in_dtype)
            attn_out = op.permute_dims(attn_out, axes=[1, 0, 2])
            attn_out = op.reshape(attn_out, (seq_len, self.dim))
            return self.proj(attn_out)

        k_t = op.permute_dims(k32, axes=[0, 2, 1])  # (h, d, s)
        scale = nn.Tensor.from_const(np.array(self.scaling, dtype="float32"))
        if os.environ.get("MLC_QWEN35_VL_PRESCALE_Q", "1") == "1":
            # `matmul(q, k^T) * c` == `matmul(q * c, k^T)`, and which side the scale
            # sits on decides how much memory the model touches. On the scores it is
            # an elementwise pass over `(h, s, s)` = **305 MB**; on `q` it is the same
            # arithmetic over `(h, s, d)` = 7.7 MB, 40x less (§20.5).
            #
            # The saving is not the pass itself — dlight fuses `matmul + multiply`
            # into one kernel, so on this path the multiply is already free. It is
            # that the fusion is what made the matmul *ineligible* for cuBLAS:
            # §20.2 measured offloading it as a net **47 ms loss**, entirely from the
            # DRAM round trip the broken fusion forced. Move the scale here and the
            # QK matmul becomes a bare GEMM with nothing to displace, so cuBLAS's
            # 3.63 ms SGEMM replaces a 9.46 ms generated kernel at no cost.
            q32 = op.multiply(q32, scale)
            attn_scores = op.matmul(q32, k_t)
        else:
            attn_scores = op.matmul(q32, k_t)
            attn_scores = op.multiply(attn_scores, scale)
        attn_probs = op.softmax(attn_scores, axis=-1)
        attn_out = op.matmul(attn_probs, v32)  # (h, s, d) fp32
        attn_out = op.astype(attn_out, in_dtype)
        # back to (s, dim)
        attn_out = op.permute_dims(attn_out, axes=[1, 0, 2])
        attn_out = op.reshape(attn_out, (seq_len, self.dim))
        return self.proj(attn_out)


class Qwen3VLPatchMerger(nn.Module):
    """2x2 spatial merge → LayerNorm → Linear → GELU → Linear.

    HF Qwen3_5VisionPatchMerger (modeling_qwen3_5.py:848-861). For 0.8B:
    in = hidden_size · spatial_merge_size² = 768·4 = 3072,
    out = out_hidden_size = 1024 (matches LM hidden).
    """

    def __init__(self, config: Qwen3VLVisionConfig):
        merge_in = config.hidden_size * (config.spatial_merge_size ** 2)
        # Pre-shuffle norm: HF uses LayerNorm over hidden_size (not merge_in)
        # because the input is reshaped from (N, hidden) to (N/4, hidden·4)
        # AFTER the norm. ``use_postshuffle_norm=False`` is the released layout.
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear_fc1 = nn.Linear(merge_in, merge_in, bias=True)
        self.linear_fc2 = nn.Linear(merge_in, config.out_hidden_size, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states: (N, hidden_size). Norm in fp32 internally; reshape
        # groups every spatial_merge_size² consecutive tokens.
        x = self.norm(hidden_states)
        # Reshape (N, hidden) → (N // merge², hidden · merge²). The grouping is
        # done by the preprocessor's index permutation upstream.
        n = x.shape[0]
        merge_sq = self.linear_fc1.weight.shape[0] // x.shape[1]
        x = op.reshape(x, (n // merge_sq, merge_sq * x.shape[1]))
        x = self.linear_fc1(x)
        x = op.gelu(x)  # plain GELU — NOT tanh-approximated (HF uses nn.GELU())
        return self.linear_fc2(x)


class Qwen3VLVisionBlock(nn.Module):
    """One ViT block: pre-LN attention + pre-LN MLP, residual on each."""

    def __init__(self, config: Qwen3VLVisionConfig):
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Qwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(self, hidden_states: Tensor, position_embeddings: Tuple[Tensor, Tensor]) -> Tensor:  # noqa: UP006
        hidden_states = op.add(
            hidden_states,
            self.attn(self.norm1(hidden_states), position_embeddings),
        )
        hidden_states = op.add(hidden_states, self.mlp(self.norm2(hidden_states)))
        return hidden_states


class Qwen3VLVisionTower(nn.Module):
    """Full Qwen3-VL vision tower minus the patch merger.

    Construction order matches HF:
        patch_embed -> pos_embed (learned 48x48) -> N x VisionBlock -> (merger)

    The learned ``pos_embed`` matrix is owned here so the loader maps the HF
    weight ``visual.pos_embed.weight`` straight in. Bilinear interpolation onto
    the per-image grid is computed externally (in Python, from
    ``image_grid_thw``) and passed in via ``pos_embeds``.
    """

    def __init__(self, config: Qwen3VLVisionConfig):
        self.config = config
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])

    def forward(
        self,
        pixel_values: Tensor,
        pos_embeds: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
    ) -> Tensor:
        """Returns the **pre-merger** hidden state (shape ``(seq_len, hidden_size)``).

        ``rotary_cos`` and ``rotary_sin`` are each shape ``(seq_len, head_dim)`` —
        the caller computes them from ``image_grid_thw`` and the static
        ``Qwen3VLVisionRotaryEmbedding`` table (``head_dim/2`` freqs, doubled).
        """
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = op.add(hidden_states, pos_embeds)
        position_embeddings = (rotary_cos, rotary_sin)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, position_embeddings)
        return hidden_states

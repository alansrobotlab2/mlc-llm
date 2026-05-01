"""Qwen3.5-VL: dense Qwen3.5 backbone + Qwen3-VL vision tower + patch merger.

Phase 10 Stage 5a — sibling module of ``qwen35``. Reuses ``Qwen35Model`` /
``Qwen35Attention`` / ``Qwen35GatedDeltaNet`` from ``qwen35_model``; adds the
visual tower (``vision/qwen3_vl_vit.py``) and an ``image_embed`` entry point.

Spec extension vs ``Qwen35LMHeadModel``:
  * ``image_embed(pixel_values, pos_embeds, rotary_cos, rotary_sin)`` — runs
    tower + merger, returns merged image-token embeddings.
  * ``prefill`` / ``batch_prefill`` add ``position_ids`` (3, 1, seq_len) and
    ``mrope_deltas`` (1, 1) inputs.
  * ``decode`` / ``batch_decode`` build ``position_ids`` from a cached delta.
  * ``create_paged_kv_cache`` flips ``rope_mode`` to ``NONE`` — the softmax
    layers apply inline mRoPE before the raw ``self_attention`` call, so K is
    stored already-rotated. **Old text-only pages (NORMAL rotation) cannot be
    reused with this build.**

Scope cuts for v1 (parity-gate first):
  * No MTP draft head (``mtp_num_hidden_layers`` forced to 0).
  * No ``forward_with_history`` / prefix-cache spec entries.
  * No ``*_to_last_hidden_states`` spec entries.
These can be reintroduced once Stage 5b passes the parity gate.
"""

import dataclasses
from typing import Any, Dict, List, Optional  # noqa: UP035

import numpy as np
from tvm import tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.model_utils import index_last_token
from mlc_llm.model.qwen35.qwen35_model import (
    Qwen35Config,
    Qwen35Model,
)
from mlc_llm.model.vision.qwen3_vl_vit import (
    Qwen3VLPatchMerger,
    Qwen3VLVisionConfig,
    Qwen3VLVisionTower,
)
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.support.config import ConfigBase


@dataclasses.dataclass
class Qwen35VLConfig(Qwen35Config):
    """Qwen3.5-VL config: text backbone + vision tower + token IDs.

    Inherits all text-side fields from ``Qwen35Config``. ``mrope_section`` and
    ``mrope_interleaved`` MUST be set (``__post_init__`` enforces this — VL
    builds without mRoPE are wrong by construction).

    Vision token IDs (image/video pad, vision start/end) are read from the HF
    config so the engine-side substitution path can find them.
    """

    # vision_config nested-dict from HF; parsed into Qwen3VLVisionConfig in __post_init__
    vision_config: Optional[Dict[str, Any]] = None  # noqa: UP006
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    def __post_init__(self):
        # Pull vision_config / vision_* token IDs out of kwargs before the parent
        # __post_init__ pops text_config / max_position_embeddings.
        if self.vision_config is None and "vision_config" in self.kwargs:
            self.vision_config = self.kwargs.pop("vision_config")
        for tok in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            if tok in self.kwargs:
                setattr(self, tok, int(self.kwargs.pop(tok)))

        # Capture mrope params from text_config.rope_parameters BEFORE the parent
        # pops text_config. Parent's __post_init__ only extracts rope_theta and
        # partial_rotary_factor (deliberately leaves mrope_* alone so text-only
        # builds don't accidentally flip rope_mode). VL builds require both.
        text_cfg = self.kwargs.get("text_config", {})
        rope_params = text_cfg.get("rope_parameters", {}) if isinstance(text_cfg, dict) else {}
        if "mrope_section" in rope_params:
            self.mrope_section = list(rope_params["mrope_section"])
        if "mrope_interleaved" in rope_params:
            self.mrope_interleaved = bool(rope_params["mrope_interleaved"])

        super().__post_init__()

        # Parse vision_config dict -> Qwen3VLVisionConfig (cached as attribute, NOT
        # a dataclass field — keeps `dataclasses.fields` pure for downstream tools).
        if self.vision_config is None:
            raise ValueError("Qwen35VLConfig requires `vision_config`.")
        vc = self.vision_config
        # Vision tower runs in fp16 unconditionally. TVM's topi::layer_norm only
        # supports fp32/fp16 (asserts at compile if asked for bf16); the tower
        # is small (~8 M params) and fp16 matches HF's autocast convention.
        # Cast the merger output to the LM dtype on the way out (image_embed).
        vision_dtype = "float16"
        if isinstance(vc, dict):
            self._vision_cfg = Qwen3VLVisionConfig(
                depth=int(vc.get("depth", 12)),
                hidden_size=int(vc.get("hidden_size", 768)),
                intermediate_size=int(vc.get("intermediate_size", 3072)),
                num_heads=int(vc.get("num_heads", 12)),
                out_hidden_size=int(vc.get("out_hidden_size", self.hidden_size)),
                in_channels=int(vc.get("in_channels", 3)),
                patch_size=int(vc.get("patch_size", 16)),
                temporal_patch_size=int(vc.get("temporal_patch_size", 2)),
                spatial_merge_size=int(vc.get("spatial_merge_size", 2)),
                num_position_embeddings=int(vc.get("num_position_embeddings", 2304)),
                hidden_act=str(vc.get("hidden_act", "gelu_pytorch_tanh")),
                rope_theta=float(vc.get("rope_theta", 10000.0)),
                dtype=vision_dtype,
            )
        elif isinstance(vc, Qwen3VLVisionConfig):
            self._vision_cfg = vc
        else:
            raise TypeError(f"vision_config must be a dict or Qwen3VLVisionConfig, got {type(vc)}")

        if self._vision_cfg.out_hidden_size != self.hidden_size:
            raise ValueError(
                f"vision_config.out_hidden_size ({self._vision_cfg.out_hidden_size}) must "
                f"equal text hidden_size ({self.hidden_size}); merger output feeds the LM directly."
            )

        if self.mrope_section is None:
            raise ValueError("Qwen35VLConfig requires `mrope_section` to be set on the text config.")

        # v1 scope cut: VL build does not include MTP. Regression: when the 35B
        # path lights up, swap to a wider check (mtp_num_hidden_layers compatible
        # with NONE-mode cache).
        if self.mtp_num_hidden_layers != 0:
            self.mtp_num_hidden_layers = 0
            self.mtp_use_dedicated_embeddings = False


class Qwen3VLVisualModel(Qwen3VLVisionTower):
    """Vision tower + patch merger, named to match HF's ``model.visual.*``.

    Subclasses ``Qwen3VLVisionTower`` (already exposes ``patch_embed``,
    ``pos_embed``, ``blocks``) and adds the ``merger`` submodule. Every HF
    visual.* weight maps onto a same-named MLC param under ``visual.``.
    """

    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__(config)
        self.merger = Qwen3VLPatchMerger(config)

    def forward(
        self,
        pixel_values: Tensor,
        pos_embeds: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
    ) -> Tensor:
        """Returns merged image-token embeddings, shape ``(N // merge², out_hidden_size)``."""
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = op.add(hidden_states, pos_embeds)
        position_embeddings = (rotary_cos, rotary_sin)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, position_embeddings)
        return self.merger(hidden_states)


class Qwen35VLLMHeadModel(nn.Module):
    """Qwen3.5-VL LM head: text backbone (Qwen35Model, mRoPE-on) + visual tower."""

    def __init__(self, config: Qwen35VLConfig):
        self.config = config
        self.model = Qwen35Model(config)
        self.visual = Qwen3VLVisualModel(config._vision_cfg)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.partial_rotary_factor = config.partial_rotary_factor
        self.kv_cache_dtype = getattr(config, "kv_cache_dtype", None) or None
        # GDN config (mirrors Qwen35LMHeadModel for create_rnn_state)
        self.num_linear_layers = config.num_linear_layers
        self.num_attention_layers = config.num_attention_layers
        self.linear_num_value_heads = config.linear_num_value_heads
        self.linear_key_head_dim = config.linear_key_head_dim
        self.linear_value_head_dim = config.linear_value_head_dim

        # Vision-side dims for spec
        self._vision_cfg = config._vision_cfg
        self._patch_dim = (
            self._vision_cfg.in_channels
            * self._vision_cfg.temporal_patch_size
            * self._vision_cfg.patch_size
            * self._vision_cfg.patch_size
        )

    def to(self, dtype: Optional[str] = None):
        # Cast every submodule, but pin the visual tower to fp16. TVM
        # topi::layer_norm only supports fp32/fp16 — a bf16 backbone needs a
        # fp16 vision tower with a dtype boundary inside image_embed. Same
        # pattern would apply to int8/fp8 backbones if those land.
        if dtype is None:
            super().to(dtype=dtype)
            return
        self.model.to(dtype=dtype)
        if not self.tie_word_embeddings:
            self.lm_head.to(dtype=dtype)
        # nn.LayerNorm defaults param dtype to fp32 if construction dtype is
        # None; force fp16 here so weights match the fp16 fwd-path inputs.
        self.visual.to(dtype=self._vision_cfg.dtype)
        self.dtype = dtype

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def image_embed(
        self,
        pixel_values: Tensor,
        pos_embeds: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
    ) -> Tensor:
        """Run vision tower + patch merger.

        Args:
            pixel_values: ``(N, C·T·P·P)`` flattened patch tensor (fp16).
            pos_embeds: ``(N, vision_hidden)`` bilinear-interpolated learned pos embed (fp16).
            rotary_cos / rotary_sin: ``(N, vision_head_dim)`` per-token vision rotary (fp16).
        Returns:
            ``(N // spatial_merge², text_hidden)`` merged embedding sequence in
            the LM's dtype, ready to scatter into text input_embed at
            ``<|image_pad|>`` positions.
        """
        out = self.visual(pixel_values, pos_embeds, rotary_cos, rotary_sin)
        # Vision tower is always fp16; cast to LM dtype before scatter.
        if self.dtype != self._vision_cfg.dtype:
            out = op.astype(out, self.dtype)
        return out

    def _lm_head(self, hidden_states: Tensor) -> Tensor:
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    # ── mRoPE delta plumbing ────────────────────────────────────────────────
    # The qwen2_5_vl_model.py pattern of stashing the delta on the cache as a
    # Python attribute is unsound — at export_tvm trace time, prefill and
    # decode receive separate PagedKVCache wrappers, so the attribute does not
    # propagate. Pass `mrope_deltas` explicitly into every entry point that
    # needs to compute position_ids (decode, batch_decode, batch_verify).

    def _build_decode_position_ids(
        self,
        seq_len: int,
        paged_kv_cache: PagedKVCache,
        batch: int,
        mrope_deltas: Tensor,
    ) -> Tensor:
        """Rebuild rank-3 position_ids from ``cache.get_query_positions`` + delta.

        For decode/verify each token sits at exactly one linear position; the
        T/H/W rows collapse to that single value plus the image-induced delta.
        """
        base = paged_kv_cache.get_query_positions(seq_len)
        base = op.reshape(base, (1, seq_len))
        base = op.broadcast_to(base, (batch, seq_len))
        base = base + mrope_deltas
        base = op.unsqueeze(base, dim=0)
        return op.broadcast_to(base, (3, batch, seq_len))

    # ── Forward methods ─────────────────────────────────────────────────────

    def _forward(
        self,
        input_embed: Tensor,
        position_ids: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states, state = self.model.forward(
            input_embed, paged_kv_cache, state, position_ids
        )
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self._lm_head(hidden_states)
        return logits, paged_kv_cache, state

    def prefill(
        self,
        input_embed: Tensor,
        position_ids: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        """Single-sequence prefill. Returns (logits at last position, cache, state)."""
        op_ext.configure()
        hidden_states, rnn_state = self.model.forward(
            input_embed, paged_kv_cache, rnn_state, position_ids
        )
        hidden_states = index_last_token(hidden_states)
        logits = self._lm_head(hidden_states)
        return logits, paged_kv_cache, rnn_state

    def decode(
        self,
        input_embed: Tensor,
        mrope_deltas: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        op_ext.configure()
        b, s, _ = input_embed.shape
        position_ids = self._build_decode_position_ids(s, paged_kv_cache, b, mrope_deltas)
        hidden_states, rnn_state = self.model.forward(
            input_embed, paged_kv_cache, rnn_state, position_ids
        )
        logits = self._lm_head(hidden_states)
        return logits, paged_kv_cache, rnn_state

    def batch_prefill(
        self,
        input_embeds: Tensor,
        position_ids: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        return self._forward(
            input_embeds, position_ids, paged_kv_cache, rnn_state, logit_positions
        )

    def batch_decode(
        self,
        input_embeds: Tensor,
        mrope_deltas: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        op_ext.configure()
        b, s, _ = input_embeds.shape
        position_ids = self._build_decode_position_ids(s, paged_kv_cache, b, mrope_deltas)
        hidden_states, rnn_state = self.model.forward(
            input_embeds, paged_kv_cache, rnn_state, position_ids
        )
        logits = self._lm_head(hidden_states)
        return logits, paged_kv_cache, rnn_state

    def batch_verify(
        self,
        input_embeds: Tensor,
        mrope_deltas: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self.batch_decode(input_embeds, mrope_deltas, paged_kv_cache, rnn_state)

    # ── Cache / RNN state factories ─────────────────────────────────────────

    def create_rnn_state(
        self,
        max_batch_size: tirx.Var,
        max_history: tirx.Var,
    ) -> RNNState:
        from tvm import relax as R

        K = self.linear_key_head_dim
        V = self.linear_value_head_dim
        n_vh = self.linear_num_value_heads
        n_kh = self.config.linear_num_key_heads
        qkv_dim = n_kh * K * 2 + n_vh * V
        conv_ks_m1 = self.config.linear_conv_kernel_dim - 1
        init_values = [
            R.const(np.zeros((n_vh, K, V), "float32")),
            R.const(np.zeros((conv_ks_m1, qkv_dim), self.dtype)),
        ]
        return RNNState.create(
            max_batch_size=max_batch_size,
            num_hidden_layers=self.num_linear_layers,
            max_history=max_history,
            init_values=init_values,
        )

    def create_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
    ) -> PagedKVCache:
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            # Phase 10 R-1: cache stores K already-rotated (inline mRoPE in
            # Qwen35Attention). Cross-build reuse with NORMAL-mode pages is
            # silently wrong; the lib SHA bump is the version key.
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rotary_dim=rotary_dim,
            dtype=self.dtype,
            dtype_kv=getattr(self, "kv_cache_dtype", None) or self.dtype,
        )

    def get_default_spec(self):
        cfg = self._vision_cfg
        head_dim = cfg.hidden_size // cfg.num_heads
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "image_embed": {
                # Vision tower is fp16 unconditionally; preprocessor emits fp16
                # tensors. Output is cast to LM dtype inside image_embed().
                "pixel_values": nn.spec.Tensor(["num_patches", self._patch_dim], cfg.dtype),
                "pos_embeds": nn.spec.Tensor(["num_patches", cfg.hidden_size], cfg.dtype),
                "rotary_cos": nn.spec.Tensor(["num_patches", head_dim], cfg.dtype),
                "rotary_sin": nn.spec.Tensor(["num_patches", head_dim], cfg.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "position_ids": nn.spec.Tensor([3, 1, "seq_len"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "mrope_deltas": nn.spec.Tensor([1, 1], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "position_ids": nn.spec.Tensor([3, 1, "seq_len"], "int32"),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "mrope_deltas": nn.spec.Tensor(["batch_size", 1], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "mrope_deltas": nn.spec.Tensor([1, 1], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
            "create_rnn_state": {
                "max_batch_size": int,
                "max_history": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)

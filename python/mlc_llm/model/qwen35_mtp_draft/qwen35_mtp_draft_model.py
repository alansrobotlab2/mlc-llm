"""Qwen3.5 MTP head as a standalone EAGLE-style draft model.

Pairs with the Qwen3.5 target (`qwen3_5`) for self-speculative decoding via the
EAGLE pipeline. The draft contains:
  - embed_tokens (full vocab embedding, shared semantically with target via tied
    weights but a separate VRAM copy here)
  - pre_fc_norm_embedding, pre_fc_norm_hidden, fc (the MTP fuse step)
  - one decoder layer (Qwen35Attention + Qwen35MLP, attn_output_gate=True)
  - final norm

No `get_logits` is exposed: the engine routes hidden states through the target's
lm_head via `CanGetLogits()=false` in the EAGLE pipeline.

Function names match the EAGLE template (`fuse_embed_hidden_states`,
`*_to_last_hidden_states`) so the existing C++ EAGLE actions drive this model
without modification.
"""

import dataclasses
from typing import Optional

from tvm import tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode

from ..qwen35.qwen35_model import (
    Qwen35Attention,
    Qwen35Config,
    Qwen35Embedding,
    Qwen35MLP,
)


@dataclasses.dataclass
class Qwen35MTPDraftConfig(Qwen35Config):
    """Reuses Qwen35Config; only the MTP-relevant fields are exercised."""


class _Qwen35MTPDraftDecoderLayer(nn.Module):
    """Single decoder block matching `_Qwen35MTPDecoderLayer` in the target model."""

    def __init__(self, config: Qwen35Config):
        self.self_attn = Qwen35Attention(config)
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, paged_kv_cache, layer_id)
        if self.tensor_parallel_shards > 1:
            hidden_states = op.ccl_allreduce(x, "sum") + residual
        else:
            hidden_states = x + residual
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        if self.tensor_parallel_shards > 1:
            hidden_states = op.ccl_allreduce(x, "sum") + residual
        else:
            hidden_states = x + residual
        return hidden_states


class Qwen35MTPDraftLM(nn.Module):
    """EAGLE-compatible draft head for Qwen3.5 self-speculative decoding."""

    def __init__(self, config: Qwen35MTPDraftConfig):
        if config.mtp_num_hidden_layers <= 0:
            raise ValueError(
                "Qwen35MTPDraftLM requires `mtp_num_hidden_layers >= 1`. "
                "Set it in the model config (Qwen3.5-0.8B ships with 1)."
            )
        self.config = config
        self.embed_tokens = Qwen35Embedding(config.vocab_size, config.hidden_size)
        self.pre_fc_norm_embedding = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )
        self.pre_fc_norm_hidden = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [_Qwen35MTPDraftDecoderLayer(config) for _ in range(config.mtp_num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

        self.num_hidden_layers = config.mtp_num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = config.dtype

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.embed_tokens(input_ids)

    def fuse_embed_hidden_states(self, input_embed: Tensor, hidden_states: Tensor):
        e_norm = self.pre_fc_norm_embedding(input_embed)
        h_norm = self.pre_fc_norm_hidden(hidden_states)
        return self.fc(op.concat([h_norm, e_norm], dim=-1))

    def forward_to_last_hidden_states(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache):
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        return self.norm(hidden_states)

    def prefill_to_last_hidden_states(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.forward_to_last_hidden_states(hidden_states, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def decode_to_last_hidden_states(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()
        hidden_states = self.forward_to_last_hidden_states(hidden_states, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_prefill_to_last_hidden_states(
        self, hidden_states: Tensor, paged_kv_cache: PagedKVCache
    ):
        op_ext.configure()
        hidden_states = self.forward_to_last_hidden_states(hidden_states, paged_kv_cache)
        return hidden_states, paged_kv_cache

    def batch_decode_to_last_hidden_states(
        self, hidden_states: Tensor, paged_kv_cache: PagedKVCache
    ):
        op_ext.configure()
        hidden_states = self.forward_to_last_hidden_states(hidden_states, paged_kv_cache)
        return hidden_states, paged_kv_cache

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
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rotary_dim=rotary_dim,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "fuse_embed_hidden_states": {
                "input_embed": nn.spec.Tensor(["seq_len", self.hidden_size], self.dtype),
                "hidden_states": nn.spec.Tensor(["seq_len", self.hidden_size], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill_to_last_hidden_states": {
                "hidden_states": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode_to_last_hidden_states": {
                "hidden_states": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill_to_last_hidden_states": {
                "hidden_states": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode_to_last_hidden_states": {
                "hidden_states": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)

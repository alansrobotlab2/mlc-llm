"""
Implementation for Qwen3.5-MoE GatedDeltaNet hybrid architecture.
75% GatedDeltaNet (recurrent linear attention) + 25% standard GQA softmax attention,
with a sparse MoE FFN (256 routed experts, top-8) plus one shared expert per layer.
Targets Qwen/Qwen3.6-35B-A3B.

Reuses the validated dense base in mlc_llm.model.qwen35: Config, Attention,
GatedDeltaNet, TIR kernel, Embedding, ACT2FN. Replaces the dense MLP with
Qwen35MoESparseMoeBlock and extends the config with MoE + mRoPE fields.
"""

import dataclasses
from typing import Any, Dict, List, Optional  # noqa: UP035

import numpy as np
from tvm import relax as R
from tvm import tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.model.qwen35.qwen35_model import (
    ACT2FN,
    Qwen35Attention,
    Qwen35Config,
    Qwen35Embedding,
    Qwen35GatedDeltaNet,
)
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.expert import MixtralExperts
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.support import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Qwen35MoEConfig(Qwen35Config):
    """Configuration for Qwen3.5-MoE (Qwen3.6-35B-A3B)."""

    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    num_experts_per_tok: int = 0
    num_experts: int = 0
    decoder_sparse_step: int = 1
    norm_topk_prob: bool = True
    # mrope_section / mrope_interleaved inherited from Qwen35Config (Phase 10).


class Qwen35MoEMLP(nn.Module):
    """Dense gated MLP used as the per-layer shared expert."""

    def __init__(self, config: Qwen35MoEConfig, intermediate_size: int):
        if intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split shared expert intermediate size {intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        # Per-token GEMV dispatch when called from the MoE block at small batch
        # (verify spec entries with seq_len pinned to literal). 5× small-batch
        # tax in dense matmul → ~13 ms savings per verify on the 35B at γ=2.
        if len(x.shape) == 2 and isinstance(x.shape[0], int) and 1 < x.shape[0] <= 5:
            num_tokens = x.shape[0]
            parts = op.split(x, indices_or_sections=num_tokens, axis=0)
            outs = []
            for t in range(num_tokens):
                concat_t = self.gate_up_proj(parts[t])
                x1, x2 = op.split(concat_t, 2, axis=-1)
                outs.append(self.down_proj(self.act_fn(x1) * x2))
            return op.concat(outs, dim=0)
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


class Qwen35MoESparseMoeBlock(nn.Module):
    """Routed-experts MoE block with a sigmoid-gated shared expert."""

    def __init__(self, config: Qwen35MoEConfig):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        if config.moe_intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MoE intermediate size {config.moe_intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.moe_intermediate_size = config.moe_intermediate_size // config.tensor_parallel_shards
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.moe_gate_up_proj = MixtralExperts(
            self.num_experts,
            in_features=config.hidden_size,
            out_features=2 * self.moe_intermediate_size,
        )
        self.moe_down_proj = MixtralExperts(
            self.num_experts,
            in_features=self.moe_intermediate_size,
            out_features=config.hidden_size,
        )

        self.shared_expert = Qwen35MoEMLP(config, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]
        self.dtype = "float32"

    def forward(self, x: Tensor):
        def _expert_forward(h: Tensor, indptr: Tensor):
            x1_x2 = self.moe_gate_up_proj(h, indptr)
            x1, x2 = op.split(x1_x2, indices_or_sections=2, axis=-1)
            return self.moe_down_proj(self.act_fn(x1) * x2, indptr)

        experts_per_tok = self.num_experts_per_tok
        num_experts = self.num_experts
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.reshape(num_tokens, hidden_size)

        gate_logits = self.gate(x_flat)
        expert_weights, expert_indices = op_ext.moe_misc.gating_softmax_topk(
            gate_logits, experts_per_tok, norm_topk_prob=self.norm_topk_prob
        )

        use_cutlass = op_ext.get_store().cutlass_group_gemm and self.dtype in [
            "float16",
            "bfloat16",
        ]
        if num_tokens == 1:
            moe_hidden_states = _expert_forward(x_flat, expert_indices)
        elif isinstance(num_tokens, int) and 1 < num_tokens <= 5:
            # Spec-decode verify (γ=1..4, num_tokens = γ+1): dispatch per-token
            # through the b=1 dequantize_gemv kernel. Bench at B=24 group_gemm
            # showed 0.058 ms/row flat at small batch vs gemv 0.008 ms/row →
            # ~7× speedup per-token. Avoids cumsum/scatter overhead entirely.
            # Triggered only when seq_len is pinned to a literal in the spec
            # (see batch_verify_g{N}_to_last_hidden_states variants).
            h_parts = op.split(x_flat, indices_or_sections=num_tokens, axis=0)
            ind_parts = op.split(expert_indices, indices_or_sections=num_tokens, axis=0)
            out_parts = [
                _expert_forward(h_parts[t], ind_parts[t]) for t in range(num_tokens)
            ]
            moe_hidden_states = op.concat(out_parts, dim=0)
        else:
            cumsum = op_ext.moe_misc.moe_cumsum(expert_indices, num_experts)
            reverse_indices, token_indices = op_ext.moe_misc.get_indices(cumsum, expert_indices)
            if use_cutlass:
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, num_experts, num_tokens, inclusive=True, out_dtype="int64"
                )
            else:
                indptr = op_ext.moe_misc.get_indptr(
                    cumsum, num_experts, num_tokens, inclusive=False, out_dtype="int32"
                )
            moe_hidden_states = op.take(x_flat, token_indices, axis=0)
            moe_hidden_states = _expert_forward(moe_hidden_states, indptr)
            moe_hidden_states = op_ext.moe_misc.scatter_output(moe_hidden_states, reverse_indices)

        expert_weights = expert_weights.reshape(num_tokens, experts_per_tok, 1)
        moe_hidden_states = (
            moe_hidden_states.reshape(num_tokens, experts_per_tok, hidden_size) * expert_weights
        )
        moe_hidden_states = op_ext.moe_misc.moe_sum(moe_hidden_states, dim=1)

        shared = self.shared_expert(x_flat)
        shared = op.sigmoid(self.shared_expert_gate(x_flat)) * shared

        out = moe_hidden_states + shared
        return out.reshape(batch_size, seq_len, hidden_size)


class Qwen35MoEDecoderLayer(nn.Module):
    def __init__(self, config: Qwen35MoEConfig, layer_id: int, category_id: int):
        self.layer_type = config.layer_types()[layer_id]
        if self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(config)
        else:
            self.linear_attn = Qwen35GatedDeltaNet(config, category_id)
        self.category_id = category_id
        assert config.num_experts > 0 and config.decoder_sparse_step == 1, (
            "Qwen3.5-MoE assumes every layer is sparse MoE."
        )
        self.mlp = Qwen35MoESparseMoeBlock(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
    ):
        out = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            out = self.self_attn(out, paged_kv_cache, self.category_id)
        else:
            out, state = self.linear_attn.forward(out, state)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states, state

    def forward_with_history(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
    ):
        """Verify-path variant — scatters per-position GDN state into history slots.

        Mirrors `Qwen35DecoderLayer.forward_with_history` from qwen35_model.py:890.
        Only the linear_attn call differs from `forward`; full_attention layers are
        rolled back via PagedKVCache PopN, which doesn't need a history-mode forward.
        """
        out = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            out = self.self_attn(out, paged_kv_cache, self.category_id)
        else:
            out, state = self.linear_attn.forward_with_history(out, state)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states, state

    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual


class Qwen35MoEModel(nn.Module):
    def __init__(self, config: Qwen35MoEConfig):
        self.embed_tokens = Qwen35Embedding(config.vocab_size, config.hidden_size)
        layer_types = config.layer_types()
        linear_idx = 0
        attn_idx = 0
        layers = []
        for i, ltype in enumerate(layer_types):
            if ltype == "linear_attention":
                layers.append(Qwen35MoEDecoderLayer(config, i, category_id=linear_idx))
                linear_idx += 1
            else:
                layers.append(Qwen35MoEDecoderLayer(config, i, category_id=attn_idx))
                attn_idx += 1
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
    ):
        hidden_states = inputs
        for layer in self.layers:
            hidden_states, state = layer.forward(hidden_states, paged_kv_cache, state)
        hidden_states = self.norm(hidden_states)
        return hidden_states, state

    def forward_with_history(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
    ):
        hidden_states = inputs
        for layer in self.layers:
            hidden_states, state = layer.forward_with_history(
                hidden_states, paged_kv_cache, state
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states, state


class Qwen35MoEForCausalLM(nn.Module):
    def __init__(self, config: Qwen35MoEConfig):
        self.config = config
        self.model = Qwen35MoEModel(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = config.dtype
        # Phase 5: optional fp8 KV cache. None or empty falls back to self.dtype.
        self.kv_cache_dtype = getattr(config, "kv_cache_dtype", None) or None
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
        self.num_linear_layers = config.num_linear_layers
        self.num_attention_layers = config.num_attention_layers
        self.linear_num_value_heads = config.linear_num_value_heads
        self.linear_key_head_dim = config.linear_key_head_dim
        self.linear_value_head_dim = config.linear_value_head_dim

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def _lm_head(self, hidden_states: Tensor) -> Tensor:
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def _forward(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states, state = self.model.forward(input_embed, paged_kv_cache, state)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self._lm_head(hidden_states), paged_kv_cache, state

    def _forward_with_history(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        logit_positions: Optional[Tensor] = None,
    ):
        """Prefill-with-history forward: scatters per-position GDN state into RNNState
        history slots so the radix prefix cache can roll the recurrent state back via
        PopN. Engine must arm `set_use_history_mode(True)` before BeginForward."""
        op_ext.configure()
        hidden_states, state = self.model.forward_with_history(
            input_embed, paged_kv_cache, state
        )
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self._lm_head(hidden_states), paged_kv_cache, state

    def _forward_to_last_hidden(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
    ):
        op_ext.configure()
        hidden_states, state = self.model.forward(input_embed, paged_kv_cache, state)
        return hidden_states, paged_kv_cache, state

    def _forward_to_last_hidden_with_history(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
    ):
        op_ext.configure()
        hidden_states, state = self.model.forward_with_history(
            input_embed, paged_kv_cache, state
        )
        return hidden_states, paged_kv_cache, state

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        op_ext.configure()
        return self._lm_head(hidden_states)

    def prefill_to_last_hidden_states(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_to_last_hidden(input_embed, paged_kv_cache, rnn_state)

    def decode_to_last_hidden_states(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_to_last_hidden(input_embed, paged_kv_cache, rnn_state)

    def batch_prefill_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_to_last_hidden(input_embeds, paged_kv_cache, rnn_state)

    def batch_prefill_to_last_hidden_states_with_history(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        # Prefix-cacheable to-last-hidden prefill: per-position GDN state -> history slots.
        # Engine must arm `set_use_history_mode(True)` before BeginForward.
        return self._forward_to_last_hidden_with_history(
            input_embeds, paged_kv_cache, rnn_state
        )

    def batch_decode_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_to_last_hidden(input_embeds, paged_kv_cache, rnn_state)

    def batch_verify_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        # Verify uses the per-position-history GDN forward so partial accept can
        # roll the recurrent state back to the accepted prefix bit-exactly via
        # PopN. Mirrors the 0.8B path (qwen35_model.py:1184). The engine pairs
        # this with `set_use_history_mode(True)` before BeginForward so EndForward
        # advances `available_history_num` by `seq_len`.
        # Dynamic seq_len entry; falls through MoE block's group_gemm path.
        # For γ ∈ {1..4}, the engine should call batch_verify_g{γ}_* below to
        # hit the per-token gemv fast path.
        return self._forward_to_last_hidden_with_history(
            input_embeds, paged_kv_cache, rnn_state
        )

    # Specialized verify entry points with literal seq_len. These trace the same
    # forward but pin num_tokens to a Python int inside the MoE block, which fires
    # the per-token-gemv dispatch branch (avoids ~7× group_gemm small-batch tax).
    # Engine selects by γ at runtime: γ=1 → g1 (seq_len=2), γ=2 → g2 (3), etc.
    def batch_verify_g1_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_to_last_hidden_with_history(
            input_embeds, paged_kv_cache, rnn_state
        )

    def batch_verify_g2_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_to_last_hidden_with_history(
            input_embeds, paged_kv_cache, rnn_state
        )

    def batch_verify_g3_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_to_last_hidden_with_history(
            input_embeds, paged_kv_cache, rnn_state
        )

    def batch_verify_g4_to_last_hidden_states(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward_to_last_hidden_with_history(
            input_embeds, paged_kv_cache, rnn_state
        )

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state, logit_positions)

    def batch_prefill_with_history(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        # Prefix-cacheable prefill: emits per-position GDN state into history slots.
        # Engine must arm `set_use_history_mode(True)` before BeginForward.
        return self._forward_with_history(
            input_embeds, paged_kv_cache, rnn_state, logit_positions
        )

    def batch_decode(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state)

    def batch_verify(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        rnn_state: RNNState,
    ):
        return self._forward(input_embeds, paged_kv_cache, rnn_state)

    def create_rnn_state(
        self,
        max_batch_size: tirx.Var,
        max_history: tirx.Var,
    ) -> RNNState:
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
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rotary_dim=rotary_dim,
            dtype=self.dtype,
            dtype_kv=getattr(self, "kv_cache_dtype", None) or self.dtype,
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
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill_with_history": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            # batch_size pinned to 1 (literal int, not SizeVar) so the MoE block's
            # `if num_tokens == 1:` resolves statically at compile time and routes
            # through `dequantize_gemv` (~6× faster than `dequantize_group_gemm` at
            # b=1 top-8 on Orin). Trade-off: this lib only supports max_batch_size=1
            # at decode (interactive mode); server mode with batched decode would
            # need either the dynamic-batch spec restored or a Relax If for runtime
            # dispatch. batch_prefill / batch_verify keep dynamic seq_len.
            "batch_decode": {
                "input_embeds": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            # EAGLE-compat entry points: hidden-state-returning variants for
            # self-spec decode. The engine pairs these with the draft model's
            # `*_to_last_hidden_states` and routes through `get_logits`.
            "get_logits": {
                "hidden_states": nn.spec.Tensor(["seq_len", self.hidden_size], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill_to_last_hidden_states": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode_to_last_hidden_states": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill_to_last_hidden_states_with_history": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            # Specialized verify entries with seq_len pinned to literal γ+1.
            # These activate the per-token-gemv fast path inside the MoE block
            # (see Qwen35MoESparseMoeBlock.forward). One per supported γ; engine
            # picks at runtime based on actual draft length.
            "batch_verify_g1_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor([1, 2, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify_g2_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor([1, 3, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify_g3_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor([1, 4, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify_g4_to_last_hidden_states": {
                "input_embeds": nn.spec.Tensor([1, 5, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "rnn_state": nn.spec.Object(object_type=RNNState),
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
            "create_rnn_state": {
                "max_batch_size": int,
                "max_history": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)

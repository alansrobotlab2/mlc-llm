"""
Implementation for Qwen3.5 GatedDeltaNet hybrid architecture.
75% GatedDeltaNet (recurrent linear attention), 25% standard GQA softmax attention.
"""

import dataclasses
import math
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
from tvm import relax as R
from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tirx as T

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.op.mrope import (
    MultimodalRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Qwen35Config(ConfigBase):
    """Configuration of the Qwen3.5 model."""

    hidden_size: int = 0
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_hidden_layers: int = 0
    num_key_value_heads: int = 0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 0
    rope_theta: int = 10000000
    head_dim: int = 256
    hidden_act: str = "silu"
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # GatedDeltaNet-specific
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    partial_rotary_factor: float = 0.25
    # MTP (Multi-Token Prediction) — self-speculative decoding draft head.
    # 0.8B ships with mtp_num_hidden_layers=1; weights live under `mtp.*` in HF.
    mtp_num_hidden_layers: int = 0
    mtp_use_dedicated_embeddings: bool = False
    # Runtime
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    dtype: str = "float32"
    max_batch_size: int = 1
    # Phase 5: dtype of the paged KV cache. None or "" -> use the model dtype.
    # Set to e.g. "float8_e4m3fn" to enable fp8 KV storage (Phase 5).
    kv_cache_dtype: Optional[str] = None
    # Phase 10: mRoPE plumbing. Default None preserves text-only behavior
    # (cache-internal RopeMode.NORMAL). When set, softmax-attention layers
    # apply inline mRoPE and use raw self_attention; spec adds position_ids +
    # mrope_deltas. The VL sibling module (qwen3_5_vl/) sets these from HF config;
    # text-only Qwen35LMHeadModel keeps them None.
    mrope_section: Optional[List[int]] = None  # noqa: UP006,UP007
    mrope_interleaved: bool = False
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)  # noqa: UP006

    def __post_init__(self):
        # Handle VLM wrapper: Qwen3.5 HF config has all text params inside text_config
        if "text_config" in self.kwargs:
            text_config = self.kwargs.pop("text_config")
            if isinstance(text_config, dict):
                field_names = {f.name for f in dataclasses.fields(self.__class__)}
                for k, v in text_config.items():
                    if k in field_names and k != "kwargs":
                        setattr(self, k, v)
                    else:
                        self.kwargs[k] = v
                # Extract rope params from nested rope_parameters
                rope_params = text_config.get("rope_parameters", {})
                if isinstance(rope_params, dict):
                    if "rope_theta" in rope_params:
                        self.rope_theta = rope_params["rope_theta"]
                    if "partial_rotary_factor" in rope_params:
                        self.partial_rotary_factor = rope_params["partial_rotary_factor"]

        # Also handle rope_parameters at top level
        if "rope_parameters" in self.kwargs:
            rope_params = self.kwargs.pop("rope_parameters")
            if isinstance(rope_params, dict):
                if "rope_theta" in rope_params:
                    self.rope_theta = rope_params["rope_theta"]
                if "partial_rotary_factor" in rope_params:
                    self.partial_rotary_factor = rope_params["partial_rotary_factor"]

        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = min(self.context_window_size, 2048)
        elif self.prefill_chunk_size > self.context_window_size:
            self.prefill_chunk_size = min(self.context_window_size, 2048)

    @property
    def num_linear_layers(self) -> int:
        """Number of GatedDeltaNet linear attention layers."""
        return self.num_hidden_layers - self.num_attention_layers

    @property
    def num_attention_layers(self) -> int:
        """Number of full attention layers."""
        return self.num_hidden_layers // self.full_attention_interval

    def layer_types(self) -> List[str]:  # noqa: UP006
        """Returns list of layer types: 'linear_attention' or 'full_attention'."""
        types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                types.append("full_attention")
            else:
                types.append("linear_attention")
        return types


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
}


class Qwen35Embedding(nn.Embedding):
    def lm_head_forward(self, x: nn.Tensor):
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        skip = os.environ.get("QWEN35_NO_QUANT_MLP", "")
        if skip:
            for n in (s.strip() for s in skip.split(",")):
                if n and hasattr(self, n):
                    getattr(self, n).no_quantization = True

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)


class Qwen35Attention(nn.Module):
    """Standard GQA attention with output gate for full_attention layers (every 4th layer).

    attn_output_gate=True: q_proj outputs 2*num_heads*head_dim, split into (Q, gate).
    Gate is sigmoid-applied to attention output before o_proj.
    """

    def __init__(self, config: Qwen35Config):
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_key_value_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.rope_theta = config.rope_theta
        # Phase 10: when set, forward() takes (cos, sin) and applies inline mRoPE
        # via op/mrope.apply_multimodal_rotary_pos_emb, using the raw
        # paged_kv_cache.self_attention path (cache rope_mode=NONE). When unset,
        # the existing attention_with_fused_qkv path runs (cache rope_mode=NORMAL).
        self.mrope_section = config.mrope_section
        self.mrope_interleaved = config.mrope_interleaved

        # c_attn: Q (2x for gate) + K + V fused projection
        self.c_attn = nn.Linear(
            in_features=config.hidden_size,
            out_features=(2 * self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(config.head_dim, -1, config.rms_norm_eps, bias=False)
        self.k_norm = nn.RMSNorm(config.head_dim, -1, config.rms_norm_eps, bias=False)

        skip = os.environ.get("QWEN35_NO_QUANT_ATTN", "")
        if skip:
            for n in (s.strip() for s in skip.split(",")):
                if n and hasattr(self, n):
                    getattr(self, n).no_quantization = True

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: PagedKVCache,
        layer_id: int,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,  # noqa: UP006
    ):
        d, h_q, h_kv = self.head_dim, self.num_attention_heads, self.num_key_value_heads
        b, s, _ = hidden_states.shape
        # c_attn per-token at small static seq: same small-batch tax fix as
        # GDN/MoE. Pinned-seq verify entries (s ∈ {2..5}) hit this branch.
        if isinstance(s, int) and 1 < s <= 5:
            h_parts = op.split(hidden_states, indices_or_sections=s, axis=1)
            proj = op.concat([self.c_attn(h_parts[t]) for t in range(s)], dim=1)
        else:
            # c_attn outputs flat: [Q_with_gate (h_q * 2 * d), K (h_kv * d), V (h_kv * d)]
            proj = self.c_attn(hidden_states)
        # Reshape to heads: (b, s, 2*h_q + 2*h_kv, d)
        proj = op.reshape(proj, (b, s, 2 * h_q + 2 * h_kv, d))
        # Split: first 2*h_q heads have interleaved [Q, gate] per head, then h_kv K, h_kv V
        q_gate, k, v = op.split(proj, [2 * h_q, 2 * h_q + h_kv], axis=2)
        # q_gate shape: (b, s, 2*h_q, d). Even heads are Q, odd heads are gate
        # But HF layout is per-head [Q_d, gate_d], so reshape to (b, s, h_q, 2*d) then split
        q_gate = op.reshape(q_gate, (b, s, h_q, 2 * d))
        q, gate = op.split(q_gate, [d], axis=3)
        # gate: (b, s, h_q, d) -> flatten to (b, s, h_q*d)
        gate = op.reshape(gate, (b, s, h_q * d))
        q = self.q_norm(q)
        k = self.k_norm(k)

        if position_embeddings is not None:
            # Phase 10 inline-mRoPE path: cos/sin already computed once at the
            # model level and broadcast to all softmax-attention layers. Cache
            # rope_mode must be NONE in this build; the cache's f_split_rotary_
            # skips rotation when rope_mode != kNormal, so we pre-rotate Q/K
            # here and feed pre-rotated qkv to attention_with_fused_qkv. The
            # cache still routes prefill→ragged kernel and decode→cached-K
            # kernel based on cur_append_lengths_, which is what we need —
            # `self_attention` is ragged-only and ignores the cached K's,
            # giving decontextualized decode (Phase 10 Stage 5b diagnosis).
            assert self.mrope_section is not None, (
                "position_embeddings provided but mrope_section is unset on the attention "
                "layer. Build config mismatch."
            )
            cos, sin = position_embeddings
            q, k = apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, self.mrope_section,
                unsqueeze_dim=2, interleaved=self.mrope_interleaved,
            )
            qkv = op.concat([q, k, v], dim=2)
            output = op.reshape(
                paged_kv_cache.attention_with_fused_qkv(
                    layer_id, qkv, self.num_attention_heads, sm_scale=self.head_dim ** -0.5
                ),
                (b, s, h_q * d),
            )
        else:
            # Existing path: cache applies 1D RoPE internally per its rope_mode setting.
            qkv = op.concat([q, k, v], dim=2)
            output = op.reshape(
                paged_kv_cache.attention_with_fused_qkv(
                    layer_id, qkv, self.num_attention_heads, sm_scale=self.head_dim ** -0.5
                ),
                (b, s, h_q * d),
            )

        # Apply output gate: sigmoid(gate) * attn_output
        output = output * op.sigmoid(gate)
        if isinstance(s, int) and 1 < s <= 5:
            o_parts = op.split(output, indices_or_sections=s, axis=1)
            return op.concat([self.o_proj(o_parts[t]) for t in range(s)], dim=1)
        return self.o_proj(output)


# ============================================================================
# GatedDeltaNet TIR kernel
# ============================================================================


def create_gated_delta_net_func(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """Creates a TIR function for the GatedDeltaNet recurrent computation.

    Thread-per-column design: each thread owns one column of the state matrix.
    State S is (key_head_dim x value_head_dim) per head, accumulated in fp32.

    The state column owned by a thread is held in a thread-local register array
    across all 5 passes per token (decay, dot_sk, delta, dot_sq, scale), so the
    state buffer hits GMEM exactly twice per kernel call: once on the initial
    load, once on the final flush. The unfused variant (`v0`) walked state
    through GMEM 4-5× per pass, leaving ~50% of kernel time bound by needless
    state traffic. For decode (seq_len=1) this halves kernel runtime; for
    prefill the win is smaller but still positive.

    Supports arbitrary sequence length via an inner `for t in range(seq_len)` loop,
    matching RWKV6's approach. During prefill (seq_len > 1), the recurrence accumulates
    state across all tokens sequentially. During decode (seq_len = 1), it's a single step.

    For GVA (num_value_heads > num_key_heads), Q/K are expanded via repeat.
    The kernel operates on value_heads (the larger dimension).
    """
    heads_per_group = num_value_heads // num_key_heads  # 1 for 0.8B, 2 for 4B
    K = key_head_dim  # 128
    V = value_head_dim  # 128

    @T.prim_func
    def gdn_func(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        gate_handle: T.handle,  # exp(g), already exponentiated
        beta_handle: T.handle,  # sigmoid(beta_raw)
        state_in_handle: T.handle,
        out_handle: T.handle,
        state_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        # q, k: (batch, seq_len, key_heads, K)
        q_buf = T.match_buffer(q_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        k_buf = T.match_buffer(k_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        # v: (batch, seq_len, value_heads, V)
        v_buf = T.match_buffer(v_handle, (batch_size, seq_len, num_value_heads, V), dtype=dtype)
        # gate and beta: (batch, seq_len, value_heads)
        gate_buf = T.match_buffer(
            gate_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        beta_buf = T.match_buffer(
            beta_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        # State: per value_head, K x V matrix in fp32
        state_in_buf = T.match_buffer(
            state_in_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )
        # Outputs: out in fp32 for numerical stability (cast to model dtype by caller)
        out_buf = T.match_buffer(
            out_handle, (batch_size, seq_len, num_value_heads, V), dtype="float32"
        )
        state_out_buf = T.match_buffer(
            state_out_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )

        scale = T.float32(1.0 / math.sqrt(K))

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                for col in T.thread_binding(V, thread="threadIdx.x"):
                    with T.sblock("gdn_thread"):
                        # Per-thread register-resident state column: 128 fp32.
                        # Persists across all (t, pass) iterations; flushed once at end.
                        state_local = T.sblock_alloc_buffer((K,), "float32", scope="local")
                        dot_sk = T.sblock_alloc_buffer((1,), "float32", scope="local")
                        dot_sq = T.sblock_alloc_buffer((1,), "float32", scope="local")

                        kh = h_idx // heads_per_group

                        # Load state_in → registers (one GMEM read per element)
                        for row in range(K):
                            state_local[row] = state_in_buf[b_idx, h_idx, row, col]

                        # Sequential loop over tokens (like RWKV6)
                        for t in range(seq_len):
                            gate_val = gate_buf[b_idx, t, h_idx]
                            beta_val = beta_buf[b_idx, t, h_idx]
                            v_val = T.cast(v_buf[b_idx, t, h_idx, col], "float32")

                            # Pass 1: decay + dot(S, k) fused
                            #   S[r] *= gate;  dot_sk += S[r] * k[r]
                            dot_sk[0] = T.float32(0)
                            for row in range(K):
                                state_local[row] = state_local[row] * gate_val
                                dot_sk[0] = dot_sk[0] + state_local[row] * T.cast(
                                    k_buf[b_idx, t, kh, row], "float32"
                                )

                            # Pass 2: delta + dot(S', q) fused
                            #   S[r] += k[r] * beta * (v - dot_sk)
                            #   dot_sq += S[r] * q[r]
                            coef = beta_val * (v_val - dot_sk[0])
                            dot_sq[0] = T.float32(0)
                            for row in range(K):
                                state_local[row] = state_local[row] + T.cast(
                                    k_buf[b_idx, t, kh, row], "float32"
                                ) * coef
                                dot_sq[0] = dot_sq[0] + state_local[row] * T.cast(
                                    q_buf[b_idx, t, kh, row], "float32"
                                )

                            # Output with scale
                            out_buf[b_idx, t, h_idx, col] = dot_sq[0] * scale

                        # Flush registers → state_out (one GMEM write per element)
                        for row in range(K):
                            state_out_buf[b_idx, h_idx, row, col] = state_local[row]

    return gdn_func


def create_gated_delta_net_func_inplace(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """GatedDeltaNet recurrence that reads and writes the RNNState slot directly.

    Same math as `create_gated_delta_net_func`, but instead of taking a pre-copied
    `state_in` tensor and emitting a `state_out` tensor for the runtime to copy back,
    this variant takes the whole state storage buffer plus the device-side slot-index
    arrays and does the slot addressing itself:

        load  from storage[seq_slot,       hist_slot,           head, row, col]
        flush into storage[seq_slot, (hist_slot + 1) % max_hist, head, row, col]

    That saves two full state copies per layer per token. On the 35B the `get`/`set`
    pair moves 4 MiB per call, 60 calls/token = 245 MiB/token, which traced at
    1.195 ms/token (6.3% of the decode budget) even though both copy kernels already
    run at 205-216 GB/s.

    **The `+ 1` in the write index is load-bearing.** It mirrors `RNNState.create_set_func`,
    which writes the *next* history slot while `create_get_func` reads the current one:
    the pair is a ring-buffer advance, not a redundant copy. The previous state has to
    survive at `hist_slot` for Phase 8's `PopN` prefix-cache rollback to read it back.

    Writing into `hist_slot` instead is the obvious implementation and it is wrong. This
    was built and measured as a negative control (2026-07-25): under
    `prefix_cache_mode="disable"` — `max_history == 1`, the two slots coincide, and what
    every bench harness sets — it passes everything, including
    `scripts/prefix_cache_roundtrip.py` 4/4. Under the *default* `"radix"` mode
    (`max_history == 64`) `EndForward` advances into a slot the kernel never wrote, so it
    fails greedy parity on all 5 prompts (3-16/50 tokens) and the round-trip gate 13
    ways. So the failure is loud **if** you gate under radix, and invisible if you only
    ever gate under the mode the benchmarks use.

    Aliasing is safe by construction, including when the two slots do coincide. Thread
    `(b_idx, h_idx, col)` owns exactly one column of the state matrix: it reads
    `storage[..., row, col]` once per `row` on entry, holds that column in registers
    across every pass and every `t`, and writes `storage[..., row, col]` once per `row`
    on exit. The read set and the write set are identical per element and disjoint
    across threads, so no thread can ever observe another thread's write. This is a
    stronger guarantee than "the recurrence is elementwise in S", and it is what
    distinguishes this from the aliasing bug in SGLang #20791, where a scheduler
    introduced aliasing into a kernel that did have cross-thread state reads.

    `max_batch_size` and `max_history` are bound from the storage tensor's own shape at
    runtime rather than baked in, so one kernel is correct for every RNNState
    configuration. Keeping the addressing dynamic — whole buffer plus *device-side*
    index arrays — is also what makes this cudagraph-safe: `EndForward` advances
    `history_slot_id` every step once `max_history > 1`, so a slot pointer baked into a
    captured graph would silently address the wrong slot.
    """
    heads_per_group = num_value_heads // num_key_heads
    K = key_head_dim
    V = value_head_dim

    @T.prim_func
    def gdn_func_inplace(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        gate_handle: T.handle,  # exp(g), already exponentiated
        beta_handle: T.handle,  # sigmoid(beta_raw)
        storage_handle: T.handle,  # the whole (max_batch, max_hist, H, K, V) state buffer
        seq_slot_handle: T.handle,  # device-side per-batch seq slot ids
        hist_slot_handle: T.handle,  # device-side per-batch history slot ids
        out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        max_batch_size, max_history = T.int64(), T.int64()
        # q, k: (batch, seq_len, key_heads, K)
        q_buf = T.match_buffer(q_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        k_buf = T.match_buffer(k_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        # v: (batch, seq_len, value_heads, V)
        v_buf = T.match_buffer(v_handle, (batch_size, seq_len, num_value_heads, V), dtype=dtype)
        # gate and beta: (batch, seq_len, value_heads)
        gate_buf = T.match_buffer(
            gate_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        beta_buf = T.match_buffer(
            beta_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        # The whole state storage, not a slot view. max_batch_size / max_history are
        # bound from the tensor's own shape here.
        storage_buf = T.match_buffer(
            storage_handle,
            (max_batch_size, max_history, num_value_heads, K, V),
            dtype="float32",
        )
        seq_slot_buf = T.match_buffer(seq_slot_handle, (batch_size,), dtype="int32")
        hist_slot_buf = T.match_buffer(hist_slot_handle, (batch_size,), dtype="int32")
        # Output in fp32 for numerical stability (cast to model dtype by caller)
        out_buf = T.match_buffer(
            out_handle, (batch_size, seq_len, num_value_heads, V), dtype="float32"
        )

        scale = T.float32(1.0 / math.sqrt(K))

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                for col in T.thread_binding(V, thread="threadIdx.x"):
                    with T.sblock("gdn_thread"):
                        # Per-thread register-resident state column: 128 fp32.
                        # Persists across all (t, pass) iterations; flushed once at end.
                        state_local = T.sblock_alloc_buffer((K,), "float32", scope="local")
                        dot_sk = T.sblock_alloc_buffer((1,), "float32", scope="local")
                        dot_sq = T.sblock_alloc_buffer((1,), "float32", scope="local")

                        kh = h_idx // heads_per_group

                        # Resolve this batch element's slot pair once.
                        seq_id: T.int64 = T.cast(seq_slot_buf[b_idx], "int64")
                        hist_in: T.int64 = T.cast(hist_slot_buf[b_idx], "int64")
                        # Ring advance — see the docstring; must match create_set_func.
                        hist_out: T.int64 = (hist_in + T.int64(1)) % max_history

                        # Load the state column straight out of its slot (one GMEM read
                        # per element — the copy this replaces did two).
                        for row in range(K):
                            state_local[row] = storage_buf[seq_id, hist_in, h_idx, row, col]

                        # Sequential loop over tokens (like RWKV6)
                        for t in range(seq_len):
                            gate_val = gate_buf[b_idx, t, h_idx]
                            beta_val = beta_buf[b_idx, t, h_idx]
                            v_val = T.cast(v_buf[b_idx, t, h_idx, col], "float32")

                            # Pass 1: decay + dot(S, k) fused
                            #   S[r] *= gate;  dot_sk += S[r] * k[r]
                            dot_sk[0] = T.float32(0)
                            for row in range(K):
                                state_local[row] = state_local[row] * gate_val
                                dot_sk[0] = dot_sk[0] + state_local[row] * T.cast(
                                    k_buf[b_idx, t, kh, row], "float32"
                                )

                            # Pass 2: delta + dot(S', q) fused
                            #   S[r] += k[r] * beta * (v - dot_sk)
                            #   dot_sq += S[r] * q[r]
                            coef = beta_val * (v_val - dot_sk[0])
                            dot_sq[0] = T.float32(0)
                            for row in range(K):
                                state_local[row] = state_local[row] + T.cast(
                                    k_buf[b_idx, t, kh, row], "float32"
                                ) * coef
                                dot_sq[0] = dot_sq[0] + state_local[row] * T.cast(
                                    q_buf[b_idx, t, kh, row], "float32"
                                )

                            # Output with scale
                            out_buf[b_idx, t, h_idx, col] = dot_sq[0] * scale

                        # Flush registers straight into the next history slot.
                        for row in range(K):
                            storage_buf[seq_id, hist_out, h_idx, row, col] = state_local[row]

    return gdn_func_inplace


def create_causal_conv1d_func_inplace(
    conv_dim: int,
    kernel_size: int,
    dtype: str,
    threads: int = 256,
):
    """Causal depthwise conv1d that reads and writes its RNNState conv slot directly.

    The conv-state analogue of `create_gated_delta_net_func_inplace`, and it collapses
    three kernels rather than two. The copy path emits `rnn_state_get_1` (slot -> temp),
    `update_conv_state` (a TE op that materializes the shifted state as a *new* tensor),
    the conv itself, then `rnn_state_set_1` (temp -> slot). Traced on the 35B at
    0.190 + 0.105 + 0.088 = 0.383 ms/token, 2.2% of the decode budget, all of it moving
    a 48 KB state around. Here the shift is just where the flush loop reads from, so
    only the conv kernel survives.

    Slot addressing mirrors the recurrent kernel exactly, including the ring advance:

        load  from storage[seq_slot,       hist_slot,            ks-1, conv_dim]
        flush into storage[seq_slot, (hist_slot + 1) % max_hist, ks-1, conv_dim]

    See `create_gated_delta_net_func_inplace` for why the `+ 1` is load-bearing and why
    writing into `hist_slot` passes every gate under `prefix_cache_mode="disable"` while
    corrupting the default `"radix"` configuration.

    **Ordering, not aliasing, is what makes this safe.** Thread `(b_idx, d_idx)` owns one
    channel: it reads only `storage[..., :, d_idx]` and writes only that same column, so
    threads never interact — same per-element ownership the recurrent kernel has. But
    unlike that kernel the reads are *not* at the same indices as the writes (the state
    shifts by `seq_len`), so when `max_history == 1` the two slots coincide and a write
    could clobber a value still to be read. The body is therefore staged strictly:

        1. compute every conv output          (reads only)
        2. stage the whole new state in registers (reads only)
        3. flush the registers to the slot    (writes only)

    All reads precede all writes, so the coincident-slot case is correct by construction
    rather than by luck of loop order. Step 2 is why the new state goes through registers
    at all -- reading it lazily inside step 3 would reintroduce the hazard.

    Every register index is a *static* Python loop var. The concatenated
    `[old_state ++ qkv]` sequence is indexed dynamically (`si + kk`, `seq_len + j`), which
    is fine against a global buffer but would force a register array to local memory, so
    the dynamic reads stay on `storage_buf` and only the staged values live in registers.

    Accumulation is in `dtype`, summed in ascending `kk`, which is what `te.sum` over the
    4-element reduction axis of the TE path lowers to. That is deliberate: it keeps the
    change bit-exact so `scripts/greedy_snapshot.py` is a valid gate for it, unlike the
    in_proj merge which altered the dlight reduction split and needed the semantic gate.
    """
    ks_m1 = kernel_size - 1
    n_blocks = (conv_dim + threads - 1) // threads

    @T.prim_func
    def conv1d_inplace(
        qkv_handle: T.handle,
        weight_handle: T.handle,
        storage_handle: T.handle,  # whole (max_batch, max_hist, ks-1, conv_dim) buffer
        seq_slot_handle: T.handle,  # device-side per-batch seq slot ids
        hist_slot_handle: T.handle,  # device-side per-batch history slot ids
        out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        max_batch_size, max_history = T.int64(), T.int64()
        qkv_buf = T.match_buffer(qkv_handle, (batch_size, seq_len, conv_dim), dtype=dtype)
        weight_buf = T.match_buffer(weight_handle, (conv_dim, 1, kernel_size), dtype=dtype)
        storage_buf = T.match_buffer(
            storage_handle, (max_batch_size, max_history, ks_m1, conv_dim), dtype=dtype
        )
        seq_slot_buf = T.match_buffer(seq_slot_handle, (batch_size,), dtype="int32")
        hist_slot_buf = T.match_buffer(hist_slot_handle, (batch_size,), dtype="int32")
        out_buf = T.match_buffer(out_handle, (batch_size, seq_len, conv_dim), dtype=dtype)

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for d_outer in T.thread_binding(n_blocks, thread="blockIdx.x"):
                for d_inner in T.thread_binding(threads, thread="threadIdx.x"):
                    with T.sblock("conv1d_thread"):
                        acc = T.sblock_alloc_buffer((1,), dtype, scope="local")
                        staged = T.sblock_alloc_buffer((ks_m1,), dtype, scope="local")

                        d_idx: T.int64 = T.cast(d_outer, "int64") * T.int64(
                            threads
                        ) + T.cast(d_inner, "int64")
                        if d_idx < T.int64(conv_dim):
                            seq_id: T.int64 = T.cast(seq_slot_buf[b_idx], "int64")
                            hist_in: T.int64 = T.cast(hist_slot_buf[b_idx], "int64")
                            # Ring advance — must match create_set_func. See docstring.
                            hist_out: T.int64 = (hist_in + T.int64(1)) % max_history

                            # --- 1. conv outputs (reads only) ---------------------
                            # out[si] = sum_kk cat[si + kk] * w[kk], where cat is the
                            # concatenation [old_state ++ qkv] of length ks-1 + seq_len.
                            for si in range(seq_len):
                                acc[0] = T.cast(0, dtype)
                                for kk in range(kernel_size):
                                    acc[0] = acc[0] + T.if_then_else(
                                        si + kk < T.int64(ks_m1),
                                        storage_buf[
                                            seq_id,
                                            hist_in,
                                            T.min(si + kk, T.int64(ks_m1 - 1)),
                                            d_idx,
                                        ],
                                        qkv_buf[
                                            b_idx,
                                            T.max(si + kk - T.int64(ks_m1), T.int64(0)),
                                            d_idx,
                                        ],
                                    ) * weight_buf[d_idx, 0, kk]
                                out_buf[b_idx, si, d_idx] = acc[0]

                            # --- 2. stage the new state (reads only) --------------
                            # The new state is the last ks-1 entries of cat, i.e.
                            # cat[seq_len + j]. Staged in registers so that step 3's
                            # writes cannot clobber a value step 2 still needs when
                            # hist_out == hist_in (max_history == 1).
                            for j in range(ks_m1):
                                staged[j] = T.if_then_else(
                                    seq_len + j < T.int64(ks_m1),
                                    storage_buf[
                                        seq_id,
                                        hist_in,
                                        T.min(seq_len + j, T.int64(ks_m1 - 1)),
                                        d_idx,
                                    ],
                                    qkv_buf[
                                        b_idx,
                                        T.max(seq_len + j - T.int64(ks_m1), T.int64(0)),
                                        d_idx,
                                    ],
                                )

                            # --- 3. flush (writes only) ---------------------------
                            for j in range(ks_m1):
                                storage_buf[seq_id, hist_out, j, d_idx] = staged[j]

    return conv1d_inplace


def create_gated_delta_net_func_with_history(
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: str,
):
    """GatedDeltaNet recurrence that emits per-position state snapshots.

    Same math as `create_gated_delta_net_func` but the state output carries an extra
    inner `seq_len` axis: `state_out_buf[vb, t, vh, K, V]` is the recurrent state
    AFTER processing position t. Used by speculative-decoding verify so a partial
    accept can roll back to any intermediate position via PopN on the RNNState.

    Memory: state_out is `seq_len` larger than the regular kernel. Acceptable for
    spec verify (seq_len = γ+1, ~5) but NOT for prefill (seq_len up to 1000s),
    which keeps using the original kernel.
    """
    heads_per_group = num_value_heads // num_key_heads
    K = key_head_dim
    V = value_head_dim

    @T.prim_func
    def gdn_func_history(
        q_handle: T.handle,
        k_handle: T.handle,
        v_handle: T.handle,
        gate_handle: T.handle,
        beta_handle: T.handle,
        state_in_handle: T.handle,
        out_handle: T.handle,
        state_out_handle: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        q_buf = T.match_buffer(q_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        k_buf = T.match_buffer(k_handle, (batch_size, seq_len, num_key_heads, K), dtype=dtype)
        v_buf = T.match_buffer(v_handle, (batch_size, seq_len, num_value_heads, V), dtype=dtype)
        gate_buf = T.match_buffer(
            gate_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        beta_buf = T.match_buffer(
            beta_handle, (batch_size, seq_len, num_value_heads), dtype="float32"
        )
        state_in_buf = T.match_buffer(
            state_in_handle, (batch_size, num_value_heads, K, V), dtype="float32"
        )
        out_buf = T.match_buffer(
            out_handle, (batch_size, seq_len, num_value_heads, V), dtype="float32"
        )
        # Per-position state: the recurrence at position t writes its post-state to
        # state_out_buf[b, t, ...] and reads its pre-state from state_out_buf[b, t-1, ...]
        # (or state_in_buf for t=0).
        state_out_buf = T.match_buffer(
            state_out_handle,
            (batch_size, seq_len, num_value_heads, K, V),
            dtype="float32",
        )

        scale = T.float32(1.0 / math.sqrt(K))

        for b_idx in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h_idx in T.thread_binding(num_value_heads, thread="blockIdx.x"):
                for col in T.thread_binding(V, thread="threadIdx.x"):
                    with T.sblock("gdn_history_thread"):
                        # Persistent register-resident state column, same as gdn_func.
                        # The history variant additionally flushes state_local back
                        # to state_out_buf[*, t, *, *, col] after each t step.
                        state_local = T.sblock_alloc_buffer((K,), "float32", scope="local")
                        dot_sk = T.sblock_alloc_buffer((1,), "float32", scope="local")
                        dot_sq = T.sblock_alloc_buffer((1,), "float32", scope="local")

                        kh = h_idx // heads_per_group

                        for row in range(K):
                            state_local[row] = state_in_buf[b_idx, h_idx, row, col]

                        for t in range(seq_len):
                            gate_val = gate_buf[b_idx, t, h_idx]
                            beta_val = beta_buf[b_idx, t, h_idx]
                            v_val = T.cast(v_buf[b_idx, t, h_idx, col], "float32")

                            # decay + dot_sk
                            dot_sk[0] = T.float32(0)
                            for row in range(K):
                                state_local[row] = state_local[row] * gate_val
                                dot_sk[0] = dot_sk[0] + state_local[row] * T.cast(
                                    k_buf[b_idx, t, kh, row], "float32"
                                )

                            # delta + dot_sq
                            coef = beta_val * (v_val - dot_sk[0])
                            dot_sq[0] = T.float32(0)
                            for row in range(K):
                                state_local[row] = state_local[row] + T.cast(
                                    k_buf[b_idx, t, kh, row], "float32"
                                ) * coef
                                dot_sq[0] = dot_sq[0] + state_local[row] * T.cast(
                                    q_buf[b_idx, t, kh, row], "float32"
                                )

                            out_buf[b_idx, t, h_idx, col] = dot_sq[0] * scale

                            # Flush per-position state into the history buffer
                            for row in range(K):
                                state_out_buf[b_idx, t, h_idx, row, col] = state_local[row]

    return gdn_func_history


# ============================================================================
# GatedDeltaNet Linear Attention Layer
# ============================================================================


class Qwen35GatedDeltaNet(nn.Module):
    """GatedDeltaNet linear attention layer."""

    def __init__(self, config: Qwen35Config, linear_layer_idx: int):
        self.config = config
        self.linear_layer_idx = linear_layer_idx  # index among linear layers only
        self.key_head_dim = config.linear_key_head_dim  # 128
        self.value_head_dim = config.linear_value_head_dim  # 128
        self.num_key_heads = config.linear_num_key_heads  # 16
        self.num_value_heads = config.linear_num_value_heads  # 16 or 32
        self.hidden_size = config.hidden_size
        self.dtype = config.dtype

        qkv_dim = (
            (self.num_key_heads * self.key_head_dim)
            + (self.num_key_heads * self.key_head_dim)
            + (self.num_value_heads * self.value_head_dim)
        )

        # Input projections, fused into ONE GEMV.
        #
        # HF ships four separate tensors (in_proj_qkv / _z / _a / _b) and this layer
        # used to mirror that 1:1. All four read the same `hidden_size` activation, so
        # they are one matmul with the outputs concatenated — exactly the fusion
        # `Qwen35Attention.c_attn` already does for q/k/v, and the loader concatenates
        # them the same way.
        #
        # Why: at batch 1 the split cost 130.6 us per GDN layer on Orin (66.1 + 48.6 +
        # 8.1 + 7.7) against 99.7 us fused — the two `->num_value_heads` projections are
        # so narrow that the 64-outputs-per-CTA GEMV schedule emits grid=(1,1,1), i.e.
        # one CTA on a 16-SM GPU, and they burned 2.5% of the token budget to move
        # 142 KB. Fused: 4.5% of the decode budget and 90 fewer launches per token.
        # See workplan-cuda-13.md §4.6/§5.
        #
        # This is bit-exact, not an approximation: group quantization groups along the
        # reduction axis (`linear_quant_axis = 1` for the NK layout, see
        # quantization/group_quantization.py), so concatenating along output rows leaves
        # every group boundary and every scale untouched.
        self.qkv_dim = qkv_dim
        self.z_dim = self.num_value_heads * self.value_head_dim
        self.in_proj_qkvzab = nn.Linear(
            config.hidden_size,
            qkv_dim + self.z_dim + self.num_value_heads + self.num_value_heads,
            bias=False,
        )
        self.out_proj = nn.Linear(
            self.num_value_heads * self.value_head_dim, config.hidden_size, bias=False
        )

        # Bisect hook: env var QWEN35_NO_QUANT=<comma-separated module names> marks
        # the listed Linear submodules with no_quantization=True so the GroupQuantize/
        # FTQuantize Mutator skips them. Used for narrowing down q4 correctness bugs.
        # Names: in_proj_qkvzab, out_proj. The legacy per-sub-projection names
        # (in_proj_qkv / _z / _a / _b) are accepted and all mark the fused Linear —
        # sub-projection granularity is no longer separable, so a bisect that needs it
        # has to un-fuse first.
        skip = os.environ.get("QWEN35_NO_QUANT", "")
        if skip:
            legacy = {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"}
            for n in (s.strip() for s in skip.split(",")):
                if n in legacy:
                    n = "in_proj_qkvzab"
                if n and hasattr(self, n):
                    getattr(self, n).no_quantization = True

        # Causal depthwise Conv1D kernel
        self.conv1d_weight = nn.Parameter(
            (qkv_dim, 1, config.linear_conv_kernel_dim),
        )

        # Decay parameters (no .weight suffix in HF)
        self.A_log = nn.Parameter((self.num_value_heads,))
        self.dt_bias = nn.Parameter((self.num_value_heads,))

        # Output gating norm — per-head RMSNorm (shared weight across heads)
        self.norm = nn.RMSNorm(self.value_head_dim, -1, config.rms_norm_eps, bias=False)

    def _in_proj(self, hidden_states: Tensor, per_token_seq: int = 0):
        """Fused input projection, split back into (qkv, z, alpha, beta_raw).

        `per_token_seq` > 0 runs the projection one token at a time and concatenates —
        the small-static-seq GEMV dispatch the verify path needs (see the comment at the
        `forward_with_history` call site). 0 means project the whole `hidden_states` in
        one call.
        """
        if per_token_seq:
            h_parts = op.split(hidden_states, indices_or_sections=per_token_seq, axis=1)
            fused = op.concat(
                [self.in_proj_qkvzab(h_parts[t]) for t in range(per_token_seq)], dim=1
            )
        else:
            fused = self.in_proj_qkvzab(hidden_states)
        n_vh = self.num_value_heads
        parts = op.split(
            fused,
            [self.qkv_dim, self.qkv_dim + self.z_dim, self.qkv_dim + self.z_dim + n_vh],
            axis=-1,
        )
        return parts[0], parts[1], parts[2], parts[3]

    def forward(
        self,
        hidden_states: Tensor,
        state: RNNState,
        state_io: Optional["_GDNStateIO"] = None,
    ) -> Tuple[Tensor, RNNState]:  # noqa: UP006
        """Forward using RNNState (for MLCEngine batch methods).

        When `state_io` is supplied the recurrent state is updated in place by the fused
        kernel and the `get`/`set` copy pair is skipped entirely; otherwise the original
        copy path runs unchanged.
        """
        b, s, _ = hidden_states.shape
        K = self.key_head_dim
        V = self.value_head_dim
        n_kh = self.num_key_heads
        n_vh = self.num_value_heads
        layer_idx = self.linear_layer_idx

        # Input projections — one GEMV, then split
        qkv, z, alpha, beta_raw = self._in_proj(hidden_states)

        qkv_dim = qkv.shape[-1]
        if state_io is not None:
            # Fused path: one kernel does the conv and advances the conv state slot, so
            # `rnn_state_get_1`, `update_conv_state` and `rnn_state_set_1` all disappear.
            conv_storage = state_io.conv_storages[layer_idx]
            conv_args = [
                qkv,
                self.conv1d_weight,
                conv_storage,
                state_io.seq_slot_ids,
                state_io.history_slot_ids,
            ]
            # Identity, not `.index()` — see the recurrent call site below.
            conv_storage_idx = next(
                i for i, a in enumerate(conv_args) if a is conv_storage
            )
            qkv, _ = op.tensor_ir_inplace_op(
                create_causal_conv1d_func_inplace(
                    conv_dim=qkv_dim,
                    kernel_size=self.config.linear_conv_kernel_dim,
                    dtype=self.dtype,
                ),
                "causal_conv1d_inplace",
                conv_args,
                # Output 0 is the conv result; output 1 aliases the storage argument, which
                # is what makes the in-place slot advance visible to the rest of the graph.
                inplace_indices=[-1, conv_storage_idx],
                out=[
                    Tensor.placeholder([b, s, qkv_dim], self.dtype),
                    Tensor(_expr=conv_storage._expr),
                ],
            )
        else:
            # Copy path (unchanged): get -> conv + shift -> set.
            conv_state = state.get(
                layer_idx,
                1,
                (b, self.config.linear_conv_kernel_dim - 1, qkv_dim),
                self.dtype,
            )
            qkv, new_conv_state = self._causal_conv1d_with_state(qkv, conv_state)
            state = state.set(layer_idx, 1, new_conv_state)

        # SiLU activation on QKV after conv
        qkv = op.silu(qkv)

        # Split QKV
        q_dim = n_kh * K
        k_dim = n_kh * K
        qkv_parts = op.split(qkv, [q_dim, q_dim + k_dim], axis=-1)
        q = op.reshape(qkv_parts[0], (b, s, n_kh, K))
        k = op.reshape(qkv_parts[1], (b, s, n_kh, K))
        v = op.reshape(qkv_parts[2], (b, s, n_vh, V))

        # L2 normalize Q and K
        q = self._l2_normalize(q)
        k = self._l2_normalize(k)

        # Gate computation
        gate, beta = self._compute_gate_beta(alpha, beta_raw)
        # beta is already (b, s, n_vh) — no GVA expansion needed.

        if state_io is not None:
            # Fused path: the kernel loads from and flushes to the state slot itself, so
            # neither the `get` copy nor the `set` copy is emitted.
            storage = state_io.storages[layer_idx]
            args = [q, k, v, gate, beta, storage, state_io.seq_slot_ids, state_io.history_slot_ids]
            # Identity, not `.index()` — `==` on a Tensor is an elementwise op, not a
            # predicate, so equality-based search is the wrong tool here.
            storage_idx = next(i for i, a in enumerate(args) if a is storage)
            out_recurrent, _ = op.tensor_ir_inplace_op(
                create_gated_delta_net_func_inplace(
                    num_key_heads=n_kh,
                    num_value_heads=n_vh,
                    key_head_dim=K,
                    value_head_dim=V,
                    dtype=self.dtype,
                ),
                "gated_delta_net_inplace",
                args,
                # Output 0 is the freshly allocated recurrent output and is the only one
                # read; output 1 aliases the storage argument, and declaring that alias is
                # what makes the in-place write visible to the rest of the graph.
                inplace_indices=[-1, storage_idx],
                out=[
                    Tensor.placeholder([b, s, n_vh, V], "float32"),
                    Tensor(_expr=storage._expr),
                ],
            )
            out_recurrent = op.astype(out_recurrent, self.dtype)
        else:
            # Copy path (unchanged): get -> compute -> set.
            state_in_layer = state.get(layer_idx, 0, (b, n_vh, K, V), "float32")

            # Recurrent computation via TIR kernel
            out_recurrent, state_out_layer = op.tensor_ir_op(
                create_gated_delta_net_func(
                    num_key_heads=n_kh,
                    num_value_heads=n_vh,
                    key_head_dim=K,
                    value_head_dim=V,
                    dtype=self.dtype,
                ),
                "gated_delta_net",
                [q, k, v, gate, beta, state_in_layer],
                [
                    Tensor.placeholder([b, s, n_vh, V], "float32"),
                    Tensor.placeholder([b, n_vh, K, V], "float32"),
                ],
            )

            # Cast recurrent output back to model dtype
            out_recurrent = op.astype(out_recurrent, self.dtype)

            # Write updated state back to RNNState (state_id=0)
            state = state.set(layer_idx, 0, state_out_layer)

        # Output gating
        out_normed = self.norm(out_recurrent)
        out_flat = op.reshape(out_normed, (b, s, n_vh * V))
        out_gated = out_flat * op.silu(z)
        return self.out_proj(out_gated), state

    def forward_with_history(
        self, hidden_states: Tensor, state: RNNState
    ) -> Tuple[Tensor, RNNState]:  # noqa: UP006
        """Forward variant that scatters per-position state into RNNState history slots.

        Used by the verify path for speculative decoding so partial accept can roll back
        the recurrent state to any intermediate position. Mirrors `forward()` step-for-step
        but the GDN kernel emits a full per-position state history and the conv state is
        also recorded per position; both are written via `state.set_with_history(...)`.
        """
        b, s, _ = hidden_states.shape
        K = self.key_head_dim
        V = self.value_head_dim
        n_kh = self.num_key_heads
        n_vh = self.num_value_heads
        layer_idx = self.linear_layer_idx

        # Per-token GEMV dispatch when seq_len is a Python int and small (1<s<=5).
        # Each linear at b=3 is ~5× slower per row than at b=1 due to tile under-
        # utilization in dlight's small-batch matmul; running per-token through
        # the dl.gpu.GEMV() path saves substantial verify cost. Triggered by the
        # seq_len-pinned `batch_verify_g{1..4}` spec entries on the 35B target;
        # 0.8B target's dynamic-seq verify falls through unchanged.
        qkv, z, alpha, beta_raw = self._in_proj(
            hidden_states, per_token_seq=s if isinstance(s, int) and 1 < s <= 5 else 0
        )

        qkv_dim = qkv.shape[-1]
        conv_state = state.get(
            layer_idx,
            1,
            (b, self.config.linear_conv_kernel_dim - 1, qkv_dim),
            self.dtype,
        )

        # Causal conv1d that also yields the per-position conv state history.
        qkv, conv_state_history = self._causal_conv1d_with_state_history(qkv, conv_state)
        # Scatter per-position conv state to history slots. The "current" state at the
        # end of position t is conv_state_history[:, t, :, :].
        state = state.set_with_history(layer_idx, 1, conv_state_history)

        qkv = op.silu(qkv)

        q_dim = n_kh * K
        k_dim = n_kh * K
        qkv_parts = op.split(qkv, [q_dim, q_dim + k_dim], axis=-1)
        q = op.reshape(qkv_parts[0], (b, s, n_kh, K))
        k = op.reshape(qkv_parts[1], (b, s, n_kh, K))
        v = op.reshape(qkv_parts[2], (b, s, n_vh, V))

        q = self._l2_normalize(q)
        k = self._l2_normalize(k)

        gate, beta = self._compute_gate_beta(alpha, beta_raw)

        state_in_layer = state.get(layer_idx, 0, (b, n_vh, K, V), "float32")

        # Recurrence kernel that emits the full (b, s, n_vh, K, V) history.
        out_recurrent, state_history_layer = op.tensor_ir_op(
            create_gated_delta_net_func_with_history(
                num_key_heads=n_kh,
                num_value_heads=n_vh,
                key_head_dim=K,
                value_head_dim=V,
                dtype=self.dtype,
            ),
            "gated_delta_net_with_history",
            [q, k, v, gate, beta, state_in_layer],
            [
                Tensor.placeholder([b, s, n_vh, V], "float32"),
                Tensor.placeholder([b, s, n_vh, K, V], "float32"),
            ],
        )

        out_recurrent = op.astype(out_recurrent, self.dtype)

        # Scatter recurrent state per position.
        state = state.set_with_history(layer_idx, 0, state_history_layer)

        out_normed = self.norm(out_recurrent)
        out_flat = op.reshape(out_normed, (b, s, n_vh * V))
        out_gated = out_flat * op.silu(z)
        # out_proj per-token at small static seq for the same reason as in_proj_*.
        if isinstance(s, int) and 1 < s <= 5:
            og_parts = op.split(out_gated, indices_or_sections=s, axis=1)
            out = op.concat([self.out_proj(og_parts[t]) for t in range(s)], dim=1)
        else:
            out = self.out_proj(out_gated)
        return out, state

    def _causal_conv1d_with_state_history(
        self, qkv: Tensor, conv_state: Tensor
    ) -> Tuple[Tensor, Tensor]:  # noqa: UP006
        """Conv1D variant that also returns the per-position conv state history.

        The conv state at the end of position t is the (kernel_size-1)-element window
        of inputs ending at position t — i.e. positions [t-ks+2, ..., t] of the combined
        [old_state, qkv_in] stream.
        """
        b, s, d = qkv.shape
        kernel_size = self.config.linear_conv_kernel_dim

        def _te_update_conv_state_history(old_state: te.Tensor, qkv_in: te.Tensor):
            ks_minus_1 = old_state.shape[1]
            seq = qkv_in.shape[1]

            # Per-position conv state history: out[bi, p, ti, di] = combined[bi, p+1+ti, di]
            # where combined = [old_state, qkv_in] (length ks_m1 + seq).
            return te.compute(
                (old_state.shape[0], seq, ks_minus_1, old_state.shape[2]),
                lambda bi, p, ti, di: tirx.if_then_else(
                    p + 1 + ti < ks_minus_1,
                    old_state[bi, p + 1 + ti, di],
                    qkv_in[bi, p + 1 + ti - ks_minus_1, di],
                ),
                name="update_conv_state_history",
            )

        conv_state_history = op.tensor_expr_op(
            _te_update_conv_state_history,
            "update_conv_state_history",
            [conv_state, qkv],
        )

        # Depthwise conv (same as the regular path)
        def _te_depthwise_conv(state: te.Tensor, qkv_in: te.Tensor, weight: te.Tensor):
            ks_m1 = state.shape[1]
            seq = qkv_in.shape[1]
            kk = te.reduce_axis((0, kernel_size), name="kk")
            return te.compute(
                (qkv_in.shape[0], seq, qkv_in.shape[2]),
                lambda bi, si, di: te.sum(
                    tirx.if_then_else(
                        si + kk < ks_m1,
                        state[bi, si + kk, di],
                        qkv_in[bi, si + kk - ks_m1, di],
                    )
                    * weight[di, 0, kk],
                    axis=kk,
                ),
                name="depthwise_conv1d",
            )

        result = op.tensor_expr_op(
            _te_depthwise_conv,
            "depthwise_conv1d",
            [conv_state, qkv, self.conv1d_weight],
            attrs={"op_pattern": 8},
        )
        return result, conv_state_history

    def _causal_conv1d_with_state(self, qkv: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: UP006
        """Causal Conv1D using a pre-extracted conv_state tensor (for RNNState path)."""
        b, s, d = qkv.shape
        kernel_size = self.config.linear_conv_kernel_dim

        # Update conv state
        def _te_update_conv_state(old_state: te.Tensor, qkv_in: te.Tensor):
            ks_minus_1 = old_state.shape[1]
            seq = qkv_in.shape[1]
            return te.compute(
                old_state.shape,
                lambda bi, ti, di: tirx.if_then_else(
                    seq + ti < ks_minus_1,
                    old_state[bi, seq + ti, di],
                    qkv_in[bi, seq + ti - ks_minus_1, di],
                ),
                name="update_conv_state",
            )

        new_conv_state = op.tensor_expr_op(
            _te_update_conv_state, "update_conv_state", [conv_state, qkv]
        )

        # Depthwise conv
        def _te_depthwise_conv(state: te.Tensor, qkv_in: te.Tensor, weight: te.Tensor):
            ks_m1 = state.shape[1]
            seq = qkv_in.shape[1]
            kk = te.reduce_axis((0, kernel_size), name="kk")
            return te.compute(
                (qkv_in.shape[0], seq, qkv_in.shape[2]),
                lambda bi, si, di: te.sum(
                    tirx.if_then_else(
                        si + kk < ks_m1,
                        state[bi, si + kk, di],
                        qkv_in[bi, si + kk - ks_m1, di],
                    )
                    * weight[di, 0, kk],
                    axis=kk,
                ),
                name="depthwise_conv1d",
            )

        result = op.tensor_expr_op(
            _te_depthwise_conv,
            "depthwise_conv1d",
            [conv_state, qkv, self.conv1d_weight],
            attrs={"op_pattern": 8},
        )
        return result, new_conv_state

    def _l2_normalize(self, x: Tensor) -> Tensor:
        """L2 normalize along last dimension with eps=1e-6."""
        # x: (b, s, h, d) — compute in float32 for numerical stability
        x_f32 = op.astype(x, "float32")
        x_sq = x_f32 * x_f32
        sum_sq = op.sum(x_sq, axis=-1, keepdims=True)  # (b, s, h, 1)
        inv_norm = op.sqrt(sum_sq + 1e-6)
        return op.astype(x_f32 / inv_norm, self.dtype)

    def _compute_gate_beta(self, alpha: Tensor, beta_raw: Tensor):
        """Compute decay gate and update rate.

        gate = exp(-exp(A_log) * softplus(alpha + dt_bias))  (per value_head)
        beta = sigmoid(beta_raw)  (per value_head)
        """

        # alpha: (b, s, n_vh), dt_bias: (n_vh,), A_log: (n_vh,)
        def _te_gate(alpha: te.Tensor, A_log: te.Tensor, dt_bias: te.Tensor):
            b, s, h = alpha.shape

            def _softplus(x):
                # softplus(x) = x if x > 20 else log(1 + exp(x))
                return tirx.if_then_else(x > 20.0, x, tirx.log(1.0 + tirx.exp(x)))

            return te.compute(
                (b, s, h),
                lambda bi, si, hi: tirx.exp(
                    -tirx.exp(A_log[hi].astype("float32"))
                    * _softplus((alpha[bi, si, hi] + dt_bias[hi]).astype("float32"))
                ),
                name="gate",
            )

        gate = op.tensor_expr_op(
            _te_gate,
            "gate",
            [alpha, self.A_log, self.dt_bias],
            attrs={"op_pattern": 8},
        )

        beta = op.sigmoid(beta_raw).astype("float32")
        return gate, beta

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype
        # A_log and dt_bias must stay float32
        self.A_log.to("float32")
        self.dt_bias.to("float32")


# ============================================================================
# In-place GDN recurrent-state access
# ============================================================================


@dataclasses.dataclass
class _GDNStateIO:
    """Hoisted handles that let the GDN kernel update its state slot in place.

    Holds the device-side slot-index arrays (shared by every layer) and one raw storage
    handle per GDN layer, so `Qwen35GatedDeltaNet.forward` can call the fused kernel
    instead of the `rnn_state_get` -> `gdn_func` -> `rnn_state_set` triple.

    These are emitted **once, above the layer loop** rather than inside each layer, and
    that placement is deliberate. TVM's cudagraph pass treats every `vm.builtin.*` call
    as non-static and calls `EndRegion()` on it
    ([rewrite_cuda_graph.cc:383](../../../3rdparty/tvm/src/relax/transform/rewrite_cuda_graph.cc#L383)),
    so emitting the handles per-layer would cut the capture region once more per layer on
    top of the cut the fused kernel itself causes. Hoisting keeps them in a single eager
    prologue; they are host-side handle fetches and launch no kernels.
    """

    seq_slot_ids: Tensor
    history_slot_ids: Tensor
    storages: Dict[int, Tensor]  # linear_layer_idx -> whole-storage handle for state 0
    conv_storages: Dict[int, Tensor]  # linear_layer_idx -> same, for state 1 (conv)


def _hoist_gdn_state_io(
    state: RNNState,
    linear_layer_ids: List[int],
    batch_size: tirx.PrimExpr,
    state_shape: Tuple[int, int, int],  # noqa: UP006
    conv_shape: Tuple[int, int],  # noqa: UP006
    conv_dtype: str,
) -> _GDNStateIO:
    """Emit the slot-id views and per-layer storage handles up front. See `_GDNStateIO`."""
    seq_slot_ids, history_slot_ids = state.slot_ids(batch_size)
    # The two leading storage dims are runtime `create_rnn_state` arguments, so nothing in
    # scope names them; fresh vars bound by `RNNState.storage`'s match_cast stand in. One
    # pair is shared by every layer — they all index the same geometry.
    max_batch_size = tirx.Var("rnn_max_batch_size", "int64")
    max_history = tirx.Var("rnn_max_history", "int64")
    return _GDNStateIO(
        seq_slot_ids=seq_slot_ids,
        history_slot_ids=history_slot_ids,
        # state_id 0 is the recurrent state: (max_batch, max_hist, n_vh, K, V).
        storages={
            idx: state.storage(idx, 0, (max_batch_size, max_history, *state_shape), "float32")
            for idx in linear_layer_ids
        },
        # state_id 1 is the conv state: (max_batch, max_hist, kernel_size - 1, qkv_dim),
        # in model dtype rather than fp32.
        conv_storages={
            idx: state.storage(idx, 1, (max_batch_size, max_history, *conv_shape), conv_dtype)
            for idx in linear_layer_ids
        },
    )


# ============================================================================
# Decoder Layer (dispatches between GDN and standard attention)
# ============================================================================


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, layer_id: int, category_id: int):
        """
        layer_id is the id of the layer within all of the layers
        category_id is the index of the layer within the category of layers that it belongs to
        ie, linear attention or regular attention
        """
        self.layer_type = config.layer_types()[layer_id]
        if self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(config)
        else:
            self.linear_attn = Qwen35GatedDeltaNet(config, category_id)
        self.category_id = category_id
        self.mlp = Qwen35MLP(config)
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
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,  # noqa: UP006
        state_io: Optional["_GDNStateIO"] = None,
    ):
        out = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            out = self.self_attn(out, paged_kv_cache, self.category_id, position_embeddings)
        else:
            # GDN ignores positions; mRoPE only routes to softmax-attn layers.
            out, state = self.linear_attn.forward(out, state, state_io)
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
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,  # noqa: UP006
    ):
        """Verify-path variant that scatters per-position GDN state into history slots."""
        out = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            out = self.self_attn(out, paged_kv_cache, self.category_id, position_embeddings)
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


# ============================================================================
# MTP (Multi-Token Prediction) head — draft for self-speculative decoding
# ============================================================================


class _Qwen35MTPDecoderLayer(nn.Module):
    """Single decoder block inside the MTP head. Always full attention (no GDN)."""

    def __init__(self, config: Qwen35Config):
        self.self_attn = Qwen35Attention(config)
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )


class Qwen35MTPHead(nn.Module):
    """Multi-Token Prediction head (DeepSeek-V3 / Qwen3.5 style).

    Given the previous step's hidden state and the embedding of the (just-sampled)
    token, predict the hidden state for the *next* token. The caller turns that
    hidden into logits via the shared lm_head (or tied embedding).

    KV cache slots: MTP's self-attention uses cache slots appended after the
    main model's attention slots — i.e. layer index `num_attention_layers + i`
    for the i-th MTP layer.
    """

    def __init__(self, config: Qwen35Config):
        self.pre_fc_norm_embedding = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )
        self.pre_fc_norm_hidden = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [_Qwen35MTPDecoderLayer(config) for _ in range(config.mtp_num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self._kv_layer_offset = config.num_attention_layers
        self._tp_shards = config.tensor_parallel_shards

    def forward(
        self,
        prev_hidden: Tensor,
        prev_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,  # noqa: UP006
    ) -> Tensor:
        # vLLM's qwen3_5_mtp.py:138 fuses as cat([embeds, hidden]) — embeds in
        # the FIRST half of fc's input. Reversed order zeros accept rate.
        e_norm = self.pre_fc_norm_embedding(prev_embed)
        h_norm = self.pre_fc_norm_hidden(prev_hidden)
        h = self.fc(op.concat([e_norm, h_norm], dim=-1))
        for i, layer in enumerate(self.layers):
            kv_layer_idx = self._kv_layer_offset + i
            residual = h
            x = layer.input_layernorm(h)
            x = layer.self_attn(x, paged_kv_cache, kv_layer_idx, position_embeddings)
            if self._tp_shards > 1:
                h = op.ccl_allreduce(x, "sum") + residual
            else:
                h = x + residual
            residual = h
            x = layer.post_attention_layernorm(h)
            x = layer.mlp(x)
            if self._tp_shards > 1:
                h = op.ccl_allreduce(x, "sum") + residual
            else:
                h = x + residual
        return self.norm(h)


class Qwen35Model(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.embed_tokens = Qwen35Embedding(config.vocab_size, config.hidden_size)
        layer_types = config.layer_types()
        linear_idx = 0
        attn_idx = 0
        layers = []
        for i, ltype in enumerate(layer_types):
            if ltype == "linear_attention":
                layers.append(Qwen35DecoderLayer(config, i, category_id=linear_idx))
                linear_idx += 1
            else:
                layers.append(Qwen35DecoderLayer(config, i, category_id=attn_idx))
                attn_idx += 1
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

        # Phase 10: instantiate the multimodal rotary table once when mRoPE is on.
        # rotary_dim = head_dim * partial_rotary_factor (0.8B: 256·0.25 = 64).
        # When config.mrope_section is None the embedding is not built and
        # forward()'s position_ids arg is ignored — existing fused-qkv path runs.
        self.use_mrope = config.mrope_section is not None
        if self.use_mrope:
            rotary_dim = int(config.head_dim * config.partial_rotary_factor)
            self.rotary_emb = MultimodalRotaryEmbedding(
                head_dim=config.head_dim,
                theta=float(config.rope_theta),
                mrope_section=config.mrope_section,
                rotary_dim=rotary_dim,
            )

    def forward(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        position_ids: Optional[Tensor] = None,
    ):
        hidden_states = inputs
        position_embeddings = None
        if self.use_mrope:
            assert position_ids is not None, (
                "mRoPE build requires position_ids in Qwen35Model.forward; got None."
            )
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        # Hoisted above the loop on purpose — see `_GDNStateIO`.
        # MLC_QWEN35_INPLACE_STATE=0 compiles the old get/set copy path instead, so an A/B
        # can be source- and flag-identical rather than a rebuild against an older tree.
        gdn_layers = [l for l in self.layers if l.layer_type != "full_attention"]
        state_io = None
        if gdn_layers and os.environ.get("MLC_QWEN35_INPLACE_STATE", "1") != "0":
            gdn = gdn_layers[0].linear_attn
            state_io = _hoist_gdn_state_io(
                state,
                [l.linear_attn.linear_layer_idx for l in gdn_layers],
                hidden_states.shape[0],
                (gdn.num_value_heads, gdn.key_head_dim, gdn.value_head_dim),
                # conv1d_weight is (qkv_dim, 1, kernel_size), so its leading dim is the
                # conv width — read it off the parameter rather than re-deriving it.
                (gdn.config.linear_conv_kernel_dim - 1, gdn.conv1d_weight.shape[0]),
                gdn.dtype,
            )
        for layer_id, layer in enumerate(self.layers):
            hidden_states, state = layer.forward(
                hidden_states, paged_kv_cache, state, position_embeddings, state_io
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states, state

    def forward_with_history(
        self,
        inputs: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        position_ids: Optional[Tensor] = None,
    ):
        hidden_states = inputs
        position_embeddings = None
        if self.use_mrope:
            assert position_ids is not None, (
                "mRoPE build requires position_ids in Qwen35Model.forward_with_history; got None."
            )
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        for layer_id, layer in enumerate(self.layers):
            hidden_states, state = layer.forward_with_history(
                hidden_states, paged_kv_cache, state, position_embeddings
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states, state


class Qwen35LMHeadModel(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.config = config
        self.model = Qwen35Model(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.mtp_num_hidden_layers > 0:
            self.mtp = Qwen35MTPHead(config)
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
        # GDN config
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

    def _forward(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        logit_positions: Optional[Tensor] = None,
    ):
        """Shared forward for batch methods using RNNState."""
        op_ext.configure()
        hidden_states, state = self.model.forward(input_embed, paged_kv_cache, state)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self._lm_head(hidden_states)
        return logits, paged_kv_cache, state

    def _forward_with_history(
        self,
        input_embed: Tensor,
        paged_kv_cache: PagedKVCache,
        state: RNNState,
        logit_positions: Optional[Tensor] = None,
    ):
        """Prefill-with-history forward: scatters per-position GDN state into RNNState
        history slots so the radix prefix cache can roll the recurrent state back to any
        intermediate position via PopN. Pair with `set_use_history_mode(True)` on the
        engine side before BeginForward so EndForward advances `available_history_num`
        by `seq_len` rather than capping at 0 for the multi-token append.
        """
        op_ext.configure()
        hidden_states, state = self.model.forward_with_history(
            input_embed, paged_kv_cache, state
        )
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self._lm_head(hidden_states)
        return logits, paged_kv_cache, state

    def _lm_head(self, hidden_states: Tensor) -> Tensor:
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

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
        # Verify uses the per-position-history GDN forward so partial accept can roll
        # the recurrent state back to the accepted prefix bit-exactly via PopN.
        # Pair this with a `set_use_history_mode(True)` call from the engine before
        # BeginForward so EndForward advances `available_history_num` by `seq_len`.
        return self._forward_to_last_hidden_with_history(
            input_embeds, paged_kv_cache, rnn_state
        )

    def mtp_decode(
        self,
        input_embeds: Tensor,
        prev_hidden: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        """One step of the MTP draft head.

        Args:
            input_embeds: (batch, 1, hidden) — embedding of the just-sampled token.
            prev_hidden: (batch, 1, hidden) — hidden state from the prior step
                (target model's last hidden, or prior MTP step's output).
        Returns:
            (logits, paged_kv_cache) — logits for the *next* token.
        """
        op_ext.configure()
        h = self.mtp(prev_hidden, input_embeds, paged_kv_cache)
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(h)
        else:
            logits = self.lm_head(h)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

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
            # Only attention layers use the KV cache
            # MTP attention reuses the same paged KV cache; append one slot per MTP layer.
            num_hidden_layers=self.num_attention_layers + self.config.mtp_num_hidden_layers,
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
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
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
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
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
        if self.config.mtp_num_hidden_layers > 0:
            mod_spec["mtp_decode"] = {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "prev_hidden": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)

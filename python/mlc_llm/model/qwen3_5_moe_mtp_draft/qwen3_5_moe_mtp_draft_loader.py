"""HuggingFace parameter mapping for the Qwen3.5-MoE MTP draft model (35B-A3B).

Pulls only the weights this draft artifact needs:
  - embed_tokens.weight              <- model.language_model.embed_tokens.weight
  - pre_fc_norm_{embedding,hidden}, fc, norm  <- mtp.<same>
  - layers.0.{self_attn, mlp, *_layernorm}    <- mtp.layers.0.<same>

Layouts vs the released checkpoint:
- self_attn: q+k+v fused into c_attn (concat axis=0).
- MoE routed experts: HF pre-stacks into single tensors with NO `.weight` suffix:
    `mtp.layers.0.mlp.experts.gate_up_proj`  shape (num_experts, 2*moe_intermediate, hidden)
    `mtp.layers.0.mlp.experts.down_proj`     shape (num_experts, hidden, moe_intermediate)
  These map directly to `MixtralExperts.weight` — direct passthrough.
- Shared expert: HF keeps gate/up/down as three linears; fuse gate+up.
- Standard RMSNorms use `output = norm(x) * (1 + weight)`, so add 1.0 at load time.
  Excludes router (`gate.weight`) and shared_expert_gate (these are linears, not norms).
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen3_5_moe_mtp_draft_model import Qwen35MoEMTPDraftConfig, Qwen35MoEMTPDraftLM


def huggingface(model_config: Qwen35MoEMTPDraftConfig, quantization: Quantization) -> ExternMapping:
    model = Qwen35MoEMTPDraftLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.mtp_num_hidden_layers):
        mlc_attn = f"layers.{i}.self_attn"
        hf_attn = f"mtp.layers.{i}.self_attn"
        mlc_name = f"{mlc_attn}.c_attn.weight"
        if mlc_name in named_parameters:
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{hf_attn}.q_proj.weight",
                    f"{hf_attn}.k_proj.weight",
                    f"{hf_attn}.v_proj.weight",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

        mlc_mlp = f"layers.{i}.mlp"
        hf_mlp = f"mtp.layers.{i}.mlp"

        # Routed experts: HF pre-stacks into [num_experts, out, in] — direct passthrough,
        # no .weight suffix on the HF names.
        for mlc_leaf, hf_leaf in [
            ("moe_gate_up_proj.weight", "experts.gate_up_proj"),
            ("moe_down_proj.weight", "experts.down_proj"),
        ]:
            mlc_name = f"{mlc_mlp}.{mlc_leaf}"
            if mlc_name in named_parameters:
                mlc_param = named_parameters[mlc_name]
                mapping.add_mapping(
                    mlc_name,
                    [f"{hf_mlp}.{hf_leaf}"],
                    functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
                )

        # Shared expert: HF keeps gate/up/down as three linears — fuse gate+up.
        mlc_name = f"{mlc_mlp}.shared_expert.gate_up_proj.weight"
        if mlc_name in named_parameters:
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{hf_mlp}.shared_expert.gate_proj.weight",
                    f"{hf_mlp}.shared_expert.up_proj.weight",
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    def _mlc_to_hf(mlc_name: str) -> str:
        if mlc_name == "embed_tokens.weight":
            return "model.language_model.embed_tokens.weight"
        # Everything else lives under top-level `mtp.*` in HF, mirroring our flat layout.
        return f"mtp.{mlc_name}"

    def _is_rmsnorm_weight(name: str) -> bool:
        # Standard RMSNorms (apply +1.0 offset). Excludes router (`mlp.gate.weight`)
        # and shared_expert_gate (these are linears, not norms).
        return (
            name.endswith("input_layernorm.weight")
            or name.endswith("post_attention_layernorm.weight")
            or name.endswith("q_norm.weight")
            or name.endswith("k_norm.weight")
            or name == "norm.weight"
            or name == "pre_fc_norm_embedding.weight"
            or name == "pre_fc_norm_hidden.weight"
        )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name in mapping.param_map:
            continue
        hf_name = _mlc_to_hf(mlc_name)
        if _is_rmsnorm_weight(mlc_name):
            mapping.add_mapping(
                mlc_name,
                [hf_name],
                functools.partial(
                    lambda x, dtype: (x.astype("float32") + 1.0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
        else:
            mapping.add_mapping(
                mlc_name,
                [hf_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    return mapping

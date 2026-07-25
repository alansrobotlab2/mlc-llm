"""
HuggingFace parameter mapping for Qwen3.5-MoE (Qwen3.6-35B-A3B).

The released checkpoint is a multimodal architecture (`Qwen3_5MoeForConditionalGeneration`),
so the text backbone is nested under `model.language_model.*`. Vision tower
(`model.visual.*`) and MTP head (`mtp.*`) are dropped.

Notable HF layouts:
- Linear-attention sub-projections are unfused in HF (`in_proj_qkv` + separate
  `in_proj_z`/`_a`/`_b`). The MLC layer fuses all four into `in_proj_qkvzab`, so
  this loader concatenates them along the output axis — same treatment as q/k/v.
- MoE experts are pre-stacked into single tensors:
    layers.{i}.mlp.experts.gate_up_proj  shape (num_experts, 2*moe_intermediate, hidden)
    layers.{i}.mlp.experts.down_proj     shape (num_experts, hidden, moe_intermediate)
  These map directly to MixtralExperts.weight; no concat/stack needed.
- Shared expert is three separate linears; we fuse gate+up into gate_up_proj.
- `lm_head.weight` is at the top level, not under `model.language_model.`.
- Standard RMSNorms use `output = norm(x) * (1 + weight)`, so we add 1.0 at load
  time. The gated norm inside linear_attn does NOT get the offset.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen3_5_moe_model import Qwen35MoEConfig, Qwen35MoEForCausalLM


def huggingface(model_config: Qwen35MoEConfig, quantization: Quantization) -> ExternMapping:
    model = Qwen35MoEForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    hf = "model.language_model"
    layer_types = model_config.layer_types()

    for i in range(model_config.num_hidden_layers):
        if layer_types[i] == "full_attention":
            mlc_attn = f"model.layers.{i}.self_attn"
            hf_attn = f"{hf}.layers.{i}.self_attn"
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
        else:
            mlc_lin = f"model.layers.{i}.linear_attn"
            hf_lin = f"{hf}.layers.{i}.linear_attn"

            # HF ships in_proj_qkv / _z / _a / _b separately; the MLC layer fuses them
            # into one GEMV. Order must match the split in Qwen35GatedDeltaNet._in_proj.
            mlc_name = f"{mlc_lin}.in_proj_qkvzab.weight"
            if mlc_name in named_parameters:
                mlc_param = named_parameters[mlc_name]
                mapping.add_mapping(
                    mlc_name,
                    [
                        f"{hf_lin}.in_proj_qkv.weight",
                        f"{hf_lin}.in_proj_z.weight",
                        f"{hf_lin}.in_proj_a.weight",
                        f"{hf_lin}.in_proj_b.weight",
                    ],
                    functools.partial(
                        lambda qkv, z, a, bb, dtype: np.concatenate(
                            [qkv, z, a, bb], axis=0
                        ).astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )

            for param_name in ["A_log", "dt_bias"]:
                mlc_name = f"{mlc_lin}.{param_name}"
                if mlc_name in named_parameters:
                    mlc_param = named_parameters[mlc_name]
                    mapping.add_mapping(
                        mlc_name,
                        [f"{hf_lin}.{param_name}"],
                        functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
                    )

            mlc_name = f"{mlc_lin}.conv1d_weight"
            if mlc_name in named_parameters:
                mlc_param = named_parameters[mlc_name]
                mapping.add_mapping(
                    mlc_name,
                    [f"{hf_lin}.conv1d.weight"],
                    functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
                )

        mlc_mlp = f"model.layers.{i}.mlp"
        hf_mlp = f"{hf}.layers.{i}.mlp"

        # Routed experts: HF pre-stacks into [num_experts, out, in] — direct passthrough.
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
        if mlc_name.startswith("model."):
            return mlc_name.replace("model.", f"{hf}.", 1)
        return mlc_name

    def _is_rmsnorm_weight(name: str) -> bool:
        return (
            name.endswith("input_layernorm.weight")
            or name.endswith("post_attention_layernorm.weight")
            or name.endswith("q_norm.weight")
            or name.endswith("k_norm.weight")
            or name == "model.norm.weight"
        )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
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

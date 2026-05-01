"""HuggingFace parameter mapping for Qwen3.5-VL.

Extends ``qwen35_loader`` with the ``visual.*`` weight mapping. The text path
is byte-equivalent to ``Qwen35LMHeadModel`` — same Q/K/V→c_attn fusion, same
gate/up→gate_up_proj fusion, same RMSNorm +1.0 offset.

HF layout:
  model.language_model.{layers, embed_tokens, norm}.*    → MLC: model.*
  model.visual.{patch_embed, pos_embed, blocks, merger}.*  → MLC: visual.*
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen3_5_vl_model import Qwen35VLConfig, Qwen35VLLMHeadModel


HF_LM_PREFIX = "model.language_model"
HF_VISUAL_PREFIX = "model.visual"


def huggingface(model_config: Qwen35VLConfig, quantization: Quantization) -> ExternMapping:
    model = Qwen35VLLMHeadModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    # ── Text backbone: replicate qwen35_loader's fusions ────────────────────
    layer_types = model_config.layer_types()
    for i in range(model_config.num_hidden_layers):
        if layer_types[i] == "full_attention":
            mlc_attn = f"model.layers.{i}.self_attn"
            hf_attn = f"{HF_LM_PREFIX}.layers.{i}.self_attn"
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
            hf_lin = f"{HF_LM_PREFIX}.layers.{i}.linear_attn"

            mlc_name = f"{mlc_lin}.in_proj_qkv.weight"
            if mlc_name in named_parameters:
                mlc_param = named_parameters[mlc_name]
                mapping.add_mapping(
                    mlc_name,
                    [f"{hf_lin}.in_proj_qkv.weight"],
                    functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
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

        # MLP gate/up fusion
        mlc_mlp = f"model.layers.{i}.mlp"
        hf_mlp = f"{HF_LM_PREFIX}.layers.{i}.mlp"
        mlc_name = f"{mlc_mlp}.gate_up_proj.weight"
        if mlc_name in named_parameters:
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{hf_mlp}.gate_proj.weight",
                    f"{hf_mlp}.up_proj.weight",
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    def _mlc_to_hf(mlc_name: str) -> str:
        """MLC → HF translation. Visual params live under model.visual.*; text
        backbone under model.language_model.*; nothing else (MTP disabled in v1).
        """
        if mlc_name.startswith("visual."):
            return mlc_name.replace("visual.", f"{HF_VISUAL_PREFIX}.", 1)
        if mlc_name.startswith("model."):
            return mlc_name.replace("model.", f"{HF_LM_PREFIX}.", 1)
        return mlc_name

    def _is_rmsnorm_weight(name: str) -> bool:
        """RMSNorm weights for the LM backbone use the +1.0 offset trick.
        Vision tower uses regular LayerNorm — does NOT need the offset.
        """
        if name.startswith("visual."):
            return False
        return (
            name.endswith("input_layernorm.weight")
            or name.endswith("post_attention_layernorm.weight")
            or name.endswith("q_norm.weight")
            or name.endswith("k_norm.weight")
            or name == "model.norm.weight"
        )

    # ── Remaining params: 1:1 mapping with prefix translation ───────────────
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

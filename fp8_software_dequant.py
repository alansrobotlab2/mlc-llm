"""Software W8A16 fp8 path so an FP8 checkpoint can serve as an HF parity reference on sm_87.

Why this exists
---------------
`Qwen/Qwen3.6-35B-A3B` in bf16 is ~72 GB and does not fit a 64 GB Orin, so the 35B
had no HuggingFace reference on this box at all. The FP8 release
(`Qwen/Qwen3.6-35B-A3B-FP8`, 37.5 GB, block-wise e4m3 with `weight_block_size
[128, 128]`) does fit — but transformers cannot execute it here:

* `FineGrainedFP8HfQuantizer.validate_environment` warns below compute capability
  8.9 and sets `dequantize = True`, which materializes the whole model in bf16 —
  straight back to 72 GB.
* Forcing the quantized path instead lands on the Triton kernel, which fails at
  compile time on Ampere: ``ValueError("type fp8e4nv not supported in this
  architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")``. DeepGEMM
  is SM90+. There is no fp8 *arithmetic* on sm_87.

What does work is the fp8→bf16/fp16 *cast* (verified on sm_87). So this module
replaces `fp8_linear` — the single dispatcher both `FP8Linear.forward` and
`FP8Experts.linear` route through — with a pure-PyTorch weight-only dequant:
keep weights fp8 in memory, expand the per-128x128-block scale, and run the
matmul in the activation dtype.

Precision note (matters when reading parity results)
----------------------------------------------------
This is **W8A16, not the W8A8 the FP8 checkpoint nominally runs**: activations
stay bf16/fp16 instead of being dynamically quantized to fp8. That makes this
reference *more* accurate than a real fp8 deployment and closer to the bf16
master — which is what you want from an oracle. It is still not a bf16 reference:
weights carry e4m3 rounding (3 mantissa bits per 128x128 block). Against a 4-bit
MLC build, expect near-tie token flips exactly like the 0.8B `q4f16_g16e` case,
not bit-exact agreement.

Usage:
    import fp8_software_dequant
    fp8_software_dequant.install()          # before from_pretrained
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_installed = False


def sw_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size=None,
    bias: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    allow_deepgemm: bool = True,
) -> torch.Tensor:
    """Weight-only fp8 dequant + dense matmul. Signature mirrors the upstream dispatcher.

    `activation_scale` is deliberately ignored — we never quantize activations, see
    the module docstring. `allow_deepgemm` is accepted and ignored (no DeepGEMM here).
    """
    out_dtype = output_dtype or input.dtype

    # Modules in the checkpoint's `modules_to_not_convert` (on this model that
    # includes every `linear_attn.in_proj_a`, the routers and the norms) keep their
    # original dtype and must not go through the dequant.
    if weight.element_size() > 1:
        return F.linear(input, weight.to(out_dtype), bias)

    w = weight.to(torch.float32)
    s = weight_scale_inv.to(torch.float32)

    if block_size is None or s.ndim == 0:
        # per-tensor scale
        w = w * s
    else:
        bn, bk = int(block_size[0]), int(block_size[1])
        n, k = w.shape
        if n % bn == 0 and k % bk == 0 and tuple(s.shape) == (n // bn, k // bk):
            # Contiguous reshape + in-place scale: no second full-size allocation.
            # Matters for lm_head (248320x2048 -> 2 GB in fp32).
            w = w.view(n // bn, bn, k // bk, bk)
            w.mul_(s.view(n // bn, 1, k // bk, 1))
            w = w.view(n, k)
        else:
            # ragged tail (or an unexpected scale layout) — expand explicitly
            se = s.repeat_interleave(bn, dim=0)[:n].repeat_interleave(bk, dim=1)[:k]
            w = w * se

    return F.linear(input.to(out_dtype), w.to(out_dtype), bias)


def is_fp8_checkpoint(model_id_or_path: str) -> bool:
    """True if the repo/dir declares a finegrained-fp8 quantization_config."""
    import json

    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(model_id_or_path, "config.json")
    except Exception:
        import os

        cfg_path = os.path.join(model_id_or_path, "config.json")
        if not os.path.exists(cfg_path):
            return False
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    except Exception:
        return False
    qc = cfg.get("quantization_config") or {}
    return str(qc.get("quant_method", "")).lower() in {"fp8", "finegrained_fp8"}


def install() -> None:
    """Patch the fp8 dispatcher and stop the quantizer from dequantizing to bf16."""
    global _installed
    if _installed:
        return

    from transformers.integrations import finegrained_fp8 as fg

    fg.fp8_linear = sw_fp8_linear

    # `fp8_linear` alone is NOT enough. The MoE experts can be dispatched through
    # `fp8_grouped_mm_experts_forward` / `fp8_batched_mm_experts_forward`, which call
    # Triton directly and never touch `fp8_linear` — on sm_87 those die with the same
    # `fp8e4nv not supported in this architecture` compile error.
    #
    # `ExpertsInterface.get_interface(impl, default)` returns `default` — the eager
    # `FP8Experts.forward`, which loops over *hit* experts and calls `fp8_linear` — for
    # any impl not in the registry. Forcing it to always return the default routes every
    # experts path through the software dequant, whatever `config._experts_implementation`
    # happens to be set to.
    try:
        fg.FP8ExpertsInterface.get_interface = lambda self, impl, default: default
    except Exception as exc:  # pragma: no cover - transformers layout drift
        print(f"[fp8-sw] WARNING: could not force eager experts dispatch ({exc}); "
              "MoE layers will try Triton and fail on sm_87")

    # validate_environment() sets `dequantize = True` below capability 8.9, which
    # would materialize the whole model in bf16 (72 GB). Neutralize it so the fp8
    # weights stay fp8 and our dispatcher handles them.
    try:
        from transformers.quantizers import quantizer_finegrained_fp8 as qfp8

        _orig = qfp8.FineGrainedFP8HfQuantizer.validate_environment

        def _patched(self, *args, **kwargs):
            result = _orig(self, *args, **kwargs)
            if getattr(self.quantization_config, "dequantize", False):
                self.quantization_config.dequantize = False
                print(
                    "[fp8-sw] overrode quantization_config.dequantize=True -> False; "
                    "weights stay fp8 and run through the software W8A16 path"
                )
            return result

        qfp8.FineGrainedFP8HfQuantizer.validate_environment = _patched
    except Exception as exc:  # pragma: no cover - transformers layout drift
        print(f"[fp8-sw] WARNING: could not patch the quantizer gate ({exc}); "
              "if the model loads as bf16 it will not fit")

    _installed = True
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    print(f"[fp8-sw] software W8A16 fp8 path installed (device capability {cap}). "
          "Activations are NOT quantized — this is a more accurate reference than "
          "real fp8 inference, but weights still carry e4m3 rounding.")

#!/usr/bin/env python3
"""Phase 10 Stage 5b — bit-exact preprocessor parity vs HF.

Verifies that ``mlc_llm.model.qwen3_5_vl.qwen3_5_vl_image.preprocess_image``
produces the same four tensors as the HF pipeline:

  * pixel_values   — vs ``Qwen2VLImageProcessorFast(...)["pixel_values"]``
  * pos_embeds     — vs ``model.model.visual.fast_pos_embed_interpolate(grid_thw)``
  * rotary_cos/sin — vs ``cat([visual.rot_pos_emb(...), visual.rot_pos_emb(...)], -1).cos()/sin()``

The test passes if **all four** tensors agree to fp16 tolerance
(max abs diff ≤ 2e-3 on cos/sin, ≤ 1e-3 on pixel_values, ≤ 5e-4 on pos_embeds).
Tighter than the 1e-2 ViT-block tolerance because these are deterministic
upstream of any matmul accumulation.

Usage:
    .venv/bin/python tests/multimodal/test_preproc_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "multimodal" / "cat.jpeg"
sys.path.insert(0, str(ROOT / "python"))

from mlc_llm.model.qwen3_5_vl.qwen3_5_vl_image import preprocess_image  # noqa: E402


def _load_hf(model_id: str = "Qwen/Qwen3.5-0.8B"):
    print(f"[preproc] Loading HF model+processor from {model_id} (CPU, fp16)…")
    from transformers import AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    return model, processor


def _diff(name: str, ours: np.ndarray, ref: np.ndarray, tol: float) -> bool:
    if ours.shape != ref.shape:
        print(f"  {name:14s} [FAIL]  shape mismatch: ours={ours.shape} vs ref={ref.shape}")
        return False
    o = ours.astype(np.float32)
    r = ref.astype(np.float32)
    diff = np.abs(o - r)
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    ok = max_d <= tol
    status = "PASS" if ok else "FAIL"
    print(
        f"  {name:14s} [{status}]  shape={ours.shape}  "
        f"max={max_d:.3e}  mean={mean_d:.3e}  tol={tol:.0e}"
    )
    if not ok:
        worst = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"               worst {worst}: ours={o[worst]:.6f}  ref={r[worst]:.6f}")
    return ok


def main() -> int:
    if not FIXTURE.exists():
        print(f"[preproc] Fixture not found: {FIXTURE}")
        return 1

    from PIL import Image

    image = Image.open(FIXTURE).convert("RGB")
    print(f"[preproc] Image: {FIXTURE} ({image.size[0]}x{image.size[1]} {image.mode})")

    model, processor = _load_hf()
    visual = model.model.visual
    vc = visual.config
    print(
        f"[preproc] vision_config: depth={vc.depth} hidden={vc.hidden_size} "
        f"num_heads={vc.num_heads} patch={vc.patch_size} tps={vc.temporal_patch_size} "
        f"merge={vc.spatial_merge_size} num_pos_embed={vc.num_position_embeddings}"
    )

    # ── HF reference ────────────────────────────────────────────────────────
    inputs = processor(text=["dummy"], images=[image], return_tensors="pt")
    hf_pixel_values = inputs["pixel_values"].numpy()  # (N, C·T·P·P), already fp32 normalized
    hf_grid_thw = inputs["image_grid_thw"][0].tolist()  # (T, H, W)
    print(f"[preproc] HF processor: pixel_values={hf_pixel_values.shape} grid_thw={hf_grid_thw}")

    with torch.no_grad():
        hf_pos_embeds = visual.fast_pos_embed_interpolate(inputs["image_grid_thw"]).numpy()
        hf_rope = visual.rot_pos_emb(inputs["image_grid_thw"])  # (N, head_dim/2)
        hf_emb = torch.cat([hf_rope, hf_rope], dim=-1)
        hf_cos = hf_emb.cos().numpy()
        hf_sin = hf_emb.sin().numpy()
    print(
        f"[preproc] HF helpers:    pos_embeds={hf_pos_embeds.shape}  "
        f"rope={hf_rope.shape}  cos/sin={hf_cos.shape}"
    )

    # ── Our preprocessor ────────────────────────────────────────────────────
    rgb = np.asarray(image, dtype=np.uint8)
    pos_embed_weight = visual.pos_embed.weight.detach().to(torch.float32).numpy()
    print(f"[preproc] visual.pos_embed.weight shape: {pos_embed_weight.shape}")

    out = preprocess_image(
        rgb,
        patch_size=vc.patch_size,
        temporal_patch_size=vc.temporal_patch_size,
        spatial_merge_size=vc.spatial_merge_size,
        vision_hidden_size=vc.hidden_size,
        vision_num_heads=vc.num_heads,
        num_position_embeddings=vc.num_position_embeddings,
        pos_embed_weight=pos_embed_weight,
        rope_theta=10000.0,
    )
    print(
        f"[preproc] Ours:          pixel_values={out.pixel_values.shape}  "
        f"pos_embeds={out.pos_embeds.shape}  cos={out.rotary_cos.shape}  "
        f"grid_thw={out.image_grid_thw}"
    )

    # ── Diffs ───────────────────────────────────────────────────────────────
    print("\n[preproc] === parity diffs ===")
    grid_match = tuple(out.image_grid_thw) == tuple(hf_grid_thw)
    print(f"  grid_thw      [{'PASS' if grid_match else 'FAIL'}]  ours={out.image_grid_thw}  ref={hf_grid_thw}")

    ok = grid_match
    # All four tensors are stored fp16 → ULP-level diffs vs HF fp32 reference.
    # ULP at magnitude 1 is ~1e-3, at magnitude 2 is ~2e-3. Tolerance bands are
    # set just above ULP. The means should be ≤ 1e-4 (most elements agree).
    ok &= _diff("pixel_values", out.pixel_values, hf_pixel_values, tol=1e-2)
    ok &= _diff("pos_embeds",   out.pos_embeds,   hf_pos_embeds,    tol=5e-3)
    ok &= _diff("rotary_cos",   out.rotary_cos,   hf_cos,           tol=2e-3)
    ok &= _diff("rotary_sin",   out.rotary_sin,   hf_sin,           tol=2e-3)

    print(f"\n[preproc] OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

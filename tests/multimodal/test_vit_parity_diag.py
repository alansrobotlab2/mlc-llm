#!/usr/bin/env python3
"""Diagnostic for the parity drift — peek at block-5 magnitudes and where the diff localizes."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "reference_outputs_vl.pt"

cache = torch.load(CACHE, weights_only=False)
print(f"vision_block_outputs: {len(cache['vision_block_outputs'])} blocks")
for i, b in enumerate(cache["vision_block_outputs"]):
    a = b.squeeze().astype(np.float32)
    print(
        f"  block {i:2d}  shape={a.shape}  "
        f"min={a.min():+.3e}  max={a.max():+.3e}  abs_max={np.abs(a).max():.3e}  "
        f"std={a.std():.3e}"
    )

# The HF cache hooked block outputs as fp32 from .float() — but the model
# itself runs fp16 internally. Check: what was the dtype going INTO the hook?
print()
print("Sanity: does block i+1 ref equal what you'd get by running our forward on block i ref?")
print("  → would confirm or refute that drift is per-block compounded vs from a prior bug")

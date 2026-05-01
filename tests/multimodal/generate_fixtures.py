#!/usr/bin/env python3
"""
Deterministic image fixtures for Phase 10 multimodal parity.

Outputs PNGs sibling to this script. Re-running must produce byte-identical
files; the parity cache (`reference_outputs_vl.pt`) keys off these images.

Resolution: 448x448 (28x28 patches at patch_size=16 → 14x14=196 vision tokens
after spatial_merge_size=2). Single-image grid_thw = (1, 28, 28).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SIZE = 448  # multiple of patch_size * spatial_merge_size = 16 * 2 = 32


def _checker_gradient() -> np.ndarray:
    """Checkerboard + RGB radial gradient. Deterministic, ViT-friendly content."""
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cx = cy = (SIZE - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (SIZE / 2.0)

    checker = ((x // 32 + y // 32) % 2).astype(np.float32)

    red = (1.0 - r) * 255.0
    green = (np.sin(x / SIZE * np.pi * 4) * 0.5 + 0.5) * 255.0
    blue = (np.cos(y / SIZE * np.pi * 4) * 0.5 + 0.5) * 255.0

    img = np.stack([red, green, blue], axis=-1)
    img = img * (0.6 + 0.4 * checker[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    arr = _checker_gradient()
    out = HERE / "fixture_448.png"
    Image.fromarray(arr, mode="RGB").save(out, format="PNG", optimize=False, compress_level=6)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({out.stat().st_size} bytes, sha256={sha[:16]}…)")


if __name__ == "__main__":
    main()

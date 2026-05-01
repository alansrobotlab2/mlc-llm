# Multimodal parity fixtures

Inputs for the Phase 10 vision-input parity harness ([validate.py](../../validate.py)
multimodal modes). Both the cached HF reference (`reference_outputs_vl.pt`) and
the MLC-side parity check key off these images — do not edit without
regenerating the cache.

## Files

| file | size | sha256 (truncated) | source |
|---|---|---|---|
| `cat.jpeg` | 960×686 | `88b9eb3e…21e5` | HF documentation-images `pipeline-cat-chonk.jpeg` — the same image referenced in `transformers/models/qwen3_5/modeling_qwen3_5.py` docstrings |
| `fixture_448.png` | 448×448 | `a2653976…7335` | Synthetic (checkerboard + radial RGB gradient). Deterministic offline backup |

## Reproducing

```sh
# Canonical: download the HF docstring reference image
curl -fsSL https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg \
    -o tests/multimodal/cat.jpeg

# Synthetic backup: regenerate from the script
python tests/multimodal/generate_fixtures.py
```

The synthetic generator must be byte-deterministic across reruns (PIL/PNG
encoder pinned via `compress_level=6`, `optimize=False`).

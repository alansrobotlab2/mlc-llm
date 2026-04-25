#!/usr/bin/env python3
"""
Parity harness for Qwen3.5-0.8B (and later Qwen3.6-35B-A3B).

Modes:
  --reference-only   Load HF model, greedy-generate, dump per-layer hidden states.
                     Caches to reference_outputs.pt. Run this first.

  --greedy-parity    Compare 50-token greedy decode between HF cache and MLC.
                     Requires a compiled MLC model dir (--mlc-model-dir).

  --layer-parity     Compare per-layer decoder hidden states between HF cache and
                     MLC. Requires a compiled MLC model with debug output support
                     (see Stage 3 in worklog.md).

Usage examples:
  # Stage 1: build reference cache
  python validate.py --reference-only --model Qwen/Qwen3.5-0.8B

  # Stage 4: greedy parity
  python validate.py --greedy-parity --model Qwen/Qwen3.5-0.8B \
      --mlc-model-dir dist/qwen3_5-0.8B-q0f16

  # Debug a single GDN layer
  python validate.py --reference-only --model Qwen/Qwen3.5-0.8B --debug-layer 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Fixed prompts used for all parity runs — do not change without regenerating
# reference_outputs.pt.
# ──────────────────────────────────────────────────────────────────────────────
FIXED_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "<|im_start|>user\nWhat is 7 multiplied by 6?<|im_end|>\n<|im_start|>assistant\n",
    "Once upon a time in a land far away, there lived a wise old owl who knew the secret of",
    "1, 1, 2, 3, 5, 8, 13, 21,",
]

CACHE_FILE = Path("reference_outputs.pt")
GREEDY_N = 50
LAYER_RTOL = 1e-3
LAYER_ATOL = 1e-3
LAYER_ATOL_LINEAR = 2e-3  # linear (GDN) layers accumulate more rounding
TOKEN_MATCH_BAR = 48  # out of GREEDY_N

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HF model ID or local path")
    p.add_argument("--cache", default=str(CACHE_FILE), help="Path to reference_outputs.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reference-only", action="store_true", help="Run HF reference; cache outputs.")
    mode.add_argument("--greedy-parity", action="store_true", help="Compare greedy decode vs MLC.")
    mode.add_argument("--layer-parity", action="store_true", help="Compare per-layer hidden states vs MLC.")

    p.add_argument("--mlc-model-dir", default=None, help="Path to compiled MLC model directory (for --greedy-parity / --layer-parity).")
    p.add_argument("--mlc-lib", default=None, help="Path to compiled MLC .so (if not auto-found in model dir).")
    p.add_argument("--debug-layer", type=int, default=None, help="Dump detailed GDN sub-step values for this layer index.")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Model dtype for HF reference (keep fp16 for parity; fp32 for debugging).")
    p.add_argument("--regen", action="store_true", help="Force regeneration even if cache exists.")
    p.add_argument("--no-layer-hooks", action="store_true", help="Skip per-layer hidden-state capture during --reference-only (greedy-only mode; needed for 35B to keep host RAM bounded).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# HF reference: load model + tokenizer
# ──────────────────────────────────────────────────────────────────────────────


def load_hf(model_id: str, dtype: str, device: str):
    """Load Qwen3.5-0.8B from HuggingFace (downloads on first run to ~/.cache/huggingface)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[ref] Loading tokenizer from {model_id!r}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    print(f"[ref] Loading model {model_id!r} on {device}, dtype={dtype}")

    if device.startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info(device)
        print(f"[ref] {device} mem before load: free={free_b/1e9:.2f} GiB / total={total_b/1e9:.2f} GiB")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        attn_implementation="eager",
        trust_remote_code=True,
        device_map=device,
        low_cpu_mem_usage=True,
    ).eval()

    # Force the requested dtype — the `dtype` kwarg is silently ignored on some
    # transformers builds and the model lands at the checkpoint's native dtype.
    if dtype == "float16":
        model = model.half()
    elif dtype == "float32":
        model = model.float()

    if device.startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info(device)
        print(f"[ref] {device} mem after load:  free={free_b/1e9:.2f} GiB / total={total_b/1e9:.2f} GiB")

    n_params = sum(p.numel() for p in model.parameters())
    p0 = next(model.parameters())
    print(f"[ref] Model loaded — {n_params/1e6:.1f}M parameters, dtype={p0.dtype}, device={p0.device}")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Hook management
# ──────────────────────────────────────────────────────────────────────────────


class HookRecorder:
    """Collects hidden states from registered forward hooks."""

    def __init__(self):
        self.layer_outputs: list[torch.Tensor] = []
        self.debug: dict[str, list[torch.Tensor]] = {}
        self._handles: list = []

    def register_decoder_hooks(self, model, debug_layer: Optional[int] = None):
        """Hook every decoder layer to capture output hidden states."""
        layers = _get_layers(model)
        for i, layer in enumerate(layers):
            handle = layer.register_forward_hook(self._make_layer_hook(i))
            self._handles.append(handle)

        if debug_layer is not None:
            self._register_gdn_debug_hooks(layers, debug_layer)

    def _make_layer_hook(self, idx: int):
        def hook(module, input, output):
            # Decoder layers return (hidden_state,) or (hidden_state, ...) tuples
            hs = output[0] if isinstance(output, (tuple, list)) else output
            self.layer_outputs.append(hs.detach().cpu().to(torch.float32))
        return hook

    def _register_gdn_debug_hooks(self, layers, debug_layer: int):
        """Register sub-step hooks on the GDN linear_attn at debug_layer."""
        layer = layers[debug_layer]
        lin_attn = getattr(layer, "linear_attn", None)
        if lin_attn is None:
            print(f"[debug] Layer {debug_layer} has no linear_attn (it's a full-attention layer). Skipping sub-step hooks.")
            return

        print(f"[debug] Registering GDN sub-step hooks on layer {debug_layer}")

        # conv1d output (pre-SiLU)
        conv1d = getattr(lin_attn, "conv1d", None)
        if conv1d is not None:
            def _conv_hook(m, inp, out, _k="conv1d_out"):
                self.debug.setdefault(_k, []).append(out.detach().cpu().float())
            self._handles.append(conv1d.register_forward_hook(_conv_hook))

        # norm output (post-RMSNorm, pre-SiLU(z) gating)
        norm = getattr(lin_attn, "norm", None)
        if norm is not None:
            def _norm_hook(m, inp, out, _k="norm_out"):
                self.debug.setdefault(_k, []).append(out.detach().cpu().float())
            self._handles.append(norm.register_forward_hook(_norm_hook))

        # out_proj input captures the gated output just before projection
        out_proj = getattr(lin_attn, "out_proj", None)
        if out_proj is not None:
            def _out_hook(m, inp, out, _k="out_proj_out"):
                self.debug.setdefault(_k, []).append(out.detach().cpu().float())
            self._handles.append(out_proj.register_forward_hook(_out_hook))

    def clear(self):
        self.layer_outputs.clear()
        self.debug.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _get_layers(model) -> list:
    """Return the list of decoder layers regardless of HF model wrapper depth."""
    # Qwen3.5 is wrapped: model.model.layers or model.language_model.model.layers
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ]
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return list(layers)
    raise RuntimeError(f"Cannot find decoder layers in {type(model).__name__}. Attributes: {dir(model)}")


# ──────────────────────────────────────────────────────────────────────────────
# Greedy decode
# ──────────────────────────────────────────────────────────────────────────────


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    n: int,
    device: str,
    recorder: Optional[HookRecorder] = None,
) -> tuple[list[int], list[list[torch.Tensor]]]:
    """
    Greedy decode `n` tokens for `prompt`.

    Returns:
        token_ids: list of generated token ids (len == n)
        layer_hidden_states: list over decode steps, each entry is a list of
            per-layer hidden-state tensors (float32, cpu).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated: list[int] = []
    all_layer_hs: list[list[torch.Tensor]] = []

    for step in range(n):
        if recorder is not None:
            recorder.clear()

        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits  # (1, seq_len, vocab)
        next_token = int(logits[0, -1].argmax())
        generated.append(next_token)

        if recorder is not None:
            all_layer_hs.append(list(recorder.layer_outputs))

        # Re-run the full sequence each step. With use_cache=False this is the
        # only correct option: replacing input_ids with just the new token wipes
        # both the prompt context and the GatedDeltaNet recurrent state.
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=device)], dim=1
        )

        if step == 0:
            # Print top-5 logits for sanity check on first step
            probs = torch.softmax(logits[0, -1].float(), dim=-1)
            top5 = probs.topk(5)
            print("[ref] First-step top-5 tokens:")
            for tok_id, prob in zip(top5.indices.tolist(), top5.values.tolist()):
                print(f"       {tok_id:6d}  {tokenizer.decode([tok_id])!r:20s}  {prob:.4f}")

    return generated, all_layer_hs


# ──────────────────────────────────────────────────────────────────────────────
# Reference mode: build and cache
# ──────────────────────────────────────────────────────────────────────────────


def run_reference(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache)
    if cache_path.exists() and not args.regen:
        print(f"[ref] Cache exists at {cache_path}. Use --regen to rebuild. Exiting.")
        return

    model, tokenizer = load_hf(args.model, args.dtype, args.device)

    recorder: Optional[HookRecorder]
    if args.no_layer_hooks:
        if args.debug_layer is not None:
            print("[ref] --no-layer-hooks set; ignoring --debug-layer.")
        recorder = None
        print("[ref] Layer hooks disabled (greedy-only reference).")
    else:
        recorder = HookRecorder()
        recorder.register_decoder_hooks(model, debug_layer=args.debug_layer)

    cache: dict = {
        "model_id": args.model,
        "dtype": args.dtype,
        "n_tokens": GREEDY_N,
        "prompts": FIXED_PROMPTS,
        "results": [],
    }

    for pi, prompt in enumerate(FIXED_PROMPTS):
        print(f"\n[ref] Prompt {pi+1}/{len(FIXED_PROMPTS)}: {prompt[:60]!r}")
        tokens, layer_hs_per_step = greedy_generate(
            model, tokenizer, prompt, GREEDY_N, args.device, recorder=recorder
        )
        text = tokenizer.decode(tokens)
        print(f"[ref] Generated: {text!r}")

        per_prompt = {
            "prompt": prompt,
            "tokens": tokens,
            "text": text,
            # Layer hidden states: list[step] of list[layer] of Tensor (1, 1, hidden).
            # Each hs is (batch, seq, hidden) — take only the last token position for
            # compactness and because that's what drives the next-token logit.
            "layer_hs": [
                [hs[:, -1:, :].numpy() for hs in step_hs]
                for step_hs in layer_hs_per_step
            ],
        }
        if recorder is not None and args.debug_layer is not None:
            per_prompt["debug"] = {k: [t.numpy() for t in v] for k, v in recorder.debug.items()}
        cache["results"].append(per_prompt)

    if recorder is not None:
        recorder.remove()
    torch.save(cache, cache_path)
    print(f"\n[ref] Saved reference cache to {cache_path}")
    print(f"[ref] Prompts: {len(FIXED_PROMPTS)}  |  Tokens per prompt: {GREEDY_N}")


# ──────────────────────────────────────────────────────────────────────────────
# Greedy-parity mode: compare MLC vs cached HF tokens
# ──────────────────────────────────────────────────────────────────────────────


def run_greedy_parity(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"[parity] No cache found at {cache_path}. Run --reference-only first.")
        sys.exit(1)

    if args.mlc_model_dir is None:
        print("[parity] --mlc-model-dir is required for --greedy-parity.")
        sys.exit(1)

    cache = torch.load(cache_path, weights_only=False)
    print(f"[parity] Loaded HF reference from {cache_path} (model={cache['model_id']!r}, dtype={cache['dtype']!r})")

    # Need HF tokenizer for re-encoding MLC text into token ids (so we can do token-level diff).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cache["model_id"], trust_remote_code=True)

    mlc_texts = _run_mlc_greedy(args, cache["prompts"])

    all_pass = True
    for pi, (prompt, ref_result, mlc_text) in enumerate(zip(cache["prompts"], cache["results"], mlc_texts)):
        ref_tokens = ref_result["tokens"]
        ref_text = ref_result["text"]

        # Re-tokenize MLC output with the same tokenizer, take the first GREEDY_N tokens.
        mlc_tokens = tokenizer.encode(mlc_text, add_special_tokens=False)[:GREEDY_N]

        # Token-level match
        n_compare = min(len(mlc_tokens), len(ref_tokens))
        matches = sum(r == m for r, m in zip(ref_tokens[:n_compare], mlc_tokens[:n_compare]))

        status = "PASS" if matches >= TOKEN_MATCH_BAR else "FAIL"
        if matches < TOKEN_MATCH_BAR:
            all_pass = False
        print(f"\n[parity] Prompt {pi+1}: {matches}/{GREEDY_N} tokens match  [{status}]")
        print(f"         REF: {ref_text!r}")
        print(f"         MLC: {mlc_text!r}")
        if matches < GREEDY_N:
            first_diff = next(
                (i for i, (r, m) in enumerate(zip(ref_tokens[:n_compare], mlc_tokens[:n_compare])) if r != m),
                n_compare,
            )
            r_tok = ref_tokens[first_diff] if first_diff < len(ref_tokens) else None
            m_tok = mlc_tokens[first_diff] if first_diff < len(mlc_tokens) else None
            print(f"         First diff @ token {first_diff}: ref={r_tok} ({tokenizer.decode([r_tok]) if r_tok else '<eof>'!r}) mlc={m_tok} ({tokenizer.decode([m_tok]) if m_tok else '<eof>'!r})")

    if all_pass:
        print(f"\n[parity] ALL PROMPTS PASS (>={TOKEN_MATCH_BAR}/{GREEDY_N} match)")
    else:
        print(f"\n[parity] SOME PROMPTS FAILED — see above")
        sys.exit(1)


def _run_mlc_greedy(args: argparse.Namespace, prompts: list[str]) -> list[list[str]]:
    """Run MLC greedy decode and return generated text per prompt (one str per prompt)."""
    try:
        from mlc_llm import MLCEngine
        from mlc_llm.protocol.generation_config import GenerationConfig
    except ImportError:
        print("[mlc] Cannot import mlc_llm. Is MLC-LLM installed in this environment?")
        sys.exit(1)

    model_dir = args.mlc_model_dir
    lib_path = args.mlc_lib
    if lib_path is None:
        so_files = list(Path(model_dir).glob("*.so"))
        if not so_files:
            print(f"[mlc] No .so found in {model_dir}. Specify --mlc-lib.")
            sys.exit(1)
        lib_path = str(so_files[0])
        print(f"[mlc] Using lib: {lib_path}")

    # mode="interactive": max_batch=1, max KV=full context window. Server mode
    # over-allocates a 6.5M-token KV cache and OOMs on a 32 GB card.
    engine = MLCEngine(
        model=model_dir,
        model_lib=lib_path,
        device=args.device,
        mode="interactive",
    )
    gen_cfg = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=GREEDY_N)

    results: list[str] = []
    for pi, prompt in enumerate(prompts):
        print(f"[mlc] Prompt {pi+1}/{len(prompts)}: {prompt[:60]!r}")
        full_text = ""
        for delta_outputs in engine._generate(prompt, gen_cfg, request_id=str(pi)):
            for delta in delta_outputs:
                full_text += delta.delta_text
        results.append(full_text)
        print(f"[mlc] Output: {full_text!r}")

    engine.terminate()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Layer-parity mode (Stage 3 — stub; fully implemented once MLC debug output is wired)
# ──────────────────────────────────────────────────────────────────────────────


def run_layer_parity(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"[layer] No cache found at {cache_path}. Run --reference-only first.")
        sys.exit(1)

    cache = torch.load(cache_path, weights_only=False)
    print(f"[layer] Loaded HF reference from {cache_path}")
    print("[layer] Layer-parity comparison requires a compiled MLC model that returns")
    print("[layer] per-layer hidden states. See Stage 3 in worklog.md for how to instrument.")
    print("[layer] Checking if reference data is present ...")

    for pi, result in enumerate(cache["results"]):
        layer_hs = result.get("layer_hs")
        if layer_hs is None or len(layer_hs) == 0:
            print(f"  Prompt {pi+1}: no layer_hs in cache (run --reference-only with --debug-layer or a full run).")
            continue
        n_steps = len(layer_hs)
        n_layers = len(layer_hs[0]) if layer_hs else 0
        print(f"  Prompt {pi+1}: {n_steps} decode steps × {n_layers} layers cached.")

    print("\n[layer] MLC-side comparison not yet implemented (Stage 3). Reference data looks complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Parity check utilities (for Stage 3)
# ──────────────────────────────────────────────────────────────────────────────


def check_layer(ref: np.ndarray, mlc: np.ndarray, layer_idx: int, layer_type: str) -> bool:
    """Check a single layer's hidden state with appropriate tolerances."""
    atol = LAYER_ATOL if layer_type == "full_attention" else LAYER_ATOL_LINEAR
    max_diff = np.abs(ref - mlc).max()
    mean_diff = np.abs(ref - mlc).mean()
    fail = max_diff > atol
    status = "FAIL" if fail else "ok  "
    print(f"  Layer {layer_idx:3d} [{layer_type[:6]}] {status}  max={max_diff:.2e}  mean={mean_diff:.2e}  (atol={atol:.0e})")
    if fail:
        worst_idx = np.unravel_index(np.abs(ref - mlc).argmax(), ref.shape)
        print(f"             worst pos {worst_idx}: ref={ref[worst_idx]:.6f}  mlc={mlc[worst_idx]:.6f}")
    return not fail


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    if args.reference_only:
        run_reference(args)
    elif args.greedy_parity:
        run_greedy_parity(args)
    elif args.layer_parity:
        run_layer_parity(args)


if __name__ == "__main__":
    main()

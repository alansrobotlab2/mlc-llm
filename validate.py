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

  --reference-vl     Phase 10 Stage 1: load HF VLM, run a fixed image+text prompt,
                     dump per-vision-block hidden states, merger output, mRoPE
                     position IDs, and 50-token greedy decode. Caches to
                     reference_outputs_vl.pt.

Usage examples:
  # Stage 1: build reference cache
  python validate.py --reference-only --model Qwen/Qwen3.5-0.8B

  # Stage 4: greedy parity
  python validate.py --greedy-parity --model Qwen/Qwen3.5-0.8B \
      --mlc-model-dir dist/qwen3_5-0.8B-q0f16

  # Debug a single GDN layer
  python validate.py --reference-only --model Qwen/Qwen3.5-0.8B --debug-layer 2

  # Phase 10 Stage 1: build the multimodal reference cache
  python validate.py --reference-vl --model Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

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
CACHE_FILE_VL = Path("reference_outputs_vl.pt")
GREEDY_N = 50
LAYER_RTOL = 1e-3
LAYER_ATOL = 1e-3
LAYER_ATOL_LINEAR = 2e-3  # linear (GDN) layers accumulate more rounding
TOKEN_MATCH_BAR = 48  # out of GREEDY_N

# ──────────────────────────────────────────────────────────────────────────────
# Phase 10 Stage 1: fixed multimodal prompt. Single image + one query — enough
# to validate the harness end to end. The Stage 5 5-prompt parity set is
# deferred until Stages 3-4 produce a runnable MLC vision tower.
# ──────────────────────────────────────────────────────────────────────────────
VL_FIXTURE_IMAGE = Path("tests/multimodal/cat.jpeg")
VL_FIXED_QUERY = "Describe this image in one short sentence."

# 5-prompt multimodal eval set. Same fixture image; 5 different queries that
# exercise different generation surfaces (description, attribute, classification,
# spatial reasoning, single-word). Built to keep the eval cheap (one image
# preprocess + 5 generations) while smoothing single-prompt noise near the bar.
VL_5PROMPT_QUERIES = [
    "Describe this image in one short sentence.",
    "What animal is shown in the image?",
    "What is the dominant color of the animal in this image?",
    "Is this a domestic pet or a wild animal?",
    "Give a one-word answer: what is the animal's facial expression?",
]
CACHE_FILE_VL5 = Path("reference_outputs_vl5.pt")
# Margin (nats) the VL gate scores at. 2.0 means top-1 was ~7.4x more likely than
# top-2, matching scripts/high_margin_gate.py's DEFAULT_TAU so the two gates are
# read on the same scale. See workplan §16.1 for why an unweighted match count is
# the wrong instrument, and §18.13 for the VL run that demonstrated it.
VL_DEFAULT_TAU = 2.0

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HF model ID or local path")
    p.add_argument("--cache", default=str(CACHE_FILE), help="Path to reference_outputs.pt")
    p.add_argument(
        "--tau", type=float, default=VL_DEFAULT_TAU,
        help="margin (nats) the VL gate scores at. A divergence at a position where the "
             "reference's own top1-top2 margin was below this is a near-tie, not a "
             "defect; only wide-margin divergences fail. Matches "
             "scripts/high_margin_gate.py's tau so the two read on one scale (§16.1).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reference-only", action="store_true", help="Run HF reference; cache outputs.")
    mode.add_argument("--greedy-parity", action="store_true", help="Compare greedy decode vs MLC.")
    mode.add_argument("--layer-parity", action="store_true", help="Compare per-layer hidden states vs MLC.")
    mode.add_argument("--reference-vl", action="store_true", help="Phase 10 Stage 1: HF multimodal reference cache.")
    mode.add_argument("--greedy-parity-vl", action="store_true", help="Phase 10 Stage 5b: drive compiled qwen3_5_vl lib + diff vs HF cache.")
    mode.add_argument("--mrope-collapse", action="store_true", help="Phase 10 Stage 5b diagnostic: drive VL lib on text-only prompt with 3 identical position rows; should reduce to 1D RoPE and match the text-only reference cache.")
    mode.add_argument("--reference-vl5", action="store_true", help="Phase 10 Stage 5b: build 5-prompt multimodal reference cache (HF VLM, cat fixture, 5 queries).")
    mode.add_argument("--greedy-parity-vl5", action="store_true", help="Phase 10 Stage 5b: drive VL lib on the 5-prompt set, aggregate match count across prompts.")
    mode.add_argument("--perf-vl5", action="store_true", help="Workplan item 0o: time image_embed / prefill / decode separately on the VL lib. No reference comparison; MLCEngine cannot drive this model's vision tower (see workplan item 0o trap 3).")

    p.add_argument("--perf-iters", type=int, default=20, help="--perf-vl5: timed iterations per call (after warmup).")
    p.add_argument("--perf-warmup", type=int, default=5, help="--perf-vl5: untimed warmup iterations per call.")
    p.add_argument("--perf-decode-steps", type=int, default=64, help="--perf-vl5: decode steps to time per repeat.")
    p.add_argument("--perf-max-history", type=int, default=None, help="--perf-vl5: RNN-state max_history. Default measures both 1 ('disable'-equivalent) and 64 ('radix'-equivalent).")

    p.add_argument("--vl-cache", default=str(CACHE_FILE_VL), help="Path to reference_outputs_vl.pt (Phase 10 Stage 1).")
    p.add_argument("--vl-image", default=str(VL_FIXTURE_IMAGE), help="Fixed image fixture for --reference-vl.")
    p.add_argument("--vl-query", default=VL_FIXED_QUERY, help="Fixed text query paired with the image fixture.")

    p.add_argument("--prefix-cache-mode", default=None, choices=["disable", "radix"], help="Override the engine prefix_cache_mode. Matters for hybrid (RNNState) models: 'radix' is the default and gives max_history=64, 'disable' clamps it to 1. Gate any change to recurrent-state handling under BOTH.")
    p.add_argument("--mlc-model-dir", default=None, help="Path to compiled MLC model directory (for --greedy-parity / --layer-parity).")
    p.add_argument("--mlc-lib", default=None, help="Path to compiled MLC .so (if not auto-found in model dir).")
    p.add_argument("--debug-layer", type=int, default=None, help="Dump detailed GDN sub-step values for this layer index.")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Model dtype for HF reference (keep fp16 for parity; fp32 for debugging).")
    p.add_argument("--regen", action="store_true", help="Force regeneration even if cache exists.")
    p.add_argument("--no-layer-hooks", action="store_true", help="Skip per-layer hidden-state capture during --reference-only (greedy-only mode; needed for 35B to keep host RAM bounded).")
    p.add_argument("--use-hf-merger", action="store_true", help="Phase 10 Stage 5b debug: bypass our image_embed and use HF's cached merger output instead.")
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

    # An fp8 checkpoint is the only way the 35B reference fits a 64 GB Orin (37.5 GB
    # vs 72 GB bf16). sm_87 has no fp8 arithmetic, so route the fp8 linears through a
    # software weight-only dequant and leave the stored weights fp8. See
    # fp8_software_dequant.py for the precision caveats — the result is W8A16.
    import fp8_software_dequant

    is_fp8 = fp8_software_dequant.is_fp8_checkpoint(model_id)
    if is_fp8:
        fp8_software_dequant.install()
        # Force bf16 for the whole model. The fp8 weights are never cast, but the
        # modules in `modules_to_not_convert` (lm_head, routers, norms, in_proj_a) are —
        # and casting only those to fp16 leaves them mismatched against the bf16
        # activations coming out of the dequant path:
        #   RuntimeError: expected mat1 and mat2 to have the same dtype,
        #                 but got: c10::BFloat16 != c10::Half
        # bf16 is also the checkpoint's native dtype, so this is the faithful choice.
        torch_dtype = torch.bfloat16
        print("[ref] fp8 checkpoint — forcing bfloat16 for the unquantized modules")

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
    # Skipped for fp8: .half()/.float() would cast the fp8 weights up and blow the
    # memory budget that picking an fp8 checkpoint bought us in the first place.
    if is_fp8:
        print("[ref] fp8 checkpoint — skipping the .half()/.float() dtype forcing")
    elif dtype == "float16":
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
    engine_kwargs = {}
    if args.prefix_cache_mode is not None:
        from mlc_llm.serve.config import EngineConfig

        engine_kwargs["engine_config"] = EngineConfig(prefix_cache_mode=args.prefix_cache_mode)
        print(f"[mlc] prefix_cache_mode={args.prefix_cache_mode}")
    engine = MLCEngine(
        model=model_dir,
        model_lib=lib_path,
        device=args.device,
        mode="interactive",
        **engine_kwargs,
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
# Phase 10 Stage 1: multimodal reference (--reference-vl)
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_vl_components(model):
    """Walk a Qwen3.5 ConditionalGeneration model to find {visual, text, get_rope_index}.

    transformers 5.6 layout (verified for Qwen3_5ForConditionalGeneration):
        model.model.visual           → Qwen3_5VisionModel
        model.model.visual.blocks    → ModuleList of Qwen3_5VisionBlock
        model.model.visual.merger    → Qwen3_5VisionPatchMerger
        model.model.language_model   → Qwen3_5TextModel
        model.model.get_rope_index   → bound method computing 3D mRoPE position ids
    """
    inner = getattr(model, "model", model)
    visual = getattr(inner, "visual", None)
    if visual is None:
        raise RuntimeError(
            f"Model {type(model).__name__} has no .model.visual — is this the multimodal checkpoint?"
        )
    blocks = list(getattr(visual, "blocks", []))
    if not blocks:
        raise RuntimeError(f"visual has no .blocks attribute (got {type(visual).__name__}).")
    merger = getattr(visual, "merger", None)
    if merger is None:
        raise RuntimeError(f"visual has no .merger attribute.")
    text = getattr(inner, "language_model", None)
    get_rope_index = getattr(inner, "get_rope_index", None)
    return {
        "visual": visual,
        "blocks": blocks,
        "merger": merger,
        "text": text,
        "get_rope_index": get_rope_index,
    }


class _VLHooks:
    """Records vision-block outputs and patch-merger output during a forward pass."""

    def __init__(self):
        self.block_outputs: list[torch.Tensor] = []
        self.merger_output: Optional[torch.Tensor] = None
        self._handles: list = []

    def attach(self, components: dict) -> None:
        for i, blk in enumerate(components["blocks"]):
            self._handles.append(blk.register_forward_hook(self._make_block_hook(i)))
        self._handles.append(components["merger"].register_forward_hook(self._merger_hook))

    def _make_block_hook(self, idx: int):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            self.block_outputs.append(t.detach().cpu().to(torch.float32))
        return hook

    def _merger_hook(self, module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        self.merger_output = t.detach().cpu().to(torch.float32)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _load_hf_vl(model_id: str, dtype: str, device: str):
    """Load Qwen3.5 multimodal model + AutoProcessor."""
    from transformers import AutoProcessor

    # Use the architecture-specific class. AutoModelForImageTextToText is the
    # canonical generic loader in transformers >= 4.45 for VLMs of this shape;
    # if it isn't registered for `qwen3_5` in this transformers build, fall back
    # to the explicit class.
    try:
        from transformers import AutoModelForImageTextToText as _AutoVL  # type: ignore
        loader = _AutoVL
        loader_name = "AutoModelForImageTextToText"
    except ImportError:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration as _CondGen,
        )
        loader = _CondGen
        loader_name = "Qwen3_5ForConditionalGeneration (direct)"

    print(f"[ref-vl] Loading processor from {model_id!r}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    print(f"[ref-vl] Loading model via {loader_name} on {device}, dtype={dtype}")

    if device.startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info(device)
        print(f"[ref-vl] {device} mem before load: free={free_b/1e9:.2f} GiB / total={total_b/1e9:.2f} GiB")

    model = loader.from_pretrained(
        model_id,
        dtype=torch_dtype,
        attn_implementation="eager",
        trust_remote_code=True,
        device_map=device,
        low_cpu_mem_usage=True,
    ).eval()

    if dtype == "float16":
        model = model.half()
    elif dtype == "float32":
        model = model.float()

    if device.startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info(device)
        print(f"[ref-vl] {device} mem after load: free={free_b/1e9:.2f} GiB / total={total_b/1e9:.2f} GiB")

    return model, processor


@torch.inference_mode()
def run_reference_vl(args: argparse.Namespace) -> None:
    """Phase 10 Stage 1 — build reference_outputs_vl.pt from a single image+text prompt."""
    cache_path = Path(args.vl_cache)
    if cache_path.exists() and not args.regen:
        print(f"[ref-vl] Cache exists at {cache_path}. Use --regen to rebuild. Exiting.")
        return

    image_path = Path(args.vl_image)
    if not image_path.exists():
        print(f"[ref-vl] Image fixture not found at {image_path}. See tests/multimodal/README.md.")
        sys.exit(1)

    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    print(f"[ref-vl] Image: {image_path} ({image.size[0]}x{image.size[1]} {image.mode})")

    model, processor = _load_hf_vl(args.model, args.dtype, args.device)
    components = _resolve_vl_components(model)
    print(
        f"[ref-vl] Vision tower: {len(components['blocks'])} blocks  |  "
        f"merger output dim: {getattr(components['merger'], 'out_features', '?')}"
    )

    # Build chat input. processor.apply_chat_template handles vision_start/end
    # token injection and image placeholder spans; the processor then expands the
    # placeholder token to the right number of <|image_pad|> tokens for the
    # post-resize grid.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.vl_query},
            ],
        },
    ]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"[ref-vl] Chat-templated prompt ({len(chat_text)} chars):\n{chat_text}")

    inputs = processor(text=[chat_text], images=[image], padding=True, return_tensors="pt").to(args.device)
    print(f"[ref-vl] Inputs: " + ", ".join(f"{k}={tuple(v.shape) if hasattr(v,'shape') else v}" for k, v in inputs.items()))
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is None:
        print("[ref-vl] WARNING: processor did not return image_grid_thw; preprocessor may not be VL.")
    else:
        print(f"[ref-vl] image_grid_thw: {image_grid_thw.tolist()}")

    # Hook visual blocks + merger for one prefill pass to capture all tower internals.
    hooks = _VLHooks()
    hooks.attach(components)
    print("[ref-vl] Running prefill forward (single pass) for tower-side captures…")
    prefill_out = model(**inputs, use_cache=False)
    hooks.remove()
    print(
        f"[ref-vl] Captured {len(hooks.block_outputs)} vision-block outputs; "
        f"merger output shape: {tuple(hooks.merger_output.shape) if hooks.merger_output is not None else None}"
    )
    prefill_logits_last = prefill_out.logits[0, -1].detach().cpu().to(torch.float32)
    print(f"[ref-vl] Prefill logits last-pos shape: {tuple(prefill_logits_last.shape)}")

    # M-RoPE position IDs via Qwen3.5's bundled helper. Signature is documented
    # to take (input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts,
    # attention_mask) and returns (position_ids[3,B,S], rope_deltas[B,1]).
    rope_position_ids = None
    rope_deltas = None
    if components["get_rope_index"] is not None and image_grid_thw is not None:
        try:
            rope_position_ids, rope_deltas = components["get_rope_index"](
                inputs["input_ids"],
                image_grid_thw=image_grid_thw,
                video_grid_thw=inputs.get("video_grid_thw"),
                attention_mask=inputs.get("attention_mask"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )
            print(
                f"[ref-vl] mRoPE position_ids: {tuple(rope_position_ids.shape)}  "
                f"rope_deltas: {rope_deltas.tolist() if rope_deltas is not None else None}"
            )
        except TypeError as exc:
            # transformers ≤ 5.5 has a different signature; retry without mm_token_type_ids.
            print(f"[ref-vl] get_rope_index TypeError: {exc!r} — retrying older signature")
            try:
                rope_position_ids, rope_deltas = components["get_rope_index"](
                    inputs["input_ids"],
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=inputs.get("video_grid_thw"),
                    attention_mask=inputs.get("attention_mask"),
                )
                print(f"[ref-vl] mRoPE position_ids: {tuple(rope_position_ids.shape)}")
            except Exception as exc2:  # noqa: BLE001
                print(f"[ref-vl] retry also failed: {exc2!r}. Cache will omit position_ids.")

    # 50-token greedy decode end-to-end. HF GenerationMixin handles VL
    # masked-scatter + cache automatically; faster than the use_cache=False loop
    # we use on the text-only path.
    print(f"[ref-vl] Greedy decode {GREEDY_N} tokens (do_sample=False)…")
    gen_out = model.generate(
        **inputs,
        max_new_tokens=GREEDY_N,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=False,
    )
    full_seq = gen_out.sequences[0]
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = full_seq[prompt_len:].detach().cpu().tolist()
    new_text = processor.batch_decode(
        [full_seq[prompt_len:]], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"[ref-vl] Generated ({len(new_tokens)} tokens): {new_text!r}")

    cache: dict = {
        "model_id": args.model,
        "dtype": args.dtype,
        "image_path": str(image_path),
        "image_size": image.size,
        "query": args.vl_query,
        "chat_text": chat_text,
        "input_ids": inputs["input_ids"].detach().cpu().numpy(),
        "attention_mask": inputs.get("attention_mask").detach().cpu().numpy() if inputs.get("attention_mask") is not None else None,
        "pixel_values_shape": tuple(inputs["pixel_values"].shape) if "pixel_values" in inputs else None,
        "image_grid_thw": image_grid_thw.detach().cpu().numpy() if image_grid_thw is not None else None,
        "vision_block_outputs": [t.numpy() for t in hooks.block_outputs],
        "merger_output": hooks.merger_output.numpy() if hooks.merger_output is not None else None,
        "prefill_logits_last": prefill_logits_last.numpy(),
        "rope_position_ids": rope_position_ids.detach().cpu().numpy() if rope_position_ids is not None else None,
        "rope_deltas": rope_deltas.detach().cpu().numpy() if rope_deltas is not None else None,
        "generated_token_ids": new_tokens,
        "generated_text": new_text,
        "n_tokens": GREEDY_N,
    }
    torch.save(cache, cache_path)
    print(f"\n[ref-vl] Saved multimodal reference cache to {cache_path}")
    sz = cache_path.stat().st_size
    print(
        f"[ref-vl] Cache: {sz/1e6:.1f} MB | "
        f"{len(cache['vision_block_outputs'])} block tensors + merger + logits + rope ids"
    )


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
# Phase 10 Stage 5b: drive compiled qwen3_5_vl lib + diff vs HF cache
# ──────────────────────────────────────────────────────────────────────────────


def _load_vl_pos_embed_weight(model_id: str) -> np.ndarray:
    """Read visual.pos_embed.weight directly from the HF safetensors.

    Avoids loading the full HF model (which costs ~3 GB CPU RAM) just to read
    one (2304, 768) tensor. Uses huggingface_hub to resolve the snapshot path,
    then safetensors directly.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    # Walk the index to find which shard holds visual.pos_embed.weight.
    import json as _json
    with open(Path(snap) / "model.safetensors.index.json", encoding="utf-8") as f:
        idx = _json.load(f)
    key = "model.visual.pos_embed.weight"
    shard = idx["weight_map"][key]
    with safe_open(Path(snap) / shard, framework="numpy") as f:
        w = f.get_tensor(key)
    return np.asarray(w, dtype=np.float32)


def run_greedy_parity_vl(args: argparse.Namespace) -> None:
    """Phase 10 Stage 5b — drive the compiled qwen3_5_vl lib end-to-end and
    diff against the cached HF reference (`reference_outputs_vl.pt`).

    Flow:
      1. Load lib + tensor-cache params via `tvm.runtime.load_module` + `tvmjs`.
      2. Run our preprocessor on the cat fixture.
      3. `image_embed(pixel_values, pos_embeds, rotary_cos, rotary_sin)` → image embeds.
      4. `embed(input_ids)` → text token embeddings; scatter image embeds at
         `<|image_pad|>` token positions.
      5. Use cached HF rope_position_ids + rope_deltas (mRoPE math is verified
         end-to-end inside the compiled model — Stage 2 chunks A+B). Stage 5b
         is a model-level test, not an mRoPE recomputation test.
      6. begin_forward → prefill → end_forward → argmax → first emitted token.
      7. Loop: embed(token) → decode → argmax for the remaining ``GREEDY_N`` steps.
      8. Diff against ``cache["generated_token_ids"]``.
    """
    cache_path = Path(args.vl_cache)
    if not cache_path.exists():
        print(f"[parity-vl] No cache at {cache_path}. Run --reference-vl first.")
        sys.exit(1)
    cache = torch.load(cache_path, weights_only=False)

    if args.mlc_model_dir is None:
        print("[parity-vl] Missing --mlc-model-dir.")
        sys.exit(1)
    model_dir = Path(args.mlc_model_dir)
    lib_path = Path(args.mlc_lib) if args.mlc_lib else model_dir / "lib.so"
    if not lib_path.exists():
        print(f"[parity-vl] No lib.so at {lib_path}.")
        sys.exit(1)

    import json as _json
    import tvm
    from tvm import relax
    from tvm.contrib import tvmjs
    from tvm.runtime import ShapeTuple

    # Pin to cuda:1 (RTX 5090, 31 GB) — Blackwell may be reserved for 35B work.
    device_str = args.device if args.device != "cuda" else "cuda:1"
    print(f"[parity-vl] Loading lib + params on {device_str}")
    dev = tvm.device(device_str)
    ex = tvm.runtime.load_module(str(lib_path))
    vm = relax.VirtualMachine(ex, device=dev)
    mod = vm.module
    metadata = _json.loads(mod["_metadata"]())
    params, meta = tvmjs.load_tensor_cache(str(model_dir), dev)
    param_names = [p["name"] for p in metadata["params"]]
    params = [params[n] for n in param_names]
    print(f"[parity-vl] Loaded {len(params)} params")

    image_token_id = 248056
    eos_token_id = 151645  # qwen3_5 conv template's <|im_end|>
    GENERATED_N = len(cache["generated_token_ids"])
    print(f"[parity-vl] HF cached {GENERATED_N} tokens; will run that many through MLC.")

    # ── Build inputs from HF cache + our preprocessor ──────────────────────
    input_ids = cache["input_ids"][0]  # (seq_len,)
    seq_len = len(input_ids)
    image_grid_thw = cache["image_grid_thw"][0]  # (T, H, W)
    rope_position_ids = cache["rope_position_ids"]  # (3, 1, seq_len) int
    rope_deltas = cache["rope_deltas"]  # (1, 1) int
    print(f"[parity-vl] seq_len={seq_len} grid_thw={image_grid_thw.tolist()}")

    print("[parity-vl] Running preprocessor on cat fixture …")
    from PIL import Image
    from mlc_llm.model.qwen3_5_vl.qwen3_5_vl_image import preprocess_image

    pos_embed_weight = _load_vl_pos_embed_weight(args.model)
    rgb = np.asarray(Image.open(cache["image_path"]).convert("RGB"), dtype=np.uint8)
    pre = preprocess_image(rgb, pos_embed_weight=pos_embed_weight)
    print(
        f"[parity-vl] preproc: pixel_values={pre.pixel_values.shape}  "
        f"pos_embeds={pre.pos_embeds.shape}  cos={pre.rotary_cos.shape}  grid={pre.image_grid_thw}"
    )

    # ── Push tensors to device ─────────────────────────────────────────────
    pixel_values_d = tvm.runtime.tensor(pre.pixel_values, device=dev)
    pos_embeds_d = tvm.runtime.tensor(pre.pos_embeds, device=dev)
    rotary_cos_d = tvm.runtime.tensor(pre.rotary_cos, device=dev)
    rotary_sin_d = tvm.runtime.tensor(pre.rotary_sin, device=dev)

    # ── Run image_embed (or use HF cache as a debug bypass) ────────────────
    if args.use_hf_merger:
        print("[parity-vl] (debug) Bypassing image_embed; using HF cache merger output.")
        image_embeds_np = cache["merger_output"].astype(np.float16)
    else:
        print("[parity-vl] Running image_embed …")
        image_embeds = mod["image_embed"](
            pixel_values_d, pos_embeds_d, rotary_cos_d, rotary_sin_d, params
        )
        image_embeds_np = image_embeds.numpy()
    print(f"[parity-vl] image_embed → shape={image_embeds_np.shape} dtype={image_embeds_np.dtype}")

    # Diff vs HF merger output (post-merger embeddings, same shape).
    ref_merger = cache["merger_output"].astype(np.float32)
    diff = np.abs(image_embeds_np.astype(np.float32) - ref_merger)
    rel = diff.max() / max(np.abs(ref_merger).max(), 1.0)
    print(
        f"[parity-vl] image_embed vs HF merger:  "
        f"max={diff.max():.3e}  mean={diff.mean():.3e}  rel={rel:.2e}  "
        f"|ref|max={np.abs(ref_merger).max():.2e}"
    )

    # ── Build text+image input embedding ───────────────────────────────────
    print("[parity-vl] Running embed(input_ids) …")
    input_ids_d = tvm.runtime.tensor(np.asarray(input_ids, dtype=np.int32), device=dev)
    text_embed = mod["embed"](input_ids_d, params)  # (seq_len, hidden)
    text_embed_np = text_embed.numpy()
    if text_embed_np.ndim == 3:
        text_embed_np = text_embed_np[0]
    print(f"[parity-vl] text_embed shape={text_embed_np.shape}")

    image_pad_pos = np.where(np.asarray(input_ids) == image_token_id)[0]
    print(f"[parity-vl] {len(image_pad_pos)} <|image_pad|> tokens at positions {image_pad_pos[:3]}…{image_pad_pos[-3:]}")
    if len(image_pad_pos) != image_embeds_np.shape[0]:
        print(
            f"[parity-vl] WARN: image_pad count ({len(image_pad_pos)}) != "
            f"image_embed count ({image_embeds_np.shape[0]}); will substitute min(N,M)."
        )
    n_sub = min(len(image_pad_pos), image_embeds_np.shape[0])
    full_embed = text_embed_np.copy()
    full_embed[image_pad_pos[:n_sub]] = image_embeds_np[:n_sub]
    full_embed_d = tvm.runtime.tensor(
        full_embed.reshape(1, seq_len, -1).astype(text_embed_np.dtype), device=dev
    )

    # ── Set up paged_kv_cache + rnn_state ──────────────────────────────────
    if mod.implements_function("create_flashinfer_paged_kv_cache"):
        kv_create = mod["create_flashinfer_paged_kv_cache"]
        kv_kind = "flashinfer"
    elif mod.implements_function("create_tir_paged_kv_cache"):
        kv_create = mod["create_tir_paged_kv_cache"]
        kv_kind = "tir"
    else:
        print("[parity-vl] No KV cache create function found.")
        sys.exit(1)
    print(f"[parity-vl] KV cache kind: {kv_kind}")

    max_total_seq_len = seq_len + GENERATED_N + 32
    prefill_chunk = max(seq_len + 32, 4096)
    kv_cache = kv_create(
        ShapeTuple([1]),
        ShapeTuple([max_total_seq_len]),
        ShapeTuple([prefill_chunk]),
        ShapeTuple([16]),
        ShapeTuple([0]),  # support_sliding_window=0
    )
    rnn_state = mod["create_rnn_state"](ShapeTuple([1]), ShapeTuple([1]))

    # RNNState inherits from KVStateObj — same kv_state_* API works for both.
    kv_add_seq = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    kv_begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    kv_end = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    kv_add_seq(kv_cache, 0)
    kv_add_seq(rnn_state, 0)

    # ── Prefill ────────────────────────────────────────────────────────────
    print(f"[parity-vl] Prefilling seq_len={seq_len} …")
    kv_begin(kv_cache, ShapeTuple([0]), ShapeTuple([seq_len]))
    kv_begin(rnn_state, ShapeTuple([0]), ShapeTuple([seq_len]))

    pos_ids_d = tvm.runtime.tensor(np.asarray(rope_position_ids, dtype=np.int32), device=dev)
    mrope_deltas_d = tvm.runtime.tensor(np.asarray(rope_deltas, dtype=np.int32), device=dev)
    logits, kv_cache, rnn_state = mod["prefill"](
        full_embed_d, pos_ids_d, kv_cache, rnn_state, params
    )
    kv_end(kv_cache)
    kv_end(rnn_state)
    logits_np = logits.numpy()
    if logits_np.ndim == 3:
        logits_np = logits_np[0, -1]
    elif logits_np.ndim == 2:
        logits_np = logits_np[-1]
    next_token = int(np.argmax(logits_np))
    print(f"[parity-vl] First token: {next_token}")

    # Diff prefill last-token logits vs HF cache
    ref_logits = cache["prefill_logits_last"]
    log_diff = np.abs(logits_np[: len(ref_logits)].astype(np.float32) - ref_logits.astype(np.float32))
    print(
        f"[parity-vl] prefill logits diff vs HF: max={log_diff.max():.3e} mean={log_diff.mean():.3e} "
        f"argmax(MLC)={next_token}  argmax(HF)={int(np.argmax(ref_logits))}"
    )

    # ── Decode loop ────────────────────────────────────────────────────────
    generated = [next_token]
    for step in range(GENERATED_N - 1):
        kv_begin(kv_cache, ShapeTuple([0]), ShapeTuple([1]))
        kv_begin(rnn_state, ShapeTuple([0]), ShapeTuple([1]))
        tok_d = tvm.runtime.tensor(np.array([next_token], dtype=np.int32), device=dev)
        tok_embed = mod["embed"](tok_d, params)
        if tok_embed.shape[0] != 1 or len(tok_embed.shape) == 2:
            tok_embed = tvm.get_global_func("vm.builtin.reshape")(
                tok_embed, ShapeTuple([1, 1, tok_embed.shape[-1]])
            )
        logits, kv_cache, rnn_state = mod["decode"](
            tok_embed, mrope_deltas_d, kv_cache, rnn_state, params
        )
        kv_end(kv_cache)
        kv_end(rnn_state)
        logits_np = logits.numpy()
        if logits_np.ndim == 3:
            logits_np = logits_np[0, -1]
        elif logits_np.ndim == 2:
            logits_np = logits_np[-1]
        next_token = int(np.argmax(logits_np))
        generated.append(next_token)
        if next_token == eos_token_id:
            print(f"[parity-vl] EOS at step {step + 1}")
            break

    # ── Diff vs HF cache ───────────────────────────────────────────────────
    ref_tokens = cache["generated_token_ids"][: len(generated)]
    matches = sum(1 for a, b in zip(generated, ref_tokens) if a == b)
    print()
    print(f"[parity-vl] HF tokens : {ref_tokens}")
    print(f"[parity-vl] MLC tokens: {generated}")
    print(f"[parity-vl] Match: {matches}/{len(ref_tokens)}")
    bar = max(int(0.96 * len(ref_tokens)), 1)
    print(f"[parity-vl] Headline gate: {'PASS' if matches >= bar else 'FAIL'} (bar = {bar}/{len(ref_tokens)})")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 10 Stage 5b — 5-prompt multimodal reference + parity
# ──────────────────────────────────────────────────────────────────────────────


@torch.inference_mode()
def run_reference_vl5(args: argparse.Namespace) -> None:
    """Build a 5-prompt multimodal reference cache.

    Same image fixture (cat), 5 different queries. Stores per-prompt:
    `chat_text`, `input_ids`, `image_grid_thw` (image-shared, but stored
    per-prompt for convenience), `rope_position_ids`, `rope_deltas`,
    `prefill_logits_last`, `generated_token_ids`. No tower-internal hooks
    (parity is the headline; tower internals are validated separately).
    """
    cache_path = CACHE_FILE_VL5
    if cache_path.exists() and not args.regen:
        print(f"[ref-vl5] Cache exists at {cache_path}. Use --regen to rebuild. Exiting.")
        return

    image_path = Path(args.vl_image)
    if not image_path.exists():
        print(f"[ref-vl5] Image fixture not found at {image_path}.")
        sys.exit(1)

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    print(f"[ref-vl5] Image: {image_path} ({image.size[0]}x{image.size[1]} {image.mode})")
    print(f"[ref-vl5] {len(VL_5PROMPT_QUERIES)} queries × {GREEDY_N} greedy tokens")

    model, processor = _load_hf_vl(args.model, args.dtype, args.device)
    components = _resolve_vl_components(model)
    print(f"[ref-vl5] Vision tower: {len(components['blocks'])} blocks")

    per_prompt = []
    for q_idx, query in enumerate(VL_5PROMPT_QUERIES):
        print(f"\n[ref-vl5] === prompt {q_idx + 1}/{len(VL_5PROMPT_QUERIES)}: {query!r} ===")
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": query}]},
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=[image], padding=True, return_tensors="pt").to(args.device)
        image_grid_thw = inputs["image_grid_thw"]

        # mRoPE position ids
        rope_position_ids, rope_deltas = components["get_rope_index"](
            inputs["input_ids"], image_grid_thw=image_grid_thw,
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        # Prefill logits
        prefill_out = model(**inputs, use_cache=False)
        prefill_logits_last = prefill_out.logits[0, -1].detach().cpu().to(torch.float32)

        # Greedy generate. `output_logits` gives the raw per-step logits (not the
        # processed `scores`), which is what a margin has to be computed from.
        gen = model.generate(
            **inputs, max_new_tokens=GREEDY_N, do_sample=False,
            return_dict_in_generate=True, output_logits=True,
        )
        full = gen.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = full[prompt_len:].detach().cpu().tolist()
        new_text = processor.batch_decode([full[prompt_len:]], skip_special_tokens=True)[0]

        # Per-position top1-top2 margin, in nats — the same quantity
        # scripts/high_margin_gate.py captures, and for the same reason: a raw match
        # count cannot tell a near-tie apart from a regression (workplan §16.1). The
        # 2026-07-26e VL run failed at 167/184 on a single ',' vs '.' divergence, which
        # is exactly the case this distinguishes (§18.13).
        margins: list[float] = []
        for step_logits in gen.logits:
            top2 = torch.topk(step_logits[0].detach().float(), k=2)
            margins.append(float(top2.values[0] - top2.values[1]))
        wide = sum(m >= VL_DEFAULT_TAU for m in margins)
        med = sorted(margins)[len(margins) // 2] if margins else float("nan")
        print(f"[ref-vl5] {len(new_tokens)} tokens; text: {new_text!r}")
        print(f"[ref-vl5]   margins: {wide}/{len(margins)} at >= {VL_DEFAULT_TAU} "
              f"(median {med:.2f})")

        per_prompt.append({
            "margins": margins,
            "query": query,
            "chat_text": chat_text,
            "input_ids": inputs["input_ids"].detach().cpu().numpy(),
            "image_grid_thw": image_grid_thw.detach().cpu().numpy(),
            "rope_position_ids": rope_position_ids.detach().cpu().numpy(),
            "rope_deltas": rope_deltas.detach().cpu().numpy(),
            "prefill_logits_last": prefill_logits_last.numpy(),
            "generated_token_ids": new_tokens,
            "generated_text": new_text,
        })

    cache = {
        "model_id": args.model,
        "dtype": args.dtype,
        "image_path": str(image_path),
        "image_size": image.size,
        "n_prompts": len(VL_5PROMPT_QUERIES),
        "prompts": per_prompt,
    }
    torch.save(cache, cache_path)
    print(f"\n[ref-vl5] Saved {len(VL_5PROMPT_QUERIES)}-prompt cache to {cache_path}")
    print(f"[ref-vl5] Cache size: {cache_path.stat().st_size / 1e6:.1f} MB")


def run_greedy_parity_vl5(args: argparse.Namespace) -> None:
    """Drive the compiled VL lib on each of the 5 prompts and aggregate."""
    cache_path = CACHE_FILE_VL5
    if not cache_path.exists():
        print(f"[parity-vl5] No cache at {cache_path}. Run --reference-vl5 first.")
        sys.exit(1)
    cache = torch.load(cache_path, weights_only=False)

    if args.mlc_model_dir is None:
        print("[parity-vl5] Missing --mlc-model-dir.")
        sys.exit(1)
    model_dir = Path(args.mlc_model_dir)
    lib_path = Path(args.mlc_lib) if args.mlc_lib else model_dir / "lib.so"
    if not lib_path.exists():
        print(f"[parity-vl5] No lib.so at {lib_path}.")
        sys.exit(1)

    import json as _json
    import tvm
    from tvm import relax
    from tvm.contrib import tvmjs
    from tvm.runtime import ShapeTuple
    from PIL import Image
    from mlc_llm.model.qwen3_5_vl.qwen3_5_vl_image import preprocess_image

    device_str = args.device if args.device != "cuda" else "cuda:0"
    print(f"[parity-vl5] Loading lib + params on {device_str}")
    dev = tvm.device(device_str)
    ex = tvm.runtime.load_module(str(lib_path))
    vm = relax.VirtualMachine(ex, device=dev)
    mod = vm.module
    metadata = _json.loads(mod["_metadata"]())
    params, _meta = tvmjs.load_tensor_cache(str(model_dir), dev)
    param_names = [p["name"] for p in metadata["params"]]
    params = [params[n] for n in param_names]
    print(f"[parity-vl5] Loaded {len(params)} params")

    # Image preprocess once (same fixture across all 5 prompts).
    pos_embed_weight = _load_vl_pos_embed_weight(args.model)
    rgb = np.asarray(Image.open(cache["image_path"]).convert("RGB"), dtype=np.uint8)
    pre = preprocess_image(rgb, pos_embed_weight=pos_embed_weight)
    pixel_values_d = tvm.runtime.tensor(pre.pixel_values, device=dev)
    pos_embeds_d = tvm.runtime.tensor(pre.pos_embeds, device=dev)
    rotary_cos_d = tvm.runtime.tensor(pre.rotary_cos, device=dev)
    rotary_sin_d = tvm.runtime.tensor(pre.rotary_sin, device=dev)

    image_token_id = 248056
    if mod.implements_function("create_flashinfer_paged_kv_cache"):
        kv_create = mod["create_flashinfer_paged_kv_cache"]
    elif mod.implements_function("create_tir_paged_kv_cache"):
        kv_create = mod["create_tir_paged_kv_cache"]
    else:
        print("[parity-vl5] No KV cache create function.")
        sys.exit(1)

    kv_add_seq = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    kv_remove_seq = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
    kv_begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    kv_end = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    total_match = 0
    total_tokens = 0
    wide_failures = 0
    per_prompt_results = []

    # One image, five prompts — so run the tower once, not five times. This used
    # to sit inside the loop under a comment reading "could cache but cheap";
    # §20.1 measured it at **337 ms**, 69% of ttft and 2.26x the prefill of the
    # whole 652-token sequence, so the loop was paying ~1.35 s per gate run for
    # five identical results. The value is input-independent: same fixture, same
    # preprocessing, same params.
    image_embeds_np = mod["image_embed"](
        pixel_values_d, pos_embeds_d, rotary_cos_d, rotary_sin_d, params
    ).numpy()

    for i, p in enumerate(cache["prompts"]):
        query = p["query"]
        ref_tokens = p["generated_token_ids"]
        n_steps = len(ref_tokens)
        input_ids = p["input_ids"][0]
        seq_len = len(input_ids)
        rope_position_ids = p["rope_position_ids"]
        rope_deltas = p["rope_deltas"]

        # Build full input embed (image_embeds_np hoisted above the loop, §20.1)
        input_ids_d = tvm.runtime.tensor(np.asarray(input_ids, dtype=np.int32), device=dev)
        text_embed = mod["embed"](input_ids_d, params).numpy()
        if text_embed.ndim == 3:
            text_embed = text_embed[0]
        image_pad_pos = np.where(np.asarray(input_ids) == image_token_id)[0]
        n_sub = min(len(image_pad_pos), image_embeds_np.shape[0])
        full_embed = text_embed.copy()
        full_embed[image_pad_pos[:n_sub]] = image_embeds_np[:n_sub]
        full_embed_d = tvm.runtime.tensor(
            full_embed.reshape(1, seq_len, -1).astype(text_embed.dtype), device=dev
        )

        # Fresh KV cache per prompt
        max_total = seq_len + n_steps + 32
        prefill_chunk = max(seq_len + 32, 4096)
        kv_cache = kv_create(
            ShapeTuple([1]), ShapeTuple([max_total]), ShapeTuple([prefill_chunk]),
            ShapeTuple([16]), ShapeTuple([0]),
        )
        rnn_state = mod["create_rnn_state"](ShapeTuple([1]), ShapeTuple([1]))
        kv_add_seq(kv_cache, 0)
        kv_add_seq(rnn_state, 0)

        # Prefill
        kv_begin(kv_cache, ShapeTuple([0]), ShapeTuple([seq_len]))
        kv_begin(rnn_state, ShapeTuple([0]), ShapeTuple([seq_len]))
        pos_ids_d = tvm.runtime.tensor(np.asarray(rope_position_ids, dtype=np.int32), device=dev)
        mrope_deltas_d = tvm.runtime.tensor(np.asarray(rope_deltas, dtype=np.int32), device=dev)
        logits, kv_cache, rnn_state = mod["prefill"](full_embed_d, pos_ids_d, kv_cache, rnn_state, params)
        kv_end(kv_cache)
        kv_end(rnn_state)
        logits_np = logits.numpy()
        if logits_np.ndim == 3:
            logits_np = logits_np[0, -1]
        elif logits_np.ndim == 2:
            logits_np = logits_np[-1]
        next_token = int(np.argmax(logits_np))
        generated = [next_token]

        # Decode loop
        for _ in range(n_steps - 1):
            kv_begin(kv_cache, ShapeTuple([0]), ShapeTuple([1]))
            kv_begin(rnn_state, ShapeTuple([0]), ShapeTuple([1]))
            tok_d = tvm.runtime.tensor(np.array([next_token], dtype=np.int32), device=dev)
            tok_embed = mod["embed"](tok_d, params)
            if tok_embed.shape[0] != 1 or len(tok_embed.shape) == 2:
                tok_embed = tvm.get_global_func("vm.builtin.reshape")(
                    tok_embed, ShapeTuple([1, 1, tok_embed.shape[-1]])
                )
            logits, kv_cache, rnn_state = mod["decode"](
                tok_embed, mrope_deltas_d, kv_cache, rnn_state, params
            )
            kv_end(kv_cache)
            kv_end(rnn_state)
            logits_np = logits.numpy()
            if logits_np.ndim == 3:
                logits_np = logits_np[0, -1]
            elif logits_np.ndim == 2:
                logits_np = logits_np[-1]
            next_token = int(np.argmax(logits_np))
            generated.append(next_token)

        kv_remove_seq(kv_cache, 0)
        kv_remove_seq(rnn_state, 0)

        match = sum(1 for a, b in zip(generated, ref_tokens) if a == b)
        total_match += match
        total_tokens += n_steps

        # Margin-gated verdict (workplan §16.1, §18.13). Two things make an unweighted
        # match count the wrong instrument here, and they compound:
        #
        #  1. A near-tie is not a defect. The 2026-07-26e run scored 167/184 and "failed"
        #     on a single ',' vs '.' after a grammatically complete clause — a position
        #     where the reference itself barely preferred one token.
        #  2. **This decode is free-running**, so once MLC picks a different token it is
        #     conditioned on a different prefix and every later position is incomparable.
        #     Counting them at all overstates the damage of one flip. `high_margin_gate.py`
        #     dodges this with teacher forcing; this driver cannot without a restructure,
        #     so the honest quantity is the *first* divergence and the reference's margin
        #     there. Everything before it is a genuine agreement; everything after is
        #     unscoreable, not wrong.
        margins = p.get("margins")
        first_div = next((k for k, (a, b) in enumerate(zip(generated, ref_tokens)) if a != b), None)
        if first_div is None:
            verdict, div_margin = "exact", None
        elif margins is None:
            verdict, div_margin = "unscored (cache has no margins; rebuild with --regen)", None
        else:
            div_margin = margins[first_div] if first_div < len(margins) else float("nan")
            verdict = ("NEAR-TIE" if div_margin < args.tau else "WIDE-MARGIN DIVERGENCE")
            if div_margin >= args.tau:
                wide_failures += 1
        per_prompt_results.append((query, match, n_steps, ref_tokens, generated))
        extra = "" if first_div is None else (
            f"  first diff @{first_div}"
            + (f", ref margin {div_margin:.2f} -> {verdict}" if div_margin is not None else "")
        )
        print(f"[parity-vl5] prompt {i+1}/{len(cache['prompts'])}: {match}/{n_steps}  "
              f"query={query!r}{extra}")

    print()
    print("=" * 70)
    print(f"[parity-vl5] AGGREGATE: {total_match}/{total_tokens}  ({100 * total_match / total_tokens:.1f}%)")
    print("=" * 70)
    bar = int(0.96 * total_tokens)
    raw = "PASS" if total_match >= bar else "FAIL"
    print(f"[parity-vl5] raw count bar = {bar}/{total_tokens} (96%): {raw}"
          "   <- informational only; see the margin verdict below (§16.1, §18.13)")
    have_margins = any(p.get("margins") for p in cache["prompts"])
    if not have_margins:
        print("[parity-vl5] MARGIN VERDICT: unavailable — this cache predates margin capture. "
              "Rebuild it with `--reference-vl5 --regen`.")
    else:
        print(f"[parity-vl5] MARGIN VERDICT at tau={args.tau}: "
              f"{wide_failures} prompt(s) diverged at a wide-margin position: "
              f"{'PASS' if wide_failures == 0 else 'FAIL'}")
    print()
    print("[parity-vl5] Per-prompt diff:")
    for query, match, n_steps, ref, mlc in per_prompt_results:
        print(f"  [{match}/{n_steps}] {query!r}")
        if match < n_steps:
            for i, (a, b) in enumerate(zip(mlc, ref)):
                if a != b:
                    print(f"    first diff at step {i}: MLC={a} HF={b}")
                    break


# ──────────────────────────────────────────────────────────────────────────────
# Workplan item 0o — the first VL performance number
# ──────────────────────────────────────────────────────────────────────────────


def _stats(samples: List[float]) -> dict:
    """Median-centred summary. Median, not mean: a single scheduler hiccup on
    this box moves the mean of a 20-sample run by more than the effects being
    measured."""
    s = sorted(samples)
    n = len(s)
    return {
        "n": n,
        "median": s[n // 2],
        "min": s[0],
        "max": s[-1],
        "p90": s[min(n - 1, int(0.9 * n))],
    }


def run_perf_vl5(args: argparse.Namespace) -> None:
    """Time image_embed / prefill / decode separately on the compiled VL lib.

    This is `--greedy-parity-vl5` minus the reference comparison, plus a sync
    and a clock around each VM call. Three deliberate differences from that
    driver, each one a trap the workplan (item 0o) names:

      1. **The host-side embedding merge is never timed.** That driver does
         `image_embeds.numpy()` -> numpy scatter -> re-upload per prompt.
         Timing the enclosing loop measures numpy and a PCIe round trip, not
         the model. Every number below brackets exactly one VM call.
      2. **`image_embed` is hoisted and timed on its own.** The parity driver
         calls it once per prompt under a comment reading "could cache but
         cheap"; that was never measured, which is the first thing to fix.
      3. **No `MLCEngine`.** `cpp/serve/model.cc` calls `image_embed` with the
         llava signature and `ImageData` hardcodes the embed size, so the
         engine cannot drive this vision tower at all. Everything here runs on
         the raw VM.
    """
    cache_path = CACHE_FILE_VL5
    if not cache_path.exists():
        print(f"[perf-vl5] No cache at {cache_path}. Run --reference-vl5 first.")
        sys.exit(1)
    cache = torch.load(cache_path, weights_only=False)

    if args.mlc_model_dir is None:
        print("[perf-vl5] Missing --mlc-model-dir.")
        sys.exit(1)
    model_dir = Path(args.mlc_model_dir)
    lib_path = Path(args.mlc_lib) if args.mlc_lib else model_dir / "lib.so"
    if not lib_path.exists():
        print(f"[perf-vl5] No lib at {lib_path}.")
        sys.exit(1)

    import json as _json
    import time
    import tvm
    from tvm import relax
    from tvm.contrib import tvmjs
    from tvm.runtime import ShapeTuple
    from PIL import Image
    from mlc_llm.model.qwen3_5_vl.qwen3_5_vl_image import preprocess_image

    device_str = args.device if args.device != "cuda" else "cuda:0"
    dev = tvm.device(device_str)
    print(f"[perf-vl5] lib   = {lib_path}")
    print(f"[perf-vl5] dev   = {device_str}")
    ex = tvm.runtime.load_module(str(lib_path))
    vm = relax.VirtualMachine(ex, device=dev)
    mod = vm.module
    metadata = _json.loads(mod["_metadata"]())
    params, _meta = tvmjs.load_tensor_cache(str(model_dir), dev)
    param_names = [p["name"] for p in metadata["params"]]
    params = [params[n] for n in param_names]

    pos_embed_weight = _load_vl_pos_embed_weight(args.model)
    rgb = np.asarray(Image.open(cache["image_path"]).convert("RGB"), dtype=np.uint8)
    pre = preprocess_image(rgb, pos_embed_weight=pos_embed_weight)
    pixel_values_d = tvm.runtime.tensor(pre.pixel_values, device=dev)
    pos_embeds_d = tvm.runtime.tensor(pre.pos_embeds, device=dev)
    rotary_cos_d = tvm.runtime.tensor(pre.rotary_cos, device=dev)
    rotary_sin_d = tvm.runtime.tensor(pre.rotary_sin, device=dev)
    n_patches = int(pre.pixel_values.shape[0])

    if mod.implements_function("create_flashinfer_paged_kv_cache"):
        kv_create = mod["create_flashinfer_paged_kv_cache"]
    elif mod.implements_function("create_tir_paged_kv_cache"):
        kv_create = mod["create_tir_paged_kv_cache"]
    else:
        print("[perf-vl5] No KV cache create function.")
        sys.exit(1)
    kv_add_seq = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    kv_remove_seq = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
    kv_begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    kv_end = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    warmup, iters = args.perf_warmup, args.perf_iters

    # ── 1. image_embed — the vision tower + patch merger, once per image ──────
    for _ in range(warmup):
        mod["image_embed"](pixel_values_d, pos_embeds_d, rotary_cos_d, rotary_sin_d, params)
    dev.sync()
    embed_samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        image_embeds = mod["image_embed"](pixel_values_d, pos_embeds_d, rotary_cos_d, rotary_sin_d, params)
        dev.sync()
        embed_samples.append((time.perf_counter() - t0) * 1e3)
    ie = _stats(embed_samples)
    n_img_tok = int(image_embeds.shape[0])
    image_embeds_np = image_embeds.numpy()

    print()
    print("=" * 74)
    print(f"[perf-vl5] image_embed  ({n_patches} patches -> {n_img_tok} tokens)")
    print("=" * 74)
    print(f"  median {ie['median']:8.2f} ms   min {ie['min']:8.2f}   p90 {ie['p90']:8.2f}   "
          f"max {ie['max']:8.2f}   (n={ie['n']})")

    # ── 2. prefill and decode, per prompt ────────────────────────────────────
    # max_history is the RNN-state knob the engine sets from prefix_cache_mode:
    # 'disable' -> 1, 'radix' (the default) -> 64. The state path is sized by
    # it (§15), so measuring only one of them would report half the model.
    histories = [args.perf_max_history] if args.perf_max_history else [1, 64]

    p = cache["prompts"][0]
    input_ids = p["input_ids"][0]
    seq_len = len(input_ids)
    rope_position_ids = p["rope_position_ids"]
    rope_deltas = p["rope_deltas"]
    image_token_id = 248056

    # Build the merged embedding ONCE, outside all timing. This is the host
    # scatter trap: it is not model work and must not land in a number.
    input_ids_d = tvm.runtime.tensor(np.asarray(input_ids, dtype=np.int32), device=dev)
    text_embed = mod["embed"](input_ids_d, params).numpy()
    if text_embed.ndim == 3:
        text_embed = text_embed[0]
    image_pad_pos = np.where(np.asarray(input_ids) == image_token_id)[0]
    n_sub = min(len(image_pad_pos), image_embeds_np.shape[0])
    full_embed = text_embed.copy()
    full_embed[image_pad_pos[:n_sub]] = image_embeds_np[:n_sub]
    full_embed_d = tvm.runtime.tensor(
        full_embed.reshape(1, seq_len, -1).astype(text_embed.dtype), device=dev
    )
    pos_ids_d = tvm.runtime.tensor(np.asarray(rope_position_ids, dtype=np.int32), device=dev)
    mrope_deltas_d = tvm.runtime.tensor(np.asarray(rope_deltas, dtype=np.int32), device=dev)

    n_steps = args.perf_decode_steps
    print()
    print("=" * 74)
    print(f"[perf-vl5] prefill + decode  (seq_len={seq_len}, {n_sub} image tokens, "
          f"{n_steps} decode steps)")
    print("=" * 74)

    results = {}
    for max_hist in histories:
        max_total = seq_len + n_steps + 32
        prefill_chunk = max(seq_len + 32, 4096)

        def fresh_state():
            kv_cache = kv_create(
                ShapeTuple([1]), ShapeTuple([max_total]), ShapeTuple([prefill_chunk]),
                ShapeTuple([16]), ShapeTuple([0]),
            )
            rnn_state = mod["create_rnn_state"](ShapeTuple([1]), ShapeTuple([max_hist]))
            kv_add_seq(kv_cache, 0)
            kv_add_seq(rnn_state, 0)
            return kv_cache, rnn_state

        def one_prefill(kv_cache, rnn_state):
            kv_begin(kv_cache, ShapeTuple([0]), ShapeTuple([seq_len]))
            kv_begin(rnn_state, ShapeTuple([0]), ShapeTuple([seq_len]))
            logits, kv_cache, rnn_state = mod["prefill"](
                full_embed_d, pos_ids_d, kv_cache, rnn_state, params
            )
            kv_end(kv_cache)
            kv_end(rnn_state)
            return logits, kv_cache, rnn_state

        # -- prefill. Fresh state per iteration, created outside the clock. --
        for _ in range(warmup):
            kv_cache, rnn_state = fresh_state()
            one_prefill(kv_cache, rnn_state)
            kv_remove_seq(kv_cache, 0)
            kv_remove_seq(rnn_state, 0)
        dev.sync()
        prefill_samples = []
        for _ in range(iters):
            kv_cache, rnn_state = fresh_state()
            dev.sync()
            t0 = time.perf_counter()
            logits, kv_cache, rnn_state = one_prefill(kv_cache, rnn_state)
            dev.sync()
            prefill_samples.append((time.perf_counter() - t0) * 1e3)
            kv_remove_seq(kv_cache, 0)
            kv_remove_seq(rnn_state, 0)
        pf = _stats(prefill_samples)

        # -- decode. One prefill to establish state, then N timed steps. --
        # Reported per-step; the KV cache grows across the run, so the spread
        # between min and p90 is signal, not noise.
        def timed_decode():
            kv_cache, rnn_state = fresh_state()
            logits, kv_cache, rnn_state = one_prefill(kv_cache, rnn_state)
            logits_np = logits.numpy()
            if logits_np.ndim == 3:
                logits_np = logits_np[0, -1]
            elif logits_np.ndim == 2:
                logits_np = logits_np[-1]
            next_token = int(np.argmax(logits_np))
            dev.sync()
            samples = []
            for _ in range(n_steps):
                tok_d = tvm.runtime.tensor(np.array([next_token], dtype=np.int32), device=dev)
                dev.sync()
                t0 = time.perf_counter()
                kv_begin(kv_cache, ShapeTuple([0]), ShapeTuple([1]))
                kv_begin(rnn_state, ShapeTuple([0]), ShapeTuple([1]))
                tok_embed = mod["embed"](tok_d, params)
                if tok_embed.shape[0] != 1 or len(tok_embed.shape) == 2:
                    tok_embed = tvm.get_global_func("vm.builtin.reshape")(
                        tok_embed, ShapeTuple([1, 1, tok_embed.shape[-1]])
                    )
                logits, kv_cache, rnn_state = mod["decode"](
                    tok_embed, mrope_deltas_d, kv_cache, rnn_state, params
                )
                kv_end(kv_cache)
                kv_end(rnn_state)
                dev.sync()
                samples.append((time.perf_counter() - t0) * 1e3)
                logits_np = logits.numpy()
                if logits_np.ndim == 3:
                    logits_np = logits_np[0, -1]
                elif logits_np.ndim == 2:
                    logits_np = logits_np[-1]
                next_token = int(np.argmax(logits_np))
            kv_remove_seq(kv_cache, 0)
            kv_remove_seq(rnn_state, 0)
            return samples

        timed_decode()  # warmup repeat, discarded
        decode_samples = timed_decode()
        dc = _stats(decode_samples)

        tag = f"max_history={max_hist}" + (
            "  ('disable')" if max_hist == 1 else "  ('radix', the default)" if max_hist == 64 else ""
        )
        print()
        print(f"  {tag}")
        print(f"    prefill   median {pf['median']:8.2f} ms  min {pf['min']:8.2f}  "
              f"p90 {pf['p90']:8.2f}   -> {1e3 * seq_len / pf['median']:8.1f} tok/s")
        print(f"    decode    median {dc['median']:8.2f} ms  min {dc['min']:8.2f}  "
              f"p90 {dc['p90']:8.2f}   -> {1e3 / dc['median']:8.1f} tok/s")
        results[max_hist] = (pf, dc)

    # ── 3. What a chat turn actually costs ───────────────────────────────────
    ref_hist = 64 if 64 in results else histories[0]
    pf, dc = results[ref_hist]
    ttft = ie["median"] + pf["median"]
    print()
    print("=" * 74)
    print(f"[perf-vl5] One chat turn at max_history={ref_hist}")
    print("=" * 74)
    print(f"  ttft (image_embed + prefill)  {ttft:8.2f} ms  "
          f"= {ie['median']:.2f} + {pf['median']:.2f}")
    print(f"  image_embed share of ttft     {100 * ie['median'] / ttft:8.1f} %")
    print(f"  ...and of a {n_steps}-token turn      "
          f"{100 * ie['median'] / (ttft + n_steps * dc['median']):8.1f} %")
    if len(results) > 1:
        p1, d1 = results[1]
        p64, d64 = results[64]
        print()
        print(f"  radix vs disable:  prefill {p64['median'] / p1['median']:.3f}x   "
              f"decode {d64['median'] / d1['median']:.3f}x")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 10 Stage 5b diagnostic — text-only mrope-collapse on the VL lib
# ──────────────────────────────────────────────────────────────────────────────


def run_mrope_collapse(args: argparse.Namespace) -> None:
    """Drive the compiled VL lib on a TEXT-ONLY prompt with 3 identical
    position rows. mRoPE math collapses to plain 1D RoPE; output should match
    the text-only HF reference (50/50 token greedy match).

    If this passes, the inline-mRoPE prefill+decode path itself is sound and
    the multimodal greedy parity drift is dominated by image_embed / fp16
    cumulative drift through the LM with image features as input.

    If this fails, chunk-B's runtime IR has a bug — text-only inputs should
    be invariant to the rotation convention swap.
    """
    if args.mlc_model_dir is None:
        print("[mrope-collapse] Missing --mlc-model-dir.")
        sys.exit(1)
    if not Path(args.cache).exists():
        print(f"[mrope-collapse] No text-only reference at {args.cache}. Run --reference-only first.")
        sys.exit(1)

    ref = torch.load(args.cache, weights_only=False)
    fixed_prompt = ref["prompts"][0]  # "The capital of France is"
    ref_tokens = ref["results"][0]["tokens"]  # 50 greedy tokens
    print(f"[mrope-collapse] Prompt: {fixed_prompt!r}")
    print(f"[mrope-collapse] HF reference: {len(ref_tokens)} tokens; first 10: {ref_tokens[:10]}")

    model_dir = Path(args.mlc_model_dir)
    lib_path = Path(args.mlc_lib) if args.mlc_lib else model_dir / "lib.so"
    if not lib_path.exists():
        print(f"[mrope-collapse] No lib.so at {lib_path}.")
        sys.exit(1)

    import json as _json
    import tvm
    from tvm import relax
    from tvm.contrib import tvmjs
    from tvm.runtime import ShapeTuple

    device_str = args.device if args.device != "cuda" else "cuda:0"
    print(f"[mrope-collapse] Loading lib + params on {device_str}")
    dev = tvm.device(device_str)
    ex = tvm.runtime.load_module(str(lib_path))
    vm = relax.VirtualMachine(ex, device=dev)
    mod = vm.module
    metadata = _json.loads(mod["_metadata"]())
    params, _meta = tvmjs.load_tensor_cache(str(model_dir), dev)
    param_names = [p["name"] for p in metadata["params"]]
    params = [params[n] for n in param_names]
    print(f"[mrope-collapse] Loaded {len(params)} params")

    # Tokenize via HF (same path as reference cache)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = tokenizer(fixed_prompt, return_tensors="np", add_special_tokens=False)["input_ids"][0]
    seq_len = int(input_ids.shape[0])
    print(f"[mrope-collapse] Tokenized: {seq_len} tokens; ids={input_ids.tolist()}")

    # Run embed
    input_ids_d = tvm.runtime.tensor(input_ids.astype(np.int32), device=dev)
    text_embed = mod["embed"](input_ids_d, params).numpy()
    if text_embed.ndim == 3:
        text_embed = text_embed[0]
    embed_d = tvm.runtime.tensor(
        text_embed.reshape(1, seq_len, -1).astype(text_embed.dtype), device=dev
    )

    # 3 identical position rows = 1D RoPE collapse; mrope_deltas = 0
    pos_1d = np.arange(seq_len, dtype=np.int32)
    pos_3d = np.tile(pos_1d[None, None, :], (3, 1, 1))  # (3, 1, seq_len)
    pos_d = tvm.runtime.tensor(pos_3d, device=dev)
    mrope_deltas = np.zeros((1, 1), dtype=np.int32)
    mrope_deltas_d = tvm.runtime.tensor(mrope_deltas, device=dev)

    # Set up cache
    if mod.implements_function("create_flashinfer_paged_kv_cache"):
        kv_create = mod["create_flashinfer_paged_kv_cache"]
    elif mod.implements_function("create_tir_paged_kv_cache"):
        kv_create = mod["create_tir_paged_kv_cache"]
    else:
        print("[mrope-collapse] No KV cache create function found.")
        sys.exit(1)

    GREEDY = 50
    max_total = seq_len + GREEDY + 32
    kv_cache = kv_create(
        ShapeTuple([1]), ShapeTuple([max_total]), ShapeTuple([max(seq_len + 32, 4096)]),
        ShapeTuple([16]), ShapeTuple([0]),
    )
    rnn_state = mod["create_rnn_state"](ShapeTuple([1]), ShapeTuple([1]))
    kv_add_seq = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    kv_begin = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    kv_end = tvm.get_global_func("vm.builtin.kv_state_end_forward")

    kv_add_seq(kv_cache, 0)
    kv_add_seq(rnn_state, 0)

    # Prefill
    print(f"[mrope-collapse] Prefilling …")
    kv_begin(kv_cache, ShapeTuple([0]), ShapeTuple([seq_len]))
    kv_begin(rnn_state, ShapeTuple([0]), ShapeTuple([seq_len]))
    logits, kv_cache, rnn_state = mod["prefill"](embed_d, pos_d, kv_cache, rnn_state, params)
    kv_end(kv_cache)
    kv_end(rnn_state)
    logits_np = logits.numpy()
    if logits_np.ndim == 3:
        logits_np = logits_np[0, -1]
    elif logits_np.ndim == 2:
        logits_np = logits_np[-1]
    next_token = int(np.argmax(logits_np))
    generated = [next_token]
    print(f"[mrope-collapse] First token: {next_token} ({tokenizer.decode([next_token])!r})")

    # Decode loop
    for step in range(GREEDY - 1):
        kv_begin(kv_cache, ShapeTuple([0]), ShapeTuple([1]))
        kv_begin(rnn_state, ShapeTuple([0]), ShapeTuple([1]))
        tok_d = tvm.runtime.tensor(np.array([next_token], dtype=np.int32), device=dev)
        tok_embed = mod["embed"](tok_d, params)
        if tok_embed.shape[0] != 1 or len(tok_embed.shape) == 2:
            tok_embed = tvm.get_global_func("vm.builtin.reshape")(
                tok_embed, ShapeTuple([1, 1, tok_embed.shape[-1]])
            )
        logits, kv_cache, rnn_state = mod["decode"](
            tok_embed, mrope_deltas_d, kv_cache, rnn_state, params
        )
        kv_end(kv_cache)
        kv_end(rnn_state)
        logits_np = logits.numpy()
        if logits_np.ndim == 3:
            logits_np = logits_np[0, -1]
        elif logits_np.ndim == 2:
            logits_np = logits_np[-1]
        next_token = int(np.argmax(logits_np))
        generated.append(next_token)

    # Diff
    matches = sum(1 for a, b in zip(generated, ref_tokens) if a == b)
    print()
    print(f"[mrope-collapse] HF tokens : {ref_tokens[:10]}…")
    print(f"[mrope-collapse] MLC tokens: {generated[:10]}…")
    print(f"[mrope-collapse] Match: {matches}/{len(ref_tokens)}")
    print(f"[mrope-collapse] HF text: {tokenizer.decode(ref_tokens)!r}")
    print(f"[mrope-collapse] MLC text: {tokenizer.decode(generated)!r}")
    bar = TOKEN_MATCH_BAR
    print(f"[mrope-collapse] Bar = {bar}/{len(ref_tokens)}: {'PASS — inline-mRoPE path is sound' if matches >= bar else 'FAIL — chunk-B inline-mRoPE has runtime drift on text-only inputs'}")


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
    elif args.reference_vl:
        run_reference_vl(args)
    elif args.greedy_parity_vl:
        run_greedy_parity_vl(args)
    elif args.mrope_collapse:
        run_mrope_collapse(args)
    elif args.reference_vl5:
        run_reference_vl5(args)
    elif args.greedy_parity_vl5:
        run_greedy_parity_vl5(args)
    elif args.perf_vl5:
        run_perf_vl5(args)


if __name__ == "__main__":
    main()

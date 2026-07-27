#!/usr/bin/env python3
"""moe_expert_histogram.py — dump the *real* expert routing distribution (workplan item 0i).

Why this exists
---------------
Every ranking decision in the 35B's MoE GEMM rests on an assumed routing distribution,
and §17.9 showed **both** instruments in use are unrepresentative:

* the bench harness's filler prompt has **11 distinct tokens per 512**, which concentrates
  the router — it inflated item 0h's end-to-end gain 3x;
* `moe_gemm_check.make_inputs`'s synthetic `even` / uniform-`random` routings bracket
  nothing real — they predicted item 0h's **wrong sign**, not merely the wrong magnitude.

This measures the thing both were standing in for. The v2 GEMM's cost is set by its real
CTA count, `sum_e ceildiv(count_e, BLK_M) * tiles_per_n` (§17.7), so the only routing
statistic that matters is the per-expert assignment histogram. That comes straight off the
routers with a forward hook — no MLC instrumentation and no compile.

How
---
The routers are `...language_model.layers.{i}.mlp.gate`, plain `Linear(2048, 256)` kept in
bf16 by the fp8 checkpoint's `modules_to_not_convert`, so they run exactly as the bf16
master would. `Qwen3_5MoeTopKRouter.forward` returns `(logits, scores, indices)`; a hook
takes `indices` — the top-8 expert ids per token — and bincounts them.

The MTP draft layer's router is deliberately excluded: it is layer 40 in the module tree
and is not on the path any of these kernels are benched on.

Prompts come from `scripts/make_prose_corpus.py` via the same `build_prompt` the bench
harness uses, salted per window, so the routing measured here is the routing the pp
numbers in the workplan were produced under — not a different sample of English.

Usage:
    source .envrc.local
    python scripts/make_prose_corpus.py --out /tmp/prose_corpus.txt
    python scripts/moe_expert_histogram.py --prompt-file /tmp/prose_corpus.txt \\
        --out tuning/expert_hist_35b.npz
    python scripts/moe_expert_histogram.py --prompt-file /tmp/prose_corpus.txt \\
        --with-filler --windows 4          # quantify §17.9 in routing terms

The `.npz` it writes is consumed by `moe_gemm_check.py --indptr-file`, which replaces the
synthetic routing in the microbench with these counts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _ROOT)

# 35B-A3B: 256 experts, top-8, 40 MoE layers (+1 MTP draft layer, excluded).
NE = 256
BLK_MS = (16, 32, 64)


def tile_stats(counts: np.ndarray, blk_m: int) -> tuple[int, float]:
    """CTA rows the kernel must cover, and the share of them that are padding.

    `counts` is per-expert assignment counts for one layer. The dispatch table gives each
    expert its own tile range, so an expert with 17 rows costs two `BLK_M=16` tiles and
    pads 15 of the 32 rows.
    """
    tiles = int(np.sum(-(-counts // blk_m)))  # ceildiv, integer
    rows = tiles * blk_m
    real = int(counts.sum())
    return tiles, (rows - real) / rows if rows else 0.0


def synthetic_counts(b: int, routing: str, seed: int = 0) -> np.ndarray:
    """The two routings `moe_gemm_check.make_inputs` currently uses, for contrast."""
    if routing == "random":
        rng = np.random.default_rng(seed)
        return np.bincount(rng.integers(0, NE, size=b), minlength=NE).astype(np.int64)
    counts = np.full(NE, b // NE, dtype=np.int64)
    counts[: b % NE] += 1
    return counts


def text_config(model_or_cfg):
    """The text sub-config, whichever level of nesting this transformers build uses.

    `AutoModelForCausalLM` on this checkpoint hands back a bare `Qwen3_5MoeTextConfig`,
    while `AutoConfig` on the same repo gives the multimodal wrapper with `.text_config`.
    Reaching through a fixed attribute path costs an 11-minute load to find out.
    """
    cfg = getattr(model_or_cfg, "config", model_or_cfg)
    getter = getattr(cfg, "get_text_config", None)
    return getter() if getter else getattr(cfg, "text_config", cfg)


def find_routers(model):
    """Return `[(layer_index, name, module)]` for the main text stack's routers.

    Matched by name rather than by a fixed attribute path — the prefix is
    `language_model.layers` on the multimodal wrapper and plain `model.layers` when
    `AutoModelForCausalLM` unwraps to the text model, and it has moved between
    transformers versions either way.

    The MTP draft layer carries its own router and must not be counted: it is not on the
    path any of these kernels are benched on. It is excluded structurally — routers are
    grouped by their parent prefix and the largest group wins — rather than by layer index,
    because the draft layer is numbered 0 under its own prefix and an index test would
    silently drop the real layer 0 instead.
    """
    pat = re.compile(r"^(.*)\.layers\.(\d+)\.mlp\.gate$")
    groups: dict[str, list] = {}
    for name, mod in model.named_modules():
        m = pat.match(name)
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), name, mod))
    if not groups:
        raise SystemExit("no `*.layers.N.mlp.gate` modules found — module layout has "
                         "drifted; print [n for n, _ in model.named_modules()] and fix "
                         "the regex")
    main = max(groups, key=lambda k: len(groups[k]))
    kept = sorted(groups[main])
    n_text = text_config(model).num_hidden_layers
    if len(kept) != n_text:
        raise SystemExit(f"found {len(kept)} routers under {main!r} but config says "
                         f"num_hidden_layers={n_text}; refusing to guess")
    others = {k: len(v) for k, v in groups.items() if k != main}
    print(f"[hist] {len(kept)} routers hooked under {main!r}"
          + (f"; excluded {others} (MTP draft)" if others else ""))
    return kept


class Recorder:
    """Bincounts `router_indices` per layer, per forward call."""

    def __init__(self, routers, n_experts: int):
        self.n_experts = n_experts
        self.by_layer: dict[int, list[np.ndarray]] = {i: [] for i, _, _ in routers}
        self.handles = [m.register_forward_hook(self._make(i)) for i, _, m in routers]

    def _make(self, layer: int):
        def hook(_mod, _inp, out):
            # Qwen3_5MoeTopKRouter returns (router_logits, router_scores, router_indices).
            idx = out[2] if isinstance(out, (tuple, list)) else out
            flat = idx.reshape(-1).to("cpu").numpy()
            self.by_layer[layer].append(
                np.bincount(flat, minlength=self.n_experts).astype(np.int64))
        return hook

    def drain(self) -> np.ndarray:
        """(n_layers, n_experts) summed over every call since the last drain."""
        out = np.stack([np.sum(self.by_layer[i], axis=0) if self.by_layer[i]
                        else np.zeros(self.n_experts, dtype=np.int64)
                        for i in sorted(self.by_layer)])
        for i in self.by_layer:
            self.by_layer[i] = []
        return out

    def close(self):
        for h in self.handles:
            h.remove()


def summarize(counts: np.ndarray, label: str) -> dict:
    """counts: (n_layers, n_experts). Reports the spread across layers, not just a mean."""
    b = int(counts[0].sum())
    hit = np.count_nonzero(counts, axis=1)
    row = {"label": label, "B": b, "n_layers": int(counts.shape[0]),
           "experts_hit": {"min": int(hit.min()), "med": int(np.median(hit)),
                           "max": int(hit.max())},
           "max_count": int(counts.max()), "blk_m": {}}
    for blk in BLK_MS:
        per_layer = [tile_stats(c, blk) for c in counts]
        tiles = np.array([t for t, _ in per_layer])
        pad = np.array([p for _, p in per_layer])
        row["blk_m"][blk] = {
            "tiles": {"min": int(tiles.min()), "med": float(np.median(tiles)),
                      "max": int(tiles.max()), "mean": float(tiles.mean())},
            "pad_share": {"min": float(pad.min()), "med": float(np.median(pad)),
                          "max": float(pad.max()), "mean": float(pad.mean())},
        }
    return row


def print_summary(row: dict, ref: dict | None = None) -> None:
    b = row["B"]
    h = row["experts_hit"]
    print(f"\n=== {row['label']}   B={b}  layers={row['n_layers']}")
    print(f"    experts hit / {NE}: min {h['min']}  med {h['med']}  max {h['max']}"
          f"   busiest expert: {row['max_count']} rows")
    print(f"    {'BLK_M':>6} {'tiles (min/med/max)':>26} {'ideal':>7} {'over':>7}"
          f" {'padding share':>22}")
    for blk in BLK_MS:
        s = row["blk_m"][blk]
        t, p = s["tiles"], s["pad_share"]
        ideal = -(-b // blk)
        over = t["mean"] / ideal
        cmp = ""
        if ref is not None:
            cmp = f"   vs {ref['label']}: {t['mean'] / ref['blk_m'][blk]['tiles']['mean']:.2f}x tiles"
        print(f"    {blk:>6} {t['min']:>7} /{t['med']:>7.0f} /{t['max']:>7}"
              f" {ideal:>7} {over:>6.2f}x"
              f"   {p['min'] * 100:>5.1f} /{p['med'] * 100:>5.1f} /{p['max'] * 100:>5.1f} %{cmp}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt-file", required=True,
                   help="natural-language corpus; build with scripts/make_prose_corpus.py")
    p.add_argument("--lengths", default="512,2048",
                   help="prompt lengths to profile. B = length * top_k, and B differs 4x "
                        "between 512 and 2048, so both are needed")
    p.add_argument("--windows", type=int, default=3,
                   help="distinct corpus windows per length; the spread across them says "
                        "how much of the result is this corpus rather than English")
    p.add_argument("--with-filler", action="store_true",
                   help="also profile the bench harness's repeated-sentence filler, which "
                        "is what §17.9 found was picking winners")
    p.add_argument("--decode-steps", type=int, default=8,
                   help="greedy decode steps to profile after each prefill (0 to skip); "
                        "decode routes B=top_k rows, a different regime entirely")
    p.add_argument("--dry-run", action="store_true",
                   help="build the model on the meta device (no weights, seconds) and check "
                        "only that the routers and config can be found. The real load is "
                        "~11 min on this box; run this first after any transformers upgrade")
    p.add_argument("--out", default=None, help="write per-layer counts to this .npz")
    p.add_argument("--json", default=None, help="write the summary table to this .json")
    cli = p.parse_args()

    import torch  # noqa: E402  — after argparse so --help is instant

    from validate import load_hf  # noqa: E402
    sys.path.insert(0, _ROOT)
    from scratch_mlc_tg_sweep import build_prompt  # noqa: E402

    source = open(cli.prompt_file).read()
    lengths = [int(v) for v in cli.lengths.split(",")]

    if cli.dry_run:
        from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
        cfg = AutoConfig.from_pretrained(cli.model, trust_remote_code=True)
        with torch.device("meta"):
            meta = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        tc = text_config(meta)
        print(f"[hist] dry run: experts={tc.num_experts} top_k={tc.num_experts_per_tok} "
              f"layers={tc.num_hidden_layers}")
        find_routers(meta)
        print("[hist] dry run OK — module layout and config access are as expected")
        return

    model, tokenizer = load_hf(cli.model, "float16", cli.device)
    tc = text_config(model)
    top_k = tc.num_experts_per_tok
    n_experts = tc.num_experts
    assert n_experts == NE, f"expected {NE} experts, config says {n_experts}"

    routers = find_routers(model)
    rec = Recorder(routers, n_experts)

    saved: dict[str, np.ndarray] = {}
    summaries: list[dict] = []

    def checkpoint() -> None:
        """Write after every phase. The load alone is ~6 min and the box is memory-tight,
        so a run that dies at pp2048 must not take pp512's result with it."""
        if cli.out and saved:
            os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
            np.savez_compressed(cli.out, top_k=np.array(top_k), **saved)
        if cli.json and summaries:
            with open(cli.json, "w") as fh:
                json.dump(summaries, fh, indent=1)

    sources = [("prose", source)] + ([("filler", None)] if cli.with_filler else [])
    for length in lengths:
        for sname, src in sources:
            per_window = []
            for w in range(cli.windows if src is not None else 1):
                text, n_tok = build_prompt(tokenizer, length, salt=f"w{w}", source=src)
                ids = tokenizer(text, return_tensors="pt").input_ids.to(cli.device)
                rec.drain()  # discard anything from a previous phase
                with torch.no_grad():
                    out = model(ids, use_cache=cli.decode_steps > 0)
                counts = rec.drain()
                b = int(counts[0].sum())
                assert b == ids.shape[1] * top_k, \
                    f"expected B={ids.shape[1] * top_k}, hooks saw {b}"
                per_window.append(counts)
                print(f"[hist] {sname} len={length} (re-encoded {n_tok}) window {w}: "
                      f"B={b}, experts hit {np.count_nonzero(counts, axis=1).mean():.1f} avg")

                if cli.decode_steps and w == 0 and sname == "prose":
                    dec = []
                    past, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)
                    for _ in range(cli.decode_steps):
                        with torch.no_grad():
                            out = model(tok, past_key_values=past, use_cache=True)
                        dec.append(rec.drain())
                        past, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)
                    d = np.stack(dec).sum(0)  # summed over steps: what a step batch sees
                    saved[f"decode_len{length}"] = d
                    summaries.append(summarize(
                        d, f"decode x{cli.decode_steps} after prose len={length}"))
                    print_summary(summaries[-1])
                    checkpoint()
                del out
                torch.cuda.empty_cache()

            stack = np.stack(per_window)  # (windows, layers, experts)
            saved[f"{sname}_len{length}"] = stack
            checkpoint()
            for w, counts in enumerate(per_window):
                summaries.append(summarize(counts, f"{sname} len={length} window{w}"))

            b = length * top_k
            real_ref = summaries[-len(per_window)]  # window 0; capture before appending
            print_summary(real_ref, ref=None)
            for r in ("even", "random"):
                s = summarize(synthetic_counts(b, r)[None, :], f"synthetic {r}")
                print_summary(s, ref=real_ref)
                summaries.append(s)

    rec.close()

    checkpoint()
    if cli.out:
        print(f"\n[hist] wrote {cli.out}: {', '.join(sorted(saved))}")
    if cli.json:
        print(f"[hist] wrote {cli.json}")


if __name__ == "__main__":
    main()

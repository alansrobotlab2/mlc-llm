#!/usr/bin/env python3
"""Prefix-cache round-trip gate for hybrid (GDN + attention) models.

Closes the gap flagged at worklog.md:946 ("the actual PopN-round-trip parity test") and
guards the risk at worklog.md:1112: the RNNState `get`/`set` indirection exists *because*
the recurrent state has to stay rollback-able, so anything that fuses the state access has
to preserve the history ring, not just the arithmetic.

Why the ordinary greedy-parity gate is not enough. It runs five *distinct* prompts through
a fresh engine, so radix caching never gets a prefix worth reusing and the rollback path is
never entered. A fused state update that wrote back into the *current* history slot instead
of the next one would sail through it — and through every bench harness, since those set
`prefix_cache_mode="disable"`, where `max_history == 1` makes the two slots coincide. The
corruption only shows up under the *default* `"radix"` mode with `max_history == 64`.

What this does instead, all inside one engine so the cache is warm:

  pass 1  five prompts, cold cache                        -> reference text
  pass 2  the same five prompts                           -> must be byte-identical (reuse)
  pass 3  extensions of those prompts, longest first      -> must be byte-identical to a
          then the bare prefixes again                       cold run of the same prompts

Pass 3 is the one that actually exercises `PopN`: asking for a *shorter* prompt after a
longer one that shares its prefix is what makes `MatchPrefixCache` fork from a recycling
parent and roll the recurrent state back by (parent_length - matched) positions.

Exit code is nonzero on any divergence, so this can gate a refactor in CI.

**This is a bit-exact gate, so it is only a pass/fail bar on a model that decodes
deterministically — in practice `q0f16`.** On the 35B `q4f16_1` it is a *comparative* signal,
for the same reason workplan-cuda-13.md §6.2 gives for the fp8 gate: 4-bit near-ties flip on
run-to-run nondeterminism. Measured 2026-07-25 — the in-place and copy-path libs each showed
exactly one divergence out of twenty comparisons, both on prompt 4, both between the same two
orderings ("red, yellow, and blue" vs "red, blue, and yellow"), but on *different* checks. Read
it that way: a mechanism bug fails a whole check (5/5 -> 0/5, as the negative control in §11
does), while a near-tie moves one prompt around between runs. If you need a hard bar on a
quantized model, run it a few times and require the *set* of failing checks to be unstable.

    python scripts/prefix_cache_roundtrip.py \
        --model-dir dist/qwen3_5-0.8B-q0f16_fused \
        --model-lib dist/qwen3_5-0.8B-q0f16_fused/lib_inplace.so
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Shared-prefix families: each base is a strict prefix of its extension, which is what
# forces a fork-with-rollback rather than a clean exact-match reuse.
PROMPT_FAMILIES = [
    ("The capital of France is", " Paris, a city on the river Seine which"),
    ("def fibonacci(n):\n    ", "if n <= 1:\n        return n\n    "),
    ("1, 1, 2, 3, 5, 8, 13, 21,", " 34, 55, 89, 144,"),
    ("Once upon a time in a land far away,", " there lived a wise old owl who"),
    ("The three primary colors are", " red, blue and"),
]
MAX_TOKENS = 40


def _gen(engine, model_name: str, max_tokens: int, prompt: str) -> str:
    """One greedy completion.

    Deliberately `completions.create` and not `engine._generate`: the latter hangs with the
    GPU idle on the 35B under `mode="interactive"` + radix, which is the auto-config loop
    documented at worklog.md:815. Pairing it with the explicit small `EngineConfig` sizes in
    `new_engine` is the combination that is known to work.
    """
    resp = engine.completions.create(
        prompt=prompt,
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        stream=False,
        extra_body={"ignore_eos": True},
    )
    return resp.choices[0].text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--model-lib", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--prefix-cache-mode",
        default="radix",
        choices=["disable", "radix"],
        help="Default 'radix' is the point of this test; 'disable' is only for a control run.",
    )
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    # Internal: the driver re-invokes this script once per phase so each engine gets its
    # own process (see the comment in main). Not meant to be passed by hand.
    p.add_argument("--phase", choices=["cold", "warm"], default=None, help=argparse.SUPPRESS)
    p.add_argument("--emit-json", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    from mlc_llm import MLCEngine
    from mlc_llm.serve.config import EngineConfig

    model_name = json.loads(
        (pathlib.Path(args.model_dir) / "mlc-chat-config.json").read_text()
    )["model_type"]
    bases = [b for b, _ in PROMPT_FAMILIES]
    extended = [b + e for b, e in PROMPT_FAMILIES]

    def new_engine():
        return MLCEngine(
            model=args.model_dir,
            model_lib=args.model_lib,
            device=args.device,
            mode="interactive",
            engine_config=EngineConfig(
                # Explicit small sizes, not the interactive auto-config path: the latter
                # (max_total=262144, prefill_chunk=2048) is half of the worklog.md:815 hang.
                max_num_sequence=2,
                max_total_sequence_length=4096,
                prefill_chunk_size=512,
                prefix_cache_mode=args.prefix_cache_mode,
            ),
        )

    # Each phase gets its own process. The two engines cannot coexist on this box — the 35B
    # is ~41 GB with a max_history=64 rnn_state — and neither `terminate()` nor a forced
    # `gc.collect()` releases the C++ engine's device memory in time, so an in-process
    # second `MLCEngine(...)` dies in cudaMalloc. Process exit is the only reliable free.
    if args.phase == "cold":
        # Cold references: each prompt seen once, so no reuse can occur.
        cold = new_engine()
        out = {
            "base": [_gen(cold, model_name, args.max_tokens, s) for s in bases],
            "ext": [_gen(cold, model_name, args.max_tokens, s) for s in extended],
        }
        pathlib.Path(args.emit_json).write_text(json.dumps(out))
        return 0

    if args.phase == "warm":
        warm = new_engine()
        out = {
            "pass1": [_gen(warm, model_name, args.max_tokens, s) for s in bases],
            "pass2": [_gen(warm, model_name, args.max_tokens, s) for s in bases],
            # Longest first, then the bare prefix -> fork from a recycling parent + PopN.
            "pass3_ext": [_gen(warm, model_name, args.max_tokens, s) for s in extended],
            "pass3_base": [_gen(warm, model_name, args.max_tokens, s) for s in bases],
        }
        pathlib.Path(args.emit_json).write_text(json.dumps(out))
        return 0

    # Driver: run both phases as subprocesses, then compare.
    tmp = tempfile.mkdtemp(prefix="prefix_cache_rt_")
    cold_json = str(pathlib.Path(tmp) / "cold.json")
    warm_json = str(pathlib.Path(tmp) / "warm.json")
    common = [
        sys.executable, __file__,
        "--model-dir", args.model_dir, "--model-lib", args.model_lib,
        "--device", args.device, "--prefix-cache-mode", args.prefix_cache_mode,
        "--max-tokens", str(args.max_tokens),
    ]
    for phase, out_path in (("cold", cold_json), ("warm", warm_json)):
        print(f"[prefix-cache] running {phase} phase...", flush=True)
        rc = subprocess.call(common + ["--phase", phase, "--emit-json", out_path])
        if rc != 0:
            print(f"[prefix-cache] {phase} phase failed (exit {rc})")
            return rc

    cold_out = json.loads(pathlib.Path(cold_json).read_text())
    warm_out = json.loads(pathlib.Path(warm_json).read_text())
    cold_base, cold_ext = cold_out["base"], cold_out["ext"]
    pass1, pass2 = warm_out["pass1"], warm_out["pass2"]
    pass3_ext, pass3_base = warm_out["pass3_ext"], warm_out["pass3_base"]
    shutil.rmtree(tmp, ignore_errors=True)

    checks = [
        ("pass1 vs cold (no reuse yet)", bases, cold_base, pass1),
        ("pass2 vs pass1 (exact-match reuse)", bases, pass1, pass2),
        ("pass3 extended vs cold (fork from base)", extended, cold_ext, pass3_ext),
        ("pass3 base vs cold (fork + PopN rollback)", bases, cold_base, pass3_base),
    ]

    failures = 0
    for label, prompts, want, got in checks:
        bad = [i for i in range(len(prompts)) if want[i] != got[i]]
        status = "PASS" if not bad else "FAIL"
        print(f"[prefix-cache] {label}: {len(prompts) - len(bad)}/{len(prompts)}  [{status}]")
        for i in bad:
            failures += 1
            print(f"    prompt {i}: {prompts[i]!r}")
            print(f"      expected: {want[i]!r}")
            print(f"      got     : {got[i]!r}")

    if failures:
        print(f"\n[prefix-cache] FAILED — {failures} divergence(s) with "
              f"prefix_cache_mode={args.prefix_cache_mode}")
        return 1
    print(f"\n[prefix-cache] ALL CHECKS PASS (prefix_cache_mode={args.prefix_cache_mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

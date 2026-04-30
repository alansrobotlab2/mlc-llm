"""Stage 8.1 smoke: confirm a freshly recompiled hybrid lib exposes the new
prefix-cacheable prefill entry points AND that the standard (non-history)
prefill path still produces coherent text after the C++ changes to
`Model::BatchPrefill` / `Model::BatchPrefillToLastHidden` (added a
`cache_prefill` flag with default false).

What this script does NOT cover (deferred to Stage 8.2):
  - The actual prefix-cache PopN-round-trip parity test. That requires the
    engine's radix cache to opt into `cache_prefill=true`, which is Stage 8.2
    work. Stage 8.1 only adds the plumbing.

Run after a `mlc compile` of a 0.8B / 35B hybrid lib that picks up the new
spec entries `batch_prefill_with_history` and
`batch_prefill_to_last_hidden_states_with_history`.

Usage:
    python scratch_phase8_stage1_smoke.py \\
        --model-dir dist/qwen3_5-0.8B-q4f16_1 \\
        --model-lib dist/qwen3_5-0.8B-q4f16_1/lib.so
"""
from __future__ import annotations

import argparse
import sys
import time


REQUIRED_FUNCS = (
    "batch_prefill",  # baseline (must still exist)
    "batch_prefill_with_history",  # new
    "batch_prefill_to_last_hidden_states",  # baseline
    "batch_prefill_to_last_hidden_states_with_history",  # new
    "batch_verify_to_last_hidden_states",  # phase 4B
)


def check_lib_exposes_funcs(model_lib: str) -> dict[str, bool]:
    """Inspect the compiled lib's exported VM functions via VirtualMachine.

    Accessing `vm[fname]` raises if the function isn't in the lib's spec —
    that's how `mlc_llm.cli.model_metadata` discovers `_metadata`.
    """
    from tvm.runtime import device, load_module
    from tvm.runtime.vm import VirtualMachine

    mod = load_module(model_lib)
    vm = VirtualMachine(mod, device("cpu"))
    found: dict[str, bool] = {}
    for fname in REQUIRED_FUNCS:
        try:
            _ = vm[fname]
            found[fname] = True
        except Exception:  # noqa: BLE001
            found[fname] = False
    return found


def smoke_engine_load_and_gen(model_dir: str, model_lib: str, n_tokens: int = 10) -> str:
    """Confirm the standard prefill path still produces coherent text. The new
    `cache_prefill` C++ flag defaults to false, so this should be a pure
    regression check on the existing decode path."""
    from mlc_llm import MLCEngine

    print(f"[smoke] loading engine: {model_lib}", flush=True)
    t0 = time.time()
    eng = MLCEngine(model_dir, mode="interactive", model_lib=model_lib)
    print(f"[smoke] engine load dt={time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    out = ""
    for tok in eng.chat.completions.create(
        stream=True,
        messages=[{"role": "user", "content": "The capital of France is"}],
        max_tokens=n_tokens,
        temperature=0.0,
    ):
        delta = tok.choices[0].delta.content
        if delta:
            out += delta
    print(f"[smoke] gen dt={time.time() - t0:.1f}s", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-lib", required=True)
    parser.add_argument("--n-tokens", type=int, default=10)
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="Only run the lib-symbol check, skip the engine smoke.",
    )
    args = parser.parse_args()

    # Step 1: lib exposes the new spec entries.
    print("[smoke] step 1: lib symbol check")
    found = check_lib_exposes_funcs(args.model_lib)
    for fname, ok in found.items():
        flag = "OK   " if ok else "MISS "
        print(f"  {flag} {fname}")
    missing_new = [
        f for f in (
            "batch_prefill_with_history",
            "batch_prefill_to_last_hidden_states_with_history",
        ) if not found.get(f, False)
    ]
    if missing_new:
        print(
            f"[smoke] FAIL: new prefill-with-history functions missing from lib: {missing_new}. "
            f"Recompile the lib after picking up the new `get_default_spec` entries.",
            flush=True,
        )
        sys.exit(2)
    missing_base = [
        f for f in ("batch_prefill", "batch_prefill_to_last_hidden_states")
        if not found.get(f, False)
    ]
    if missing_base:
        print(
            f"[smoke] FAIL: baseline prefill functions missing from lib: {missing_base}. "
            f"Lib likely doesn't match the qwen35 model module.",
            flush=True,
        )
        sys.exit(2)

    if args.skip_engine:
        print("[smoke] OK (lib-symbol-only)")
        return

    # Step 2: engine load + tiny gen (regression: cache_prefill=false default
    # path must still produce coherent text after the C++ signature change).
    print("[smoke] step 2: engine smoke")
    out = smoke_engine_load_and_gen(args.model_dir, args.model_lib, args.n_tokens)
    print(f"[smoke] out: {out!r}", flush=True)
    if not out.strip():
        print("[smoke] FAIL: empty output from standard prefill path", flush=True)
        sys.exit(1)
    print("[smoke] OK (lib symbols + standard prefill regression)")


if __name__ == "__main__":
    main()

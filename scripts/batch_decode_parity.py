#!/usr/bin/env python3
"""Concurrent-batch parity gate for hybrid (GDN + attention) models.

Both the greedy-parity gate and `prefix_cache_roundtrip.py` run one request at a time in
`mode="interactive"` (max_batch_size 1), so every decode step sees batch_size == 1 and the
recurrent state is always read from sequence slot 0. That leaves the per-batch slot
indexing untested: a kernel that reads the recurrent state itself does
`storage[seq_slot_ids[b], ...]` per batch element, and getting that indexing wrong is
invisible until two sequences are in flight at once.

This runs the same prompts twice — once serially (batch 1, the configuration the other
gates cover) and once with all of them in flight concurrently — and requires the generated
text to be identical. Greedy decode makes each sequence independent of its batch-mates, so
any difference means state is leaking across sequence slots.

    python scripts/batch_decode_parity.py \
        --model-dir dist/qwen3_5-0.8B-q0f16_fused \
        --model-lib dist/qwen3_5-0.8B-q0f16_fused/lib_inplace.so

Both phases are timed, so the run also answers "what does concurrency buy" — the serial
phase is the same work one request at a time, so `concurrent speedup` is the end-to-end win
from batching decode.

History: until 2026-07-25 the concurrent phase could not run at all on hybrid models,
because `Model::BatchPrefill` packs the batch into a single `(1, total_len, h)` row while
`RNNState::BeginForward` is told about N sequences, and `rnn_state_get_1` then raised
"Mismatched output.shape[0] ... expected to match seq_slot_ids.shape[0]". That was an
engine-side limitation, not a model or kernel one; see workplan-cuda-13.md §12 for the fix.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "1, 1, 2, 3, 5, 8, 13, 21,",
    "Once upon a time in a land far away,",
    "The three primary colors are",
    "The chemical symbol for gold is",
]
MAX_TOKENS = 40


def _gen(engine, model_name: str, max_tokens: int, prompt: str) -> str:
    """One greedy completion.

    Uses `completions.create` rather than `engine._generate`: the latter never returns for
    some engine configurations (worklog.md:815), and it deadlocks outright when driven from
    several threads at once, which is exactly what this gate needs to do.
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


async def _gen_concurrent(args, model_name: str, max_seq: int) -> list[str]:
    """Run every prompt at once through an AsyncMLCEngine, preserving prompt order."""
    from mlc_llm import AsyncMLCEngine
    from mlc_llm.serve.config import EngineConfig

    engine = AsyncMLCEngine(
        model=args.model_dir,
        model_lib=args.model_lib,
        device=args.device,
        engine_config=EngineConfig(
            max_num_sequence=max_seq,
            max_total_sequence_length=4096,
            prefill_chunk_size=512,
            prefix_cache_mode=args.prefix_cache_mode,
        ),
        mode="interactive",
    )

    async def one(prompt: str) -> str:
        resp = await engine.completions.create(
            prompt=prompt,
            model=model_name,
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            stream=False,
            extra_body={"ignore_eos": True},
        )
        return resp.choices[0].text

    try:
        # Same warmup as the serial phase, for the same reason.
        await engine.completions.create(
            prompt=PROMPTS[0], model=model_name, max_tokens=4, temperature=0.0,
            top_p=1.0, stream=False, extra_body={"ignore_eos": True},
        )
        t0 = time.perf_counter()
        out = list(await asyncio.gather(*(one(s) for s in PROMPTS)))
        return out, time.perf_counter() - t0
    finally:
        engine.terminate()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-dir", required=True)
    p.add_argument("--model-lib", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument(
        "--prefix-cache-mode", default="disable", choices=["disable", "radix"],
        help="Default 'disable' isolates batch-slot indexing from prefix-cache reuse.",
    )
    # Internal: the driver re-invokes this script once per phase so each engine gets its
    # own process. Not meant to be passed by hand.
    p.add_argument("--phase", choices=["serial", "concurrent"], default=None, help=argparse.SUPPRESS)
    p.add_argument("--emit-json", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    from mlc_llm import MLCEngine
    from mlc_llm.serve.config import EngineConfig

    n = len(PROMPTS)
    model_name = json.loads(
        (pathlib.Path(args.model_dir) / "mlc-chat-config.json").read_text()
    )["model_type"]

    def new_engine(max_seq: int):
        return MLCEngine(
            model=args.model_dir,
            model_lib=args.model_lib,
            device=args.device,
            engine_config=EngineConfig(
                max_num_sequence=max_seq,
                max_total_sequence_length=4096,
                prefill_chunk_size=512,
                prefix_cache_mode=args.prefix_cache_mode,
            ),
            mode="interactive",
        )

    # One engine per process — see the same note in prefix_cache_roundtrip.py: neither
    # terminate() nor gc frees the C++ engine's device memory in time for a second one.
    if args.phase == "serial":
        # One sequence in flight, so batch_size is always 1.
        engine = new_engine(1)
        # Warm up outside the timed region: the first request pays kernel JIT and
        # cudagraph capture, which would otherwise land entirely on the serial phase and
        # flatter the concurrent one.
        _gen(engine, model_name, 4, PROMPTS[0])
        t0 = time.perf_counter()
        out = [_gen(engine, model_name, args.max_tokens, s) for s in PROMPTS]
        elapsed = time.perf_counter() - t0
        pathlib.Path(args.emit_json).write_text(json.dumps({"out": out, "elapsed": elapsed}))
        return 0

    if args.phase == "concurrent":
        # All prompts in flight, so decode steps run with batch_size > 1 across distinct
        # sequence slots — the thing this gate exists to cover.
        #
        # `AsyncMLCEngine` + asyncio.gather, not the sync engine driven from a thread pool:
        # the sync `MLCEngine.completions.create` hangs with the GPU idle when several
        # threads call it at once (measured — the serial phase completes, the concurrent
        # phase never returns). Async is the supported way to get requests in flight
        # together, which is the entire point of this phase.
        out, elapsed = asyncio.run(_gen_concurrent(args, model_name, n))
        pathlib.Path(args.emit_json).write_text(json.dumps({"out": out, "elapsed": elapsed}))
        return 0

    tmp = tempfile.mkdtemp(prefix="batch_parity_")
    common = [
        sys.executable, __file__,
        "--model-dir", args.model_dir, "--model-lib", args.model_lib,
        "--device", args.device, "--prefix-cache-mode", args.prefix_cache_mode,
        "--max-tokens", str(args.max_tokens),
    ]
    results = {}
    for phase in ("serial", "concurrent"):
        out_path = str(pathlib.Path(tmp) / f"{phase}.json")
        print(f"[batch-parity] running {phase} phase...", flush=True)
        rc = subprocess.call(common + ["--phase", phase, "--emit-json", out_path])
        if rc != 0:
            print(f"[batch-parity] {phase} phase failed (exit {rc})")
            return rc
        results[phase] = json.loads(pathlib.Path(out_path).read_text())
    shutil.rmtree(tmp, ignore_errors=True)
    serial, batched = results["serial"]["out"], results["concurrent"]["out"]
    t_serial, t_conc = results["serial"]["elapsed"], results["concurrent"]["elapsed"]

    print(f"[batch-parity] wall clock: serial {t_serial:.2f}s, concurrent {t_conc:.2f}s "
          f"-> {t_serial / t_conc:.2f}x for {n} requests x {args.max_tokens} tokens")
    bad = [i for i in range(n) if serial[i] != batched[i]]
    print(f"[batch-parity] serial vs concurrent: {n - len(bad)}/{n} "
          f"[{'PASS' if not bad else 'FAIL'}]")
    for i in bad:
        print(f"    prompt {i}: {PROMPTS[i]!r}")
        print(f"      serial    : {serial[i]!r}")
        print(f"      concurrent: {batched[i]!r}")

    if bad:
        print("\n[batch-parity] FAILED — recurrent state is not isolated per sequence slot")
        return 1
    print("\n[batch-parity] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

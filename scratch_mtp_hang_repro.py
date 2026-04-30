"""Bisect the interactive-auto-config + MTP + prefix_cache hang.

Run with `--cfg <name>`:
  baseline    = working: explicit max_num_seq=2, max_total=4096, chunk=512, recycling=2
  interactive = hangs (we hope): no overrides; let mode="interactive" auto-config pick
  mns1        = explicit: max_num_seq=1 (from auto-config), but otherwise small
  mts262k     = explicit: max_total=262144 (from auto-config), but otherwise small
  chunk2048   = explicit: prefill_chunk=2048 (from auto-config), but otherwise small
  recycling0  = explicit: max_num_seq=2 but prefix_cache_max_num_recycling_seqs=0
  recycling1  = explicit: max_num_seq=2 but prefix_cache_max_num_recycling_seqs=1
  noprefix    = explicit small + prefix_cache="disable" (sanity)
"""
import argparse
import sys
import time

from mlc_llm.serve import EngineConfig, MLCEngine
from transformers import AutoTokenizer

TARGET = "dist/qwen3_6-35B-A3B-q4f16_1"
DRAFT = "dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft"


def cfg_for(name: str) -> tuple[EngineConfig, str]:
    base = dict(
        additional_models=[(DRAFT, f"{DRAFT}/lib.so")],
        speculative_mode="eagle",
        spec_draft_length=1,
        prefix_cache_mode="radix",
    )
    if name == "baseline":
        return EngineConfig(**base, max_num_sequence=2, max_total_sequence_length=4096,
                            prefill_chunk_size=512), "working baseline"
    if name == "interactive":
        return EngineConfig(**base), "auto-config (mode=interactive picks values)"
    if name == "mns1":
        return EngineConfig(**base, max_num_sequence=1, max_total_sequence_length=4096,
                            prefill_chunk_size=512), "auto's max_num_seq=1, rest small"
    if name == "mts262k":
        return EngineConfig(**base, max_num_sequence=2, max_total_sequence_length=262144,
                            prefill_chunk_size=512), "auto's max_total=262144"
    if name == "chunk2048":
        return EngineConfig(**base, max_num_sequence=2, max_total_sequence_length=4096,
                            prefill_chunk_size=2048), "auto's prefill_chunk=2048"
    if name == "recycling0":
        return EngineConfig(**base, max_num_sequence=2, max_total_sequence_length=4096,
                            prefill_chunk_size=512, prefix_cache_max_num_recycling_seqs=0
                            ), "max_num_seq=2 + recycling=0"
    if name == "recycling1":
        return EngineConfig(**base, max_num_sequence=2, max_total_sequence_length=4096,
                            prefill_chunk_size=512, prefix_cache_max_num_recycling_seqs=1
                            ), "max_num_seq=2 + recycling=1"
    if name == "noprefix":
        cfg = dict(base); cfg["prefix_cache_mode"] = "disable"
        return EngineConfig(**cfg, max_num_sequence=2, max_total_sequence_length=4096,
                            prefill_chunk_size=512), "explicit small, prefix_cache=disable"
    if name == "mns1_noprefix":
        cfg = dict(base); cfg["prefix_cache_mode"] = "disable"
        return EngineConfig(**cfg, max_num_sequence=1, max_total_sequence_length=4096,
                            prefill_chunk_size=512), "max_num_seq=1 + prefix_cache=disable"
    if name == "mns1_target_only":
        cfg = {"prefix_cache_mode": "radix"}
        return EngineConfig(**cfg, max_num_sequence=1, max_total_sequence_length=4096,
                            prefill_chunk_size=512), "max_num_seq=1 + prefix_cache=radix, no MTP"
    raise SystemExit(f"unknown cfg {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gen-timeout", type=float, default=60.0,
                        help="seconds before declaring hang")
    args = parser.parse_args()

    cfg, descr = cfg_for(args.cfg)
    print(f"[repro] cfg={args.cfg}: {descr}", flush=True)

    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tok.encode(filler, add_special_tokens=False)[: args.prompt_len]
    prompt = tok.decode(ids)
    print(f"[repro] prompt token len = {len(ids)}", flush=True)

    t_load = time.time()
    eng = MLCEngine(model=TARGET, model_lib=f"{TARGET}/lib.so",
                    device="cuda:0", mode="interactive", engine_config=cfg)
    print(f"[repro] engine constructed in {time.time()-t_load:.1f}s", flush=True)

    # Wrap gen in a watchdog: if no response in `gen_timeout` sec, declare hang.
    import signal

    def alarm_handler(signum, frame):
        raise TimeoutError(f"hang detected (>{args.gen_timeout:.0f}s)")
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(int(args.gen_timeout))

    try:
        t_gen = time.time()
        out = eng.completions.create(
            prompt=prompt, model="qwen3_5_moe", max_tokens=args.max_tokens,
            temperature=0.0, stream=False, extra_body={"ignore_eos": True},
        )
        signal.alarm(0)
        print(f"[repro] req-A completed in {time.time()-t_gen:.2f}s", flush=True)
        print(f"[repro] out: {out.choices[0].text!r}", flush=True)
        eng.terminate()
        return 0
    except TimeoutError as e:
        signal.alarm(0)
        print(f"[repro] HANG: {e}", flush=True)
        # Don't terminate — let the parent kill us so we get a clean stack
        return 2


if __name__ == "__main__":
    sys.exit(main())

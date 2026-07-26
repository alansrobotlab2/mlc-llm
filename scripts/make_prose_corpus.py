#!/usr/bin/env python3
"""make_prose_corpus.py — build a natural-language corpus for `--prompt-file` benching.

`scratch_mlc_tg_sweep.py`'s built-in filler is one sentence repeated, which gives a
512-token prompt **11 distinct tokens**. On a dense model that is harmless; on this MoE
it concentrates the router and changes which kernel configuration wins (workplan §17.9).
Any pp measurement whose conclusion depends on expert routing needs real text instead.

This strips code fences and markdown furniture out of the repo's own prose so the result
is ordinary English at ordinary token diversity (~40% distinct at 512 tokens), and is
committed so the §17.10 numbers can be reproduced exactly.

Usage:
    python scripts/make_prose_corpus.py --out /tmp/prose_corpus.txt
    python scratch_mlc_tg_sweep.py ... --prompt-file /tmp/prose_corpus.txt
"""
from __future__ import annotations

import argparse
import pathlib
import re

DEFAULT_SOURCES = ["worklog.md"]


def build(sources: list[str]) -> str:
    root = pathlib.Path(__file__).resolve().parent.parent
    text = "\n".join((root / s).read_text() for s in sources)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)   # code blocks are not prose
    text = re.sub(r"[`|#*\-_>]+", " ", text)             # markdown furniture
    return re.sub(r"\s+", " ", text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
    cli = p.parse_args()
    text = build(cli.sources)
    pathlib.Path(cli.out).write_text(text)
    print(f"wrote {cli.out}: {len(text)} chars from {', '.join(cli.sources)}")


if __name__ == "__main__":
    main()

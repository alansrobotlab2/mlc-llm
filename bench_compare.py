#!/usr/bin/env python3
"""
Apples-to-apples bench: llama.cpp Q4_K_S vs MLC across short/medium/long context.

Drives llama-bench and bench_mlc.py at the same pp lengths, parses both, emits a
markdown table.

  pp_tps : prefill rate when processing N tokens
  tg_tps : decode rate at depth N (i.e., generating after a context of N tokens)

Run:
  source .envrc.local && .venv/bin/python bench_compare.py \\
      --label 0.8B \\
      --gguf-path dist/gguf/Qwen3.5-0.8B-Q4_K_S.gguf \\
      --mlc-dir dist/qwen3_5-0.8B-q4f16_1 \\
      --mlc-label 'MLC q4f16_1' \\
      --ctx 128,1024,4096
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import subprocess
import sys
from pathlib import Path

LLAMA_BENCH = "/home/alfie/llama.cpp/build/bin/llama-bench"
MLC_BENCH = Path(__file__).resolve().parent / "bench_mlc.py"
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="Model label (e.g. '0.8B', '35B-A3B')")
    p.add_argument("--gguf-path", help="Path to llama.cpp gguf for the same model+quant")
    p.add_argument("--mlc-dir", help="Path to MLC-compiled artifact")
    p.add_argument("--mlc-label", default="MLC", help="Label for the MLC row in the table")
    p.add_argument("--ctx", default="128,1024,4096",
                   help="Comma-separated context (prefill) lengths")
    p.add_argument("--tg", type=int, default=128, help="Tokens decoded after each prefill")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", help="Write markdown table to this file (default: stdout only)")
    p.add_argument("--skip-llamacpp", action="store_true")
    p.add_argument("--skip-mlc", action="store_true")
    return p.parse_args()


def run_llamacpp(gguf_path: str, ctx_list: list[int], tg: int, runs: int) -> dict[int, dict[str, float]]:
    """Returns {ctx: {"pp_tps": ..., "tg_tps": ...}}."""
    if not Path(gguf_path).exists():
        print(f"[compare] llama.cpp gguf missing: {gguf_path}", file=sys.stderr)
        return {}

    # -p N for prefill rate at N. -d N -n tg for decode rate at depth N.
    pp_csv = ",".join(str(c) for c in ctx_list)
    depth_csv = ",".join(str(c) for c in ctx_list)
    cmd = [
        LLAMA_BENCH,
        "-m", gguf_path,
        "-p", pp_csv,
        "-n", "0",  # prefill-only for the -p tests
        "-r", str(runs),
        "-o", "csv",
    ]
    print(f"[compare] llama-bench (prefill): {' '.join(cmd)}", file=sys.stderr)
    pp_csv_out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    cmd = [
        LLAMA_BENCH,
        "-m", gguf_path,
        "-p", "0",
        "-n", str(tg),
        "-d", depth_csv,
        "-r", str(runs),
        "-o", "csv",
    ]
    print(f"[compare] llama-bench (decode):  {' '.join(cmd)}", file=sys.stderr)
    tg_csv_out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    results: dict[int, dict[str, float]] = {c: {} for c in ctx_list}

    for csv_blob, key in ((pp_csv_out, "pp_tps"), (tg_csv_out, "tg_tps")):
        reader = csv.DictReader(io.StringIO(csv_blob))
        for row in reader:
            n_prompt = int(row.get("n_prompt", "0") or 0)
            n_gen = int(row.get("n_gen", "0") or 0)
            n_depth = int(row.get("n_depth", "0") or 0)
            avg_ts = float(row["avg_ts"])
            if key == "pp_tps" and n_gen == 0 and n_prompt in results:
                results[n_prompt][key] = avg_ts
            elif key == "tg_tps" and n_prompt == 0 and n_depth in results:
                results[n_depth][key] = avg_ts

    return results


def run_mlc(mlc_dir: str, ctx_list: list[int], tg: int, runs: int, device: str, label: str) -> dict[int, dict[str, float]]:
    if not Path(mlc_dir).exists():
        print(f"[compare] mlc-dir missing: {mlc_dir}", file=sys.stderr)
        return {}

    pp_csv = ",".join(str(c) for c in ctx_list)
    json_out = Path(mlc_dir).name + "_bench.json"
    cmd = [
        PYTHON, "-u", str(MLC_BENCH),
        "--model-dir", mlc_dir,
        "--device", device,
        "--pp", pp_csv,
        "--tg", str(tg),
        "--runs", str(runs),
        "--label", label,
        "--json-out", json_out,
    ]
    print(f"[compare] {' '.join(cmd)}", file=sys.stderr)
    # Stream stdout/stderr live so the engine's background threads never block on a full pipe buffer.
    subprocess.run(cmd, check=True)

    results: dict[int, dict[str, float]] = {c: {} for c in ctx_list}
    if not Path(json_out).exists():
        print(f"[compare] expected json at {json_out} not found", file=sys.stderr)
        return results
    import json
    data = json.loads(Path(json_out).read_text())
    for pp_str, vals in data.items():
        pp = int(pp_str)
        if pp in results:
            results[pp]["pp_tps"] = vals["pp_tps"]
            results[pp]["tg_tps"] = vals["tg_tps"]
    return results


def fmt_table(label: str, ctx_list: list[int], llama_res: dict, mlc_res: dict, mlc_label: str) -> str:
    lines = []
    lines.append(f"### {label} — apples-to-apples (Orin AGX)")
    lines.append("")
    lines.append("| ctx | backend | pp_tps | tg_tps | tg vs llama.cpp |")
    lines.append("|---:|---|---:|---:|---:|")
    for ctx in ctx_list:
        ll = llama_res.get(ctx, {})
        ml = mlc_res.get(ctx, {})
        ll_pp = ll.get("pp_tps")
        ll_tg = ll.get("tg_tps")
        ml_pp = ml.get("pp_tps")
        ml_tg = ml.get("tg_tps")

        def f(x): return f"{x:.2f}" if x is not None else "—"

        if ll_pp or ll_tg:
            lines.append(f"| {ctx} | llama.cpp Q4_K_S | {f(ll_pp)} | {f(ll_tg)} | 1.000× |")
        if ml_pp or ml_tg:
            ratio = (ml_tg / ll_tg) if (ll_tg and ml_tg) else None
            ratio_s = f"{ratio:.3f}×" if ratio else "—"
            lines.append(f"| {ctx} | {mlc_label} | {f(ml_pp)} | {f(ml_tg)} | {ratio_s} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ctx_list = [int(x) for x in args.ctx.split(",") if x.strip()]

    llama_res: dict[int, dict[str, float]] = {}
    mlc_res: dict[int, dict[str, float]] = {}

    if not args.skip_llamacpp and args.gguf_path:
        llama_res = run_llamacpp(args.gguf_path, ctx_list, args.tg, args.runs)
    if not args.skip_mlc and args.mlc_dir:
        mlc_res = run_mlc(args.mlc_dir, ctx_list, args.tg, args.runs, args.device, args.mlc_label)

    table = fmt_table(args.label, ctx_list, llama_res, mlc_res, args.mlc_label)
    print()
    print(table)
    if args.output:
        Path(args.output).write_text(table + "\n")
        print(f"\n[compare] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

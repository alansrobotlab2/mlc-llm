#!/usr/bin/env python3
"""Per-token decode breakdown from an nsys sqlite export.

Answers two questions the raw `nsys stats --report cuda_gpu_kern_sum` output
cannot:

1. **Which kernels are actually at the memory wall?** For the weight-streaming
   GEMV kernels we know the exact parameter count behind each one, so achieved
   bandwidth is computed against real bytes rather than guessed.
2. **How much of the token budget is GPU idle?** Steps are delimited by the
   sampling kernel, so per-step wall time is measured from the trace itself
   instead of being inferred from an end-to-end tps figure.

Prefill and decode are separated by step duration — a trace that contains both
(as any `MLCEngine._generate` run does) would otherwise blend a 4 ms prefill
instance of a kernel into the average with its 11 us decode instances.

Usage:
    nsys profile -t cuda,nvtx --cuda-graph-trace=node -o dec -f true \\
        python scripts/profile_decode_35b.py --model-dir ... --model-lib ...
    nsys export --type sqlite -o dec.sqlite dec.nsys-rep     # or --force-export
    python scripts/analyze_decode_trace.py dec.sqlite --model 35b
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

# Achievable (not spec-sheet) read bandwidth, measured on Orin AGX sm_87 with a
# 128-bit vectorized read kernel over a 2 GiB buffer. Spec peak is 204.8 GB/s;
# 156.0 is what a kernel can actually reach. See scripts/bw_probe.cu.
ACHIEVABLE_GBPS = 156.0

# ---------------------------------------------------------------------------
# Kernel -> weight identification.
#
# Every entry was pinned down from launch geometry in the trace, not guessed:
# the dlight GEMV schedule emits block=(16,32,1) and one CTA per 64 output
# elements, so gridX * 64 == output width. Instances-per-token gives the layer
# multiplicity (40 = every layer, 30 = GDN layers, 10 = full-attention layers,
# 1 = once per token). Both together identify the projection uniquely.
#
# `params` is the source-weight element count for ONE invocation. Bytes are
# params * bits/8, where bits includes the group scales (4.345 for q4f16_1 as
# reported by convert_weight).
# ---------------------------------------------------------------------------
QWEN3_6_35B_A3B = {
    # grid, per-token, output width -> projection
    "fused_dequantize_fused_NT_matmul9_cast4_kernel": (
        2048 * 248320, "lm_head (2048->248320)"),
    "moe_dequantize_gemv_kernel": (
        8 * 2048 * 1024, "routed experts gate_up, top-8 (2048->2*512)"),
    "moe_dequantize_gemv1_kernel": (
        8 * 512 * 2048, "routed experts down, top-8 (512->2048)"),
    "fused_dequantize1_NT_matmul_kernel": (
        2048 * 8192, "GDN in_proj_qkv (2048->8192)"),
    "fused_dequantize2_fused_NT_matmul1_silu1_multiply1_kernel": (
        2048 * 4096, "GDN in_proj_z + silu*gate (2048->4096)"),
    "fused_dequantize4_NT_matmul3_kernel": (
        4096 * 2048, "GDN out_proj AND attn o_proj (4096->2048, shared kernel)"),
    "fused_dequantize7_NT_matmul8_kernel": (
        2048 * 9216, "attn c_attn (2048->9216)"),
    "fused_dequantize5_NT_matmul5_kernel": (
        2048 * 1024, "shared expert gate_up (2048->2*512)"),
    "fused_dequantize6_fused_NT_matmul6_multiply5_add1_kernel": (
        512 * 2048, "shared expert down + gate*add (512->2048)"),
    "fused_dequantize3_NT_matmul2_kernel": (
        2048 * 32, "GDN in_proj_a (2048->32)"),
    "fused_dequantize3_fused_NT_matmul2_tir_sigmoid_cast2_kernel": (
        2048 * 32, "GDN in_proj_b + sigmoid (2048->32)"),
    # fp16, not quantized
    "NT_matmul4_kernel": (2048 * 256, "MoE router gate (2048->256, fp16)"),
    # Recurrent state traffic. Not weights: a get reads one 2 MiB state slot and
    # writes a 2 MiB destination, so 2x the slot size moves per call.
    "rnn_state_get_0_kernel": (2 * 32 * 128 * 128, "GDN recurrent state get (fp32 2 MiB slot)"),
    "rnn_state_set_0_kernel": (2 * 32 * 128 * 128, "GDN recurrent state set (fp32 2 MiB slot)"),
    "rnn_state_get_1_kernel": (2 * 3 * 8192, "GDN conv state get (3x8192)"),
    "rnn_state_set_1_kernel": (2 * 3 * 8192, "GDN conv state set (3x8192)"),
}

# element size in bytes for the byte estimate
BITS = {
    "NT_matmul4_kernel": 16.0,
    "rnn_state_get_0_kernel": 32.0,
    "rnn_state_set_0_kernel": 32.0,
    "rnn_state_get_1_kernel": 16.0,
    "rnn_state_set_1_kernel": 16.0,
}
DEFAULT_BITS = 4.345  # q4f16_1 incl. group scales

MODELS = {"35b": QWEN3_6_35B_A3B}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", help="nsys sqlite export")
    ap.add_argument("--model", default="35b", choices=sorted(MODELS),
                    help="which kernel->weight table to use for the BW column")
    ap.add_argument("--anchor", default="parallel_sampling_from_prob_kernel",
                    help="kernel that fires exactly once per engine step; used to "
                         "cut the trace into steps")
    ap.add_argument("--decode-max-ms", type=float, default=60.0,
                    help="steps longer than this are prefill, not decode")
    ap.add_argument("--bits", type=float, default=DEFAULT_BITS,
                    help="bits per weight element incl. scales")
    ap.add_argument("--top", type=int, default=25, help="rows to print in full")
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    names = dict(cur.execute("SELECT id, value FROM StringIds").fetchall())
    rname = {v: k for k, v in names.items()}

    if args.anchor not in rname:
        sys.exit(f"anchor kernel {args.anchor!r} not in trace; pass --anchor")

    kernels = cur.execute(
        "SELECT shortName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    if not kernels:
        sys.exit("no kernel rows — export the .nsys-rep with `nsys export --type sqlite`")

    # ---- cut into steps on the anchor kernel -------------------------------
    anchor_id = rname[args.anchor]
    bounds = [end for sn, _s, end in kernels if sn == anchor_id]
    if len(bounds) < 3:
        sys.exit(f"only {len(bounds)} anchor firings; need >=3 to define steps")

    steps = list(zip(bounds[:-1], bounds[1:]))          # (step_start, step_end)
    durs = [(e - s) / 1e6 for s, e in steps]
    decode = [(s, e) for (s, e), d in zip(steps, durs) if d <= args.decode_max_ms]
    prefill_n = len(steps) - len(decode)
    if not decode:
        sys.exit("no steps under --decode-max-ms; raise the threshold")

    n_steps = len(decode)
    wall_ms = sum((e - s) for s, e in decode) / 1e6 / n_steps

    print(f"trace: {len(kernels)} kernel launches, {len(steps)} steps "
          f"({prefill_n} prefill / {n_steps} decode)")
    print(f"measured decode wall: {wall_ms:.3f} ms/token  "
          f"({1e3 / wall_ms:.2f} tps)\n")

    # ---- attribute kernels to decode steps --------------------------------
    # steps are contiguous and sorted, so walk both lists once
    agg: dict[int, list[float]] = {}
    busy_ns = 0.0
    di = 0
    for sn, s, e in kernels:
        while di < n_steps and s >= decode[di][1]:
            di += 1
        if di >= n_steps:
            break
        lo, hi = decode[di]
        if s < lo:
            continue
        agg.setdefault(sn, []).append(e - s)
        busy_ns += e - s

    rows = []
    table = MODELS[args.model]
    for sn, ds in agg.items():
        name = names.get(sn, str(sn))
        n = len(ds)
        tot_ms = sum(ds) / 1e6 / n_steps
        per_tok = n / n_steps
        gbps = pct = None
        if name in table:
            params, _desc = table[name]
            bits = BITS.get(name, args.bits)
            byts = params * bits / 8 * per_tok
            gbps = byts / (tot_ms / 1e3) / 1e9
            pct = gbps / ACHIEVABLE_GBPS * 100
        rows.append((name, n, per_tok, sum(ds) / n / 1e3, tot_ms, gbps, pct))

    rows.sort(key=lambda r: -r[4])

    hdr = (f"{'kernel':56s} {'/tok':>5s} {'avg us':>7s} {'ms/tok':>7s} "
           f"{'%budget':>7s} {'GB/s':>7s} {'%wall':>6s}")
    print(hdr)
    print("-" * len(hdr))
    shown = 0
    tail_ms = tail_n = 0
    for name, _n, per_tok, avg_us, ms, gbps, pct in rows:
        if shown < args.top:
            g = f"{gbps:7.1f}" if gbps else f"{'-':>7s}"
            p = f"{pct:5.0f}%" if pct else f"{'-':>6s}"
            print(f"{name[:56]:56s} {per_tok:5.1f} {avg_us:7.2f} {ms:7.3f} "
                  f"{ms / wall_ms * 100:6.1f}% {g} {p}")
            shown += 1
        else:
            tail_ms += ms
            tail_n += 1
    if tail_n:
        print(f"{f'+ {tail_n} smaller kernels':56s} {'':5s} {'':7s} {tail_ms:7.3f} "
              f"{tail_ms / wall_ms * 100:6.1f}%")

    busy = busy_ns / 1e6 / n_steps
    print("-" * len(hdr))
    print(f"{'sum of kernel time':56s} {'':5s} {'':7s} {busy:7.3f} "
          f"{busy / wall_ms * 100:6.1f}%")
    print(f"{'GPU idle / launch gap':56s} {'':5s} {'':7s} {wall_ms - busy:7.3f} "
          f"{(wall_ms - busy) / wall_ms * 100:6.1f}%")

    # ---- roll-ups ---------------------------------------------------------
    def group(pred) -> float:
        return sum(r[4] for r in rows if pred(r[0]))

    print()
    state = group(lambda n: n.startswith("rnn_state_"))
    tiny = group(lambda n: n in (
        "fused_dequantize3_NT_matmul2_kernel",
        "fused_dequantize3_fused_NT_matmul2_tir_sigmoid_cast2_kernel"))
    print(f"rnn_state get/set (all 4)      {state:7.3f} ms/tok "
          f"({state / wall_ms * 100:.1f}%)")
    print(f"GDN in_proj_a + in_proj_b      {tiny:7.3f} ms/tok "
          f"({tiny / wall_ms * 100:.1f}%)  <- 142 KB of weights")

    known = [r for r in rows if r[5] is not None and not r[0].startswith("rnn_state")]
    if known:
        w_ms = sum(r[4] for r in known)
        print(f"identified weight GEMVs        {w_ms:7.3f} ms/tok "
              f"({w_ms / wall_ms * 100:.1f}%)")


if __name__ == "__main__":
    main()

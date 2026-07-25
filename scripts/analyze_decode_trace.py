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
# GEMV-schedule weight kernels, as `(K, N, label)` for a K->N projection.
#
# N is CHECKED against launch geometry rather than trusted. The dlight GEMV schedule
# emits `block=(16,32,1)` with one CTA per 64 output columns, so `gridX * 64` is the
# real output width and `gridY` is the expert multiplicity (1 for dense kernels, 8 for
# the top-8 MoE gemvs). The *observed* width is what the byte count uses; a table N that
# disagrees prints a `!` and a warning.
#
# That check exists because the failure it catches already happened. These names are
# assigned by the Relax fusion pass in emission order, so *fusing two projections
# renumbers unrelated kernels and can reuse a name for a different weight*. After the
# in_proj_qkvzab merge, `fused_dequantize1_NT_matmul_kernel` went on meaning "the GDN
# input projection" but its width went 8192 -> 12352, and the stale entry reported
# 94.2 GB/s (60% of wall) for a kernel actually running at 142 GB/s (91%) — the
# difference between "biggest remaining headroom" and "done". Never key these on name
# alone.
QWEN3_6_35B_A3B = {
    "fused_dequantize_fused_NT_matmul7_cast4_kernel": (
        2048, 248320, "lm_head"),
    "moe_dequantize_gemv_kernel": (
        2048, 1024, "routed experts gate_up, top-8"),
    "moe_dequantize_gemv1_kernel": (
        512, 2048, "routed experts down, top-8"),
    "fused_dequantize1_NT_matmul_kernel": (
        2048, 12352, "GDN in_proj_qkvzab (fused qkv|z|a|b)"),
    "fused_dequantize2_NT_matmul1_kernel": (
        4096, 2048, "GDN out_proj AND attn o_proj (shared kernel)"),
    "fused_dequantize5_NT_matmul6_kernel": (
        2048, 9216, "attn c_attn"),
    "fused_dequantize3_NT_matmul3_kernel": (
        2048, 1024, "shared expert gate_up"),
    "fused_dequantize4_fused_NT_matmul4_multiply5_add1_kernel": (
        512, 2048, "shared expert down + gate*add"),
    # fp16, not quantized — see BITS
    "NT_matmul2_kernel": (2048, 256, "MoE router gate (fp16)"),
}

# Non-GEMV traffic, as `(bytes_per_call, label)`. No geometry check applies — these do
# not use the GEMV schedule, so gridX*64 means nothing for them.
STATE_TRAFFIC_35B = {
    # A get reads one state slot and writes a same-sized destination, so 2x the slot.
    "rnn_state_get_1_kernel": (2 * 3 * 8192 * 2, "GDN conv state get (3x8192 fp16)"),
    "rnn_state_set_1_kernel": (2 * 3 * 8192 * 2, "GDN conv state set (3x8192 fp16)"),
    # The in-place fused kernel reads its 2 MiB slot and writes the next one. Its time
    # also covers the recurrence itself, so this is a floor on traffic, not a pure
    # bandwidth measurement — read the %wall column accordingly.
    "gdn_func_inplace_kernel": (
        2 * 32 * 128 * 128 * 4, "GDN recurrence, in-place state (reads slot h, writes h+1)"),
    # Pre-in-place builds only. Kept so old traces still analyze.
    "rnn_state_get_0_kernel": (2 * 32 * 128 * 128 * 4, "GDN recurrent state get (fp32 2 MiB slot)"),
    "rnn_state_set_0_kernel": (2 * 32 * 128 * 128 * 4, "GDN recurrent state set (fp32 2 MiB slot)"),
}

# element size in bits for the weight byte estimate
BITS = {
    "NT_matmul2_kernel": 16.0,
}
DEFAULT_BITS = 4.345  # q4f16_1 incl. group scales

MODELS = {"35b": (QWEN3_6_35B_A3B, STATE_TRAFFIC_35B)}


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

    # Launch geometry per kernel name, for the N check on GEMV-schedule kernels.
    # Only the (16,32,1) dlight GEMV schedule has the "one CTA per 64 output columns"
    # property; anything else is left alone.
    geom: dict[int, tuple[int, int]] = {}
    for sn, gx, gy, bx, by in cur.execute(
        "SELECT shortName, gridX, gridY, blockX, blockY FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "GROUP BY shortName, gridX, gridY, blockX, blockY "
        "ORDER BY COUNT(*) DESC"
    ).fetchall():
        if sn not in geom and (bx, by) == (16, 32):
            geom[sn] = (gx * 64, gy)
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
    gemv_table, state_table = MODELS[args.model]
    warnings: list[str] = []
    for sn, ds in agg.items():
        name = names.get(sn, str(sn))
        n = len(ds)
        tot_ms = sum(ds) / 1e6 / n_steps
        per_tok = n / n_steps
        gbps = pct = None
        flag = " "
        if name in gemv_table:
            k, n_table, _desc = gemv_table[name]
            n_real, experts = geom.get(sn, (n_table, 1))
            if n_real != n_table:
                flag = "!"
                warnings.append(
                    f"{name}: table says N={n_table}, launch geometry says N={n_real} "
                    f"(gridX*64). Using {n_real}. The kernel this name refers to has "
                    f"changed — update the table."
                )
            bits = BITS.get(name, args.bits)
            byts = k * n_real * experts * bits / 8 * per_tok
            gbps = byts / (tot_ms / 1e3) / 1e9
            pct = gbps / ACHIEVABLE_GBPS * 100
        elif name in state_table:
            byts_per_call, _desc = state_table[name]
            byts = byts_per_call * per_tok
            gbps = byts / (tot_ms / 1e3) / 1e9
            pct = gbps / ACHIEVABLE_GBPS * 100
        rows.append((name + flag, n, per_tok, sum(ds) / n / 1e3, tot_ms, gbps, pct))

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
        return sum(r[4] for r in rows if pred(r[0].rstrip("! ")))

    print()
    state = group(lambda n: n.startswith("rnn_state_"))
    # Everything the conv-state fusion would absorb: the two copies plus the separate
    # shift kernel that materializes the new state before `set` writes it back.
    conv = group(lambda n: n in (
        "rnn_state_get_1_kernel", "rnn_state_set_1_kernel", "update_conv_state1_kernel"))
    gemvs = group(lambda n: n in gemv_table)
    print(f"rnn_state copies still in decode  {state:7.3f} ms/tok "
          f"({state / wall_ms * 100:.1f}%)")
    print(f"conv-state fusion candidate       {conv:7.3f} ms/tok "
          f"({conv / wall_ms * 100:.1f}%)  <- get_1 + set_1 + update_conv_state1")
    print(f"identified weight GEMVs           {gemvs:7.3f} ms/tok "
          f"({gemvs / wall_ms * 100:.1f}%)")

    if warnings:
        print("\n!! kernel table is stale — bandwidth numbers below the flagged rows "
              "would have been wrong:")
        for w in warnings:
            print(f"   {w}")



if __name__ == "__main__":
    main()

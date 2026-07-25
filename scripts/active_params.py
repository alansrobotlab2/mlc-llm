"""Exact per-token ACTIVE parameter count for Qwen3.6-35B-A3B text-only, target-only decode.

Reads safetensors headers only. Excludes the vision tower and the MTP draft head
(neither runs in target-only text decode) and the embedding table (one row per token,
not streamed). Routed experts counted at top-k/num_experts.
"""
import argparse
import glob
import json
import re
import struct
from collections import defaultdict

DEFAULT_SNAP = (
    "/home/alfie/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
    "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--snapshot", default=DEFAULT_SNAP, help="HF snapshot dir")
_ap.add_argument("--bits", type=float, default=4.345,
                 help="bits/param incl. group scales (convert_weight reports this)")
_ap.add_argument("--tps", type=float, default=54.13,
                 help="measured decode tps to convert into achieved GB/s")
_args = _ap.parse_args()
SNAP = _args.snapshot

cfg = json.load(open(f"{SNAP}/config.json"))
tc = cfg.get("text_config", cfg)
TOPK, NEXP = tc["num_experts_per_tok"], tc["num_experts"]

shapes = {}
for f in sorted(glob.glob(f"{SNAP}/*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k != "__metadata__":
            shapes[k] = v["shape"]


def numel(s):
    r = 1
    for d in s:
        r *= d
    return r


b = defaultdict(int)
for name, shp in shapes.items():
    n = numel(shp)
    if name.startswith("mtp."):
        b["EXCLUDED mtp draft head"] += n
    elif name.startswith("model.visual."):
        b["EXCLUDED vision tower"] += n
    elif "embed_tokens" in name:
        b["EXCLUDED embed table"] += n
    elif name.startswith("lm_head"):
        b["lm_head"] += n
    elif re.search(r"\.experts\.", name) and "shared" not in name:
        b["moe routed (full)"] += n
    elif "shared_expert" in name:
        b["moe shared expert"] += n
    elif "linear_attn" in name:
        b["gdn linear-attn layers"] += n
    elif "self_attn" in name:
        b["full-attention layers"] += n
    elif re.search(r"mlp\.gate\.weight$", name):
        b["moe router"] += n
    else:
        b["norms / misc"] += n

ACTIVE_KEYS = [k for k in b if not k.startswith("EXCLUDED")]
active = 0.0
print(f"top-k / experts = {TOPK}/{NEXP}\n")
print(f"{'bucket':28s} {'full params':>16s} {'active/token':>16s}")
for k in sorted(b):
    full = b[k]
    if k.startswith("EXCLUDED"):
        act = 0.0
    elif k == "moe routed (full)":
        act = full * TOPK / NEXP
    else:
        act = float(full)
    active += act
    print(f"{k:28s} {full:16,d} {act:16,.0f}")

print(f"\n{'TOTAL in checkpoint':28s} {sum(b.values()):16,d}")
print(f"{'ACTIVE / token':28s} {'':16s} {active:16,.0f}")

MEASURED_TPS = _args.tps
PEAK = 204.8          # spec sheet: 256-bit LPDDR5 @ 3200 MT/s
ACHIEVABLE = 156.0    # measured by scripts/bw_probe.cu on this box

gb = active * _args.bits / 8 / 1e9
eff = gb * MEASURED_TPS
print(f"\n@ {_args.bits} bits/param -> {gb:.3f} GB/token")
print(f"  roofline @ {PEAK} GB/s spec peak    : {PEAK/gb:6.1f} tps")
print(f"  roofline @ {ACHIEVABLE} GB/s achievable : {ACHIEVABLE/gb:6.1f} tps")
print(f"  measured {MEASURED_TPS} tps           : {eff:6.1f} GB/s "
      f"= {eff/PEAK*100:.1f}% of spec, {eff/ACHIEVABLE*100:.1f}% of achievable")
print("\nNote: this is the WEIGHT roofline only. It omits the ~0.25 GB/token of "
      "\nrecurrent-state copy traffic that rnn_state_get/set actually move "
      "\n(see scripts/analyze_decode_trace.py), so the real byte floor is higher.")

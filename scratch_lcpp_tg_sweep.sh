#!/usr/bin/env bash
# llama.cpp TG sweep — parity with scratch_mlc_tg_sweep.py
# Each test is pp=512 prefill + tg ∈ {512,1024,2048,4096,8192} decode (-pg form)
# Run with the box locked: sudo nvpmodel -m 0 && sudo jetson_clocks
set -euo pipefail

MODEL="${MODEL:-/home/alfie/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
LB="${LB:-$HOME/llama.cpp/build/bin/llama-bench}"
REPS="${REPS:-3}"
FA="${FA:-1}"
OUT="${OUT:-tuning/lcpp_tg_sweep_$(date +%Y%m%d_%H%M%S).md}"

mkdir -p "$(dirname "$OUT")"

if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

echo "model:  $MODEL"
echo "reps:   $REPS"
echo "fa:     $FA"
echo "out:    $OUT"
echo

# -pg pp,tg: prefill pp tokens, then decode tg, reports pp/tg tps separately.
# Each -pg pair becomes its own row in the table.
"$LB" \
  -m "$MODEL" \
  -pg 512,512 \
  -pg 512,1024 \
  -pg 512,2048 \
  -pg 512,4096 \
  -pg 512,8192 \
  -fa "$FA" \
  -r "$REPS" \
  -o md \
  | tee "$OUT"

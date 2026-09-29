#!/usr/bin/env bash
# llama-bench prefill at KV depth for one or more engine builds, in one GPU-lock hold (same GPU state for all builds).
#   bench/pp_depth.sh OUT_FILE BIN_DIR... (env: DEPTHS=0,65536,131072,196608 P=2048 REPS=2 KV=q5_0 UB=1024)
# Production KV/batch flags: -fa 1 -ctk/-ctv $KV -ub $UB -b 2048. Appends one markdown table per build to OUT_FILE.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1"); shift
mkdir -p "$(dirname "$OUT")"
BINS=(); for b in "$@"; do BINS+=("$(realpath "$b")"); done
exec "$HERE/../deploy/gpu-lock.sh" bash -c '
set -uo pipefail
OUT=$1; shift
MODEL=${MODEL:-/models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf}
for bin in "$@"; do
  echo "== $bin ($(date -Is))" >> "$OUT"
  docker run --rm --gpus all -v "$bin:/e:ro" -v "$HOME/models:/models:ro" lovelace-engine-runtime:cuda13.1 \
    /e/llama-bench -m $MODEL -ngl 99 -fa 1 -ub ${UB:-1024} -b 2048 -ctk ${KV:-q5_0} -ctv ${KV:-q5_0} \
      -p ${P:-2048} -n 0 -d ${DEPTHS:-0,65536,131072,196608} -r ${REPS:-2} >> "$OUT" 2>&1
done
' _ "$OUT" "${BINS[@]}"

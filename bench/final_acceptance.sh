#!/usr/bin/env bash
# Final acceptance on one engine build at the production context: single-stream depth sweep (B1), 6 x 60k replay
# (B2/C2), 6 x 150k replay with warm-up (C3), 3 agents past 200k with swapping (C1). Each step is skipped when its
# output exists, so the script can be re-run after a failure. All GPU work goes through bench/h2h.sh (gpu-lock).
#   bench/final_acceptance.sh OUT_DIR   (env: BIN engine bin dir, CTX unified KV cells, default 327680)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "${1:?OUT_DIR}")
export BIN CTX SERVER_ARGS="--spec-draft-vocab /draft-vocab/qwen3.8-27b.txt"
mkdir -p "$OUT"
run() { # LABEL LANES MODE [WARMUP]
    local label=$1 lanes=$2 mode=$3 warm=${4:-0}
    if [[ -s $OUT/$label.sweep.out || -s $OUT/$label/requests.jsonl ]]; then echo "skip $label"; return; fi
    echo "== $label ($(date +%T))"
    WARMUP=$warm "$HERE/h2h.sh" lovelace "$lanes" "$mode" "$OUT" "$label" || echo "FAILED $label"
}
run sweep 1 sweep
run 6a-60k 6 replay:6:60000:1200
run 6a-150k 6 replay:6:150000:1200 1
run 3a-200k 3 replay:3:200000:1200 1
tail -7 "$OUT/sweep.sweep.out" || true
uv run -q --directory "$HERE/replay" python "$HERE/bakeoff/analyze.py" "$OUT" || true
for d in "$OUT"/*a-*/; do uv run -q --directory "$HERE/replay" python "$HERE/sched_report.py" "$d" || true; done

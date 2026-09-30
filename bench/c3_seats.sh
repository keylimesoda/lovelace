#!/usr/bin/env bash
# C3 (6 agents at depth) measurement runner: one replay through bench/h2h.sh (gpu-lock, one server lifecycle), with
# the RAM sampler/guard of bench/final_acceptance.sh and a /telemetry sample every 5 s (loop phase seconds, session
# tiers) for bench/seat_timeline.py. Skips a run whose requests.jsonl exists.
#   bench/c3_seats.sh OUT_DIR LABEL MODE [WARMUP]   e.g. bench/c3_seats.sh bench/results/c3-seats base-150k replay:6:150000:1200 1
#   env: BIN engine bin dir, CTX (default 376832), SERVER_ARGS appended to the draft-vocab flag
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "${1:?OUT_DIR}") LABEL=${2:?LABEL} MODE=${3:?MODE} WARM=${4:-0}
export BIN CTX=${CTX:-376832} SERVER_ARGS="--spec-draft-vocab /draft-vocab/qwen3.8-27b.txt ${SERVER_ARGS:-}"
mkdir -p "$OUT"
if [[ -s $OUT/$LABEL/requests.jsonl ]]; then echo "skip $LABEL"; exit 0; fi
( while :; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    used=$(docker stats --no-stream --format '{{.MemUsage}}' h2h 2>/dev/null || true)
    echo "$(date +%T) avail_gib=$avail load=$(cut -d" " -f1 /proc/loadavg) h2h=${used:-none}" >> "$OUT/$LABEL.mem.log"
    if (( avail < 4 )); then echo "$(date +%T) GUARD: stopping h2h" >> "$OUT/$LABEL.mem.log"; docker rm -f h2h >/dev/null 2>&1; fi
    sleep 10
  done ) &
SAMPLER=$!
( while :; do
    # only the server of this run (an earlier run may still hold the GPU lock and the port)
    if docker inspect -f '{{range .Mounts}}{{.Source}} {{end}}' h2h 2>/dev/null | grep -q "$(realpath "$BIN") " &&
       t=$(curl -sf --max-time 2 127.0.0.1:18300/telemetry); then
      printf '{"t_unix":%s,"telemetry":%s}\n' "$(date +%s.%N)" "$t" >> "$OUT/$LABEL.telemetry.jsonl"
    fi
    sleep 5
  done ) &
TSAMPLER=$!
trap 'kill $SAMPLER $TSAMPLER 2>/dev/null' EXIT
echo "== $LABEL ($(date +%T)) BIN=$BIN"
WARMUP=$WARM "$HERE/h2h.sh" lovelace 6 "$MODE" "$OUT" "$LABEL" || echo "FAILED $LABEL"

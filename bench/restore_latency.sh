#!/usr/bin/env bash
# Session-switch latency, deterministic: N agents take greedy turns one request at a time on a KV pool too small to
# hold them all, so every turn after the first contact parks one session and restores another. One GPU-lock hold.
#   bench/restore_latency.sh BIN_DIR OUT_DIR LABEL [CTX=131072] [AGENTS=3] [MIN_DEPTH=60000] [TURNS=6] [server args...]
# Writes OUT_DIR/LABEL/{outputs.jsonl,requests.server.jsonl,server.log,telemetry-end.json,TABLE.md}.
# env BUSY=K: K more agents keep decoding short-context requests meanwhile, so that the switches happen under load
# (outputs are then not deterministic). Greedy identity check (BUSY=0): run twice, a pool that parks (CTX=131072) and
# one that holds all (CTX=327680), then (cd bench/replay && uv run python restore_latency.py --compare OUT/A OUT/B)
set -euo pipefail
BIN=$(realpath "$1"); OUT=$(realpath -m "$2"); LABEL=$3; CTX=${4:-131072}; AGENTS=${5:-3}; MIND=${6:-60000}; TURNS=${7:-6}
BUSY=${BUSY:-0}
shift $(( $# < 7 ? $# : 7 ))
EXTRA=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/$LABEL"
exec "$HERE/../deploy/gpu-lock.sh" bash -c '
  set -uo pipefail
  BIN=$1 OUT=$2 LABEL=$3 CTX=$4 AGENTS=$5 MIND=$6 TURNS=$7 HERE=$8 BUSY=$9; shift 9
  PORT=18400
  docker rm -f restore-lat >/dev/null 2>&1
  docker run -d --name restore-lat --gpus all --network host --ulimit memlock=-1:-1 --memory ${ENGINE_MEM:-34g} --memory-swap ${ENGINE_MEM:-34g} \
    -v "$BIN:/e" -v "$HOME/models:/models:ro" \
    -v "$OUT/$LABEL:/out" lovelace-engine-runtime:cuda13.1 /e/llama-server \
    -m /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf -ngl 99 -np $((AGENTS + BUSY)) --kv-unified -c "$CTX" -ctk q5_0 -ctv q5_0 \
    --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q5_0 -ctvd q5_0 -fa on -ub 1024 -b 2048 --jinja --reasoning-preserve \
    --session-ram ${SESSION_RAM:-24576} --session-dir /tmp/sessions -ctxcp 8 --kv-budget --prefill-chunk-gen 1024 \
    --alias qwen3.8-27b --telemetry --request-log-jsonl /out/requests.server.jsonl --host 127.0.0.1 --port $PORT "$@" >/dev/null
  for i in $(seq 180); do curl -sf 127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
  (cd "$HERE/replay" && uv run python restore_latency.py --base-url http://127.0.0.1:$PORT/v1 --agents "$AGENTS" \
     --min-depth "$MIND" --turns "$TURNS" --busy "$BUSY" --out "$OUT/$LABEL") > "$OUT/$LABEL/driver.out" 2>&1
  curl -s 127.0.0.1:$PORT/telemetry > "$OUT/$LABEL/telemetry-end.json"
  docker logs restore-lat > "$OUT/$LABEL/server.log" 2>&1
  docker rm -f restore-lat >/dev/null
  (cd "$HERE/replay" && uv run python restore_latency.py --table "$OUT/$LABEL/requests.server.jsonl") > "$OUT/$LABEL/TABLE.md"
  cat "$OUT/$LABEL/TABLE.md"
' _ "$BIN" "$OUT" "$LABEL" "$CTX" "$AGENTS" "$MIND" "$TURNS" "$HERE" "$BUSY" "${EXTRA[@]}"

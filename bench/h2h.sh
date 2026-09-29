#!/usr/bin/env bash
# Head-to-head runner: one GPU-lock hold per run; the server is started, measured and removed inside the hold.
#   bench/h2h.sh ENGINE LANES MODE OUT_DIR LABEL
#     ENGINE: lovelace | ninfer     LANES: slots/lanes (1..6)
#     MODE:   sweep                 single-stream depth sweep (bench/depth_sweep.py)
#   env BIN: lovelace build dir (default <repo>/engine/build-sm89/bin)
#             replay:AGENTS:MINDEPTH:DURATION   multi-agent replay (bench/replay)
#   env SERVER_ARGS: extra lovelace server args (deploy/draft-vocab is mounted at /draft-vocab)
#   env: WARMUP=1 runs every agent's first contact before the timed replay window (replay.py --warmup)
#   env CTX: lovelace unified KV cells (default 327680)
set -euo pipefail
ENGINE=$1 LANES=$2 MODE=$3 OUT=$(realpath -m "$4") LABEL=$5
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$(realpath "${BIN:-$HERE/../engine/build-sm89/bin}")
mkdir -p "$OUT"
exec "$HERE/../deploy/gpu-lock.sh" bash -c '
set -uo pipefail
ENGINE=$1 LANES=$2 MODE=$3 OUT=$4 LABEL=$5 HERE=$6 BIN=$7
PORT=18300
start_lovelace() {
  mkdir -p /tmp/h2h-sessions
  docker run -d --name h2h --gpus all --network host --ulimit memlock=-1:-1 -v "$BIN:/e" -v "$HOME/models:/models:ro" \
    -v /tmp/h2h-sessions:/sessions -v "$HERE/../deploy/draft-vocab:/draft-vocab:ro" lovelace-engine-runtime:cuda13.1 /e/llama-server \
    -m /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf -ngl 99 -np "$LANES" --kv-unified -c "${CTX:-327680}" -ctk q5_0 -ctv q5_0 \
    --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q5_0 -ctvd q5_0 -fa on -ub 1024 -b 2048 --jinja --reasoning-preserve \
    --session-ram 30720 --session-dir /sessions -ctxcp 8 --kv-budget --prefill-chunk-gen 1024 \
    --alias qwen3.8-27b --telemetry --metrics --host 127.0.0.1 --port $PORT ${SERVER_ARGS:-} >/dev/null
}
start_ninfer() { # $1 = max context
  docker run -d --name h2h --gpus all --network host -v "$HOME/Work/ninfer-artifacts:/models:ro" ninfer-4090:hybrid \
    ninfer-serve /models/qwen3_8_27b.ninfer --model-id qwen3.8-27b --host 127.0.0.1 --port $PORT \
    --max-context "$1" --kv-capacity auto --max-concurrency "$LANES" --max-pending-requests 64 --pending-timeout-ms 3600000 \
    --prefill-chunk 1024 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking \
    --host-kv-mib 30720 --host-state-slots 24 --auto-long-anchors 2 --max-long-anchors-per-continuation 2 >/dev/null
}
up() { for i in $(seq 180); do [ "$(docker inspect -f {{.State.Running}} h2h 2>/dev/null)" = true ] || return 1
         curl -sf 127.0.0.1:$PORT/health >/dev/null && return 0; sleep 2; done; return 1; }
docker rm -f h2h >/dev/null 2>&1
if [ "$ENGINE" = lovelace ]; then start_lovelace; up || { docker logs h2h 2>&1 | tail -5; docker rm -f h2h; exit 1; }
else
  ok=0
  for C in 262144 229376 196608 163840 131072 98304; do
    start_ninfer $C
    if up; then ok=1; echo "ninfer lanes=$LANES max-context=$C" > "$OUT/$LABEL.config"; break; fi
    docker logs h2h 2>&1 | grep -iE "fatal|error|must" | head -2; docker rm -f h2h >/dev/null 2>&1
  done
  [ $ok = 1 ] || exit 1
fi
docker logs h2h 2>&1 | grep -iE "capacity|kv|reservation" | head -8 > "$OUT/$LABEL.startup.txt"
cd "$HERE/replay"
case "$MODE" in
  sweep) uv run python ../depth_sweep.py --base-url http://127.0.0.1:$PORT/v1 --label "$LABEL" --out "$OUT" > "$OUT/$LABEL.sweep.out" 2>&1 ;;
  replay:*) IFS=: read -r _ A MD DUR <<< "$MODE"
    WFLAG=(); [ "${WARMUP:-0}" = 1 ] && WFLAG=(--warmup)
    uv run replay.py --base-url http://127.0.0.1:$PORT/v1 --model qwen3.8-27b "${WFLAG[@]}" --agents "$A" --duration "$DUR" --no-drain \
      --start-turn-min-depth "$MD" --think-scale 0.25 --max-think 20 --output-tokens recorded --temperature 1.0 \
      --extra-body "{\"top_p\":0.95,\"top_k\":20}" --results-dir "$OUT" --label "$LABEL" > "$OUT/$LABEL.replay.out" 2>&1
    echo 180 > "$OUT/$LABEL/warmup_s" ;;
esac
docker logs h2h > "$OUT/$LABEL.server.log" 2>&1
docker rm -f h2h >/dev/null
' _ "$ENGINE" "$LANES" "$MODE" "$OUT" "$LABEL" "$HERE" "$BIN"

#!/usr/bin/env bash
# One GPU-lock hold: start the production-profile server from a build dir, replay N agents, stop it.
# Usage: bench/run_replay.sh BIN_DIR OUT_DIR LABEL [CTX] [AGENTS] [DURATION] [MIN_DEPTH] [extra server args...]
set -euo pipefail
BIN=$(realpath "$1"); OUT=$(realpath -m "$2"); LABEL=$3; CTX=${4:-327680}; AGENTS=${5:-6}; DUR=${6:-1200}; MIND=${7:-60000}
shift $(( $# < 7 ? $# : 7 ))
EXTRA=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
exec "$HERE/../deploy/gpu-lock.sh" bash -c '
  set -uo pipefail
  BIN=$1 OUT=$2 LABEL=$3 CTX=$4 AGENTS=$5 DUR=$6 MIND=$7 HERE=$8; shift 8
  docker rm -f replay-srv >/dev/null 2>&1
  docker run -d --name replay-srv --gpus all --network host --ulimit memlock=-1:-1 -v "$BIN:/e" -v "$HOME/models:/models:ro" \
    lovelace-engine-runtime:cuda13.1 /e/llama-server -m /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf -ngl 99 -fa on \
    --jinja -ub 1024 -b 2048 -np 6 --kv-unified -c "$CTX" -ctk q5_0 -ctv q5_0 --spec-type draft-mtp --spec-draft-n-max 3 \
    -ctkd q5_0 -ctvd q5_0 --cache-ram 32768 -ctxcp 8 --reasoning-preserve --alias qwen3.8-27b --telemetry --metrics \
    --host 127.0.0.1 --port 18000 "$@" >/dev/null
  for i in $(seq 150); do curl -sf 127.0.0.1:18000/health >/dev/null && break; sleep 2; done
  (cd "$HERE/replay" && /tmp/uvtool/bin/uv run replay.py --base-url http://127.0.0.1:18000/v1 --model qwen3.8-27b \
     --agents "$AGENTS" --duration "$DUR" --no-drain --start-turn-min-depth "$MIND" --think-scale 0.25 --max-think 20 \
     --output-tokens recorded --temperature 1.0 --extra-body "{\"top_p\":0.95,\"top_k\":20}" \
     --results-dir "$OUT" --label "$LABEL") > "$OUT/$LABEL.replay.out" 2>&1
  echo 180 > "$OUT/$LABEL/warmup_s"
  curl -s 127.0.0.1:18000/telemetry > "$OUT/$LABEL/telemetry-end.json"
  docker logs replay-srv > "$OUT/$LABEL/server.log" 2>&1
  docker rm -f replay-srv >/dev/null
' _ "$BIN" "$OUT" "$LABEL" "$CTX" "$AGENTS" "$DUR" "$MIND" "$HERE" "${EXTRA[@]}"

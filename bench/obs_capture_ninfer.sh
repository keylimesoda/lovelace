#!/usr/bin/env bash
# Capture real tensorninja/ninfer-4090 observability payloads (bench/obs-fixtures/ninfer/) under the GPU lock.
# The engine is tensorninja/ninfer-4090 at f408990, built from its own Dockerfile's build stage:
#   git -C ~/ninfer-4090 archive f408990 | tar -x -C ~/Work/ninfer-ref
#   docker build --target build -t ninfer-tn:f408990-build ~/Work/ninfer-ref
# with the official artifact (sha256 eec39564...bf3e, the one its Dockerfile pins) and its Dockerfile's default
# serving flags (context and cache tiers scaled down so the capture is short).
# Usage: bench/obs_capture_ninfer.sh    env: IMAGE, ARTIFACTS, PORT
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-ninfer-tn:f408990-build}
ARTIFACTS=${ARTIFACTS:-$HOME/Work/ninfer-artifacts}
PORT=${PORT:-18480}
OUT=$HERE/obs-fixtures/ninfer
CACHE=$HOME/.cache/lovelace-bench/obs-ninfer
rm -rf "$CACHE" && mkdir -p "$CACHE" "$OUT"
exec "$HERE/../deploy/gpu-lock.sh" bash -c '
set -euo pipefail
IMAGE=$1 ARTIFACTS=$2 PORT=$3 OUT=$4 CACHE=$5 HERE=$6
docker rm -f obs-ninfer >/dev/null 2>&1 || true
docker run -d --name obs-ninfer --gpus all --network host --user "$(id -u):$(id -g)" \
  --memory 34g --memory-swap 34g \
  -v "$ARTIFACTS:/models:ro" -v "$CACHE:/var/cache/ninfer" "$IMAGE" \
  /opt/ninfer/bin/ninfer-serve /models/qwen3_8_27b.ninfer --model-id qwen3.8-27b --host 127.0.0.1 --port "$PORT" \
  --max-context 65536 --kv-capacity auto --max-concurrency 4 --max-pending-requests 16 --prefill-chunk 1024 \
  --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 3 --lm-head-draft --prefix-checkpoint-policy rolling-tool \
  --continuation-cache l1-l2-l3 --continuation-cache-dir /var/cache/ninfer --continuation-cache-namespace local \
  --continuation-cache-l1-mib 768 --continuation-cache-l2-mib 4096 --continuation-cache-l3-mib 8192 \
  --continuation-cache-filesystem-reserve-mib 4096 --preserve-thinking \
  --request-log-jsonl /var/cache/ninfer/request-log.jsonl >/dev/null
ok=0
for i in $(seq 240); do
  [ "$(docker inspect -f {{.State.Running}} obs-ninfer 2>/dev/null)" = true ] || break
  curl -sf "127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }; sleep 2
done
if [ $ok = 1 ]; then
  python3 "$HERE/obs_capture.py" --base-url "http://127.0.0.1:$PORT" --model qwen3.8-27b --out "$OUT" || ok=0
  cp "$CACHE/request-log.jsonl" "$OUT/request-log.jsonl" || true
fi
docker logs obs-ninfer > "$OUT/server.log" 2>&1 || true
docker rm -f obs-ninfer >/dev/null
[ $ok = 1 ]
' _ "$IMAGE" "$ARTIFACTS" "$PORT" "$OUT" "$CACHE" "$HERE"

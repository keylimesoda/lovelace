#!/usr/bin/env bash
# Largest context that starts, and the compute buffers, for the production profile and several engine builds.
# One GPU-lock hold for all builds, so every build sees the same free VRAM. For each build, llama-server is started
# with deploy/profiles/PROFILE.args (host-side paths mapped into the container, --web-dir dropped) at -c START, then
# -c grows by STEP until the server fails to start; if START already fails, -c shrinks by STEP until it starts.
#   bench/max_ctx.sh OUT_DIR BIN_DIR... (env: PROFILE=balanced START=327680 STEP=8192)
# Writes OUT_DIR/<build>-c<ctx>.log per attempt and OUT_DIR/max_ctx.tsv (build, largest ctx that starts, compute
# buffer MiB per device buffer at that ctx).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1"); shift
mkdir -p "$OUT"
BINS=(); for b in "$@"; do BINS+=("$(realpath "$b")"); done
exec "$HERE/../deploy/gpu-lock.sh" bash -c '
set -uo pipefail
OUT=$1 HERE=$2; shift 2
PROFILE=${PROFILE:-balanced} START=${START:-327680} STEP=${STEP:-8192} PORT=18400 C=max-ctx
mkdir -p /tmp/max-ctx-sessions
args_for() { # ctx -> profile args, one per line
  sed -e "s/\${CTX}/$1/" -e "s#/var/lib/lovelace/sessions#/sessions#" -e "s#/var/log/lovelace/#/tmp/#" \
      -e "s/^--port .*/--port $PORT/" "$HERE/../deploy/profiles/$PROFILE.args" | grep -v "^--web-dir"
  echo "-lv 4"   # info level: the llama_context buffer sizes
}
try() { # bin ctx log -> 0 if the server becomes healthy
  local bin=$1 ctx=$2 log=$3 ok=1
  docker rm -f $C >/dev/null 2>&1
  mapfile -t A < <(args_for "$ctx" | tr " " "\n")
  docker run -d --name $C --gpus all --network host --ulimit memlock=-1:-1 -v "$bin:/e" -v "$HOME/models:/models:ro" \
    -v /tmp/max-ctx-sessions:/sessions -v "$HERE/../deploy/draft-vocab:/opt/lovelace/draft-vocab:ro" \
    lovelace-engine-runtime:cuda13.1 /e/llama-server "${A[@]}" >/dev/null
  for i in $(seq 200); do
    [ "$(docker inspect -f {{.State.Running}} $C 2>/dev/null)" = true ] || break
    curl -sf 127.0.0.1:$PORT/health >/dev/null && { ok=0; break; }; sleep 2
  done
  docker logs $C > "$log" 2>&1
  docker rm -f $C >/dev/null
  while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1) > 2048 )); do sleep 1; done
  return $ok
}
for bin in "$@"; do
  name=$(echo "${bin#$HOME/Work/}" | tr / _)
  ctx=$START best=0
  if try "$bin" $ctx "$OUT/$name-c$ctx.log"; then
    best=$ctx
    while ctx=$((ctx + STEP)); try "$bin" $ctx "$OUT/$name-c$ctx.log"; do best=$ctx; done
  else
    while ctx=$((ctx - STEP)); (( ctx > 0 )); do
      if try "$bin" $ctx "$OUT/$name-c$ctx.log"; then best=$ctx; break; fi
    done
  fi
  bufs=$(grep -hoE "[A-Za-z0-9_]+ compute buffer size = +[0-9.]+ MiB" "$OUT/$name-c$best.log" 2>/dev/null | tr -s " " | paste -sd ";")
  printf "%s\t%s\t%s\t%s\n" "$name" "$bin" "$best" "$bufs" | tee -a "$OUT/max_ctx.tsv"
done
' _ "$OUT" "$HERE" "${BINS[@]}"

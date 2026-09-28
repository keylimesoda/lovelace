#!/usr/bin/env bash
# Unattended engine bake-off under replayed OMP multi-agent traffic.
# Stops the production ninfer-qwen container for the duration and ALWAYS restarts it on exit.
# Usage: run_matrix.sh OUTDIR [RUN_ID ...]   (no RUN_IDs = full matrix)
set -uo pipefail

OUT=${1:?outdir}; shift
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
HERE=$(cd "$(dirname "$0")" && pwd)
REPLAY_DIR=$HERE/../replay
UV=${UV:-/tmp/uvtool/bin/uv}
DURATION=${DURATION:-1200}
IMG=lovelace-engine-runtime:cuda13.1
SRC=$HOME/Work/src

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/matrix.log"; }

restore_prod() {
  docker rm -f bake >/dev/null 2>&1
  docker start ninfer-qwen >/dev/null 2>&1
  log "production ninfer-qwen restarted"
}
trap restore_prod EXIT

wait_health() { # port -> 0 healthy, 1 dead/timeout
  local port=$1 t=0
  while (( t < 420 )); do
    if [[ "$(docker inspect -f '{{.State.Running}}' bake 2>/dev/null)" != "true" ]]; then return 1; fi
    curl -sf "127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    sleep 3; t=$((t + 3))
  done
  return 1
}

gpu_sampler() { # file
  while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)" >> "$1"
    sleep 5
  done
}

replay() { # id port agents
  local id=$1 port=$2 agents=$3 dir=$OUT/$1
  gpu_sampler "$dir/gpu.csv" & local sp=$!
  (cd "$REPLAY_DIR" && "$UV" run replay.py --base-url "http://127.0.0.1:$port/v1" --model qwen3.8-27b \
      --agents "$agents" --duration "$DURATION" --no-drain --start-turn-min-depth 60000 \
      --think-scale 0.25 --max-think 20 --output-tokens recorded --temperature 1.0 \
      --extra-body '{"top_p":0.95,"top_k":20}' --results-dir "$OUT" --label "$id" ) > "$dir/replay.out" 2>&1
  kill "$sp" 2>/dev/null
  docker logs bake > "$dir/server.log" 2>&1
  docker rm -f bake >/dev/null 2>&1
}

LLAMA_COMMON=(-m /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf --host 127.0.0.1 --port 18000 -ngl 99 -fa on
  --jinja -ub 1024 -b 2048 --kv-unified --cache-ram 24576 --cache-idle-slots --reasoning-preserve
  --alias qwen3.8-27b --metrics --slots)
DFLASH=(-md /models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ngld 99 --spec-type draft-dflash)

run_llama() { # id engine agents extra...
  local id=$1 engine=$2 agents=$3; shift 3
  local dir=$OUT/$id; mkdir -p "$dir"
  for c in 393216 327680 262144 229376 196608 163840 131072; do
    docker rm -f bake >/dev/null 2>&1
    local cmd=(/e/llama-server "${LLAMA_COMMON[@]}" -c "$c" "$@")
    docker run -d --name bake --gpus all --network host -v "$SRC/$engine/build-sm89/bin:/e" \
      -v "$HOME/models:/models:ro" "$IMG" "${cmd[@]}" >/dev/null
    if wait_health 18000; then
      printf '%q ' "${cmd[@]}" > "$dir/server_cmd.txt"; echo "ctx=$c" >> "$dir/server_cmd.txt"
      log "$id up with -c $c; replaying $agents agents"
      replay "$id" 18000 "$agents"
      log "$id done"
      return 0
    fi
    docker logs bake > "$dir/fail-c$c.log" 2>&1
    log "$id failed at -c $c"
  done
  log "$id FAIL: no context size fit"
}

NINFER_BASE=(--model-id qwen3.8-27b --host 127.0.0.1 --port 18080 --max-context 262144 --max-pending-requests 32
  --pending-timeout-ms 600000 --prefill-chunk 1024 --kv-dtype rk4v4-e8 --lm-head-draft --preserve-thinking)

run_ninfer() { # id agents lanes artifact spec-args...
  local id=$1 agents=$2 lanes=$3 artifact=$4; shift 4
  local dir=$OUT/$id; mkdir -p "$dir"
  for cap in 393216 327680 262144 229376 196608 163840; do
    docker rm -f bake >/dev/null 2>&1
    local cmd=(ninfer-serve "/models/$artifact" "${NINFER_BASE[@]}" "$@" --max-concurrency "$lanes" --kv-capacity "$cap")
    docker run -d --name bake --gpus all --network host -v "$HOME/Work/ninfer-artifacts:/models:ro" \
      ninfer-4090:hybrid "${cmd[@]}" >/dev/null
    if wait_health 18080; then
      printf '%q ' "${cmd[@]}" > "$dir/server_cmd.txt"; echo "kv_capacity=$cap" >> "$dir/server_cmd.txt"
      log "$id up with kv-capacity $cap; replaying $agents agents"
      replay "$id" 18080 "$agents"
      log "$id done"
      return 0
    fi
    docker logs bake > "$dir/fail-cap$cap.log" 2>&1
    log "$id failed at kv-capacity $cap"
  done
  log "$id FAIL: no kv capacity fit"
}

run() {
  case $1 in
    L-nospec-a6) run_llama "$1" llama.cpp 6 -np 6 -ctk q4_0 -ctv q4_0 ;;
    L-mtp-a6)    run_llama "$1" llama.cpp 6 -np 6 -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 3 ;;
    L-dflash-a4) run_llama "$1" llama.cpp 4 -np 4 -ctk q4_0 -ctv q4_0 "${DFLASH[@]}" --spec-draft-n-max 7 ;;
    L-dflash-a6) run_llama "$1" llama.cpp 6 -np 6 -ctk q4_0 -ctv q4_0 "${DFLASH[@]}" --spec-draft-n-max 3 ;;
    B-dflash-a6) run_llama "$1" beellama.cpp 6 -np 6 -ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024 "${DFLASH[@]}" -ubd 128 --spec-draft-n-max 3 ;;
    N-mtp-a4)    run_ninfer "$1" 4 4 qwen3_8_27b.ninfer --spec mtp --draft-tokens 3 ;;
    N-mtp-a6)    run_ninfer "$1" 6 6 qwen3_8_27b.ninfer --spec mtp --draft-tokens 3 ;;
    N-dflash-a6) run_ninfer "$1" 6 6 qwen3_8_27b_dflash2.ninfer --spec dflash2 --draft-tokens 3 ;;
    *) log "unknown run $1" ;;
  esac
}

RUNS=("$@")
(( ${#RUNS[@]} )) || RUNS=(N-mtp-a6 L-mtp-a6 L-nospec-a6 B-dflash-a6 L-dflash-a6 N-dflash-a6 N-mtp-a4 L-dflash-a4)
log "stopping production for bake-off: ${RUNS[*]}"
docker stop ninfer-qwen >/dev/null
for r in "${RUNS[@]}"; do run "$r"; done
"$UV" run --directory "$REPLAY_DIR" python "$HERE/analyze.py" "$OUT" > "$OUT/REPORT.md" 2>&1
log "report: $OUT/REPORT.md"

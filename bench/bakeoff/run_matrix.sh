#!/usr/bin/env bash
# Unattended engine bake-off under replayed OMP multi-agent traffic.
# Phase 1 (preflight): start every selected config once at exactly the KV capacity its workload needs and record
#   whether it fits (seconds per config). Phase 2: replay only the configs that fit.
# Workloads are sized so agents x (max_depth + max output) <= capacity: no run can overcommit the KV cache.
# Usage: run_matrix.sh OUTDIR [RUN_ID ...]      env: RESTORE_PROD=1 restarts ninfer-qwen on exit
set -uo pipefail

OUT=${1:?outdir}; shift
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
HERE=$(cd "$(dirname "$0")" && pwd)
REPLAY_DIR=$HERE/../replay
UV=${UV:-uv}
IMG=lovelace-engine-runtime:cuda13.1
SRC=$HOME/Work/src

# workload profiles: agents min_depth max_depth capacity duration warmup
declare -A WL=(
  [batch]="3 45000 60000 196608 600 180"
  [deep]="1 120000 140000 147456 480 120"
)

log() { echo "[$(date +%T)] $*" | tee -a "$OUT/matrix.log"; }

cleanup() {
  docker rm -f bake >/dev/null 2>&1
  if [[ ${RESTORE_PROD:-0} == 1 ]]; then docker start ninfer-qwen >/dev/null 2>&1 && log "production restarted"; fi
}
trap cleanup EXIT

wait_health() { # port -> 0 healthy, 1 dead/timeout
  local port=$1 t=0
  while (( t < 420 )); do
    [[ "$(docker inspect -f '{{.State.Running}}' bake 2>/dev/null)" == "true" ]] || return 1
    curl -sf "127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    sleep 2; t=$((t + 2))
  done
  return 1
}

gpu_sampler() {
  while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)" >> "$1"
    sleep 5
  done
}

LLAMA_COMMON=(-m /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf --host 127.0.0.1 --port 18000 -ngl 99 -fa on
  --jinja -ub 1024 -b 2048 --kv-unified --cache-ram 24576 --cache-idle-slots --reasoning-preserve
  --alias qwen3.8-27b --metrics --slots)
DFLASH=(-md /models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ngld 99 --spec-type draft-dflash)
NINFER_BASE=(--model-id qwen3.8-27b --host 127.0.0.1 --port 18080 --max-pending-requests 32
  --pending-timeout-ms 600000 --prefill-chunk 1024 --kv-dtype rk4v4-e8 --lm-head-draft --preserve-thinking)

# config registry: id -> "engine|workload|args..."  (engine: llama.cpp | beellama.cpp | ninfer:<artifact>)
declare -A CFG=(
  [L-nospec-b]="llama.cpp|batch|-ctk q4_0 -ctv q4_0"
  [L-mtp-b]="llama.cpp|batch|-ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 3"
  [L-dflash7-b]="llama.cpp|batch|-ctk q4_0 -ctv q4_0 ${DFLASH[*]} --spec-draft-n-max 7"
  [L-dflash3-b]="llama.cpp|batch|-ctk q4_0 -ctv q4_0 ${DFLASH[*]} --spec-draft-n-max 3"
  [B-dflash-b]="beellama.cpp|batch|-ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024 ${DFLASH[*]} -ubd 128 --spec-draft-n-max 7"
  [N-mtp-b]="ninfer:qwen3_8_27b.ninfer|batch|--spec mtp --draft-tokens 3"
  [N-dflash-b]="ninfer:qwen3_8_27b_dflash2.ninfer|batch|--spec dflash2 --draft-tokens 3"
  [L-nospec-d]="llama.cpp|deep|-ctk q4_0 -ctv q4_0"
  [L-mtp-d]="llama.cpp|deep|-ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 3"
  [B-dflash-d]="beellama.cpp|deep|-ctk kvarn5 -ctv kvarn4 --kv-tail-tokens 1024 ${DFLASH[*]} -ubd 128 --spec-draft-n-max 7"
  [N-mtp-d]="ninfer:qwen3_8_27b.ninfer|deep|--spec mtp --draft-tokens 3"
)
ORDER=(L-nospec-b L-mtp-b L-dflash7-b L-dflash3-b B-dflash-b N-mtp-b N-dflash-b L-nospec-d L-mtp-d B-dflash-d N-mtp-d)

start_server() { # id -> 0 if healthy; sets PORT
  local id=$1 spec=${CFG[$1]}
  local engine=${spec%%|*} rest=${spec#*|}; local wl=${rest%%|*} args=${rest#*|}
  read -r agents dmin dmax cap dur warm <<< "${WL[$wl]}"
  local dir=$OUT/$id; mkdir -p "$dir"
  docker rm -f bake >/dev/null 2>&1
  local cmd
  if [[ $engine == ninfer:* ]]; then
    PORT=18080
    # shellcheck disable=SC2206
    cmd=(ninfer-serve "/models/${engine#ninfer:}" "${NINFER_BASE[@]}" $args --max-concurrency "$agents"
      --max-context "$cap" --kv-capacity "$cap")
    docker run -d --name bake --gpus all --network host -v "$HOME/Work/ninfer-artifacts:/models:ro" \
      ninfer-4090:hybrid "${cmd[@]}" >/dev/null
  else
    PORT=18000
    # shellcheck disable=SC2206
    cmd=(/e/llama-server "${LLAMA_COMMON[@]}" -np "$agents" -c "$cap" $args)
    docker run -d --name bake --gpus all --network host -v "$SRC/$engine/build-sm89/bin:/e" \
      -v "$HOME/models:/models:ro" "$IMG" "${cmd[@]}" >/dev/null
  fi
  printf '%q ' "${cmd[@]}" > "$dir/server_cmd.txt"
  if wait_health "$PORT"; then return 0; fi
  docker logs bake > "$dir/preflight-fail.log" 2>&1
  return 1
}

replay() { # id
  local id=$1 spec=${CFG[$1]}; local rest=${spec#*|}; local wl=${rest%%|*}
  read -r agents dmin dmax cap dur warm <<< "${WL[$wl]}"
  local dir=$OUT/$id
  echo "$warm" > "$dir/warmup_s"
  gpu_sampler "$dir/gpu.csv" & local sp=$!
  (cd "$REPLAY_DIR" && "$UV" run replay.py --base-url "http://127.0.0.1:$PORT/v1" --model qwen3.8-27b \
      --agents "$agents" --duration "$dur" --no-drain --start-turn-min-depth "$dmin" --max-depth "$dmax" \
      --think-scale 0.25 --max-think 20 --output-tokens recorded --temperature 1.0 \
      --extra-body '{"top_p":0.95,"top_k":20}' --results-dir "$OUT" --label "$id") > "$dir/replay.out" 2>&1
  kill "$sp" 2>/dev/null
  docker logs bake > "$dir/server.log" 2>&1
  docker rm -f bake >/dev/null 2>&1
}

RUNS=("$@"); (( ${#RUNS[@]} )) || RUNS=("${ORDER[@]}")
docker stop ninfer-qwen >/dev/null 2>&1
log "preflight: ${RUNS[*]}"
FIT=()
for id in "${RUNS[@]}"; do
  if start_server "$id"; then FIT+=("$id"); log "FIT    $id"; else log "NOFIT  $id (see $id/preflight-fail.log)"; fi
  docker rm -f bake >/dev/null 2>&1
done
for id in "${FIT[@]}"; do
  if start_server "$id"; then log "run $id"; replay "$id"; log "done $id"; else log "restart failed $id"; fi
done
"$UV" run --directory "$REPLAY_DIR" python "$HERE/analyze.py" "$OUT" > "$OUT/REPORT.md" 2>&1
log "report: $OUT/REPORT.md"

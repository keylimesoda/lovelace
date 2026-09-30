#!/usr/bin/env bash
# Observability overhead (CHARTER E1: no measurable cost on decode/prefill), one GPU-lock hold, same binary, flags only.
# Alternates server starts telemetry OFF / ON / OFF / ON (ORDER); each start measures
#   (a) single-stream decode at DEPTH: bench/depth_sweep.py, one cold + REPS warm repetitions (decode tok/s = median of
#       the warm ones, same prompt and seed every time);
#   (b) six concurrent sequences: bench/step_drive.py at SIX_DEPTHS (bench/step_profile.sh's six=52000 x 6), capture
#       window CAPTURE_S (aggregate tok/s, per-sequence tok/s, step ms).
# OFF: deploy/profiles/balanced.args without --telemetry --request-log-jsonl --metrics --slots --web-dir.
# ON:  the same plus --telemetry --request-log-jsonl (on the NVMe) --metrics --slots, while bench/obs_overhead.py load
#      polls /telemetry at 1 Hz and /metrics every 4 s and holds an /events subscription for the whole run.
# Container conventions of bench/h2h.sh (lovelace-engine-runtime:cuda13.1, BIN mounted at /e, --memory 34g), with the
# profile's own in-container paths mounted (sessions and request log under ~/.cache/lovelace-bench/obs-overhead/).
#   bench/obs_overhead.sh [OUT_DIR]      (default bench/results/obs-parity/overhead)
# env: BIN engine bin dir with llama-server + libs (default <repo>/engine/build-sm89/bin), CTX (profile context,
#      default 376832 = deploy/serve.sh balanced), ORDER (default "off on off on"), DEPTH (64000), REPS (3),
#      MAX_TOKENS (512, single stream), SIX_DEPTHS (52000,52000,52000,52000,52000,52000), SIX_TOKENS (1024),
#      CAPTURE_S (10), PORT (18500), ENGINE_MEM (34g), PROFILE (balanced),
#      DRY_RUN=1 prints each run's server command and exits (no lock, no GPU),
#      IN_GPU_LOCK=1 when the caller already holds deploy/gpu-lock.sh (bench/obs_parity.sh).
# Writes OUT_DIR/NN-{off,on}/{server_args.txt,single.jsonl,single.out,six.out,load.json,server.log} and
# OUT_DIR/RESULTS.md (bench/obs_overhead.py table).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=$(realpath -m "${1:-$HERE/results/obs-parity/overhead}")
BIN=$(realpath -m "${BIN:-$ROOT/engine/build-sm89/bin}")
CTX=${CTX:-376832} PORT=${PORT:-18500} PROFILE=${PROFILE:-balanced}
read -r -a ORDER <<< "${ORDER:-off on off on}"
C=obs-overhead
CACHE=$HOME/.cache/lovelace-bench/obs-overhead   # NVMe, never /tmp (tmpfs is RAM)

server_args() { # MODE -> the profile's args with the observability flags removed (off) or kept (on)
  local mode=$1 line skip=0 tok
  local -a toks out=()
  while IFS= read -r line || [[ -n $line ]]; do
    [[ -z $line || $line == \#* ]] && continue
    line=${line//\$\{CTX\}/$CTX}
    read -r -a toks <<< "$line"
    out+=("${toks[@]}")
  done < "$ROOT/deploy/profiles/$PROFILE.args"
  local -a res=()
  for tok in "${out[@]}"; do
    if ((skip)); then skip=0; continue; fi
    case $tok in
      --telemetry|--metrics|--slots) continue ;;
      --request-log-jsonl|--web-dir) skip=1; continue ;;
      --port) skip=1; res+=(--port "$PORT"); continue ;;
    esac
    res+=("$tok")
  done
  [[ $mode == on ]] && res+=(--telemetry --request-log-jsonl /var/log/lovelace/requests.jsonl --metrics --slots)
  printf '%s\n' "${res[@]}"
}

if [[ ${DRY_RUN:-0} == 1 ]]; then
  for i in "${!ORDER[@]}"; do
    echo "# run $(printf %02d $((i + 1)))-${ORDER[i]}"
    echo "llama-server $(server_args "${ORDER[i]}" | tr '\n' ' ')"
  done
  exit 0
fi
[[ ${IN_GPU_LOCK:-0} == 1 ]] || exec "$ROOT/deploy/gpu-lock.sh" env IN_GPU_LOCK=1 "$0" "$@"
[[ -x $BIN/llama-server ]] || { echo "obs_overhead: no llama-server in BIN=$BIN" >&2; exit 1; }
mkdir -p "$OUT"
up() {
  for _ in $(seq 300); do
    [[ $(docker inspect -f '{{.State.Running}}' $C 2>/dev/null) == true ]] || return 1
    curl -sf 127.0.0.1:$PORT/health >/dev/null && return 0; sleep 2
  done; return 1
}
fail=0
for i in "${!ORDER[@]}"; do
  mode=${ORDER[i]} R=$OUT/$(printf %02d $((i + 1)))-${ORDER[i]}
  rm -rf "$R" "$CACHE" && mkdir -p "$R" "$CACHE/sessions" "$CACHE/log"
  mapfile -t ARGS < <(server_args "$mode")
  printf '%s\n' "${ARGS[@]}" > "$R/server_args.txt"
  echo "== $(basename "$R") ($(date +%T))"
  docker rm -f $C >/dev/null 2>&1 || true
  docker run -d --name $C --gpus all --network host --user "$(id -u):$(id -g)" --ulimit memlock=-1:-1 \
    --memory "${ENGINE_MEM:-34g}" --memory-swap "${ENGINE_MEM:-34g}" \
    -v "$BIN:/e" -v "$HOME/models:/models:ro" -v "$ROOT/deploy/draft-vocab:/opt/lovelace/draft-vocab:ro" \
    -v "$CACHE/sessions:/var/lib/lovelace/sessions" -v "$CACHE/log:/var/log/lovelace" \
    lovelace-engine-runtime:cuda13.1 /e/llama-server "${ARGS[@]}" >/dev/null
  if up; then
    LOAD=
    if [[ $mode == on ]]; then
      python3 "$HERE/obs_overhead.py" load --base-url "http://127.0.0.1:$PORT" --out "$R" & LOAD=$!
    fi
    (cd "$HERE/replay" && uv run python ../depth_sweep.py --base-url "http://127.0.0.1:$PORT/v1" --label single \
       --out "$R" --depths "${DEPTH:-64000}" --reps "${REPS:-3}" --max-tokens "${MAX_TOKENS:-512}") > "$R/single.out" 2>&1 ||
      { echo "single-stream drive failed ($R/single.out)"; fail=1; }
    (cd "$HERE/replay" && uv run python ../step_drive.py --base-url "http://127.0.0.1:$PORT/v1" \
       --depths "${SIX_DEPTHS:-52000,52000,52000,52000,52000,52000}" --max-tokens "${SIX_TOKENS:-1024}" \
       --capture-s "${CAPTURE_S:-10}") > "$R/six.out" 2>&1 || { echo "six-sequence drive failed ($R/six.out)"; fail=1; }
    [[ -n $LOAD ]] && { kill -TERM "$LOAD"; wait "$LOAD" || true; }
  else
    echo "server did not start"; fail=1
  fi
  docker logs $C > "$R/server.log" 2>&1 || true
  [[ $mode == on ]] && { cp "$CACHE/log/requests.jsonl" "$R/requests.jsonl" || echo "no request log"; }
  docker rm -f $C >/dev/null
done
rm -rf "$CACHE"
python3 "$HERE/obs_overhead.py" table "$OUT"
exit $fail

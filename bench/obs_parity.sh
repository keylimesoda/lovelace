#!/usr/bin/env bash
# Observability parity evidence in ONE deploy/gpu-lock.sh hold (docs/observability-parity.md):
#   1. start lovelace (BIN, deploy/profiles/$PROFILE.args: --telemetry --request-log-jsonl --metrics --slots and the
#      dashboard via --web-dir apps/web/dist) as a container named ninfer-qwen on 127.0.0.1:8080, where the
#      omarchy-ninfer widget looks for its server;
#   2. bench/obs_capture.py -> bench/obs-fixtures/lovelace/ (+ request-log.jsonl, server.log);
#   3. bench/obs_conformance.py (vs bench/obs-fixtures/ninfer/) -> bench/results/obs-parity/conformance.md;
#   4. bench/widget_check.sh (while one streamed request decodes) -> bench/results/obs-parity/widget/;
#   5. dashboard screenshot after the traffic: bun bench/screenshot.mjs http://127.0.0.1:8080/ (headless Chromium over
#      the DevTools protocol, full page, SCREENSHOT_WAIT ms for the SSE backlog and polls)
#      -> bench/results/obs-parity/dashboard.png;
#   6. remove the container; with OVERHEAD=1 run bench/obs_overhead.sh (same hold) -> bench/results/obs-parity/overhead/.
# A stopped container already named ninfer-qwen (the widget's own ninfer server) is renamed aside for the run and
# restored after; a running one aborts the script.
#   bench/obs_parity.sh
# env: BIN engine bin dir with llama-server + libs (default <repo>/engine/build-sm89/bin), CTX (default 376832),
#      PROFILE (balanced), WEB_DIST (default <repo>/apps/web/dist, built with `cd apps/web && bun run build`),
#      MODEL_ID (qwen3.8-27b), SCREENSHOT_WAIT (8000), OVERHEAD=1 (also run the overhead measurement), ENGINE_MEM (34g),
#      DRY_RUN=1 (print the server command and the plan; no lock, no docker)
# Exit 1 when a step fails (conformance, widget or screenshot); every step runs regardless.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BIN=$(realpath -m "${BIN:-$ROOT/engine/build-sm89/bin}")
WEB=$(realpath -m "${WEB_DIST:-$ROOT/apps/web/dist}")
CTX=${CTX:-376832} PROFILE=${PROFILE:-balanced} MODEL_ID=${MODEL_ID:-qwen3.8-27b}
RES=$HERE/results/obs-parity FIX=$HERE/obs-fixtures/lovelace
C=ninfer-qwen SAVED=ninfer-qwen.obs-parity-saved PORT=8080 BASE=http://127.0.0.1:8080
CACHE=$HOME/.cache/lovelace-bench/obs-parity   # NVMe, never /tmp (tmpfs is RAM)

args=()
while IFS= read -r line || [[ -n $line ]]; do
  [[ -z $line || $line == \#* ]] && continue
  read -r -a toks <<< "${line//\$\{CTX\}/$CTX}"
  args+=("${toks[@]}")
done < "$ROOT/deploy/profiles/$PROFILE.args"
for i in "${!args[@]}"; do [[ ${args[i]} == --port ]] && args[i+1]=$PORT; done
for f in --telemetry --metrics --slots --request-log-jsonl --web-dir; do
  [[ " ${args[*]} " == *" $f "* ]] || { echo "obs_parity: $PROFILE.args lacks $f" >&2; exit 1; }
done

if [[ ${DRY_RUN:-0} == 1 ]]; then   # print the plan, touch nothing
  echo "container $C (docker run --gpus all --network host, BIN=$BIN at /e, dashboard $WEB at /opt/lovelace/web)"
  echo "llama-server ${args[*]}"
  echo "then: obs_capture.py -> $FIX; obs_conformance.py -> $RES/conformance.md; widget_check.sh -> $RES/widget;"
  echo "      screenshot.mjs -> $RES/dashboard.png$([[ ${OVERHEAD:-0} == 1 ]] && echo "; obs_overhead.sh -> $RES/overhead")"
  exit 0
fi
if [[ ${IN_GPU_LOCK:-0} != 1 ]]; then   # checks that need no GPU, before queueing for the lock
  [[ -x $BIN/llama-server ]] || { echo "obs_parity: no llama-server in BIN=$BIN" >&2; exit 1; }
  [[ -f $WEB/index.html ]] || { echo "obs_parity: no dashboard build in $WEB (cd apps/web && bun run build)" >&2; exit 1; }
  [[ -d $HERE/obs-fixtures/ninfer ]] || echo "obs_parity: warning: bench/obs-fixtures/ninfer/ missing (conformance will fail)" >&2
  command -v bun >/dev/null || { echo "obs_parity: bun not found" >&2; exit 1; }
  exec "$ROOT/deploy/gpu-lock.sh" env IN_GPU_LOCK=1 "$0" "$@"
fi

if [[ $(docker container inspect -f '{{.State.Running}}' $C 2>/dev/null) == true ]]; then
  echo "obs_parity: a container named $C is running (the widget's own server?); stop it first" >&2; exit 1
fi
renamed=0
if docker container inspect $C >/dev/null 2>&1; then
  docker container inspect $SAVED >/dev/null 2>&1 && { echo "obs_parity: both $C and $SAVED exist" >&2; exit 1; }
  docker rename $C $SAVED && renamed=1
fi
restore() {
  docker rm -f $C >/dev/null 2>&1 || true
  ((renamed)) && docker rename $SAVED $C
  true
}
trap restore EXIT
trap 'exit 130' INT TERM   # run the EXIT trap (restore the renamed container) on Ctrl-C / kill too

rm -rf "$CACHE" "$FIX" && mkdir -p "$CACHE/sessions" "$CACHE/log" "$FIX" "$RES"
printf '%s\n' "${args[@]}" > "$RES/server_args.txt"
echo "== start lovelace as $C ($(date +%T))"
docker run -d --name $C --gpus all --network host --user "$(id -u):$(id -g)" --ulimit memlock=-1:-1 \
  --memory "${ENGINE_MEM:-34g}" --memory-swap "${ENGINE_MEM:-34g}" \
  -v "$BIN:/e" -v "$HOME/models:/models:ro" -v "$ROOT/deploy/draft-vocab:/opt/lovelace/draft-vocab:ro" \
  -v "$CACHE/sessions:/var/lib/lovelace/sessions" -v "$CACHE/log:/var/log/lovelace" -v "$WEB:/opt/lovelace/web:ro" \
  lovelace-engine-runtime:cuda13.1 /e/llama-server "${args[@]}" >/dev/null
ok=0
for _ in $(seq 300); do
  [[ $(docker inspect -f '{{.State.Running}}' $C 2>/dev/null) == true ]] || break
  curl -sf $BASE/health >/dev/null && { ok=1; break; }; sleep 2
done
fail=0
if ((ok)); then
  echo "== capture ($(date +%T))"
  python3 "$HERE/obs_capture.py" --base-url $BASE --model "$MODEL_ID" --out "$FIX" || fail=1
  cp "$CACHE/log/requests.jsonl" "$FIX/request-log.jsonl" || fail=1

  echo "== conformance"
  python3 "$HERE/obs_conformance.py" --lovelace "$FIX" --out "$RES/conformance.md" > /dev/null || fail=1
  tail -n +1 "$RES/conformance.md" | sed -n '/^| \*\*total/p;/^## Failures/,+12p'

  echo "== widget ($(date +%T))"
  curl -sN $BASE/v1/chat/completions -H 'content-type: application/json' -d "{\"model\":\"$MODEL_ID\",\"stream\":true,
    \"max_tokens\":400,\"messages\":[{\"role\":\"user\",\"content\":\"Write a long story about a lighthouse keeper.\"}],
    \"chat_template_kwargs\":{\"enable_thinking\":false}}" >/dev/null & STREAM=$!
  sleep 2
  "$HERE/widget_check.sh" "$RES/widget" > /dev/null || fail=1
  sed -n '/^## Checks/,/^## omarchy/p' "$RES/widget/labels.md"
  wait $STREAM || true

  echo "== dashboard screenshot ($(date +%T))"
  bun "$HERE/screenshot.mjs" $BASE/ "$RES/dashboard.png" "${SCREENSHOT_WAIT:-8000}" || fail=1
else
  echo "obs_parity: server did not start" >&2; fail=1
fi
docker logs $C > "$FIX/server.log" 2>&1 || true
cp "$FIX/server.log" "$RES/server.log"
restore
trap - EXIT

if [[ ${OVERHEAD:-0} == 1 ]]; then
  echo "== overhead ($(date +%T))"
  IN_GPU_LOCK=1 BIN=$BIN CTX=$CTX PROFILE=$PROFILE "$HERE/obs_overhead.sh" "$RES/overhead" || fail=1
fi
exit $fail

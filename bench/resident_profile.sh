#!/usr/bin/env bash
# Cold-prefill cost with and without another session resident in the unified KV cache (production server flags).
# One GPU-lock hold per spec: start llama-server (under nsys unless NSYS=0), prefill the spec's depths in order with
# bench/resident_drive.py (all but the last stay resident), capture the last (measured) prefill, remove the server.
#   bench/resident_profile.sh OUT_DIR SPEC [SPEC ...] [-- extra server args]
#     SPEC = LABEL=DEPTH[,DEPTH...][+decode]   e.g. alone=128000 resident=150000,128000 mixed=150000,128000+decode
#     (+decode: the first session generates during the measured prefill, resident_drive.py --decode-during)
# env: BIN (default <worktree>/engine/build-sm89/bin), NSYS (1), CTX (327680)
# Writes OUT_DIR/LABEL.drive.out, LABEL.server.log and with NSYS=1 LABEL.nsys-rep, LABEL.sqlite, LABEL.table.txt
# (bench/resident_table.py: per-ubatch GPU ms per kernel class, uploads and host gap).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1"); shift
SPECS=(); while (($#)) && [[ $1 != -- ]]; do SPECS+=("$1"); shift; done
(($#)) && shift
EXTRA=("$@")
BIN=$(realpath "${BIN:-$HERE/../engine/build-sm89/bin}")
export NSYS=${NSYS:-1} CTX=${CTX:-327680}
mkdir -p "$OUT"
docker build -q -f "$HERE/../deploy/nsys.Dockerfile" -t lovelace-nsys:cuda13.1 "$HERE/../deploy" >/dev/null
for spec in "${SPECS[@]}"; do
  L=${spec%%=*} D=${spec#*=} DD=
  [[ $D == *+decode ]] && { D=${D%+decode}; DD=--decode-during; }
  export DD
  "$HERE/../deploy/gpu-lock.sh" bash -c '
  set -uo pipefail
  OUT=$1 BIN=$2 HERE=$3 L=$4 D=$5; shift 5
  PORT=18410 C=resprof
  docker rm -f $C >/dev/null 2>&1
  rm -rf /tmp/resprof-sessions; mkdir -p /tmp/resprof-sessions
  LAUNCH=(); START=true; STOP=true
  if [ "$NSYS" = 1 ]; then
    LAUNCH=(nsys launch --session-new=$C --trace=cuda,osrt --cuda-graph-trace=node)
    START="docker exec $C nsys start --session=$C --sample=none --cpuctxsw=none --output=/out/$L --force-overwrite=true >/dev/null"
    STOP="docker exec $C nsys stop --session=$C >/dev/null"
  fi
  docker run -d --name $C --gpus all --network host --ulimit memlock=-1:-1 -e LD_LIBRARY_PATH=/e \
    -v "$BIN:/e" -v "$HOME/models:/models:ro" -v "$OUT:/out" -v /tmp/resprof-sessions:/sessions \
    lovelace-nsys:cuda13.1 "${LAUNCH[@]}" \
    /e/llama-server -m /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf -ngl 99 -np 6 --kv-unified -c $CTX \
      -ctk q5_0 -ctv q5_0 --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q5_0 -ctvd q5_0 -fa on -ub 1024 -b 2048 \
      --jinja --reasoning-preserve --session-ram 30720 --session-dir /sessions -ctxcp 8 --kv-budget --prefill-chunk-gen 1024 \
      --alias qwen3.8-27b --telemetry --metrics --slots --host 127.0.0.1 --port $PORT "$@" >/dev/null
  ok=0
  for i in $(seq 300); do
    [ "$(docker inspect -f {{.State.Running}} $C 2>/dev/null)" = true ] || break
    curl -sf 127.0.0.1:$PORT/health >/dev/null && { ok=1; break; }; sleep 2
  done
  if [ $ok = 1 ]; then
    (cd "$HERE/replay" && uv run python ../resident_drive.py --base-url http://127.0.0.1:$PORT/v1 --depths "$D" $DD \
       --on-start "$START" --on-stop "$STOP") > "$OUT/$L.drive.out" 2>&1 || ok=0
    if [ "$NSYS" = 1 ] && [ $ok = 1 ]; then
      docker exec $C bash -c "[ -f /out/$L.nsys-rep ] || { \$NSYS_IMPORTER -i /out/$L.qdstrm >/dev/null && rm /out/$L.qdstrm; }"
      docker exec $C nsys export --type=sqlite --force-overwrite=true --output=/out/$L.sqlite /out/$L.nsys-rep >/dev/null
    fi
  fi
  docker logs $C > "$OUT/$L.server.log" 2>&1
  docker exec $C chown -R "$(id -u):$(id -g)" /out 2>/dev/null
  docker rm -f $C >/dev/null
  [ $ok = 1 ]
  ' _ "$OUT" "$BIN" "$HERE" "$L" "$D" "${EXTRA[@]}"
  cat "$OUT/$L.drive.out"
  if [ "$NSYS" = 1 ] && [ -f "$OUT/$L.sqlite" ]; then
    python3 "$HERE/resident_table.py" "$OUT/$L.sqlite" > "$OUT/$L.table.txt" && cat "$OUT/$L.table.txt"
  fi
done

#!/usr/bin/env bash
# Deterministic decode-step profile of the production server under nsys, in one GPU-lock hold.
#   bench/step_profile.sh OUT_DIR SPEC [SPEC ...] [-- extra server args]
#     SPEC = LABEL=DEPTH[,DEPTH...]   one capture per spec: one concurrent request per depth (bench/step_drive.py)
#   e.g. bench/step_profile.sh bench/results/step-profile/base six=52000,52000,52000,52000,52000,52000 one=64000
#   deploy/draft-vocab is mounted at /draft-vocab (e.g. -- --spec-draft-vocab /draft-vocab/qwen3.8-27b.txt)
# env: BIN (default <worktree>/engine/build-sm89/bin), CAPTURE_S (2), MAX_TOKENS (256), SPEC_N (3),
#      CPU_SAMPLE=1 adds host CPU sampling with DWARF backtraces (nsys stats --report osrt/cpu views)
# Writes OUT_DIR/LABEL.nsys-rep, LABEL.sqlite, LABEL.drive.out, LABEL.table.txt (bench/step_table.py) and server.log.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1"); shift
SPECS=(); while (($#)) && [[ $1 != -- ]]; do SPECS+=("$1"); shift; done
(($#)) && shift
EXTRA=("$@")
BIN=$(realpath "${BIN:-$HERE/../engine/build-sm89/bin}")
mkdir -p "$OUT"
docker build -q -f "$HERE/../deploy/nsys.Dockerfile" -t lovelace-nsys:cuda13.1 "$HERE/../deploy" >/dev/null
SAMPLE="--sample=none --cpuctxsw=none"; CAPS=()
[[ ${CPU_SAMPLE:-0} = 1 ]] && { SAMPLE="--sample=process-tree --backtrace=dwarf --cpuctxsw=none"; CAPS=(--cap-add SYS_ADMIN --security-opt seccomp=unconfined); }
export SAMPLE CAPS_STR="${CAPS[*]}"
"$HERE/../deploy/gpu-lock.sh" bash -c '
set -uo pipefail
OUT=$1 BIN=$2 HERE=$3 NSPEC=$4; shift 4
SPECS=("${@:1:$NSPEC}"); shift $NSPEC
PORT=18400 C=stepprof
docker rm -f $C >/dev/null 2>&1
mkdir -p /tmp/stepprof-sessions
docker run -d --name $C --gpus all $CAPS_STR --network host --ulimit memlock=-1:-1 -e LD_LIBRARY_PATH=/e \
  -v "$BIN:/e" -v "$HOME/models:/models:ro" -v "$OUT:/out" -v /tmp/stepprof-sessions:/sessions -v "$HERE/../deploy/draft-vocab:/draft-vocab:ro" \
  lovelace-nsys:cuda13.1 nsys launch --session-new=$C --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  /e/llama-server -m /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf -ngl 99 -np 6 --kv-unified -c 327680 \
    -ctk q5_0 -ctv q5_0 --spec-type draft-mtp --spec-draft-n-max ${SPEC_N:-3} -ctkd q5_0 -ctvd q5_0 -fa on -ub 1024 -b 2048 \
    --jinja --reasoning-preserve --session-ram 30720 --session-dir /sessions -ctxcp 8 --kv-budget --prefill-chunk-gen 1024 \
    --alias qwen3.8-27b --telemetry --metrics --host 127.0.0.1 --port $PORT "$@" >/dev/null
ok=0
for i in $(seq 300); do
  [ "$(docker inspect -f {{.State.Running}} $C 2>/dev/null)" = true ] || break
  curl -sf 127.0.0.1:$PORT/health >/dev/null && { ok=1; break; }; sleep 2
done
if [ $ok = 1 ]; then
  for spec in "${SPECS[@]}"; do
    L=${spec%%=*} D=${spec#*=}
    (cd "$HERE/replay" && uv run python ../step_drive.py --base-url http://127.0.0.1:$PORT/v1 --depths "$D" \
       --max-tokens ${MAX_TOKENS:-256} --capture-s ${CAPTURE_S:-2} \
       --on-start "docker exec $C nsys start --session=$C $SAMPLE --output=/out/$L --force-overwrite=true >/dev/null" \
       --on-stop "docker exec $C nsys stop --session=$C >/dev/null") > "$OUT/$L.drive.out" 2>&1
    docker exec $C bash -c "[ -f /out/$L.nsys-rep ] || { \$NSYS_IMPORTER -i /out/$L.qdstrm >/dev/null && rm /out/$L.qdstrm; }"
    docker exec $C nsys export --type=sqlite --force-overwrite=true --output=/out/$L.sqlite /out/$L.nsys-rep >/dev/null
  done
fi
docker logs $C > "$OUT/server.log" 2>&1
docker exec $C chown -R "$(id -u):$(id -g)" /out 2>/dev/null
docker rm -f $C >/dev/null
[ $ok = 1 ]
' _ "$OUT" "$BIN" "$HERE" "${#SPECS[@]}" "${SPECS[@]}" "${EXTRA[@]}"
for spec in "${SPECS[@]}"; do
  L=${spec%%=*}
  [ -f "$OUT/$L.sqlite" ] && python3 "$HERE/step_table.py" "$OUT/$L.sqlite" > "$OUT/$L.table.txt" && cat "$OUT/$L.table.txt"
done

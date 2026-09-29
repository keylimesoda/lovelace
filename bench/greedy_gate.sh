#!/usr/bin/env bash
# Greedy e2e token-identity gate: the production server from BIN_A and from BIN_B, one GPU-lock hold each, the same
# greedy requests (sequential, then all concurrent), then bench/greedy_check.py compare.
#   bench/greedy_gate.sh OUT_DIR BIN_A BIN_B [DEPTHS] [-- extra server args for B]
# A side that already has results in OUT_DIR is not rerun. NO_SPEC=1 serves without MTP (top_logprobs then cover
# every token, so the compare reports the baseline's top-1/top-2 margin at each divergence).
# DEPTHS default: six distinct sessions at ~20k (a ~120k-cell unified cache with six live sequences).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1") A=$(realpath "$2") B=$(realpath "$3"); shift 3
DEPTHS=20000,20000,20000,20000,20000,20000
if (($#)) && [[ $1 != -- ]]; then DEPTHS=$1; shift; fi
(($#)) && shift
EXTRA=("$@")
mkdir -p "$OUT"
for side in A B; do
  BIN=${!side}
  [[ -f $OUT/$side.par.json ]] && continue   # already measured (rerun after an interrupted hold)
  SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3 -ctkd q5_0 -ctvd q5_0"
  [[ ${NO_SPEC:-0} = 1 ]] && SPEC_ARGS=""
  export SPEC_ARGS
  "$HERE/../deploy/gpu-lock.sh" bash -c '
    set -uo pipefail
    BIN=$1 OUT=$2 SIDE=$3 DEPTHS=$4 HERE=$5; shift 5
    PORT=18500 C=greedy-gate
    docker rm -f $C >/dev/null 2>&1
    docker run -d --name $C --gpus all --network host --ulimit memlock=-1:-1 -v "$BIN:/e" -v "$HOME/models:/models:ro" \
      lovelace-engine-runtime:cuda13.1 /e/llama-server -m /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf -ngl 99 \
      -np 6 --kv-unified -c 327680 -ctk q5_0 -ctv q5_0 $SPEC_ARGS \
      -fa on -ub 1024 -b 2048 --jinja --reasoning-preserve -ctxcp 8 --kv-budget --prefill-chunk-gen 1024 \
      --alias qwen3.8-27b --host 127.0.0.1 --port $PORT "$@" >/dev/null
    ok=0
    for i in $(seq 300); do
      [ "$(docker inspect -f {{.State.Running}} $C 2>/dev/null)" = true ] || break
      curl -sf 127.0.0.1:$PORT/health >/dev/null && { ok=1; break; }; sleep 2
    done
    if [ $ok = 1 ]; then
      cd "$HERE/replay"
      uv run python ../greedy_check.py run --base-url http://127.0.0.1:$PORT/v1 --depths "$DEPTHS" --sequential \
        --out "$OUT/$SIDE.seq.json" > "$OUT/$SIDE.seq.out" 2>&1
      uv run python ../greedy_check.py run --base-url http://127.0.0.1:$PORT/v1 --depths "$DEPTHS" \
        --out "$OUT/$SIDE.par.json" > "$OUT/$SIDE.par.out" 2>&1
    fi
    docker logs $C > "$OUT/$SIDE.server.log" 2>&1
    docker rm -f $C >/dev/null
    [ $ok = 1 ]
  ' _ "$BIN" "$OUT" "$side" "$DEPTHS" "$HERE" $([[ $side = B ]] && printf '%s\n' "${EXTRA[@]}")
done
cd "$HERE/replay"
for mode in seq par; do
  echo "== $mode"; uv run python ../greedy_check.py compare "$OUT/A.$mode.json" "$OUT/B.$mode.json"
done | tee "$OUT/COMPARE.txt"

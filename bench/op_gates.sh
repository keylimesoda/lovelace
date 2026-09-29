#!/usr/bin/env bash
# Correctness gates for a build, in one GPU-lock hold: test-backend-ops (test mode) for the given ops,
# test-gdn-replay and test-kq-mask (when built). Output in OUT_FILE; exit status non-zero if any gate fails.
#   bench/op_gates.sh OUT_FILE BIN_DIR [OP ...]      (default ops: MUL_MAT TOP_K FLASH_ATTN_EXT)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1") BIN=$(realpath "$2"); shift 2
OPS=("$@"); ((${#OPS[@]})) || OPS=(MUL_MAT TOP_K FLASH_ATTN_EXT)
mkdir -p "$(dirname "$OUT")"
"$HERE/../deploy/gpu-lock.sh" bash -c '
  OUT=$1 BIN=$2; shift 2
  rc=0
  : > "$OUT"
  run() { docker run --rm --gpus all -v "$BIN:/e" lovelace-engine-runtime:cuda13.1 "$@"; }
  for op in "$@"; do
    echo "== test-backend-ops -o $op" >> "$OUT"
    run /e/test-backend-ops test -o "$op" -b CUDA0 > /tmp/op_gates.$$ 2>&1 || rc=1
    grep -E "FAIL|passed|tests passed|Backend" /tmp/op_gates.$$ | grep -v "OK$" | tail -20 >> "$OUT"
  done
  echo "== test-gdn-replay" >> "$OUT"
  run /e/test-gdn-replay > /tmp/op_gates.$$ 2>&1 || rc=1
  tail -5 /tmp/op_gates.$$ >> "$OUT"
  if [ -x "$BIN/test-kq-mask" ]; then
    echo "== test-kq-mask" >> "$OUT"
    run /e/test-kq-mask > /tmp/op_gates.$$ 2>&1 || rc=1
    cat /tmp/op_gates.$$ >> "$OUT"
  fi
  rm -f /tmp/op_gates.$$
  echo "exit $rc" >> "$OUT"
  exit $rc
' _ "$OUT" "$BIN" "${OPS[@]}"

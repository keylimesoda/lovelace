#!/usr/bin/env bash
# test-backend-ops perf for one or more library builds in one GPU-lock hold, with the same test binary.
#   bench/op_perf.sh OUT_FILE TEST_BIN_DIR LIB_DIR[,LIB_DIR...] -- test-backend-ops args (e.g. perf -o MUL_MAT -p 5120)
# Each LIB_DIR supplies libggml*.so (LD_LIBRARY_PATH), so a new test case can time an old and a new kernel build.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1") TB=$(realpath "$2") LIBS=$3; shift 3
[[ ${1:-} == -- ]] && shift
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
MOUNTS=(-v "$TB:/t")
i=0; LDS=()
IFS=, read -ra DIRS <<< "$LIBS"
for d in "${DIRS[@]}"; do MOUNTS+=(-v "$(realpath "$d"):/l$i"); LDS+=("/l$i"); i=$((i+1)); done
"$HERE/../deploy/gpu-lock.sh" bash -c '
  OUT=$1; shift; n=$1; shift; LDS=("${@:1:$n}"); shift $n
  MOUNTS=(); while [[ $1 != -- ]]; do MOUNTS+=("$1"); shift; done; shift
  for l in "${LDS[@]}"; do
    echo "== libs $l" >> "$OUT"
    docker run --rm --gpus all "${MOUNTS[@]}" -e LD_LIBRARY_PATH=$l lovelace-engine-runtime:cuda13.1 /t/test-backend-ops "$@" -b CUDA0 >> "$OUT" 2>&1
  done
' _ "$OUT" "${#LDS[@]}" "${LDS[@]}" "${MOUNTS[@]}" -- "$@"

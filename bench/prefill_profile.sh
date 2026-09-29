#!/usr/bin/env bash
# Kernel-class profile of one prefill ubatch (1024 tokens) at several KV depths, production KV flags (q5_0, FA, -ub 1024).
# One nsys run of llama-bench per depth (one GPU-lock hold each); the last ubatch of the run is the measured one.
#   bench/prefill_profile.sh [BIN_DIR] [OUT_DIR] [DEPTHS]
#     BIN_DIR default engine/build-sm89/bin, OUT_DIR default bench/results/prefill-profile, DEPTHS default 0,65536,131072
# Writes OUT_DIR/pp-d<depth>.nsys-rep, OUT_DIR/pp-d<depth>.trace.csv and OUT_DIR/REPORT.md (table per kernel class).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$(realpath "${1:-$HERE/../engine/build-sm89/bin}")
OUT=$(realpath -m "${2:-$HERE/results/prefill-profile}")
DEPTHS=${3:-0,65536,131072}
MODEL=${MODEL:-/models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf}
mkdir -p "$OUT"
for D in ${DEPTHS//,/ }; do
  "$HERE/../deploy/gpu-lock.sh" docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -e LD_LIBRARY_PATH=/e -v "$BIN:/e:ro" -v "$HOME/models:/models:ro" \
    -v "$OUT:/out" lovelace-engine-devel:cuda13.1 bash -c "
      nsys profile -f true -t cuda --cuda-graph-trace=node -o /out/pp-d$D \
        /e/llama-bench -m $MODEL -ngl 99 -fa 1 -ub 1024 -b 1024 -ctk q5_0 -ctv q5_0 -p 1024 -n 0 -d $D -r 1 --no-warmup \
        > /out/pp-d$D.bench.txt 2>&1 &&
      nsys stats -r cuda_gpu_trace -f csv -o /out/pp-d$D /out/pp-d$D.nsys-rep > /dev/null &&
      mv /out/pp-d${D}_cuda_gpu_trace.csv /out/pp-d$D.trace.csv"
done
python3 "$HERE/prefill_profile/classify.py" "$OUT" ${DEPTHS//,/ } > "$OUT/REPORT.md"
cat "$OUT/REPORT.md"

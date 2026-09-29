#!/usr/bin/env bash
# Byte-exact restore gate of the session store (engine/tests/test-session-restore.cpp): a session mirrored in two
# deltas is parked and restored into other cells, then demoted to disk and restored by a fresh store; its KV (all
# layers, K and V) and recurrent state must match the pre-park state byte for byte. One GPU-lock hold.
#   bench/session_restore_check.sh [BIN_DIR=engine/build-sm89/bin] [MODEL=byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$(realpath "${1:-$HERE/../engine/build-sm89/bin}")
MODEL=${2:-byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf}
exec "$HERE/../deploy/gpu-lock.sh" docker run --rm --name session-restore-check --gpus all --ulimit memlock=-1:-1 \
  -v "$BIN:/e" -v "$HOME/models:/models:ro" lovelace-engine-runtime:cuda13.1 \
  /e/test-session-restore "/models/$MODEL" /tmp/session-restore-check

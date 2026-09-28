#!/usr/bin/env bash
# Hold the lovelace GPU mutex while running a command. Every GPU job on this box goes through here, so concurrent
# agents and benchmarks never overlap. Hold it for one bounded step (<= ~30 min for interactive work).
# Usage: deploy/gpu-lock.sh <command...>
# env: GPU_LOCK (default /tmp/lovelace-gpu.lock)
#      GPU_LOCK_LOW=1  low priority: acquire only when no other job holds or waits for the lock (flock is not FIFO,
#                      so long batch jobs would otherwise starve interactive ones)
set -euo pipefail
LOCK=${GPU_LOCK:-/tmp/lovelace-gpu.lock}
if [[ ${GPU_LOCK_LOW:-0} == 1 ]]; then
  others() { pgrep -fc "^flock $LOCK" || true; }
  while true; do
    if (( $(others) == 0 )); then
      exec 9>"$LOCK"
      if flock -n 9; then exec "$@"; fi
      exec 9>&-
    fi
    sleep 20
  done
fi
exec flock "$LOCK" "$@"

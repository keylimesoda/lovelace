#!/usr/bin/env bash
# Exit (and so notify whoever launched this in the background) when the GPU mutex has been held while the GPU sat
# idle for IDLE_MIN minutes: a hung or waiting job is blocking the queue. Prints the holder and the queue.
# Usage: deploy/gpu-watchdog.sh     env: IDLE_MIN (default 3), UTIL_MAX (default 5 %)
set -uo pipefail
IDLE_MIN=${IDLE_MIN:-3}
UTIL_MAX=${UTIL_MAX:-5}
idle=0
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1)
  held=$(pgrep -fc 'deploy/gpu-lock.sh' || true)
  if (( held > 0 && util <= UTIL_MAX )); then idle=$((idle + 1)); else idle=0; fi
  if (( idle >= IDLE_MIN * 6 )); then
    echo "GPU idle (${util}% util) for ${IDLE_MIN}+ min while the lock is held/queued at $(date +%T)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    ps -eo pid,etime,args | grep -E 'deploy/gpu-lock.sh' | grep -v grep | cut -c1-160
    docker ps --format '{{.Names}} {{.Status}}' | grep -v -E 'searx|ninfer-build'
    exit 0
  fi
  sleep 10
done

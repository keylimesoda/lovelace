#!/usr/bin/env bash
# Hold the lovelace GPU mutex while running a command. Every GPU job on this box goes through here, so concurrent
# agents and benchmarks never overlap. Hold it for one bounded step (<= ~30 min for interactive work).
# Jobs run in ARRIVAL ORDER (ticket queue): a job waits until its ticket is the oldest live ticket, then takes the
# flock. Tickets of dead processes are reaped, so a killed waiter never blocks the queue.
# Usage: deploy/gpu-lock.sh <command...>
# env: GPU_LOCK (default /tmp/lovelace-gpu.lock), GPU_QUEUE (default /tmp/lovelace-gpu.q)
set -euo pipefail
LOCK=${GPU_LOCK:-/tmp/lovelace-gpu.lock}
QUEUE=${GPU_QUEUE:-/tmp/lovelace-gpu.q}
mkdir -p "$QUEUE"
ticket="$QUEUE/$(date +%s%N)-$$"
printf '%s\n' "$*" > "$ticket"
trap 'rm -f "$ticket"' EXIT

head_ticket() { # oldest ticket whose owner is alive; reap the rest
  local t pid
  for t in $(ls "$QUEUE" | sort); do
    pid=${t##*-}
    if kill -0 "$pid" 2>/dev/null; then echo "$QUEUE/$t"; return; fi
    rm -f "$QUEUE/$t"
  done
}

while [[ "$(head_ticket)" != "$ticket" ]]; do sleep 2; done
exec 9>"$LOCK"
flock 9
rm -f "$ticket"   # dequeue once we hold the lock; the next ticket waits on the flock
trap - EXIT

# Keep holding the lock until the GPU is back to its idle baseline, so a detached container started inside this
# hold (docker run -d) can't leak GPU memory into the next job. Baseline: desktop/compositor usage (~1 GiB).
idle_mib=${GPU_IDLE_MIB:-2048}
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1; }
rc=0; "$@" || rc=$?
waited=0
while (( $(used) > idle_mib )); do
  (( waited == 0 )) && echo "gpu-lock: command exited but $(used) MiB still in use; holding the lock until the GPU is idle" >&2
  (( waited >= 1800 )) && { echo "gpu-lock: GPU still busy after 30 min; releasing anyway" >&2; break; }
  sleep 5; waited=$((waited + 5))
done
exit "$rc"

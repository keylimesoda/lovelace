#!/usr/bin/env bash
# Hold the lovelace GPU mutex while running a command. Every GPU job on this box goes through here, so concurrent
# agents and benchmarks never overlap. Hold it for one bounded step (<= ~30 min for interactive work).
# Usage: deploy/gpu-lock.sh <command...>     env: GPU_LOCK (default /tmp/lovelace-gpu.lock)
set -euo pipefail
exec flock "${GPU_LOCK:-/tmp/lovelace-gpu.lock}" "$@"

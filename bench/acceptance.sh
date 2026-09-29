#!/usr/bin/env bash
# Charter acceptance, end to end, scripts only (docs/CHARTER.md). Re-runnable: each step skips work whose output exists.
#   bench/acceptance.sh            run every missing measurement, then write bench/results/acceptance/REPORT.md
#   bench/acceptance.sh report     only regenerate the report
# Production (deploy/serve.sh) must be stopped: every GPU step takes the GPU lock.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
O=$ROOT/bench/results/h2h
cd "$ROOT"
if [[ ${1:-} != report ]]; then
  # A1: quality. lovelace production config vs the Q8_0 reference (ninfer's row comes from eval/kld/ninfer-score, see README)
  (cd eval/kld && ./run_kld.sh >/dev/null)
  # B1: single-stream depth sweep, same prompts
  [[ -s $O/lovelace-sweep.jsonl ]] || bench/h2h.sh lovelace 6 sweep "$O" lovelace-sweep
  [[ -s $O/ninfer-sweep.jsonl ]]   || bench/h2h.sh ninfer 1 sweep "$O" ninfer-sweep
  # B2: 6-agent replay at real depth, 20 min, both engines (ninfer at 6 and 3 lanes; the best one counts)
  [[ -s $O/lovelace-6a/requests.jsonl ]]       || bench/h2h.sh lovelace 6 replay:6:60000:1200 "$O" lovelace-6a
  [[ -s $O/ninfer-6lane-6a/requests.jsonl ]]   || bench/h2h.sh ninfer 6 replay:6:60000:1200 "$O" ninfer-6lane-6a
  [[ -s $O/ninfer-3lane-6a/requests.jsonl ]]   || bench/h2h.sh ninfer 3 replay:6:60000:1200 "$O" ninfer-3lane-6a
fi
python3 bench/acceptance_report.py

#!/usr/bin/env bash
# omarchy-ninfer widget data path against lovelace: runs the widget's own helper (`ninfer-qwen info`, unmodified) and
# feeds its output through the widget's Model.js (bench/widget_check.mjs). No GPU of its own: the caller arranges a
# running container named ninfer-qwen serving the API on 127.0.0.1:8080 (bench/obs_parity.sh does), with traffic
# already served (prefix cache, draft acceptance, generated tokens, device KV and host spill must be live).
#   bench/widget_check.sh OUT_DIR
# env: WIDGET_DIR (default ~/Work/omarchy-ninfer), SAMPLE_GAP seconds between the two feed samples the widget's
#      Generation / Prefill rates are derived from (default 3)
# Writes OUT_DIR/{info-1.txt,info-2.txt,labels.md,widget-repo.txt}; exit 1 when a check fails or Model.js /
# ninfer-qwen changed while it ran.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "${1:?OUT_DIR}")
W=${WIDGET_DIR:-$HOME/Work/omarchy-ninfer}
GAP=${SAMPLE_GAP:-3}
mkdir -p "$OUT"
[[ $(docker container inspect -f '{{.State.Running}}' ninfer-qwen 2>/dev/null) == true ]] ||
  { echo "widget_check: no running container named ninfer-qwen" >&2; exit 1; }
curl -sf --max-time 5 127.0.0.1:8080/v1/models >/dev/null ||
  { echo "widget_check: 127.0.0.1:8080/v1/models does not answer" >&2; exit 1; }

repo() {
  echo "HEAD $(git -C "$W" rev-parse HEAD)"
  echo "git status --porcelain:"; git -C "$W" status --porcelain | sed 's/^/  /'
  (cd "$W" && sha256sum Model.js ninfer-qwen)
}
before=$(repo)
t1=$(date +%s.%N); "$W/ninfer-qwen" info > "$OUT/info-1.txt"
sleep "$GAP"
t2=$(date +%s.%N); "$W/ninfer-qwen" info > "$OUT/info-2.txt"
after=$(repo)
dt=$(awk -v a="$t1" -v b="$t2" 'BEGIN { printf "%.3f", b - a }')
{ echo "$before"; [[ $before == "$after" ]] && echo "unchanged during the check" || { echo "CHANGED during the check:"; echo "$after"; }; } \
  > "$OUT/widget-repo.txt"
rc=0
WIDGET_DIR=$W bun "$HERE/widget_check.mjs" "$OUT/info-2.txt" "$OUT/info-1.txt" "$dt" > "$OUT/labels.md" || rc=$?
{ echo; echo "## omarchy-ninfer checkout"; echo; sed 's/^/    /' "$OUT/widget-repo.txt"; } >> "$OUT/labels.md"
cat "$OUT/labels.md"
[[ $before == "$after" ]] || { echo "widget_check: Model.js / ninfer-qwen changed while the check ran" >&2; rc=1; }
exit "$rc"

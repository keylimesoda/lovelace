#!/usr/bin/env bash
# Lost wake-up repro (fix/sched-wakeup), one GPU-lock hold: the balanced profile with -np 3 (-c 376832 q5_0, --kv-budget,
# --session-ram 24576, MTP n=3, draft vocab), 3 replay agents whose first contacts are ~200k tokens each
# (replay.py --warmup --start-turn-min-depth 200000). One first contact fits the cells at a time, so every other one is
# admitted by parking a finished session, whose mirror sync is in flight when its slot goes idle.
# A 1 s telemetry sampler records stalls: no slot busy while requests wait. A stall longer than PROBE_S is ended by one
# tiny unrelated request (what unstuck production), so that a hanging build still completes; the probe is recorded.
#   bench/wakeup_repro.sh BIN_DIR OUT_DIR LABEL
# Pass: every first contact completes, no stall longer than 30 s, no probe needed. Writes OUT_DIR/LABEL/{RESULT.md,
# samples.jsonl,warmup.jsonl,requests.jsonl,server.log,telemetry-end.json}.
# env: PROBE_S (default 90), DURATION (timed window after the warm-up, default 30), ENGINE_MEM (default 34g)
set -euo pipefail
BIN=$(realpath "$1"); OUT=$(realpath -m "$2"); LABEL=$3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/$LABEL"
exec "$HERE/../deploy/gpu-lock.sh" bash -c '
set -uo pipefail
BIN=$1 OUT=$2 LABEL=$3 HERE=$4
R=$OUT/$LABEL PORT=18500 C=wakeup-repro PROBE_S=${PROBE_S:-90}
SD=$HOME/.cache/lovelace-bench/wakeup-sessions; rm -rf "$SD"; mkdir -p "$SD"   # NVMe, never /tmp (tmpfs is RAM)
mapfile -t A < <(sed -e "s/\${CTX}/376832/" -e "s/^-np .*/-np 3/" -e "s#/var/lib/lovelace/sessions#/sessions#" \
    -e "s#/var/log/lovelace/#/out/server-#" -e "s/^--port .*/--port $PORT/" "$HERE/../deploy/profiles/balanced.args" |
  grep -v "^--web-dir" | tr " " "\n")
docker rm -f $C >/dev/null 2>&1
docker run -d --name $C --gpus all --network host --user "$(id -u):$(id -g)" --ulimit memlock=-1:-1 \
  --memory ${ENGINE_MEM:-34g} --memory-swap ${ENGINE_MEM:-34g} \
  -v "$BIN:/e" -v "$HOME/models:/models:ro" -v "$SD:/sessions" -v "$R:/out" \
  -v "$HERE/../deploy/draft-vocab:/opt/lovelace/draft-vocab:ro" lovelace-engine-runtime:cuda13.1 /e/llama-server "${A[@]}" >/dev/null
for i in $(seq 200); do curl -sf 127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done

# sampler: one line per second; a stall longer than PROBE_S gets one tiny request
python3 - "$PORT" "$PROBE_S" > "$R/samples.jsonl" 2> "$R/sampler.err" <<"PY" &
import json, sys, time, urllib.request
port, probe_s = sys.argv[1], float(sys.argv[2])
base = f"http://127.0.0.1:{port}"
t0 = time.time(); stall_t0 = None
while True:
    try:
        d = json.load(urllib.request.urlopen(base + "/telemetry", timeout=10))
        s = d.get("scheduler", {}); k = s.get("kv_budget", {})
        rec = {"t": round(time.time() - t0, 1), "busy": s.get("slots_busy", 0), "waiting": k.get("queue_waiting", 0),
               "parks": k.get("totals", {}).get("parks", 0), "cells_used": d.get("memory", {}).get("kv", {}).get("cells_used")}
    except Exception as e:
        rec = {"t": round(time.time() - t0, 1), "error": str(e)[:200]}
    stalled = rec.get("waiting", 0) > 0 and rec.get("busy", 1) == 0
    stall_t0 = (stall_t0 if stall_t0 is not None else rec["t"]) if stalled else None
    if stall_t0 is not None and rec["t"] - stall_t0 >= probe_s:
        rec["probe"] = True
        body = json.dumps({"model": "qwen3.8-27b", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", body,
                                                          {"Content-Type": "application/json"}), timeout=5).read()
        except Exception as e:
            rec["probe_error"] = str(e)[:200]
        stall_t0 = None
    print(json.dumps(rec), flush=True)
    time.sleep(1)
PY
SAMPLER=$!

(cd "$HERE/replay" && uv run replay.py --base-url http://127.0.0.1:$PORT/v1 --model qwen3.8-27b --warmup --agents 3 \
   --start-turn-min-depth 200000 --duration "${DURATION:-30}" --no-drain --think-scale 0.25 --max-think 20 \
   --output-tokens recorded --temperature 1.0 --extra-body "{\"top_p\":0.95,\"top_k\":20}" \
   --results-dir "$OUT" --label "$LABEL") > "$R/replay.out" 2>&1
kill $SAMPLER 2>/dev/null
curl -s 127.0.0.1:$PORT/telemetry > "$R/telemetry-end.json"
docker logs $C > "$R/server.log" 2>&1
docker rm -f $C >/dev/null
rm -rf "$SD"

python3 - "$R" <<"PY" | tee "$R/RESULT.md"
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
samples = [json.loads(l) for l in (r / "samples.jsonl").read_text().splitlines() if l.strip()]
warm = [json.loads(l) for l in (r / "warmup.jsonl").read_text().splitlines() if l.strip()] if (r / "warmup.jsonl").exists() else []
stalls, cur = [], None
for s in samples:
    if s.get("waiting", 0) > 0 and s.get("busy", 1) == 0:
        cur = cur or [s["t"], s["t"]]; cur[1] = s["t"]
    elif cur:
        stalls.append(cur); cur = None
if cur:
    stalls.append(cur)
longest = max((b - a + 1 for a, b in stalls), default=0)
probes = sum(1 for s in samples if s.get("probe"))
ok = sum(1 for w in warm if w.get("status") == "ok")
passed = len(warm) == 3 and ok == 3 and longest <= 30 and probes == 0
prompts = [w.get("prompt_n") for w in warm]
parks = samples[-1].get("parks") if samples else None
verdict = "PASS" if passed else "FAIL"
print(f"# wakeup repro: {r.name}\n")
print(f"- first contacts: {ok}/{len(warm)} ok, prompt tokens {prompts}")
print(f"- stalls (no slot busy while requests wait): {len(stalls)}, longest {longest:.0f} s, probes sent {probes}")
print(f"- parks at the end: {parks}, samples {len(samples)}")
print(f"- result: {verdict}")
PY
' _ "$BIN" "$OUT" "$LABEL" "$HERE"

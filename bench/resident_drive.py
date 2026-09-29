"""Cold prefills against a running llama-server, one after the other: every prompt but the last stays resident in the
unified KV cache, the last one is the measured prefill (used by bench/resident_profile.sh).

For each target depth it picks a recorded OMP turn from a distinct session (bench/step_drive.py), sends it with
max_tokens=1 and prints the server's prompt timings. --on-start/--on-stop run around the last request only.
Afterwards it prints the server's slot table (cached tokens per slot) so a run shows whether the earlier sessions
stayed resident.

Usage (from bench/replay): uv run python ../resident_drive.py --base-url http://127.0.0.1:18410/v1 --depths 150000,128000
         [--on-start CMD] [--on-stop CMD]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_drive import body, pick_distinct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--depths", required=True, help="comma-separated prompt depths; the last one is measured")
    ap.add_argument("--on-start", default=None)
    ap.add_argument("--on-stop", default=None)
    a = ap.parse_args()
    picks = pick_distinct([int(x) for x in a.depths.split(",")])
    client = httpx.Client(base_url=a.base_url, timeout=7200)
    for i, (target, depth, name, turn) in enumerate(picks):
        last = i == len(picks) - 1
        if last and a.on_start:
            subprocess.run(a.on_start, shell=True, check=True)
        t0 = time.monotonic()
        r = client.post("/chat/completions", json=body(turn, 1, {"temperature": 0, "seed": 1000 + i}, False))
        wall = time.monotonic() - t0
        if last and a.on_stop:
            subprocess.run(a.on_stop, shell=True, check=True)
        r.raise_for_status()
        tm = r.json().get("timings") or {}
        print(json.dumps({"req": i, "measured": last, "target": target, "session": name,
                          "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                          "prompt_ms": tm.get("prompt_ms"), "prompt_tps": tm.get("prompt_per_second"),
                          "wall_s": round(wall, 1)}), flush=True)
    root = a.base_url.rsplit("/v1", 1)[0]
    try:
        slots = httpx.get(f"{root}/slots", timeout=30).json()
        print(json.dumps({"slots": [{"id": s.get("id"), "n_ctx_used": s.get("n_ctx_used"),
                                     "is_processing": s.get("is_processing")} for s in slots]}), flush=True)
    except Exception as e:  # noqa: BLE001 - informational only
        print(json.dumps({"slots_error": str(e)}), flush=True)


if __name__ == "__main__":
    main()

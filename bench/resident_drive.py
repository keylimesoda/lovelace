"""Cold prefills against a running llama-server, one after the other: every prompt but the last stays resident in the
unified KV cache, the last one is the measured prefill (used by bench/resident_profile.sh).

For each target depth it picks a recorded OMP turn from a distinct session (bench/step_drive.py), sends it with
max_tokens=1 and prints the server's prompt timings. --on-start/--on-stop run around the last request only.
With --decode-during, the first session generates (streaming, ignore_eos) while the last prompt is prefilled, so
the measured prefill shares its ubatches with another session's decode (the server batches both).

Usage (from bench/replay): uv run python ../resident_drive.py --base-url http://127.0.0.1:18410/v1 --depths 150000,128000
         [--decode-during] [--on-start CMD] [--on-stop CMD]
"""

import argparse
import json
import subprocess
import sys
import threading
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
    ap.add_argument("--decode-during", action="store_true", help="the first session decodes during the measured prefill")
    a = ap.parse_args()
    picks = pick_distinct([int(x) for x in a.depths.split(",")])
    client = httpx.Client(base_url=a.base_url, timeout=7200)
    stop = threading.Event()
    decoded = [0]

    def decode(turn):
        b = body(turn, 4096, {"temperature": 0, "seed": 1000, "ignore_eos": True}, True)
        with httpx.Client(base_url=a.base_url, timeout=7200) as c, c.stream("POST", "/chat/completions", json=b) as resp:
            for line in resp.iter_lines():
                if stop.is_set():
                    break
                if line.startswith("data: ") and line != "data: [DONE]" and json.loads(line[6:]).get("choices"):
                    decoded[0] += 1

    for i, (target, depth, name, turn) in enumerate(picks):
        last = i == len(picks) - 1
        dec = None
        if last and a.decode_during:
            dec = threading.Thread(target=decode, args=(picks[0][3],), daemon=True)
            dec.start()
            while decoded[0] < 16 and dec.is_alive():
                time.sleep(0.05)
        if last and a.on_start:
            subprocess.run(a.on_start, shell=True, check=True)
        t0, d0 = time.monotonic(), decoded[0]
        r = client.post("/chat/completions", json=body(turn, 1, {"temperature": 0, "seed": 1000 + i}, False))
        wall = time.monotonic() - t0
        d1 = decoded[0]
        if last and a.on_stop:
            subprocess.run(a.on_stop, shell=True, check=True)
        if dec:
            stop.set()
            dec.join()
        r.raise_for_status()
        tm = r.json().get("timings") or {}
        print(json.dumps({"req": i, "measured": last, "target": target, "session": name,
                          "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                          "prompt_ms": tm.get("prompt_ms"), "prompt_tps": tm.get("prompt_per_second"),
                          "wall_s": round(wall, 1),
                          **({"concurrent_decode_events": d1 - d0,
                              "concurrent_decode_eps": round((d1 - d0) / wall, 2)} if dec else {})}), flush=True)


if __name__ == "__main__":
    main()

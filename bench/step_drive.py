"""Drive N concurrent decodes at fixed depths for a step profile (used by bench/step_profile.sh).

1. For each target depth, picks the recorded OMP turn closest to it, one distinct session per target.
2. Prefills each prompt alone (max_tokens=1), so every slot holds its prompt.
3. Sends all prompts concurrently (streaming, ignore_eos, production sampling, fixed seeds). Once every stream has
   produced --warm-tokens tokens it runs --on-start, waits --capture-s, runs --on-stop, then lets the requests finish
   and prints per-request decode rate and draft acceptance.

Usage (from bench/replay): uv run python ../step_drive.py --base-url http://127.0.0.1:18400/v1 --depths 52000,52000
         [--max-tokens 256] [--warm-tokens 24] [--capture-s 2] [--on-start CMD] [--on-stop CMD] [--greedy]
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent / "replay"))
from corpusfmt import CORPUS_DIR, iter_turns, load_tools  # noqa: E402

TOOLS = load_tools(CORPUS_DIR)


def pick_distinct(depths):
    """For each target depth, the turn closest to it from a session not used by an earlier target."""
    by_session = {}
    for f in sorted((CORPUS_DIR / "sessions").glob("*.jsonl")):
        by_session[f.name] = [(t.get("rec_prompt_tokens") or t.get("est_prompt_tokens") or 0, t) for t in iter_turns(f)]
    used, picks = set(), []
    for target in depths:
        best = min(((abs(d - target), name, d, t) for name, ts in by_session.items() if name not in used for d, t in ts),
                   key=lambda c: (c[0], c[1]))
        used.add(best[1])
        picks.append((target, best[2], best[1], best[3]))
    return picks


def body(turn, max_tokens, sampling, stream):
    b = {"model": "qwen3.8-27b", "messages": turn["messages"], "max_tokens": max_tokens, "stream": stream, **sampling}
    if turn.get("tools"):
        b["tools"] = [TOOLS[n] for n in turn["tools"] if n in TOOLS]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--depths", required=True, help="comma-separated target prompt depths, one request each")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warm-tokens", type=int, default=24)
    ap.add_argument("--capture-s", type=float, default=2.0)
    ap.add_argument("--on-start", default=None)
    ap.add_argument("--on-stop", default=None)
    ap.add_argument("--greedy", action="store_true")
    a = ap.parse_args()
    sampling = {"temperature": 0} if a.greedy else {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    picks = pick_distinct([int(x) for x in a.depths.split(",")])
    client = httpx.Client(base_url=a.base_url, timeout=3600)

    for i, (target, depth, name, turn) in enumerate(picks):
        t0 = time.monotonic()
        r = client.post("/chat/completions", json=body(turn, 1, {**sampling, "seed": 1000 + i}, False))
        r.raise_for_status()
        u = r.json().get("usage") or {}
        print(json.dumps({"prefill": i, "target": target, "rec_depth": depth, "session": name,
                          "prompt_tokens": u.get("prompt_tokens"), "wall_s": round(time.monotonic() - t0, 1)}), flush=True)

    counts = [0] * len(picks)
    finals = [None] * len(picks)

    def run(i, turn):
        b = body(turn, a.max_tokens, {**sampling, "seed": 1000 + i, "ignore_eos": True}, True)
        b["stream_options"] = {"include_usage": True}
        with client.stream("POST", "/chat/completions", json=b) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                j = json.loads(line[6:])
                if j.get("choices"):
                    counts[i] += 1
                if j.get("timings"):
                    finals[i] = j

    threads = [threading.Thread(target=run, args=(i, p[3]), daemon=True) for i, p in enumerate(picks)]
    for t in threads:
        t.start()
    while min(counts) < a.warm_tokens:
        if not any(t.is_alive() for t in threads):
            sys.exit("requests ended before every stream warmed up")
        time.sleep(0.02)
    if a.on_start:
        subprocess.run(a.on_start, shell=True, check=True)
    c0, t0 = list(counts), time.monotonic()
    time.sleep(a.capture_s)
    c1, t1 = list(counts), time.monotonic()
    if a.on_stop:
        subprocess.run(a.on_stop, shell=True, check=True)
    for t in threads:
        t.join()
    agg = sum(c1) - sum(c0)
    print(json.dumps({"capture_s": round(t1 - t0, 3), "stream_events_in_capture": agg,
                      "approx_aggregate_tps": round(agg / (t1 - t0), 1)}), flush=True)
    for i, f in enumerate(picks):
        tm = (finals[i] or {}).get("timings") or {}
        dn, da = tm.get("draft_n") or 0, tm.get("draft_n_accepted") or 0
        print(json.dumps({"req": i, "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                          "predicted_n": tm.get("predicted_n"), "decode_tps": tm.get("predicted_per_second"),
                          "draft_n": dn, "draft_accepted": da, "acceptance": round(da / dn, 3) if dn else None}),
              flush=True)


if __name__ == "__main__":
    main()

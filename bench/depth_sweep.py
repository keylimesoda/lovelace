"""Single-stream depth sweep against an OpenAI-compatible server, identical prompts for every engine.

For each target depth it picks the recorded OMP turn whose prompt is closest to that depth, sends it cold (prefill
rate), then sends the same prompt again (prefix cached, so the later runs measure decode on their own). It reads the
server's llama.cpp-style `timings` (both lovelace and ninfer report them). Request bodies match replay.py
(messages + the turn's tool schemas).

Usage (from bench/replay): uv run python ../depth_sweep.py --base-url http://127.0.0.1:18000/v1 --label NAME --out DIR
           [--depths 2000,32000,64000,128000,200000] [--max-tokens 512] [--reps 2] [--greedy]
"""

import argparse
import json
import time
from pathlib import Path

import httpx

from corpusfmt import CORPUS_DIR, iter_turns, load_tools

TOOLS = load_tools(CORPUS_DIR)


def pick_turns(depths):
    """For each target depth, the turn whose recorded prompt size is closest: {target: (depth, session, turn)}."""
    cands = []
    for f in sorted((CORPUS_DIR / "sessions").glob("*.jsonl")):
        for t in iter_turns(f):
            d = t.get("rec_prompt_tokens") or t.get("est_prompt_tokens")
            if d:
                cands.append((d, f.name, t))
    return {target: min(cands, key=lambda c: abs(c[0] - target)) for target in depths}


def run_one(client, base, model, turn, max_tokens, sampling):
    body = {"model": model, "messages": turn["messages"], "max_tokens": max_tokens, "stream": False, **sampling}
    if turn.get("tools"):
        body["tools"] = [TOOLS[n] for n in turn["tools"] if n in TOOLS]
    t0 = time.monotonic()
    r = client.post(f"{base}/chat/completions", json=body, timeout=1800)
    wall = time.monotonic() - t0
    r.raise_for_status()
    j = r.json()
    tm = j.get("timings") or {}
    u = j.get("usage") or {}
    return {
        "wall_s": round(wall, 3),
        "prompt_tokens": u.get("prompt_tokens"),
        "cached": tm.get("cache_n", (u.get("prompt_tokens_details") or {}).get("cached_tokens")),
        "prompt_n": tm.get("prompt_n"),
        "prompt_tps": tm.get("prompt_per_second"),
        "predicted_n": tm.get("predicted_n", u.get("completion_tokens")),
        "decode_tps": tm.get("predicted_per_second"),
        "draft_n": tm.get("draft_n"),
        "draft_accepted": tm.get("draft_n_accepted"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depths", default="2000,32000,64000,128000,200000")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=2, help="warm repetitions after the cold one")
    ap.add_argument("--greedy", action="store_true", help="temperature 0 instead of production sampling")
    a = ap.parse_args()
    sampling = {"temperature": 0} if a.greedy else {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1234}
    turns = pick_turns([int(x) for x in a.depths.split(",")])
    rows = []
    with httpx.Client() as client:
        for target, (depth, name, turn) in turns.items():
            for rep in range(a.reps + 1):
                r = run_one(client, a.base_url, a.model, turn, a.max_tokens, sampling)
                r.update({"target": target, "rec_depth": depth, "session": name, "rep": rep})
                rows.append(r)
                print(json.dumps(r), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\n{a.label}: target | prompt | cold prefill tok/s | warm decode tok/s (mean) | draft acceptance (warm)")
    for target in turns:
        rs = [r for r in rows if r["target"] == target]
        cold = rs[0]
        warm = [r for r in rs[1:] if r["decode_tps"]]
        dec = sum(r["decode_tps"] for r in warm) / len(warm) if warm else float("nan")
        dn = sum(r["draft_n"] or 0 for r in warm)
        da = sum(r["draft_accepted"] or 0 for r in warm)
        print(f"{target:>7} | {cold['prompt_tokens']} | {cold['prompt_tps']} | {dec:.1f} | {da / dn if dn else float('nan'):.2f}")


if __name__ == "__main__":
    main()

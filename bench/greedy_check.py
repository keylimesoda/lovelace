"""Greedy end-to-end token-identity check between two engine builds.

run:     sends one greedy request per depth (concurrently, or one at a time with --sequential) and records the generated
         tokens with their top-2 log-probabilities.
compare: for every request, the first position where the two runs differ and the baseline's top-1/top-2 margin there
         (a small margin means a numerically legitimate flip; a large one means a bug).

Usage (from bench/replay):
  uv run python ../greedy_check.py run --base-url URL --depths 20000,20000 --out A.json [--max-tokens 128] [--sequential]
  uv run python ../greedy_check.py compare A.json B.json
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_drive import body, pick_distinct  # noqa: E402


def run(a):
    picks = pick_distinct([int(x) for x in a.depths.split(",")])
    client = httpx.Client(base_url=a.base_url, timeout=3600)
    results = [None] * len(picks)

    def one(i, turn):
        b = body(turn, a.max_tokens, {"temperature": 0, "logprobs": True, "top_logprobs": 2}, False)
        r = client.post("/chat/completions", json=b)
        r.raise_for_status()
        j = r.json()
        toks = []
        for c in (j["choices"][0].get("logprobs") or {}).get("content") or []:
            top = c.get("top_logprobs") or []
            toks.append({"t": c["token"], "lp": c["logprob"],
                         "lp2": top[1]["logprob"] if len(top) > 1 else None})
        results[i] = {"session": picks[i][2], "rec_depth": picks[i][1], "tokens": toks,
                      "timings": j.get("timings")}

    if a.sequential:
        for i, p in enumerate(picks):
            one(i, p[3])
    else:
        ts = [threading.Thread(target=one, args=(i, p[3])) for i, p in enumerate(picks)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    Path(a.out).write_text(json.dumps(results))
    for r in results:
        tm = r["timings"] or {}
        print(json.dumps({"session": r["session"], "n": len(r["tokens"]), "draft_n": tm.get("draft_n"),
                          "draft_accepted": tm.get("draft_n_accepted")}))


def compare(a):
    A, B = json.loads(Path(a.a).read_text()), json.loads(Path(a.b).read_text())
    n_ident = 0
    for ra, rb in zip(A, B):
        ta, tb = ra["tokens"], rb["tokens"]
        k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x["t"] != y["t"]), None)
        if k is None and len(ta) == len(tb):
            n_ident += 1
            print(f"{ra['session']}: identical ({len(ta)} tokens)")
            continue
        k = min(len(ta), len(tb)) if k is None else k
        x = ta[k] if k < len(ta) else None
        margin = (x["lp"] - x["lp2"]) if x and x["lp2"] is not None else None
        print(f"{ra['session']}: diverges at token {k}/{len(ta)}: {x and x['t']!r} vs {tb[k]['t'] if k < len(tb) else None!r}"
              f", baseline top-1/top-2 logprob margin {margin if margin is None else round(margin, 4)}")
    print(f"{n_ident}/{len(A)} identical")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--depths", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=128)
    r.add_argument("--sequential", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()

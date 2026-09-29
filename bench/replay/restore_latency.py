"""Deterministic session-switch driver: N agents take turns one request at a time, greedy, on a KV pool too small to hold
them all, so every turn after the first contact restores a parked session. Writes outputs.jsonl (for the greedy
identity check) and, from the server's request log, a per-phase timing table.

  uv run python restore_latency.py --base-url URL --agents 3 --min-depth 60000 --turns 6 --out DIR
  uv run python restore_latency.py --table DIR/requests.server.jsonl          # table only
  uv run python restore_latency.py --compare DIR_A DIR_B                      # greedy outputs identical?
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from corpusfmt import CORPUS_DIR, iter_turns, load_manifest, load_tools


def depth(turn):
    return turn.get("rec_prompt_tokens") or turn["est_prompt_tokens"]


def pick(n_agents, min_depth, n_turns):
    """The first n_agents sessions (by name) with n_turns turns at or past min_depth."""
    tools = load_tools()
    picked = []
    for s in sorted(load_manifest()["sessions"], key=lambda s: s["name"]):
        turns = [t for t in iter_turns(CORPUS_DIR / s["file"]) if depth(t) >= min_depth]
        if len(turns) >= n_turns:
            picked.append((s["name"], turns[:n_turns]))
        if len(picked) == n_agents:
            return tools, picked
    sys.exit(f"only {len(picked)} sessions have {n_turns} turns past {min_depth} tokens")


def busy(base_url, stop, k):
    """A short-context agent that keeps decoding, so that the switches happen while other slots run."""
    with httpx.Client(timeout=3600) as c:
        while not stop.is_set():
            c.post(base_url + "/chat/completions", json={
                "model": "qwen3.8-27b", "max_tokens": 1024, "temperature": 0.0,
                "messages": [{"role": "user", "content": f"Agent {k}: write a long, detailed essay on the history of computing."}],
            })


def run(args):
    import threading

    tools, picked = pick(args.agents, args.min_depth, args.turns)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    threads = [threading.Thread(target=busy, args=(args.base_url, stop, k), daemon=True) for k in range(args.busy)]
    for th in threads:
        th.start()
    try:
        foreground(args, tools, picked, out)
    finally:
        stop.set()
        for th in threads:
            th.join()


def foreground(args, tools, picked, out):
    with httpx.Client(timeout=3600) as client, open(out / "outputs.jsonl", "w") as f:
        for i in range(args.turns):
            for a, (name, turns) in enumerate(picked):
                t = turns[i]
                body = {
                    "model": "qwen3.8-27b",
                    "messages": t["messages"],
                    "tools": [tools[n] for n in t["tools"] if n in tools],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "top_k": 1,
                    "seed": 0,
                }
                t0 = time.monotonic()
                r = client.post(args.base_url + "/chat/completions", json=body)
                dt = time.monotonic() - t0
                ok = r.status_code == 200
                msg = r.json()["choices"][0]["message"] if ok else {}
                rec = {
                    "agent": a, "session": name, "turn": t["turn"], "status": r.status_code, "e2e_s": round(dt, 3),
                    "content": msg.get("content"), "reasoning": msg.get("reasoning_content"),
                    "tool_calls": msg.get("tool_calls"), "usage": r.json().get("usage") if ok else None,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"agent {a} turn {t['turn']:>4} depth {depth(t):>7} status {r.status_code} e2e {dt:7.2f}s", flush=True)


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def table(path):
    """Median and p90 of each phase over the requests that restored or prefetched a parked session."""
    rows = []
    for line in open(path):
        e = json.loads(line)
        if e.get("event") != "request_done":
            continue
        s = e.get("session") or {}
        tm = e["timings_ms"]
        rows.append({
            "tier": s.get("tier"), "bytes": s.get("restore_bytes", 0), "prefetch": s.get("prefetch_hit", False),
            "queue_ms": tm["queue_wait"], "park_ms": s.get("park_ms"), "issue_ms": s.get("restore_issue_ms"),
            "ready_ms": s.get("restore_ready_ms"), "prompt_start_ms": s.get("prompt_start_ms"),
            "restore_dev_ms": s.get("restore_ms"), "prefill_ms": tm["prefill"],
            "prefill_tokens": e["result"]["n_prefill_tokens"],
            "first_token_after_admit_ms": (tm["ttft"] - tm["queue_wait"]) if tm.get("ttft") is not None else None,
            "error": e["result"]["error"],
        })
    sw = [r for r in rows if r["tier"] in ("ram", "disk") or r["prefetch"]]
    cols = ["queue_ms", "park_ms", "issue_ms", "restore_dev_ms", "ready_ms", "prompt_start_ms", "prefill_ms",
            "prefill_tokens", "first_token_after_admit_ms"]
    lines = [f"requests {len(rows)}, session switches {len(sw)}, prefetch hits {sum(r['prefetch'] for r in sw)}, "
             f"errors {sum(1 for r in rows if r['error'])}, restored GiB p50 {(pct([r['bytes'] for r in sw], 50) or 0) / 2**30:.2f}",
             "", "| phase (switch requests) | p50 | p90 |", "|---|---|---|"]
    for c in cols:
        a, b = pct([r[c] for r in sw], 50), pct([r[c] for r in sw], 90)
        lines.append(f"| {c} | {'-' if a is None else f'{a:.1f}'} | {'-' if b is None else f'{b:.1f}'} |")
    return "\n".join(lines)


def compare(a, b):
    ra = [json.loads(l) for l in open(Path(a) / "outputs.jsonl")]
    rb = [json.loads(l) for l in open(Path(b) / "outputs.jsonl")]
    key = lambda r: (r["content"], r["reasoning"], json.dumps(r["tool_calls"], sort_keys=True))
    diff = [(x["agent"], x["turn"]) for x, y in zip(ra, rb) if key(x) != key(y)]
    print(f"{len(ra)} vs {len(rb)} outputs, {len(diff)} differ: {diff}")
    return 0 if not diff and len(ra) == len(rb) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--min-depth", type=int, default=60000)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--busy", type=int, default=0, help="background agents that keep decoding (0 = deterministic)")
    ap.add_argument("--out")
    ap.add_argument("--table")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.table:
        print(table(a.table))
    elif a.compare:
        sys.exit(compare(*a.compare))
    else:
        run(a)

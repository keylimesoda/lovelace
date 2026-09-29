"""Multi-agent trace replay against an OpenAI-compatible /v1/chat/completions server.

N concurrent agents each walk a distinct recorded session (corpus/sessions/*.jsonl)
turn by turn. History always comes from the recording (never from the server's
generated output), so every engine sees byte-identical prompt prefixes and prefix
caching behaves as it would for the real agents. Between turns an agent sleeps
for the recorded tool/think gap * --think-scale (capped by --max-think).

Example:
  uv run replay.py --base-url http://127.0.0.1:8080/v1 --model qwen3.8-27b \
      --agents 4 --duration 1800 --start-turn-min-depth 30000 --label ninfer-4agents
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from corpusfmt import CORPUS_DIR, HERE, dumps, iter_turns, load_manifest, load_tools, loads

MAX_ERROR_CHARS = 2000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None, help="sent as Bearer token if given")
    ap.add_argument("--agents", type=int, default=4, help="concurrent agents (distinct sessions)")
    stop = ap.add_argument_group("stop condition (default --duration 600)")
    stop.add_argument("--duration", type=float, default=None, help="seconds before agents stop issuing new turns")
    stop.add_argument("--max-turns", type=int, default=None, help="total requests across all agents")
    ap.add_argument("--no-drain", action="store_true", help="cancel in-flight requests at --duration instead of waiting")
    ap.add_argument("--think-scale", type=float, default=0.25, help="multiplier on recorded inter-turn gaps")
    ap.add_argument("--max-think", type=float, default=20.0, help="cap on a single think sleep, seconds")
    ap.add_argument("--stagger", type=float, default=2.0, help="agent i starts after i*STAGGER seconds")
    ap.add_argument(
        "--output-tokens",
        default="recorded",
        help="'recorded' (max_tokens = recorded output tokens, clamped to [64, 4096]) or 'fixed:N'",
    )
    ap.add_argument("--start-turn-min-depth", type=int, default=0, help="skip leading turns shallower than this")
    ap.add_argument("--max-depth", type=int, default=0,
                    help="end a session once its next turn is deeper than this (0 = no cap); keeps N*depth under KV capacity")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--reasoning-effort", default=None, help="passed through as reasoning_effort")
    ap.add_argument("--extra-body", default=None, help="JSON object merged into every request body")
    ap.add_argument("--request-timeout", type=float, default=1800.0, help="per-request deadline incl. queueing")
    ap.add_argument("--warmup", action="store_true",
                    help="run each agent's first contact (its first turn) before the timed window: the window and "
                         "--duration start once every agent has finished it; those turns go to warmup.jsonl, not "
                         "requests.jsonl or the summary's aggregates")
    ap.add_argument("--warmup-timeout", type=float, default=3600.0, help="per-request deadline of the warm-up turns")
    ap.add_argument("--seed", type=int, default=0, help="session order shuffle seed")
    ap.add_argument("--sessions", default=None, help="comma-separated session names to restrict to")
    ap.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    ap.add_argument("--label", default=None, help="results/<label>/ (default: timestamp)")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = ap.parse_args(argv)
    if args.duration is None and args.max_turns is None:
        args.duration = 600.0
    if args.output_tokens != "recorded":
        kind, _, n = args.output_tokens.partition(":")
        if kind != "fixed" or not n.isdigit() or int(n) < 1:
            ap.error("--output-tokens must be 'recorded' or 'fixed:N'")
    args.extra = loads(args.extra_body) if args.extra_body else {}
    if not isinstance(args.extra, dict):
        ap.error("--extra-body must be a JSON object")
    args.label = args.label or datetime.now().strftime("%Y%m%d-%H%M%S")
    return args


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))]


def depth_of(turn: dict) -> int:
    return turn.get("rec_prompt_tokens") or turn["est_prompt_tokens"]


class SessionPool:
    """Hands out sessions so that concurrently running agents never share one."""

    def __init__(self, sessions: list[dict], seed: int):
        self.sessions = sessions
        self.rng = random.Random(seed)
        self.order: list[dict] = []
        self.in_use: set[str] = set()

    def acquire(self) -> dict:
        for _ in range(2):
            if not self.order:
                self.order = self.sessions[:]
                self.rng.shuffle(self.order)
            for i, s in enumerate(self.order):
                if s["name"] not in self.in_use:
                    self.order.pop(i)
                    self.in_use.add(s["name"])
                    return s
            self.order = []
        raise RuntimeError("more agents than sessions")

    def release(self, s: dict) -> None:
        self.in_use.discard(s["name"])


@dataclass
class Run:
    args: argparse.Namespace
    client: httpx.AsyncClient
    tools: dict[str, dict]
    pool: SessionPool
    out: object
    t0: float = field(default_factory=time.monotonic)
    issued: int = 0
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    records: list[dict] = field(default_factory=list)
    started_at: str = ""
    # --warmup: the agents still running their first contact; the timed window (t0) starts when the last one is done
    warm_out: object = None
    warm_pending: set[int] = field(default_factory=set)
    warm_done: asyncio.Event = field(default_factory=asyncio.Event)
    warm_records: list[dict] = field(default_factory=list)
    warm_wall_s: float | None = None

    def warming(self) -> bool:
        return self.warm_out is not None and not self.warm_done.is_set()

    def may_issue(self) -> bool:
        if self.stop.is_set():
            return False
        if self.warming():
            return True
        a = self.args
        if a.duration is not None and time.monotonic() - self.t0 >= a.duration:
            self.stop.set()
            return False
        if a.max_turns is not None and self.issued >= a.max_turns:
            self.stop.set()
            return False
        return True

    def write(self, rec: dict) -> None:
        self.records.append(rec)
        self.out.write(dumps(rec) + b"\n")
        self.out.flush()

    def write_warm(self, rec: dict) -> None:
        self.warm_records.append(rec)
        self.warm_out.write(dumps(rec) + b"\n")
        self.warm_out.flush()

    def warm_finished(self, idx: int) -> None:
        """Agent idx is done with its first contact; the last one starts the timed window."""
        self.warm_pending.discard(idx)
        if not self.warm_pending and not self.warm_done.is_set():
            self.t0 = time.monotonic()
            self.warm_done.set()


def max_tokens_for(turn: dict, spec: str) -> int:
    if spec.startswith("fixed:"):
        return int(spec.split(":", 1)[1])
    n = turn.get("rec_output_tokens") or turn.get("est_output_tokens") or 0
    return max(64, min(n, 4096))


def build_body(run: Run, turn: dict) -> dict:
    a = run.args
    body = {
        "model": a.model,
        "messages": turn["messages"],
        "tools": [run.tools[n] for n in turn["tools"] if n in run.tools],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens_for(turn, a.output_tokens),
        "temperature": a.temperature,
    }
    if a.reasoning_effort:
        body["reasoning_effort"] = a.reasoning_effort
    body.update(a.extra)
    return body


async def no_lines():
    return
    yield


async def send(run: Run, body: bytes, timeout: float | None = None) -> dict:
    """POST one streaming chat completion and measure it. Never raises for server errors."""
    timeout = timeout or run.args.request_timeout
    r: dict = {"status": "ok"}
    t_send = time.monotonic()
    r["t_send"] = round(t_send - run.t0, 4)
    first = last = None
    n_events = 0
    out_chars = reasoning_chars = 0
    tool_names: dict[int, str] = {}
    usage = timings = None
    finish = None
    try:
        async with asyncio.timeout(timeout):
            http_timeout = httpx.Timeout(connect=10.0, read=timeout, write=120.0, pool=None)
            async with run.client.stream("POST", "/chat/completions", content=body, timeout=http_timeout) as resp:
                r["http_status"] = resp.status_code
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    r.update(status="error", error=text[:MAX_ERROR_CHARS])
                lines = resp.aiter_lines() if resp.status_code == 200 else no_lines()
                async for line in lines:
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        ev = loads(data)
                    except ValueError:
                        continue
                    if ev.get("error"):
                        r.update(status="error", error=str(ev["error"])[:MAX_ERROR_CHARS])
                        break
                    if ev.get("usage"):
                        usage = ev["usage"]
                    if ev.get("timings"):
                        timings = ev["timings"]
                    for ch in ev.get("choices") or ():
                        d = ch.get("delta") or {}
                        got = False
                        if c := d.get("content"):
                            out_chars += len(c)
                            got = True
                        if c := d.get("reasoning_content") or d.get("reasoning"):
                            reasoning_chars += len(c)
                            got = True
                        for tc in d.get("tool_calls") or ():
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                tool_names[tc.get("index", 0)] = fn["name"]
                            out_chars += len(fn.get("arguments") or "") + len(fn.get("name") or "")
                            got = True
                        if got:
                            now = time.monotonic()
                            first = first or now
                            last = now
                            n_events += 1
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
    except TimeoutError:
        r.update(status="timeout", error=f"no completion within {timeout}s")
    except asyncio.CancelledError:
        r.update(status="cancelled")
    except httpx.HTTPError as e:
        r.update(status="error", error=f"{type(e).__name__}: {e}"[:MAX_ERROR_CHARS])
    finally:
        t_end = time.monotonic()
        r["t_end"] = round(t_end - run.t0, 4)
        r["e2e_s"] = round(t_end - t_send, 4)
    if r["status"] == "ok" and first is None:
        r.update(status="error", error="stream ended without any tokens")
    r["ttft_s"] = round(first - t_send, 4) if first else None
    r["decode_s"] = round(last - first, 4) if first else None
    r["stream_events"] = n_events
    r["output_chars"] = out_chars
    r["reasoning_chars"] = reasoning_chars
    r["tool_calls"] = [tool_names[k] for k in sorted(tool_names)]
    r["finish_reason"] = finish
    r["usage"] = usage
    r["timings"] = timings
    u = usage or {}
    t = timings or {}
    r["prompt_tokens"] = u.get("prompt_tokens") or (
        (t.get("prompt_n") or 0) + (t.get("cache_n") or 0) if "prompt_n" in t else None
    )
    r["completion_tokens"] = u.get("completion_tokens") or t.get("predicted_n") or (n_events or None)
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    r["cached_tokens"] = cached if cached is not None else t.get("cache_n")
    r["prompt_n"] = t.get("prompt_n")
    r["prompt_ms"] = t.get("prompt_ms")
    r["predicted_n"] = t.get("predicted_n")
    r["predicted_ms"] = t.get("predicted_ms")
    r["draft_n"] = t.get("draft_n")
    r["draft_n_accepted"] = t.get("draft_n_accepted")
    ct = r["completion_tokens"]
    r["decode_tps"] = round((ct - 1) / r["decode_s"], 3) if ct and ct > 1 and r["decode_s"] else None
    if r["prompt_n"] and r["prompt_ms"]:
        r["prefill_tps"] = round(r["prompt_n"] / (r["prompt_ms"] / 1000), 1)
    return r


async def sleep_or_stop(run: Run, seconds: float) -> None:
    if seconds <= 0:
        return
    try:
        await asyncio.wait_for(run.stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


async def agent(run: Run, idx: int) -> None:
    # --warmup: this agent's first contact runs before the timed window, the agents start together
    try:
        await agent_turns(run, idx, run.warming())
    finally:
        if run.warm_out is not None:
            run.warm_finished(idx)  # also when the agent ends early


async def agent_turns(run: Run, idx: int, warm: bool) -> None:
    a = run.args
    if not warm:
        await sleep_or_stop(run, idx * a.stagger)
    while run.may_issue():
        sess = run.pool.acquire()
        try:
            started = False
            for turn in iter_turns(a.corpus / sess["file"]):
                if a.max_depth and depth_of(turn) > a.max_depth:
                    break
                if not started:
                    if depth_of(turn) < a.start_turn_min_depth:
                        continue
                    started = True
                    think = 0.0
                else:
                    think = min((turn.get("gap_s") or 0.0) * a.think_scale, a.max_think)
                    await sleep_or_stop(run, think)
                if not run.may_issue():
                    return
                body = dumps(build_body(run, turn))
                meta = {
                    "agent": idx,
                    "session": sess["name"],
                    "turn": turn["turn"],
                    "n_messages": turn["n_messages"],
                    "est_prompt_tokens": turn["est_prompt_tokens"],
                    "rec_prompt_tokens": turn.get("rec_prompt_tokens"),
                    "rec_output_tokens": turn.get("rec_output_tokens"),
                    "max_tokens": max_tokens_for(turn, a.output_tokens),
                    "think_s": round(think, 3),
                    "body_bytes": len(body),
                }
                if warm:
                    res = await send(run, body, a.warmup_timeout)
                    rec = {**meta, **res, "warmup": True}
                    run.write_warm(rec)
                    print("warm-up " + fmt_line(rec), flush=True)
                    if res["status"] == "cancelled":
                        return
                    warm = False
                    run.warm_finished(idx)
                    await run.warm_done.wait()
                    continue
                run.issued += 1
                res = await send(run, body)
                rec = {**meta, **res}
                run.write(rec)
                print(fmt_line(rec), flush=True)
                if res["status"] == "cancelled":
                    return
        finally:
            run.pool.release(sess)


def fmt_line(r: dict) -> str:
    def f(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    s = (
        f"[{r['t_end']:8.1f}s] a{r['agent']} {r['session'][:36]:<36} t{r['turn']:<4} {r['status']:<9}"
        f" prompt {f(r.get('prompt_tokens') or r['est_prompt_tokens'], '>7')}"
        f" cached {f(r.get('cached_tokens'), '>7')} ttft {f(r.get('ttft_s'), '7.2f')}s"
        f" out {f(r.get('completion_tokens'), '>5')} @ {f(r.get('decode_tps'), '6.1f')} tok/s"
    )
    if r["status"] != "ok" and r.get("error"):
        s += "  " + r["error"][:120].replace("\n", " ")
    return s


def summarize(run: Run, wall_s: float) -> dict:
    recs = run.records
    ok = [r for r in recs if r["status"] == "ok"]
    by_status: dict[str, int] = {}
    for r in recs:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    t_first = min((r["t_send"] for r in recs), default=0.0)
    t_last = max((r["t_end"] for r in recs), default=0.0)
    span = max(t_last - t_first, 1e-9)
    ctoks = sum(r["completion_tokens"] or 0 for r in ok)
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    dec = [r["decode_tps"] for r in ok if r["decode_tps"]]
    depth = [r["prompt_tokens"] or r["est_prompt_tokens"] for r in recs if r["status"] in ("ok", "error")]
    cache_rows = [r for r in ok if r["cached_tokens"] is not None and r["prompt_tokens"]]
    prefill = [r["prefill_tps"] for r in ok if r.get("prefill_tps")]
    drafts = [r for r in ok if r.get("draft_n")]
    per_agent = {}
    for i in sorted({r["agent"] for r in recs}):
        mine = [r for r in recs if r["agent"] == i]
        mine_ok = [r for r in mine if r["status"] == "ok"]
        d = [r["decode_tps"] for r in mine_ok if r["decode_tps"]]
        per_agent[str(i)] = {
            "turns": len(mine),
            "ok": len(mine_ok),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in mine_ok),
            "mean_decode_tps": round(statistics.fmean(d), 2) if d else None,
            "ttft_p50_s": pct([r["ttft_s"] for r in mine_ok if r["ttft_s"] is not None], 0.5),
            "sessions": sorted({r["session"] for r in mine}),
        }
    agent_means = [v["mean_decode_tps"] for v in per_agent.values() if v["mean_decode_tps"]]
    a = run.args
    return {
        "label": a.label,
        "base_url": a.base_url,
        "model": a.model,
        "config": {
            k: v
            for k, v in vars(a).items()
            if k not in ("corpus", "results_dir", "extra_body", "api_key") and not callable(v)
        },
        "started_at": run.started_at,
        "wall_s": round(wall_s, 3),
        "active_span_s": round(span, 3),
        "total_turns": len(recs),
        "by_status": by_status,
        "errors": sum(v for k, v in by_status.items() if k in ("error", "timeout")),
        "warmup": {
            "turns": len(run.warm_records),
            "errors": sum(1 for r in run.warm_records if r["status"] in ("error", "timeout")),
            "wall_s": round(run.warm_wall_s, 3) if run.warm_wall_s is not None else None,
        }
        if run.warm_out is not None
        else None,
        "completion_tokens": ctoks,
        "agg_output_tps": round(ctoks / span, 3),
        "mean_decode_tps_per_agent": round(statistics.fmean(agent_means), 3) if agent_means else None,
        "decode_tps": {"p10": pct(dec, 0.1), "p50": pct(dec, 0.5), "p90": pct(dec, 0.9)},
        "ttft_s": {"p50": pct(ttft, 0.5), "p90": pct(ttft, 0.9), "p99": pct(ttft, 0.99), "max": max(ttft, default=None)},
        "e2e_s": {"p50": pct([r["e2e_s"] for r in ok], 0.5), "p90": pct([r["e2e_s"] for r in ok], 0.9)},
        "prompt_depth_tokens": {"p50": pct(depth, 0.5), "p90": pct(depth, 0.9), "max": max(depth, default=None)},
        "prompt_tokens_total": sum(r["prompt_tokens"] or 0 for r in ok),
        "cache_hit_ratio": round(
            sum(r["cached_tokens"] for r in cache_rows) / sum(r["prompt_tokens"] for r in cache_rows), 4
        )
        if cache_rows
        else None,
        "cache_reported_requests": len(cache_rows),
        "prefill_tps_p50": pct(prefill, 0.5),
        "draft_acceptance": round(sum(r["draft_n_accepted"] or 0 for r in drafts) / sum(r["draft_n"] for r in drafts), 4)
        if drafts
        else None,
        "per_agent": per_agent,
    }


def print_summary(s: dict) -> None:
    def f(v, spec=".2f"):
        return format(v, spec) if isinstance(v, (int, float)) else "n/a"

    rows = [
        ("label", s["label"]),
        ("turns (ok/total)", f"{s['by_status'].get('ok', 0)}/{s['total_turns']}  status={s['by_status']}"),
        ("errors+timeouts", s["errors"]),
        *(
            [("warm-up (turns/errors/wall)", f"{s['warmup']['turns']} / {s['warmup']['errors']} / {f(s['warmup']['wall_s'], '.1f')} s")]
            if s["warmup"]
            else []
        ),
        ("active span", f"{f(s['active_span_s'], '.1f')} s (wall {f(s['wall_s'], '.1f')} s)"),
        ("aggregate output tok/s", f(s["agg_output_tps"])),
        ("mean decode tok/s per agent", f(s["mean_decode_tps_per_agent"])),
        ("decode tok/s p10/p50/p90", " / ".join(f(s["decode_tps"][k]) for k in ("p10", "p50", "p90"))),
        ("TTFT s p50/p90/p99", " / ".join(f(s["ttft_s"][k]) for k in ("p50", "p90", "p99"))),
        ("e2e s p50/p90", " / ".join(f(s["e2e_s"][k]) for k in ("p50", "p90"))),
        ("prompt depth p50/p90/max", " / ".join(f(s["prompt_depth_tokens"][k], "") for k in ("p50", "p90", "max"))),
        ("cache hit ratio", f"{f(s['cache_hit_ratio'], '.3f')} ({s['cache_reported_requests']} reqs reported)"),
        ("prefill tok/s p50", f(s["prefill_tps_p50"], ".0f")),
        ("draft acceptance", f(s["draft_acceptance"], ".3f")),
    ]
    w = max(len(k) for k, _ in rows)
    print("\n" + "=" * 72)
    for k, v in rows:
        print(f"{k:<{w}}  {v}")
    print("-" * 72)
    print(f"{'agent':<6}{'turns':>6}{'ok':>5}{'out tok':>9}{'decode tok/s':>14}{'ttft p50':>10}  sessions")
    for i, v in s["per_agent"].items():
        print(
            f"{i:<6}{v['turns']:>6}{v['ok']:>5}{v['completion_tokens']:>9}{f(v['mean_decode_tps']):>14}"
            f"{f(v['ttft_p50_s']):>10}  {','.join(x[:30] for x in v['sessions'])}"
        )
    print("=" * 72)


async def amain(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.corpus)
    sessions = manifest["sessions"]
    if args.sessions:
        want = set(args.sessions.split(","))
        sessions = [s for s in sessions if s["name"] in want]
    deep = [s for s in sessions if (s.get("rec_depth_tokens") or s["est_depth_tokens"])["max"] >= args.start_turn_min_depth]
    if len(deep) < args.agents:
        print(f"only {len(deep)} sessions reach depth {args.start_turn_min_depth}; need {args.agents}", file=sys.stderr)
        return 2
    out_dir = args.results_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"content-type": "application/json", "accept": "text/event-stream"}
    if args.api_key:
        headers["authorization"] = f"Bearer {args.api_key}"
    timeout = httpx.Timeout(connect=10.0, read=args.request_timeout, write=120.0, pool=None)
    limits = httpx.Limits(max_connections=args.agents + 4, max_keepalive_connections=args.agents + 4)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=headers, timeout=timeout, limits=limits) as client:
        with open(out_dir / "requests.jsonl", "wb") as out, open(out_dir / "warmup.jsonl", "wb") if args.warmup else nullcontext() as warm_out:
            run = Run(args=args, client=client, tools=load_tools(args.corpus), pool=SessionPool(deep, args.seed), out=out)
            run.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
            print(
                f"replay {args.label}: {args.agents} agents over {len(deep)} sessions -> {out_dir}"
                f" (duration={args.duration} max_turns={args.max_turns} warmup={args.warmup})",
                flush=True,
            )
            if warm_out is not None:
                run.warm_out = warm_out
                run.warm_pending = set(range(args.agents))
            tasks = [asyncio.create_task(agent(run, i)) for i in range(args.agents)]
            if warm_out is not None:
                t_warm = time.monotonic()
                await run.warm_done.wait()
                run.warm_wall_s = time.monotonic() - t_warm
                print(f"warm-up done in {run.warm_wall_s:.1f} s, timed window starts", flush=True)
            if args.duration is not None:
                asyncio.get_running_loop().call_later(args.duration, run.stop.set)
            if args.duration is not None and args.no_drain:
                _, pending = await asyncio.wait(tasks, timeout=args.duration)
                for t in pending:
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=args.no_drain)
            wall = time.monotonic() - run.t0
    summary = summarize(run, wall)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_summary(summary)
    return 0


def main() -> None:
    args = parse_args()
    try:
        sys.exit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()

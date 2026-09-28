"""Build long-depth KL-divergence windows from the replay corpus.

A window is one recorded agent turn rendered exactly as the model sees it:

    render(system + tools + messages, add_generation_prompt=True)   <- context (depth)
    + the turn's RECORDED assistant reply (reasoning, content, tool calls, <|im_end|>)  <- scored

rendered with the model's own template (bench/replay/corpus/chat_template.jinja). The
recorded reply of turn i is the assistant message that turn i+1 appends right after turn
i's messages (extract.py stores no separate response field).

Selection targets depth buckets (prompt tokens before the reply). Two modes:

  1. estimate:  `uv run build_corpus.py`
     depth = rendered prompt chars / 3.2 (corpusfmt.CHARS_PER_TOKEN).
  2. exact:     `uv run build_corpus.py --counts counts.jsonl`
     depth = real token counts produced by `lovelace-kld tokcount` over the candidate
     pool this script writes (--pool, default candidates.jsonl). The pool is a pure
     function of the corpus, so the two runs see the same candidates.

Buckets deeper than any real turn (250k: the deepest recorded prompt is ~225k tokens)
are filled with synthetic concatenations: a full, finished main session (ending on a
final assistant answer with no pending tool calls) followed by the history of a
different main session. All main sessions share one system prompt and tool set, so the
concatenation is a valid conversation for the template (one leading system message,
alternating user/assistant/tool turns). Real turns are always preferred over concats.

Output windows.jsonl lines: {id, session, depth_bucket, text, score_start_char, ...meta}.
`score_start_char` is a Python code-point offset into `text`; lovelace-kld tokenizes
text[:start] and text[start:] separately (parse_special) and scores the second part.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent.parent / "bench" / "replay"
sys.path.insert(0, str(REPLAY))
import corpusfmt  # noqa: E402

BUCKETS = {"32k": 32_000, "64k": 64_000, "128k": 128_000, "170k": 170_000, "250k": 250_000}
CONCAT_BUCKETS = {"250k"}
MODEL_CTX = 262_144
MAX_SCORED = 1024
WINDOW_TOKEN_CAP = 262_000  # depth + scored tokens per window (the tool's n_ctx)
POOL_BAND = (0.78, 1.25)  # estimate band (ratio to target) for pool candidates
POOL_PER_SESSION = 6  # evenly spaced candidates per (bucket, session)
POOL_CONCATS = 16
EXACT_TOL = 0.06  # accepted |depth/target - 1| with exact counts
EST_TOL = 0.08  # accepted |depth/target - 1| with estimates
REPLY_CHAR_CAP = 16_384  # >> 1024 tokens; the tool caps scored tokens exactly
MIN_REPLY_CHARS = 200


def _raise(msg):
    raise jinja2.exceptions.TemplateError(msg)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # same output as HF transformers and llama.cpp's jinja tojson (", " / ": ", key order kept)
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def make_env() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    return env


def template_message(m: dict) -> dict:
    """OpenAI message -> template message (tool-call arguments parsed like the server does)."""
    if m["role"] != "assistant" or not m.get("tool_calls"):
        return m
    out = dict(m)
    out["tool_calls"] = [
        {**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"] or "{}")}}
        for c in m["tool_calls"]
    ]
    return out


@dataclass
class Turn:
    session: str
    turn: int
    messages: list[dict]  # prompt messages (system first)
    reply: dict  # recorded assistant message
    tools: list[str]
    est_prompt_tokens: int
    kind: str  # main | subagent
    final: bool = field(default=False)  # reply ends the task (no tool calls)


def load_turns(corpus_dir: Path) -> dict[str, list[Turn]]:
    manifest = corpusfmt.load_manifest(corpus_dir)
    kinds = {s["name"]: s["kind"] for s in manifest["sessions"]}
    sessions: dict[str, list[Turn]] = {}
    for path in sorted((corpus_dir / "sessions").glob("*.jsonl")):
        name = path.stem
        turns: list[Turn] = []
        prev = None
        for t in corpusfmt.iter_turns(path):
            if prev is not None:
                n = len(prev["messages"])
                m = t["messages"]
                if len(m) > n and m[n]["role"] == "assistant" and m[:n] == prev["messages"]:
                    turns.append(
                        Turn(
                            session=name,
                            turn=prev["turn"],
                            messages=prev["messages"],
                            reply=m[n],
                            tools=prev["tools"],
                            est_prompt_tokens=prev["est_prompt_tokens"],
                            kind=kinds.get(name, "main"),
                            final=not m[n].get("tool_calls"),
                        )
                    )
            prev = t
        sessions[name] = turns
    return sessions


class Renderer:
    def __init__(self, corpus_dir: Path):
        self.tmpl = make_env().from_string((corpus_dir / "chat_template.jinja").read_text())
        self.tools = corpusfmt.load_tools(corpus_dir)

    def render(self, messages: list[dict], tools: list[str], reply: dict) -> tuple[str, int]:
        msgs = [template_message(m) for m in messages]
        tool_list = [self.tools[n] for n in tools]
        prompt = self.tmpl.render(messages=msgs, tools=tool_list, add_generation_prompt=True)
        full = self.tmpl.render(messages=msgs + [template_message(reply)], tools=tool_list, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise ValueError("template is not prefix-stable for the reply")
        # the model's generation ends with <|im_end|>; the template's trailing newline is not generated
        full = full.removesuffix("\n")
        reply_text = full[len(prompt) :]
        if len(reply_text) > REPLY_CHAR_CAP:
            full = full[: len(prompt) + REPLY_CHAR_CAP]
        # lone surrogates cannot be tokenized as UTF-8; replacement keeps code-point offsets intact
        full = full.encode("utf-8", "replace").decode("utf-8")
        return full, len(prompt)


def spread(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def build_pool(sessions: dict[str, list[Turn]], renderer: Renderer) -> list[dict]:
    pool: list[dict] = []

    def add(bucket: str, wid: str, session: str, kind: str, parts: list, messages, tools, reply):
        text, start = renderer.render(messages, tools, reply)
        if len(text) - start < MIN_REPLY_CHARS:
            return
        pool.append(
            {
                "id": wid,
                "session": session,
                "depth_bucket": bucket,
                "kind": kind,
                "parts": parts,
                "est_depth_tokens": round(start / corpusfmt.CHARS_PER_TOKEN),
                "text": text,
                "score_start_char": start,
            }
        )

    for bucket, target in BUCKETS.items():
        lo, hi = POOL_BAND[0] * target, min(POOL_BAND[1] * target, MODEL_CTX - 4 * MAX_SCORED)
        for name, turns in sessions.items():
            cands = [t for t in turns if lo <= t.est_prompt_tokens <= hi]
            for t in spread(cands, POOL_PER_SESSION):
                add(bucket, f"{bucket}/{name}/t{t.turn}", name, "turn", [[name, t.turn]], t.messages, t.tools, t.reply)
        if bucket not in CONCAT_BUCKETS:
            continue
        # synthetic: finished main session A (prefix ending on a final answer) + main session B's history
        mains = {n: ts for n, ts in sessions.items() if ts and ts[0].kind == "main"}
        system = next(iter(mains.values()))[0].messages[0]
        tools = next(iter(mains.values()))[0].tools
        mains = {n: ts for n, ts in mains.items() if ts[0].messages[0] == system and ts[0].tools == tools}
        b_cands = []  # deepest turn of each session within [0.55, 0.9] * target
        for name, ts in mains.items():
            inb = [t for t in ts if 0.55 * target <= t.est_prompt_tokens <= 0.9 * target]
            if inb:
                b_cands.append(inb[-1])
        a_ends = [t for ts in mains.values() for t in ts if t.final]
        made = 0
        for b in sorted(b_cands, key=lambda t: -t.est_prompt_tokens):
            need = target - b.est_prompt_tokens
            opts = [a for a in a_ends if a.session != b.session and 0.3 * need <= a.est_prompt_tokens <= 1.1 * need]
            # A's cost in the concat is its whole prompt + reply minus the shared system/tool block
            for a in sorted(opts, key=lambda a: (abs(a.est_prompt_tokens - need), a.session, a.turn))[:2]:
                messages = a.messages + [a.reply] + b.messages[1:]
                wid = f"{bucket}/{a.session}/t{a.turn}+{b.session}/t{b.turn}"
                add(
                    bucket,
                    wid,
                    b.session,
                    "concat",
                    [[a.session, a.turn], [b.session, b.turn]],
                    messages,
                    b.tools,
                    b.reply,
                )
                made += 1
            if made >= POOL_CONCATS:
                break
    return pool


def select(pool: list[dict], per_bucket: int, seed: int, exact: bool) -> list[dict]:
    tol = EXACT_TOL if exact else EST_TOL
    key = "depth_tokens" if exact else "est_depth_tokens"

    def fits(w: dict) -> bool:
        scored = min(w["reply_tokens"], MAX_SCORED) if exact else MAX_SCORED
        return w[key] + scored <= WINDOW_TOKEN_CAP

    out = []
    for bucket, target in BUCKETS.items():
        rng = random.Random(f"{seed}:{bucket}")
        in_bucket = [w for w in pool if w["depth_bucket"] == bucket and fits(w)]
        cands = [w for w in in_bucket if abs(w[key] / target - 1) <= tol]
        if not cands:  # nothing in band: deepest available, actual depth recorded
            cands = sorted(in_bucket, key=lambda w: -w[key])[:per_bucket]
        chosen: list[dict] = []
        for kind in ("turn", "concat"):  # real turns first
            by_session: dict[str, list[dict]] = {}
            for w in sorted((w for w in cands if w["kind"] == kind), key=lambda w: w["id"]):
                by_session.setdefault(w["session"], []).append(w)
            names = sorted(by_session)
            rng.shuffle(names)
            for w_list in by_session.values():
                rng.shuffle(w_list)
            # round-robin across sessions: distinct sessions first, repeats only if short
            while len(chosen) < per_bucket and any(by_session[n] for n in names):
                for n in names:
                    if by_session[n] and len(chosen) < per_bucket:
                        chosen.append(by_session[n].pop())
        out.extend(sorted(chosen, key=lambda w: w[key]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=REPLAY / "corpus")
    ap.add_argument("--pool", type=Path, default=HERE / "candidates.jsonl", help="candidate pool (tokcount input)")
    ap.add_argument("--counts", type=Path, help="lovelace-kld tokcount output for --pool: exact depths")
    ap.add_argument("--out", type=Path, default=HERE / "windows.jsonl")
    ap.add_argument("--per-bucket", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    sessions = load_turns(args.corpus)
    pool = build_pool(sessions, Renderer(args.corpus))
    pool.sort(key=lambda w: w["id"])
    with open(args.pool, "w") as f:
        for w in pool:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    exact = args.counts is not None
    if exact:
        counts = {}
        for line in args.counts.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                counts[c["id"]] = c
        missing = [w["id"] for w in pool if w["id"] not in counts]
        if missing:
            sys.exit(f"{args.counts}: no counts for {len(missing)} pool windows (e.g. {missing[0]}); rerun tokcount on {args.pool}")
        for w in pool:
            w["depth_tokens"] = counts[w["id"]]["depth_tokens"]
            w["reply_tokens"] = counts[w["id"]]["reply_tokens"]

    windows = select(pool, args.per_bucket, args.seed, exact)
    with open(args.out, "w") as f:
        for w in windows:
            w["depth_exact"] = exact
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    key = "depth_tokens" if exact else "est_depth_tokens"
    print(f"pool: {len(pool)} candidates -> {args.pool}")
    print(f"windows: {len(windows)} -> {args.out}  (depth = {'exact tokens' if exact else 'chars/3.2 estimate'})")
    for bucket in BUCKETS:
        ws = [w for w in windows if w["depth_bucket"] == bucket]
        if not ws:
            print(f"  {bucket:>5}: 0 windows")
            continue
        d = [w[key] for w in ws]
        kinds = sum(w["kind"] == "concat" for w in ws)
        print(
            f"  {bucket:>5}: {len(ws)} windows, {len({w['session'] for w in ws})} sessions, "
            f"depth {min(d)}..{max(d)}" + (f", {kinds} concat" if kinds else "")
        )


if __name__ == "__main__":
    main()

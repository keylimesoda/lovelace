"""Extract replayable multi-agent traffic from OMP session logs.

Each OMP session file (~/.omp/agent/sessions/**/*.jsonl) is a tree of entries
(id/parentId). For every assistant message produced by the local model we rebuild
the OpenAI chat `messages` array the model saw right before generating it, using
the same rules as OMP's context builder:

- walk the parentId chain root -> parent (so forked branches are handled);
- apply the latest compaction on that path: summary message, then entries from
  firstKeptEntryId up to the compaction, then everything after it;
- custom_message entries (async results, IRC, reminders) become user messages
  (OMP sends them as `developer`, downgraded to `user` for local servers);
- errored/aborted assistant messages (and their tool results) are dropped from
  history, as are tool calls that never got a result;
- tool results that were pruned *after* this request are restored (display
  content if saved, otherwise deterministic filler of the recorded token size).

Usage: uv run extract.py [--sessions-root DIR] [--out corpus] [--reqlog FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

from corpusfmt import CHARS_PER_TOKEN, CORPUS_DIR, dumps, loads

DEFAULT_ROOT = Path.home() / ".omp/agent/sessions"
# customTypes OMP never sends to the model (packages/coding-agent context builder)
HIDDEN_CUSTOM = {"prewalk-plan", "vibe-mode-context"}
COMPACTION_TEMPLATE = (
    "Prior model work/tool state available.\nMUST build on prior work; NEVER duplicate prior work.\n\n"
    "<summary>\n{summary}\n</summary>\n"
)
HANDOFF_TEMPLATE = (
    "Context replaced. The <handoff> below is a handoff document a prior instance of you wrote from the "
    "full conversation. It is your own working memory, not user input.\n"
    "- First person inside it refers to you (the prior instance).\n"
    '- "Next Steps" is your own resumed plan; re-check it against the latest user message before acting.\n'
    "- The handoff already exists and is complete: NEVER write another handoff document unless the user "
    "explicitly asks.\nMUST build on prior work; NEVER duplicate prior work.\n\n<handoff>\n{summary}\n</handoff>\n"
)
BRANCH_TEMPLATE = "Branch-return summary:\n\n<summary>\n{summary}\n</summary>\n"
SHAKEN_RE = re.compile(r"\[shaken ~([\d,]+) tokens[^\]]*\]")
SUPERSEDED = "[Superseded by a newer read of this file]"
# Short common words: " to" etc. are one token each and ~3.2 chars incl. the space, so filler
# of N words is ~N real tokens and also ~N tokens by the chars/3.2 estimate.
FILLER_WORDS = "to of in is it on at by or as an be we if no so up do go me my us".split()
MAIN_TOOLS = ["read", "bash", "edit", "write", "grep", "glob", "eval", "todo", "web_search", "task", "wait", "ask"]


def ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def text_of(parts, sep: str = "\n") -> str:
    if isinstance(parts, str):
        return parts
    return sep.join(p["text"] for p in parts if p.get("type") == "text" and p.get("text"))


def filler(n_tokens: int, seed: str) -> str:
    h = int(hashlib.blake2s(seed.encode(), digest_size=4).hexdigest(), 16)
    k = len(FILLER_WORDS)
    return " ".join(FILLER_WORDS[(h + i * 7) % k] for i in range(n_tokens))


def tool_result_text(msg: dict, request_ms: int) -> str:
    text = text_of(msg["content"])
    pruned = msg.get("prunedAt")
    if pruned is None or pruned <= request_ms:
        return text
    # Request happened before OMP pruned this result: the model saw the original.
    shown = ((msg.get("details") or {}).get("displayContent") or {}).get("text")
    if text.strip() == SUPERSEDED and shown:
        return shown
    return SHAKEN_RE.sub(lambda m: filler(int(m.group(1).replace(",", "")), msg.get("toolCallId", "")), text)


def is_dropped_assistant(m: dict) -> bool:
    return m.get("stopReason") in ("error", "aborted") or bool(m.get("retryRecovery"))


def convert(path_entries: list[dict], request_ms: int) -> list[dict]:
    """OMP entries on the root->parent path -> OpenAI chat messages (without system)."""
    comp_idx = max((i for i, e in enumerate(path_entries) if e["type"] == "compaction"), default=-1)
    prefix: list[dict] = []
    if comp_idx >= 0:
        comp = path_entries[comp_idx]
        tmpl = HANDOFF_TEMPLATE if comp.get("method") == "handoff" else COMPACTION_TEMPLATE
        prefix.append({"role": "user", "content": tmpl.format(summary=comp["summary"])})
        first = next(
            (i for i, e in enumerate(path_entries[:comp_idx]) if e.get("id") == comp.get("firstKeptEntryId")), None
        )
        seq = (path_entries[first:comp_idx] if first is not None else []) + path_entries[comp_idx + 1 :]
    else:
        seq = path_entries

    raw: list[dict] = []  # OMP-level messages
    for e in seq:
        t = e["type"]
        if t == "message":
            raw.append(e["message"])
        elif t == "custom_message" and e.get("customType") not in HIDDEN_CUSTOM:
            raw.append({"role": "developer", "content": e.get("content")})
        elif t == "branch_summary" and e.get("summary"):
            raw.append({"role": "user", "content": BRANCH_TEMPLATE.format(summary=e["summary"])})

    dropped_calls = {
        p["id"]
        for m in raw
        if m.get("role") == "assistant" and is_dropped_assistant(m)
        for p in m["content"]
        if p.get("type") == "toolCall"
    }
    answered = {m.get("toolCallId") for m in raw if m.get("role") == "toolResult"} - dropped_calls

    out: list[dict] = list(prefix)
    for m in raw:
        role = m.get("role")
        if role in ("user", "developer"):
            content = m.get("content")
            text = text_of(content) if content is not None else ""
            if isinstance(content, list) and any(p.get("type") == "image" for p in content):
                text = (text + "\n" if text else "") + "[image omitted]"
            if text.strip():
                out.append({"role": "user", "content": text})
        elif role == "assistant":
            if is_dropped_assistant(m):
                continue
            parts = m["content"]
            text = "".join(p["text"] for p in parts if p.get("type") == "text" and p.get("text", "").strip())
            thinking = [p["thinking"] for p in parts if p.get("type") == "thinking" and p.get("thinking", "").strip()]
            calls = [
                {
                    "id": p["id"],
                    "type": "function",
                    "function": {
                        "name": p["name"],
                        "arguments": p["arguments"]
                        if isinstance(p["arguments"], str)
                        else json.dumps(p["arguments"], ensure_ascii=False, separators=(",", ":")),
                    },
                }
                for p in parts
                if p.get("type") == "toolCall" and p["id"] in answered
            ]
            if not text and not calls and not thinking:
                continue
            msg: dict = {"role": "assistant", "content": text}
            if thinking:
                msg["reasoning_content"] = "\n".join(thinking)
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif role == "toolResult":
            if m.get("toolCallId") in dropped_calls:
                continue
            content = tool_result_text(m, request_ms)
            if not content and any(p.get("type") == "image" for p in m["content"]):
                content = "(see attached image)"
            out.append({"role": "tool", "tool_call_id": m["toolCallId"], "content": content})
    return out


def message_chars(m: dict) -> int:
    n = len(m.get("content") or "") + len(m.get("reasoning_content") or "") + 12  # + role/template framing
    for c in m.get("tool_calls") or ():
        n += len(c["function"]["name"]) + len(c["function"]["arguments"]) + 24
    return n


def estimate_prompt_tokens(messages: list[dict], tools_chars: int) -> int:
    """chars / CHARS_PER_TOKEN estimate. The model's chat template (corpus/chat_template.jinja)
    renders reasoning_content of *every* assistant message unless preserve_thinking=false,
    so all reasoning counts toward depth."""
    return round((tools_chars + sum(message_chars(m) for m in messages)) / CHARS_PER_TOKEN)


def session_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    stem = parts[-1].removesuffix(".jsonl")
    if len(parts) >= 3:  # <cwd>/<parent-session-dir>/<agent>.jsonl
        parent = parts[-2]
        return f"{parent[:19]}_{parent.split('_')[-1][:8]}__{stem}"
    return f"{stem[:19]}_{stem.split('_')[-1][:8]}"


def extract_session(path: Path, root: Path, system_prompt: str, tools: dict[str, dict], provider: str):
    entries = [loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    by_id = {e["id"]: e for e in entries if e.get("id") and e["type"] != "session"}
    init = next((e for e in entries if e["type"] == "session_init"), None)
    header = next((e for e in entries if e["type"] == "session"), {})
    if init:
        system = init["systemPrompt"]
        names = init["tools"] if isinstance(init["tools"], list) else []
        tool_names = [n for n in names if n in tools]
    else:
        system = system_prompt
        tool_names = MAIN_TOOLS
    tools_chars = len(system) + sum(len(json.dumps(tools[n])) for n in tool_names)

    turns: list[dict] = []
    prev_msgs: list[dict] = []
    turn_of_entry: dict[str, dict] = {}
    for e in entries:
        if e["type"] != "message" or e["message"].get("role") != "assistant":
            continue
        am = e["message"]
        if am.get("provider") != provider or is_dropped_assistant(am):
            continue
        # root -> parent path
        path_entries = []
        cur = by_id.get(e.get("parentId"))
        while cur is not None:
            path_entries.append(cur)
            cur = by_id.get(cur.get("parentId"))
        path_entries.reverse()
        start_ms = am.get("timestamp") or ms(e["timestamp"])
        msgs = [{"role": "system", "content": system}, *convert(path_entries, start_ms)]
        if msgs[-1]["role"] == "assistant":  # would be a prefill-continuation; not how OMP calls
            continue

        prev_turn = next((turn_of_entry[p["id"]] for p in reversed(path_entries) if p["id"] in turn_of_entry), None)
        gap_s = None
        after_user = False
        if prev_turn is not None:
            gap_s = max(0.0, (start_ms - prev_turn["end_ms"]) / 1000)
            idx = next(i for i, p in enumerate(path_entries) if p["id"] == prev_turn["entry_id"])
            after_user = any(
                p["type"] == "message" and p["message"].get("role") == "user" for p in path_entries[idx + 1 :]
            )

        keep = 0
        for a, b in zip(prev_msgs, msgs):
            if a != b:
                break
            keep += 1
        parts = am["content"]
        out_chars = sum(len(p.get("text", "")) for p in parts if p.get("type") == "text") + sum(
            len(p.get("name", "")) + len(json.dumps(p.get("arguments"), ensure_ascii=False))
            for p in parts
            if p.get("type") == "toolCall"
        )
        reasoning_chars = sum(len(p.get("thinking", "")) for p in parts if p.get("type") == "thinking")
        usage = am.get("usage") or {}
        rec_prompt = (usage.get("input") or 0) + (usage.get("cacheRead") or 0) + (usage.get("cacheWrite") or 0)
        end_ms = am.get("completedAt") or (start_ms + int(am.get("duration") or 0))
        turn = {
            "turn": len(turns),
            "entry_id": e["id"],
            "ts_ms": start_ms,
            "end_ms": end_ms,
            "gap_s": gap_s,
            "after_user": after_user,
            "tools": tool_names,
            "n_messages": len(msgs),
            "est_prompt_tokens": estimate_prompt_tokens(msgs, tools_chars - len(system)),
            "rec_prompt_tokens": rec_prompt or None,
            "rec_output_tokens": usage.get("output") or None,
            "output_chars": out_chars,
            "reasoning_chars": reasoning_chars,
            "est_output_tokens": round((out_chars + reasoning_chars) / CHARS_PER_TOKEN),
            "rec_ttft_ms": am.get("ttft"),
            "rec_duration_ms": am.get("duration"),
            "stop_reason": am.get("stopReason"),
            "compacted": any(p["type"] == "compaction" for p in path_entries),
            "keep": keep,
            "append": msgs[keep:],
        }
        turns.append(turn)
        turn_of_entry[e["id"]] = turn
        prev_msgs = msgs
    meta = {
        "source": str(path),
        "kind": "subagent" if init else "main",
        "agent": (init or {}).get("agent"),
        "cwd": header.get("cwd"),
        "real_system_prompt": init is not None,
        "tools": tool_names,
        "compactions": sum(e["type"] == "compaction" for e in entries),
    }
    return turns, meta


def pct(values: list[float], p: float):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))]


def dist(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p10": pct(values, 0.10),
        "p50": pct(values, 0.50),
        "p90": pct(values, 0.90),
        "p99": pct(values, 0.99),
        "max": max(values) if values else None,
        "mean": round(statistics.fmean(values)) if values else None,
    }


def reqlog_prompt_tokens(path: Path) -> list[int]:
    out = []
    for line in path.read_bytes().splitlines():
        try:
            d = loads(line)
        except Exception:
            continue
        if d.get("event") == "request_done":
            pt = (d.get("result") or {}).get("prompt_tokens")
            if pt:
                out.append(pt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=CORPUS_DIR)
    ap.add_argument("--provider", default="ninfer-local", help="only turns generated by this OMP provider")
    ap.add_argument("--min-turns", type=int, default=5)
    ap.add_argument("--reqlog", type=Path, default=Path("/tmp/reqlog.jsonl"), help="ninfer request log to compare")
    args = ap.parse_args()

    out_dir: Path = args.out
    sess_dir = out_dir / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    for old in sess_dir.glob("*.jsonl"):
        old.unlink()
    system_prompt = (out_dir / "system_prompt.txt").read_text()
    tools = {t["function"]["name"]: t for t in json.loads((out_dir / "tools.json").read_text())}

    manifest: dict = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sessions_root": str(args.sessions_root),
        "provider": args.provider,
        "chars_per_token": CHARS_PER_TOKEN,
        "note": f"est_* token counts are chars/{CHARS_PER_TOKEN} estimates of the rebuilt prompt (system prompt and tool schemas "
        "reconstructed for main sessions); rec_* are the token counts the server reported when the session ran.",
        "sessions": [],
        "skipped": [],
    }
    all_est, all_rec, all_ratio = [], [], []
    files = sorted(p for p in args.sessions_root.rglob("*.jsonl") if p.is_file())
    for path in files:
        turns, meta = extract_session(path, args.sessions_root, system_prompt, tools, args.provider)
        name = session_name(path, args.sessions_root)
        if len(turns) < args.min_turns:
            manifest["skipped"].append({"name": name, "source": str(path), "turns": len(turns)})
            continue
        with open(sess_dir / f"{name}.jsonl", "wb") as f:
            for t in turns:
                f.write(dumps(t) + b"\n")
        est = [t["est_prompt_tokens"] for t in turns]
        rec = [t["rec_prompt_tokens"] for t in turns if t["rec_prompt_tokens"]]
        all_est += est
        all_rec += rec
        all_ratio += [t["est_prompt_tokens"] / t["rec_prompt_tokens"] for t in turns if t["rec_prompt_tokens"]]
        gaps = [t["gap_s"] for t in turns if t["gap_s"] is not None and not t["after_user"]]
        manifest["sessions"].append(
            {
                "name": name,
                "file": f"sessions/{name}.jsonl",
                **meta,
                "turns": len(turns),
                "est_depth_tokens": {"p50": pct(est, 0.5), "p90": pct(est, 0.9), "max": max(est)},
                "rec_depth_tokens": {"p50": pct(rec, 0.5), "p90": pct(rec, 0.9), "max": max(rec)} if rec else None,
                "agent_gap_s_p50": pct(gaps, 0.5),
                "output_tokens_rec_sum": sum(t["rec_output_tokens"] or 0 for t in turns),
            }
        )
    manifest["totals"] = {
        "sessions": len(manifest["sessions"]),
        "turns": sum(s["turns"] for s in manifest["sessions"]),
        "est_depth_tokens": dist(all_est),
        "rec_depth_tokens": dist(all_rec),
        "est_over_rec_ratio": dist([round(r, 3) for r in all_ratio]),
    }
    if args.reqlog.exists():
        manifest["totals"]["reqlog_prompt_tokens"] = dist(reqlog_prompt_tokens(args.reqlog))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    tot = manifest["totals"]
    print(f"sessions kept: {tot['sessions']}  skipped(<{args.min_turns} turns or other provider): {len(manifest['skipped'])}")
    print(f"turns: {tot['turns']}")
    fmt = "{:<34}{:>8}{:>9}{:>9}{:>9}{:>9}{:>9}"
    print(fmt.format("prompt depth (tokens)", "n", "p10", "p50", "p90", "p99", "max"))
    rows = [
        (f"corpus est (chars/{CHARS_PER_TOKEN})", tot["est_depth_tokens"]),
        ("corpus recorded (server usage)", tot["rec_depth_tokens"]),
    ]
    if "reqlog_prompt_tokens" in tot:
        rows.append((f"reqlog {args.reqlog.name}", tot["reqlog_prompt_tokens"]))
    for label, d in rows:
        print(fmt.format(label, d["n"], d["p10"], d["p50"], d["p90"], d["p99"], d["max"]))
    r = tot["est_over_rec_ratio"]
    print(f"est/recorded ratio: p10 {r['p10']}  p50 {r['p50']}  p90 {r['p90']}")


if __name__ == "__main__":
    main()

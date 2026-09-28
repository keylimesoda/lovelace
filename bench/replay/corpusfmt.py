"""Shared corpus format helpers (reader side).

Corpus layout (written by extract.py):
  corpus/system_prompt.txt   representative system prompt for top-level sessions
  corpus/tools.json          OpenAI tool schemas; each turn names the subset it carries
  corpus/manifest.json       per-session stats
  corpus/sessions/<name>.jsonl  one turn per line

Turn lines are delta-encoded against the previous line of the same file so the
corpus stays O(total messages) instead of O(turns^2):
  messages(turn i) = messages(turn i-1)[:keep] + append
Turn 0 always has keep == 0 and starts with the system message. Use iter_turns()
to get fully materialized OpenAI `messages` arrays.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

try:
    import orjson

    def loads(b: bytes | str):
        return orjson.loads(b)

    def dumps(obj) -> bytes:
        return orjson.dumps(obj)

except ImportError:  # pragma: no cover - orjson is a declared dependency

    def loads(b: bytes | str):
        return json.loads(b)

    def dumps(obj) -> bytes:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


# chars per Qwen3.8 token for this traffic (code/JSON/logs heavy). Calibrated against the
# server-reported prompt tokens stored in the sessions (median 3.2); it is only an estimate.
CHARS_PER_TOKEN = 3.2
HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"


def load_tools(corpus_dir: Path = CORPUS_DIR) -> dict[str, dict]:
    tools = loads((corpus_dir / "tools.json").read_bytes())
    return {t["function"]["name"]: t for t in tools}


def load_manifest(corpus_dir: Path = CORPUS_DIR) -> dict:
    return loads((corpus_dir / "manifest.json").read_bytes())


def iter_turns(path: Path) -> Iterator[dict]:
    """Yield turns with a materialized `messages` list (a fresh list per turn)."""
    messages: list[dict] = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            turn = loads(line)
            keep = turn.pop("keep")
            if keep > len(messages):
                raise ValueError(f"{path}: turn {turn.get('turn')} keeps {keep} > {len(messages)} messages")
            messages = messages[:keep] + turn.pop("append")
            turn["messages"] = messages
            yield turn

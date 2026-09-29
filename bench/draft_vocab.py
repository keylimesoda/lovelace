"""Build the MTP draft vocabulary (the token ids the reduced draft LM head computes) from agent outputs.

The draft head only proposes tokens; the target verifies over the full vocabulary, so a token outside the set
costs acceptance, never correctness. Tokens are ranked by their frequency in the assistant turns the agents
produced in the replay corpus (bench/replay/corpus), rendered the way the chat template renders them (reasoning,
content, tool calls), plus --input-weight times their frequency in the tool results and user messages the agents
read (outputs echo that text; it also ranks tokens the outputs never used). Both are relative frequencies, and the
text is tokenized with the served model's vocabulary (llama-tokenize, vocab only, no GPU).

Regenerate (from bench/replay, after re-extracting the corpus or changing the model):
  uv run python ../draft_vocab.py --bin ../../engine/build-sm89/bin \
      --model /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf --out ../../deploy/draft-vocab/qwen3.8-27b.txt
The server uses it with --spec-draft-vocab <file> (MTP only).

Prints the corpus coverage of the chosen set and an estimate for unseen traffic (leave one session out: each
session's outputs against the set built from the other sessions).

--exclude S1,S2,... leaves those sessions (file names or name prefixes) out of the set and reports the set's
coverage of their outputs (held_out_coverage): replaying them (bench/step_drive.py) then measures acceptance on
traffic the set never saw, e.g. the six 50k sessions step_drive picks:
  --exclude 2026-09-27T08-42-53,2026-09-27T08-18-54,2026-09-25T07-15-46,2026-09-26T02-32-58,2026-09-26T09-54-32,2026-09-27T09-11-19
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "replay"))
from corpusfmt import CORPUS_DIR, iter_turns  # noqa: E402

IMAGE = "lovelace-engine-runtime:cuda13.1"


def render(msg):
    """The assistant message as the chat template renders it after '<|im_start|>assistant\\n'."""
    reasoning = (msg.get("reasoning_content") or "").strip() if isinstance(msg.get("reasoning_content"), str) else ""
    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    out = "<think>\n" + reasoning + "\n</think>\n\n" + content
    calls = msg.get("tool_calls") or []
    for i, tc in enumerate(calls):
        fn = tc.get("function", tc)
        if i == 0:
            out += "\n\n<tool_call>\n<function=" if content.strip() else "<tool_call>\n<function="
        else:
            out += "\n<tool_call>\n<function="
        out += fn["name"] + ">\n"
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        for k, v in args.items():
            out += "<parameter=" + k + ">\n" + (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) + "\n</parameter>\n"
        out += "</function>\n</tool_call>"
    return out + "<|im_end|>"


def session_outputs(path):
    """Distinct assistant messages of one session (every turn's context, so compactions don't lose any)."""
    seen, outs = set(), []
    for turn in iter_turns(path):
        for m in turn["messages"]:
            if m.get("role") != "assistant":
                continue
            text = render(m)
            if text not in seen:
                seen.add(text)
                outs.append(text)
    return outs


def session_inputs(path):
    """Distinct tool results and user messages of one session."""
    seen, outs = set(), []
    for turn in iter_turns(path):
        for m in turn["messages"]:
            if m.get("role") not in ("tool", "user"):
                continue
            c = m.get("content") or ""
            if not isinstance(c, str):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            if c and c not in seen:
                seen.add(c)
                outs.append(c)
    return outs


def tokenize(bin_dir, model, texts):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "text.txt").write_text("\n".join(texts))
        # the binaries link libcuda, so the driver is mounted, but no device is visible: no GPU work
        cmd = ["docker", "run", "--rm", "--gpus", "all", "-e", "CUDA_VISIBLE_DEVICES=",
               "-v", f"{Path(bin_dir).resolve()}:/e:ro", "-v", f"{Path.home() / 'models'}:/models:ro",
               "-v", f"{tmp}:/w:ro", IMAGE, "/e/llama-tokenize", "-m", model, "-f", "/w/text.txt", "--ids", "--no-bos",
               "--log-disable"]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    line = next(ln for ln in out.splitlines() if ln.startswith("["))
    return json.loads(line)


def coverage(counts, ids):
    total = sum(counts.values())
    return sum(counts[t] for t in ids) / total if total else 0.0


def top(out_counts, in_counts, n, w):
    to = sum(out_counts.values()) or 1
    ti = sum(in_counts.values()) or 1
    score = {t: out_counts[t] / to + w * in_counts[t] / ti for t in out_counts.keys() | in_counts.keys()}
    return set(sorted(score, key=lambda t: (-score[t], t))[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="engine bin dir with llama-tokenize")
    ap.add_argument("--model", required=True, help="model path inside the container (~/models is /models)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--input-weight", type=float, default=0.25)
    ap.add_argument("--exclude", default="", help="comma-separated session files (or name prefixes) kept out of the set")
    a = ap.parse_args()

    all_files = sorted((CORPUS_DIR / "sessions").glob("*.jsonl"))
    excl = [p for p in a.exclude.split(",") if p]
    held = [f for f in all_files if any(f.name.startswith(p) for p in excl)]
    missing = [p for p in excl if not any(f.name.startswith(p) for f in held)]
    if missing:
        sys.exit(f"--exclude: no session matches {missing}")
    files = [f for f in all_files if f not in held]
    per_session = [Counter(tokenize(a.bin, a.model, session_outputs(f))) for f in files]
    per_session_in = [Counter(tokenize(a.bin, a.model, session_inputs(f))) for f in files]
    counts = sum(per_session, Counter())
    counts_in = sum(per_session_in, Counter())

    # unseen-traffic estimate: each session's outputs against the set built from all the other sessions
    hit = 0
    for c, c_in in zip(per_session, per_session_in):
        ids_c = top(counts - c, counts_in - c_in, a.n, a.input_weight)
        hit += sum(v for t, v in c.items() if t in ids_c)

    ids = top(counts, counts_in, a.n, a.input_weight)
    stats = {
        "sessions": len(files),
        "output_tokens": sum(counts.values()),
        "input_tokens": sum(counts_in.values()),
        "distinct_output": len(counts),
        "input_weight": a.input_weight,
        "n": len(ids),
        "coverage": round(coverage(counts, ids), 5),
        "leave_one_session_out_coverage": round(hit / sum(counts.values()), 5),
    }
    if held:
        held_counts = Counter(tokenize(a.bin, a.model, [m for f in held for m in session_outputs(f)]))
        stats["excluded"] = [f.name for f in held]
        stats["held_out_output_tokens"] = sum(held_counts.values())
        stats["held_out_coverage"] = round(coverage(held_counts, ids), 5)
    print(json.dumps(stats))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# MTP draft vocabulary (--spec-draft-vocab), built by bench/draft_vocab.py\n")
        f.write(f"# {json.dumps(stats)}\n")
        f.writelines(f"{t}\n" for t in sorted(ids))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

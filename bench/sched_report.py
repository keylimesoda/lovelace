"""Scheduling v2 table for replay runs (bench/run_replay.sh output dirs): steady throughput and TTFT from
bakeoff/analyze.py, plus the scheduler's own view from telemetry-end.json and the server request log.

Usage: python bench/sched_report.py RUN_DIR [RUN_DIR ...]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "bakeoff"))
from analyze import pct, summarize  # noqa: E402


def sched_row(run_dir: Path) -> dict:
    s = summarize(run_dir, 180.0) or {}
    t = json.loads((run_dir / "telemetry-end.json").read_text()) if (run_dir / "telemetry-end.json").exists() else {}
    kb = (t.get("scheduler") or {}).get("kv_budget") or {}
    tot = kb.get("totals") or {}
    v2 = kb.get("scheduler")  # absent before scheduling v2
    ss = (t.get("memory") or {}).get("sessions") or {}
    loop = (t.get("scheduler") or {}).get("totals") or {}

    # per-request queue waits from the server request log (timed window and warm-up alike)
    waits = []
    log = run_dir / "requests.server.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("event") == "request_done":
                waits.append((rec.get("timings_ms") or {}).get("queue_wait", 0.0) / 1e3)

    warm = {}
    if (run_dir / "summary.json").exists():
        warm = json.loads((run_dir / "summary.json").read_text()).get("warmup") or {}

    return {
        "run": run_dir.name,
        "turns": s.get("turns"),
        "errors": (s.get("errors") or 0) + (warm.get("errors") or 0),
        "agg_steady": s.get("agg_tps_steady"),
        "ttft_p50": s.get("ttft_p50"),
        "ttft_p90": s.get("ttft_p90"),
        "depth_p50": s.get("depth_p50"),
        "busy_slots/decode": loop.get("n_busy_slots_per_decode"),
        "wait_p90": pct(waits, 0.9) if waits else None,
        "max_wait": v2["max_wait_s"] if v2 else (max(waits) if waits else None),
        "aged": v2["admits_aged"] if v2 else None,
        "lane_occ": v2["lane"]["occupancy"] if v2 else None,
        "preempts": tot.get("preempts"),
        "demotions": ss.get("demotions"),
        "prefetch_hit": f"{ss.get('prefetch_hits', 0)}/{tot.get('unparks', 0)}" if ss else None,
        "warmup_s": warm.get("wall_s"),
    }


def main() -> None:
    rows = [sched_row(Path(d)) for d in sys.argv[1:]]
    cols = list(rows[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else ("-" if r[c] is None else str(r[c])) for c in cols) + " |")


if __name__ == "__main__":
    main()

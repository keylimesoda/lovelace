"""Observability overhead (CHARTER E1): client load for the telemetry-ON runs and the OFF/ON table.

  python3 bench/obs_overhead.py load --base-url http://127.0.0.1:18500 --out RUN_DIR
      polls /telemetry at 1 Hz and /metrics every 4 s and holds one /events subscription until SIGTERM/SIGINT, then
      writes RUN_DIR/load.json (request counts, errors, latencies, SSE bytes and frames)
  python3 bench/obs_overhead.py table OUT_DIR
      reads OUT_DIR/NN-{off,on}/{single.jsonl,six.out,load.json} (bench/obs_overhead.sh) and writes OUT_DIR/RESULTS.md

Figures per run:
  single decode tok/s   median of the warm repetitions of bench/depth_sweep.py at one depth (server `timings`)
  six agg tok/s         stream events per second over step_drive.py's capture window, all six streams decoding
  six seq tok/s         median over the six requests of the server's per-request decode rate
  six step ms           median over the six requests of predicted_ms / (predicted_n - draft_accepted): wall time per
                        decode step (each step commits 1 + accepted draft tokens per sequence)
"""

import argparse
import json
import signal
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def load(base, out):
    stop = threading.Event()
    stats = {"telemetry": {"ok": 0, "err": 0, "ms": []}, "metrics": {"ok": 0, "err": 0, "ms": []},
             "events": {"bytes": 0, "frames": 0, "err": 0}}

    def poll(path, period, key):
        nxt = time.monotonic()
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(base + path, timeout=10) as r:
                    r.read()
                stats[key]["ok"] += 1
                stats[key]["ms"].append((time.monotonic() - t0) * 1e3)
            except OSError:
                stats[key]["err"] += 1
            nxt += period
            stop.wait(max(0.0, nxt - time.monotonic()))

    def events():
        while not stop.is_set():
            try:
                with urllib.request.urlopen(base + "/events", timeout=60) as r:
                    while not stop.is_set():
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        stats["events"]["bytes"] += len(chunk)
                        stats["events"]["frames"] += chunk.count(b"\n\n")
            except OSError:
                stats["events"]["err"] += 1
                stop.wait(1.0)

    threads = [threading.Thread(target=poll, args=("/telemetry", 1.0, "telemetry"), daemon=True),
               threading.Thread(target=poll, args=("/metrics", 4.0, "metrics"), daemon=True),
               threading.Thread(target=events, daemon=True)]
    for t in threads:
        t.start()
    t0 = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.is_set():
            stop.wait(0.5)
    except KeyboardInterrupt:
        stop.set()
    for k in ("telemetry", "metrics"):
        ms = stats[k].pop("ms")
        stats[k]["p50_ms"] = round(statistics.median(ms), 2) if ms else None
        stats[k]["max_ms"] = round(max(ms), 2) if ms else None
    stats["seconds"] = round(time.monotonic() - t0, 1)
    Path(out, "load.json").write_text(json.dumps(stats, indent=1) + "\n")


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def run_figures(d):
    f = {"run": d.name, "mode": d.name.split("-", 1)[1]}
    p = d / "single.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        warm = [r["decode_tps"] for r in rows if r.get("rep", 0) > 0 and r.get("decode_tps")]
        f["single_depth"] = rows[0].get("prompt_tokens") if rows else None
        f["single_reps"] = [round(x, 2) for x in warm]
        f["single_tps"] = med(warm)
    p = d / "six.out"
    if p.exists():
        reqs, steps = [], []
        for line in p.read_text().splitlines():
            if not line.startswith("{"):
                continue
            j = json.loads(line)
            if "approx_aggregate_tps" in j:
                f["six_agg_tps"] = j["approx_aggregate_tps"]
            elif "req" in j and j.get("decode_tps"):
                reqs.append(j["decode_tps"])
                n, acc = j.get("predicted_n") or 0, j.get("draft_accepted") or 0
                if n > acc:
                    steps.append(1e3 * n / j["decode_tps"] / (n - acc))
        f["six_seq_tps"] = med(reqs)
        f["six_step_ms"] = med(steps)
    p = d / "load.json"
    if p.exists():
        f["load"] = json.loads(p.read_text())
    return f


def table(out):
    runs = [run_figures(d) for d in sorted(Path(out).iterdir()) if d.is_dir() and d.name[:2].isdigit()]
    keys = [("single_tps", "single decode tok/s", True), ("six_agg_tps", "six agg tok/s", True),
            ("six_seq_tps", "six seq tok/s", True), ("six_step_ms", "six step ms", False)]
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    lines = ["# Observability overhead: telemetry OFF vs ON (same binary, flags only)", "",
             "ON = `--telemetry --request-log-jsonl --metrics --slots` with /telemetry polled at 1 Hz, /metrics every "
             "4 s and one /events subscription held for the whole run (bench/obs_overhead.py load).", "",
             "| run | mode | " + " | ".join(k[1] for k in keys) + " | single warm reps | client load |",
             "|---|---|" + "---:|" * len(keys) + "---|---|"]
    for r in runs:
        ld = r.get("load")
        load_s = (f"telemetry {ld['telemetry']['ok']} ok / {ld['telemetry']['err']} err, metrics {ld['metrics']['ok']}"
                  f" / {ld['metrics']['err']}, events {ld['events']['frames']} frames" if ld else "-")
        lines.append(f"| {r['run']} | {r['mode']} | " + " | ".join(fmt(r.get(k[0])) for k in keys)
                     + f" | {r.get('single_reps', '-')} | {load_s} |")
    lines += ["", "| figure | OFF median | ON median | ON vs OFF | OFF spread | ON spread |", "|---|---:|---:|---:|---:|---:|"]
    for k, label, higher_better in keys:
        off = [r[k] for r in runs if r["mode"] == "off" and r.get(k) is not None]
        on = [r[k] for r in runs if r["mode"] == "on" and r.get(k) is not None]
        mo, mn = med(off), med(on)
        rel = f"{(mn / mo - 1) * 100:+.2f}%" if mo and mn else "-"
        spread = lambda xs, m: f"{(max(xs) - min(xs)) / m * 100:.2f}%" if len(xs) > 1 and m else "-"
        lines.append(f"| {label}{'' if higher_better else ' (lower is better)'} | {fmt(mo)} | {fmt(mn)} | {rel} | "
                     f"{spread(off, mo)} | {spread(on, mn)} |")
    lines += ["", "spread = (max - min) / median over the runs of one mode; an ON vs OFF difference inside the spread "
                  "is not a measurable cost."]
    text = "\n".join(lines) + "\n"
    Path(out, "RESULTS.md").write_text(text)
    print(text, end="")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("load")
    a1.add_argument("--base-url", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("table")
    a2.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "load":
        load(a.base_url.rstrip("/"), a.out)
    else:
        table(a.out)


if __name__ == "__main__":
    main()

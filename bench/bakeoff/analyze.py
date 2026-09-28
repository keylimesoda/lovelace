"""Summarize a bake-off directory: one row per run, steady-state window metrics.

Usage: python analyze.py OUTDIR [--warmup 360]
"""

import json
import statistics
import sys
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else float("nan")


def gpu_stats(path: Path):
    mem, pwr = [], []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                try:
                    mem.append(float(parts[1]))
                    pwr.append(float(parts[3]))
                except ValueError:
                    pass
    return (max(mem) if mem else float("nan"), statistics.fmean(pwr) if pwr else float("nan"))


def summarize(run_dir: Path, warmup: float):
    reqs_path = run_dir / "requests.jsonl"
    if not reqs_path.exists():
        return None
    reqs = [json.loads(line) for line in reqs_path.read_text().splitlines() if line.strip()]
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    end = max((r.get("t_end") or 0) for r in reqs) if reqs else 0
    ok = [r for r in reqs if r.get("status") == "ok"]
    steady = [r for r in ok if (r.get("t_end") or 0) >= warmup]
    window = max(1e-9, end - warmup)

    def ctoks(r):
        return (r.get("usage") or {}).get("completion_tokens") or 0

    def prompt(r):
        return (r.get("usage") or {}).get("prompt_tokens") or 0

    def cached(r):
        u = r.get("usage") or {}
        c = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        if c is None:
            c = (r.get("timings") or {}).get("cache_n")
        return c or 0

    def decode_tps(r):
        d = r.get("decode_s") or 0
        return ctoks(r) / d if d > 0 and ctoks(r) > 1 else None

    drafts = sum((r.get("timings") or {}).get("draft_n") or 0 for r in steady)
    accepted = sum((r.get("timings") or {}).get("draft_n_accepted") or 0 for r in steady)
    ttft = [r["ttft_s"] for r in steady if r.get("ttft_s") is not None]
    dtps = [x for x in (decode_tps(r) for r in steady) if x]
    ptot = sum(prompt(r) for r in steady)
    recompute = [prompt(r) - cached(r) for r in steady]
    peak_mem, mean_pwr = gpu_stats(run_dir / "gpu.csv")
    cmd = (run_dir / "server_cmd.txt").read_text().strip().splitlines()[-1] if (run_dir / "server_cmd.txt").exists() else "?"
    return {
        "run": run_dir.name,
        "fit": cmd,
        "turns": len(reqs),
        "errors": len(reqs) - len(ok) - sum(1 for r in reqs if r.get("status") == "cancelled"),
        "agg_tps_full": summary.get("agg_output_tps", float("nan")),
        "agg_tps_steady": sum(ctoks(r) for r in steady) / window,
        "ttft_p50": pct(ttft, 0.5),
        "ttft_p90": pct(ttft, 0.9),
        "decode_tps_mean": statistics.fmean(dtps) if dtps else float("nan"),
        "cache_hit": (sum(cached(r) for r in steady) / ptot) if ptot else float("nan"),
        "recompute_p50": pct(recompute, 0.5),
        "recompute_p90": pct(recompute, 0.9),
        "depth_p50": pct([prompt(r) for r in steady], 0.5),
        "accept": (accepted / drafts) if drafts else float("nan"),
        "peak_mib": peak_mem,
        "mean_w": mean_pwr,
    }


def main():
    out = Path(sys.argv[1])
    warmup = float(sys.argv[sys.argv.index("--warmup") + 1]) if "--warmup" in sys.argv else 360.0
    rows = [s for d in sorted(out.iterdir()) if d.is_dir() for s in [summarize(d, warmup)] if s]
    cols = ["run", "fit", "turns", "errors", "agg_tps_full", "agg_tps_steady", "ttft_p50", "ttft_p90",
            "decode_tps_mean", "cache_hit", "recompute_p50", "recompute_p90", "depth_p50", "accept", "peak_mib", "mean_w"]
    print(f"# Bake-off {out.name} (steady state = requests finishing after t={warmup:.0f}s)\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        cells = [f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        print("| " + " | ".join(cells) + " |")
    for d in sorted(out.iterdir()):
        log = d / "server.log"
        if d.is_dir() and log.exists():
            bad = [l for l in log.read_text(errors="replace").splitlines()
                   if any(k in l.lower() for k in ("error", "failed", "out of memory", "oom", "abort"))]
            if bad:
                print(f"\n## {d.name} server log errors ({len(bad)} lines, first 5)\n")
                print("```\n" + "\n".join(bad[:5]) + "\n```")


if __name__ == "__main__":
    main()

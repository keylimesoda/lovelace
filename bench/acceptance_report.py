"""Charter acceptance report: lovelace vs the previous setup (ninfer), from files the acceptance scripts wrote.

Reads (missing inputs are reported as "not run", never guessed):
  A1  eval/kld/results/wq-iq4xs-q5_0.json (lovelace production) vs eval/kld/results/wq-ninfer-rk4v4-e8.json
      (ninfer production), both against the same Q8_0 reference; plus the high-precision controls.
  B1  bench/results/h2h/{lovelace,ninfer}-sweep.jsonl (bench/depth_sweep.py)
  B2  bench/results/h2h/<label>/requests.jsonl for the 6-agent replays (bench/replay)

Usage: python3 bench/acceptance_report.py [--out bench/results/acceptance/REPORT.md]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KLD = ROOT / "eval/kld/results"
H2H = ROOT / "bench/results/h2h"
sys.path.insert(0, str(ROOT / "bench/bakeoff"))
from analyze import summarize  # noqa: E402


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def a1(lines):
    lines += ["## A1 quality (KL vs Q8_0 reference, same 24 windows, 32k-170k; lower KL is better)", "",
              "| setup | KL mean | KL p99 | top-1 agree | ΔNLL |", "|---|---:|---:|---:|---:|"]
    rows = {
        "lovelace production (GPU-5 + q5_0 KV)": KLD / "wq-iq4xs-q5_0.json",
        "lovelace control (GPU-5 + f16 KV)": KLD / "wq-iq4xs-f16.json",
        "ninfer production (artifact + rk4v4-e8)": KLD / "wq-ninfer-rk4v4-e8.json",
    }
    rows.update({f"ninfer control ({p.stem.removeprefix('wq-ninfer-')})": p
                 for p in sorted(KLD.glob("wq-ninfer-*.json")) if "e8" not in p.stem})
    got = {}
    for name, p in rows.items():
        d = load(p)
        if not d:
            lines.append(f"| {name} | not run | | | |")
            continue
        o = d["overall"]
        got[name] = o
        lines.append(f"| {name} | {o['kl_mean']:.4f} | {o['kl_p99']:.3f} | {o['top1_agree_pct']:.2f}% | {o['delta_nll']:+.4f} |")
    lov, nin = got.get("lovelace production (GPU-5 + q5_0 KV)"), got.get("ninfer production (artifact + rk4v4-e8)")
    verdict = "not decided (missing input)" if not (lov and nin) else (
        "PASS" if lov["kl_mean"] <= nin["kl_mean"] * 1.05 else "FAIL")
    lines += ["", f"**A1 (lovelace KL <= ninfer KL, 5% tolerance): {verdict}**", ""]


def b1(lines):
    lines += ["## B1 single-stream speed (bench/depth_sweep.py; same prompts; decode = mean of warm reps)", "",
              "| depth | lovelace prefill | ninfer prefill | lovelace decode | ninfer decode | lovelace accept | ninfer accept |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    data = {}
    for eng in ("lovelace", "ninfer"):
        p = H2H / f"{eng}-sweep.jsonl"
        data[eng] = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    if not all(data.values()):
        lines += ["| not run | | | | | | |", "", "**B1: not decided (missing input)**", ""]
        return
    wins = []
    for target in sorted({r["target"] for r in data["lovelace"]}):
        cells = {}
        for eng, rows in data.items():
            rs = [r for r in rows if r["target"] == target]
            warm = [r for r in rs if r["rep"] > 0 and r.get("decode_tps")]
            dn = sum(r.get("draft_n") or 0 for r in warm)
            da = sum(r.get("draft_accepted") or 0 for r in warm)
            cells[eng] = (rs[0].get("prompt_tps") if rs else None,
                          sum(r["decode_tps"] for r in warm) / len(warm) if warm else None,
                          da / dn if dn else None, rs[0].get("prompt_tokens") if rs else None)
        L, N = cells["lovelace"], cells["ninfer"]
        f = lambda x, s="{:.0f}": s.format(x) if x is not None else "-"
        lines.append(f"| {f(L[3])} | {f(L[0])} | {f(N[0])} | {f(L[1], '{:.1f}')} | {f(N[1], '{:.1f}')} | {f(L[2], '{:.2f}')} | {f(N[2], '{:.2f}')} |")
        if L[1] and N[1]:
            wins.append(L[1] >= N[1] * 0.95)
    verdict = "PASS" if wins and all(wins) else "FAIL" if wins else "not decided"
    lines += ["", f"**B1 (lovelace decode >= ninfer at every depth, 5% tolerance): {verdict}**", ""]


def b2(lines):
    lines += ["## B2 multi-agent (6 agents, real depth, 20 min; steady state)", "",
              "| run | errors | steady agg tok/s | TTFT p50 / p90 s | decode/agent | cache hit | recompute p90 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    runs = {}
    for d in sorted(p for p in H2H.iterdir() if p.is_dir()) if H2H.exists() else []:
        s = summarize(d, 180.0)
        if s:
            runs[d.name] = s
            lines.append(f"| {d.name} | {s['errors']} | {s['agg_tps_steady']:.1f} | {s['ttft_p50']:.1f} / {s['ttft_p90']:.1f}"
                         f" | {s['decode_tps_mean']:.1f} | {s['cache_hit']:.2f} | {s['recompute_p90']} |")
    lov = [s for n, s in runs.items() if n.startswith("lovelace")]
    nin = [s for n, s in runs.items() if n.startswith("ninfer")]
    if not (lov and nin):
        lines += ["", "**B2: not decided (missing input)**", ""]
        return
    best_n = max(nin, key=lambda s: s["agg_tps_steady"])
    best_l = max(lov, key=lambda s: s["agg_tps_steady"])
    ok = best_l["errors"] == 0 and best_l["agg_tps_steady"] >= best_n["agg_tps_steady"] * 0.95
    lines += ["", f"**B2 (lovelace errors 0 and aggregate >= best ninfer, 5% tolerance): {'PASS' if ok else 'FAIL'}**", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/acceptance/REPORT.md")
    a = ap.parse_args()
    lines = ["# Acceptance vs previous setup (ninfer)", "",
             "Generated by bench/acceptance_report.py from script outputs; see docs/CHARTER.md.", ""]
    a1(lines)
    b1(lines)
    b2(lines)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

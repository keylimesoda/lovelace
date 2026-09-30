#!/usr/bin/env python3
"""README charts (docs/img/*.svg) from committed benchmark results. Standard library only; deterministic output.

Reads:
  bench/results/final/REPORT.md        lovelace 6 x 60k replay row (steady aggregate, TTFT p50/p90)
  bench/results/acceptance/REPORT.md   ninfer-3lane-6a row of the B2 table (same replay workload on ninfer)
  bench/results/final/sweep.sweep.out  lovelace single-stream depth sweep (cold prefill, warm decode)
  bench/results/h2h/ninfer-sweep.sweep.out  the same sweep on ninfer

Writes docs/img/multi-agent.svg and docs/img/depth-sweep.svg.
Usage: python3 docs/make_charts.py      (from anywhere; paths are relative to the repo root)

Colors are mid-tones and the background is transparent, so the charts read on GitHub's light and dark themes.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

LOVELACE = "#4493f8"
NINFER = "#e0823d"
INK = "#8b949e"  # GitHub's muted text: readable on white and on #0d1117
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------- inputs

def table_row(path: Path, first_cell: str) -> list[str]:
    """Cells of the first markdown table row whose first cell is `first_cell`."""
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and cells[0] == first_cell:
            return cells
    raise SystemExit(f"{path.relative_to(ROOT)}: no table row '{first_cell}'")


def header_index(path: Path, first_cell: str, column: str) -> int:
    """Column index of `column` in the header of the table that contains the row `first_cell`."""
    header: list[str] = []
    for line in path.read_text().splitlines():
        if not line.lstrip().startswith("|"):
            header = []
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not header:
            header = cells
        elif cells[0] == first_cell:
            return header.index(column)
    raise SystemExit(f"{path.relative_to(ROOT)}: no table with row '{first_cell}'")


def replay_point(path: Path, row: str, agg_col: str, ttft_col: str) -> tuple[float, float, float]:
    cells = table_row(path, row)
    agg = float(cells[header_index(path, row, agg_col)])
    p50, p90 = (float(x) for x in cells[header_index(path, row, ttft_col)].split("/"))
    return agg, p50, p90


def sweep(path: Path) -> list[tuple[int, float, float]]:
    """(prompt tokens, cold prefill tok/s, warm decode tok/s) from the summary lines of a depth_sweep.py output."""
    pts = []
    for line in path.read_text().splitlines():
        m = re.fullmatch(r"\s*(\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| [\d.]+", line)
        if m:
            pts.append((int(m[2]), float(m[3]), float(m[4])))
    if not pts:
        raise SystemExit(f"{path.relative_to(ROOT)}: no sweep summary lines")
    return pts


# ---------------------------------------------------------------- SVG helpers

def fmt(v: float, decimals: int) -> str:
    return f"{v:,.{decimals}f}"


def text(x, y, s, size=13, anchor="start", weight="normal", fill=INK, opacity=1.0):
    op = "" if opacity == 1.0 else f' fill-opacity="{opacity}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{op}>{s}</text>')


def svg(w: int, h: int, body: list[str], title: str) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'font-family="{FONT}" role="img" aria-label="{title}">',
        f"<title>{title}</title>",
        *body,
        "</svg>",
        "",
    ])


def legend(x: float, y: float, items: list[tuple[str, str, float]]) -> list[str]:
    out = []
    for label, color, opacity in items:
        out.append(f'<rect x="{x:.1f}" y="{y - 10:.1f}" width="12" height="12" rx="2" fill="{color}" '
                   f'fill-opacity="{opacity}"/>')
        out.append(text(x + 18, y, label, 12))
        x += 26 + 7.2 * len(label)
    return out


def hbars(x0, y0, w, title, subtitle, groups, vmax, label_w=78, bar_h=22, gap=6, group_gap=18):
    """Horizontal bar panel. groups: [(label, [(value, color, opacity, value_text)])]."""
    out = [text(x0, y0, title, 14, weight="600"), text(x0, y0 + 18, subtitle, 12)]
    y = y0 + 38
    plot_w = w - label_w - 60
    for label, bars in groups:
        gh = len(bars) * bar_h + (len(bars) - 1) * gap
        out.append(text(x0 + label_w - 10, y + gh / 2 + 5, label, 13, "end", "600"))
        for value, color, opacity, vtext in bars:
            bw = max(plot_w * value / vmax, 1.0)
            out.append(f'<rect x="{x0 + label_w:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h}" rx="3" '
                       f'fill="{color}" fill-opacity="{opacity}"/>')
            out.append(text(x0 + label_w + bw + 6, y + bar_h / 2 + 5, vtext, 12, weight="600"))
            y += bar_h + gap
        y += group_gap - gap
    return out


def line_panel(x0, y0, w, h, title, subtitle, series, xmax, ymax, ystep, xticks, decimals):
    """Line chart panel. series: [(color, [(x, y)])], all with the same x points. At each x the higher point is
    labelled above and the lower one below, so the value labels never collide."""
    out = [text(x0, y0, title, 14, weight="600"), text(x0, y0 + 18, subtitle, 12)]
    px0, py0 = x0 + 44, y0 + 36          # plot area origin (top-left)
    pw, ph = w - 58, h - 36 - 30
    sx = lambda v: px0 + pw * v / xmax
    sy = lambda v: py0 + ph * (1 - v / ymax)
    v = 0
    while v <= ymax + 1e-9:
        out.append(f'<line x1="{px0:.1f}" y1="{sy(v):.1f}" x2="{px0 + pw:.1f}" y2="{sy(v):.1f}" stroke="{INK}" '
                   f'stroke-opacity="0.25" stroke-width="1"/>')
        out.append(text(px0 - 6, sy(v) + 4, f"{v:,.0f}", 11, "end"))
        v += ystep
    for t in xticks:
        out.append(text(sx(t), py0 + ph + 18, f"{t // 1000}k", 11, "middle"))
    out.append(text(px0 + pw / 2, py0 + ph + 34, "prompt depth (tokens)", 11, "middle"))
    for color, pts in series:
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
        for j, (x, y) in enumerate(pts):
            below = any(other[j][1] > y for _, other in series if other is not pts)
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>')
            out.append(text(sx(x), sy(y) + (17 if below else -9), fmt(y, decimals), 11, "middle", "600", fill=color))
    return out


# ---------------------------------------------------------------- charts

def multi_agent() -> str:
    final = ROOT / "bench/results/final/REPORT.md"
    accept = ROOT / "bench/results/acceptance/REPORT.md"
    l_agg, l50, l90 = replay_point(final, "6 x 60k (B2/C2)", "steady agg tok/s", "TTFT p50 / p90 s")
    n_agg, n50, n90 = replay_point(accept, "ninfer-3lane-6a", "steady agg tok/s", "TTFT p50 / p90 s")

    w, h = 820, 250
    body = [text(20, 26, "6 coding agents at once, each at 60k+ tokens of context: 20-minute replay of real sessions",
                 15, weight="600")]
    left = hbars(20, 60, 380, "Aggregate output", "tokens/s, steady state (higher is better)",
                 [("lovelace", [(l_agg, LOVELACE, 1.0, f"{l_agg:.1f}  ({l_agg / n_agg:.1f}x)")]),
                  ("ninfer", [(n_agg, NINFER, 1.0, f"{n_agg:.1f}")])],
                 vmax=max(l_agg, n_agg) * 1.12, bar_h=30)
    right = hbars(420, 60, 400, "Time to first token", "seconds, p50 and p90 (lower is better)",
                  [("lovelace", [(l50, LOVELACE, 1.0, f"p50 {l50:.1f}"), (l90, LOVELACE, 0.55, f"p90 {l90:.1f}")]),
                   ("ninfer", [(n50, NINFER, 1.0, f"p50 {n50:.1f}"), (n90, NINFER, 0.55, f"p90 {n90:.1f}")])],
                  vmax=max(n90, l90) * 1.08, bar_h=14, gap=4)
    body += left + right
    body.append(text(20, h - 26, "RTX 4090 24 GB, Qwen3.8-27B. ninfer: best configuration that starts (3 lanes).", 11))
    body.append(text(20, h - 10, "Sources: bench/results/final/REPORT.md (lovelace), "
                                 "bench/results/acceptance/REPORT.md (ninfer).", 11))
    return svg(w, h, body, "lovelace vs ninfer: 6-agent aggregate throughput and time to first token")


def depth_sweep() -> str:
    lov = sweep(ROOT / "bench/results/final/sweep.sweep.out")
    nin = sweep(ROOT / "bench/results/h2h/ninfer-sweep.sweep.out")
    xmax, xticks = 200_000, [0, 50_000, 100_000, 150_000, 200_000]
    w, h = 820, 330
    body = [text(20, 26, "One agent: speed vs context depth (same prompts on both engines)", 15, weight="600")]
    body += line_panel(20, 60, 390, 240, "Decode", "tokens/s, warm cache, MTP speculation",
                       [(LOVELACE, [(p, d) for p, _, d in lov]), (NINFER, [(p, d) for p, _, d in nin])],
                       xmax, 160, 40, xticks, 1)
    body += line_panel(430, 60, 390, 240, "Prefill", "tokens/s, cold prompt",
                       [(LOVELACE, [(p, f) for p, f, _ in lov]), (NINFER, [(p, f) for p, f, _ in nin])],
                       xmax, 3500, 500, xticks, 0)
    body += legend(640, 26, [("lovelace", LOVELACE, 1.0), ("ninfer", NINFER, 1.0)])
    body.append(text(20, h - 8, "RTX 4090 24 GB, Qwen3.8-27B. Sources: bench/results/final/sweep.sweep.out, "
                                "bench/results/h2h/ninfer-sweep.sweep.out (bench/depth_sweep.py).", 11))
    return svg(w, h, body, "lovelace vs ninfer: single-stream decode and prefill by context depth")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("multi-agent.svg", multi_agent), ("depth-sweep.svg", depth_sweep)):
        (OUT / name).write_text(fn())
        print(f"wrote docs/img/{name}")


if __name__ == "__main__":
    main()

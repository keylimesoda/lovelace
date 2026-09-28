"""Summary table over lovelace-kld compare results: one row per (run, depth bucket) plus overall.

usage: python3 summarize.py results/wq-iq4xs-f16.json results/kv-iq4xs-*.json
"""

import json
import sys
from pathlib import Path

COLS = [
    ("pos", "positions", "{:>7d}"),
    ("KL mean", "kl_mean", "{:>9.5f}"),
    ("KL med", "kl_median", "{:>9.5f}"),
    ("KL p99", "kl_p99", "{:>8.4f}"),
    ("KL p99.9", "kl_p999", "{:>8.4f}"),
    ("top1 %", "top1_agree_pct", "{:>7.2f}"),
    ("dNLL", "delta_nll", "{:>+9.5f}"),
    ("p_tail", "p_tail_mean", "{:>8.5f}"),
]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs = []
    for p in map(Path, sys.argv[1:]):
        r = json.loads(p.read_text())
        base = r["base"]["meta"]
        label = f"{Path(r['model']).stem} kv {r['run']['ctk']}/{r['run']['ctv']}"
        ref = f"{Path(base['model']).stem} kv {base['ctk']}/{base['ctv']}"
        runs.append((p.stem, label, ref, r))
    w_run = max(len(x[0]) for x in runs)
    head = f"{'run':<{w_run}}  {'bucket':<8}" + "".join(f" {h:>{len(f.format(0))}}" for h, _, f in COLS)
    last_ref = None
    for name, label, ref, r in runs:
        if ref != last_ref:
            print(f"\nreference: {ref}\n{head}")
            last_ref = ref
        print(f"{name:<{w_run}}  # {label}")
        rows = list(r["buckets"].items()) + [("overall", r["overall"])]
        for bucket, s in rows:
            print(f"{'':<{w_run}}  {bucket:<8}" + "".join(" " + f.format(s[k]) for _, k, f in COLS))


if __name__ == "__main__":
    main()

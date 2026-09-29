"""Per-ubatch cost table of a llama-server prefill capture (bench/resident_profile.sh).

Segmentation as in bench/step_table.py: the decode thread's runtime calls are split at synchronizing calls; a segment
that runs a Gated-DeltaNet kernel is a target ubatch, the segments until the next target ubatch are the MTP draft
context's pass over the same tokens. One row = one target ubatch plus its MTP pass. Kernel time is summed per class
(FA split into the FA kernel, its mask pre-pass and the K/V -> f16 conversion); uploads are H2D memcpys; host gap =
row wall time during which the GPU runs nothing (graph build, input/mask build on the host, sync latency).

Usage: python3 bench/resident_table.py CAPTURE.sqlite [--bands 3]
"""

import argparse
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_table import SYNC_CALLS, classify  # noqa: E402


def kclass(short, dem):
    s = short.lower()
    if "mask_to_kv_max" in s or "mask_to_sparse" in s or "fattn_mma_q_visible" in s or "attn_ext_mma_q_visible" in s:
        return "FA mask pre-pass"
    if ("dequantize" in s or "convert_unary" in s or "to_fp16" in s) and "q5_0" in dem.lower() or "k_dequantize" in s:
        return "KV->f16 convert"
    if "kq_mask" in s:
        return "mask (GPU build)"
    c, _ = classify(short, dem)
    return "FA kernel" if c == "FA" else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--bands", type=int, default=3, help="report means over this many equal depth bands too")
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    q = lambda sql: db.execute(sql).fetchall()  # noqa: E731
    strings = dict(q("SELECT id, value FROM StringIds"))
    rt = q("SELECT start, end, correlationId, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")
    ker = q("SELECT start, end, correlationId, shortName, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    mem = []
    for tbl, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        try:
            cols = {r[1] for r in q(f"PRAGMA table_info({tbl})")}
            ck = "copyKind" if "copyKind" in cols else "0"
            mem += [(s, e, c, kind, b, k) for s, e, c, b, k in q(f"SELECT start, end, correlationId, bytes, {ck} FROM {tbl}")]
        except sqlite3.OperationalError:
            pass

    corr_tid = {c: tid for _, _, c, _, tid in rt}
    launches = defaultdict(int)
    for k in ker:
        if k[2] in corr_tid:
            launches[corr_tid[k[2]]] += 1
    main_tid = max(launches, key=launches.get)

    seg_of_corr, seg = {}, 0
    for s, e, c, n, tid in rt:
        if tid != main_tid:
            continue
        seg_of_corr[c] = seg
        if strings.get(n, "").startswith(SYNC_CALLS):
            seg += 1
    segs = defaultdict(lambda: {"k": [], "m": []})
    for s, e, c, sn, dn in ker:
        if (sg := seg_of_corr.get(c)) is not None:
            segs[sg]["k"].append((s, e, strings.get(sn, "?"), strings.get(dn, "?")))
    for s, e, c, kind, b, ck in mem:
        if (sg := seg_of_corr.get(c)) is not None:
            segs[sg]["m"].append((s, e, kind, b, ck))
    order = sorted(i for i in segs if segs[i]["k"] or segs[i]["m"])
    is_target = {i: any(("gated_delta" in x[2].lower() or "ssm_conv" in x[2].lower()) for x in segs[i]["k"]) for i in order}
    targets = [i for i in order if is_target[i]]

    rows = []
    for j in range(len(targets) - 1):
        members = [i for i in order if targets[j] <= i < targets[j + 1]]
        cls, acts = defaultdict(float), []
        h2d_bytes = 0
        for i in members:
            pre = "" if is_target[i] else "MTP "
            for s, e, sn, dn in segs[i]["k"]:
                cls[pre + kclass(sn, dn)] += (e - s) / 1e6
                acts.append((s, e))
            for s, e, kind, b, ck in segs[i]["m"]:
                name = "upload H2D" if (kind == "memcpy" and ck == 1) else f"{kind} kind{ck}"
                cls[name] += (e - s) / 1e6
                if name == "upload H2D":
                    h2d_bytes += b
                acts.append((s, e))
        nxt = min(x[0] for x in segs[targets[j + 1]]["k"])
        acts.sort()
        busy, cs, ce = 0, None, None
        for s, e in acts:
            e = min(e, nxt)
            if ce is None or s > ce:
                if ce is not None:
                    busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += ce - cs
        wall = (nxt - acts[0][0]) / 1e6
        cls["host gap (GPU idle)"] = wall - busy / 1e6
        rows.append((wall, cls, h2d_bytes))
    # the capture starts and ends mid-request: drop a partial first row and rows that are not full prefill ubatches
    if not rows:
        print("no ubatches found")
        return
    med = statistics.median(r[0] for r in rows)
    rows = [r for r in rows if r[0] > 0.3 * med]

    def table(sel, title):
        n = len(sel)
        wall = statistics.mean(r[0] for r in sel)
        keys = sorted({k for r in sel for k in r[1]}, key=lambda k: -sum(r[1].get(k, 0) for r in sel))
        print(f"{title}: {n} ubatches, wall {wall:.1f} ms/ubatch, H2D {statistics.mean(r[2] for r in sel) / 2**20:.0f} MiB/ubatch")
        print(f"  {'class':30s} {'ms/ubatch':>9s} {'share':>6s}")
        for k in keys:
            v = sum(r[1].get(k, 0) for r in sel) / n
            if v >= 0.05:
                print(f"  {k:30s} {v:9.2f} {100 * v / wall:5.1f}%")

    print(f"{a.db}")
    table(rows, "all")
    nb = max(1, a.bands)
    for b in range(nb):
        sel = rows[b * len(rows) // nb:(b + 1) * len(rows) // nb]
        if sel:
            table(sel, f"band {b + 1}/{nb} (ubatches {b * len(rows) // nb}-{(b + 1) * len(rows) // nb - 1})")


if __name__ == "__main__":
    main()

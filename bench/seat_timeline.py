"""Per-seat timeline of a replay run (charter C3): where the seats' time goes in the timed window.

A seat is room in the KV pool for one session at the run's depth: floor(cells / (median prompt depth + 8.4k reserve))
(2 at 150k, 4 at 60k), or --seats. The script rebuilds every request's life on the server from the server log
(launch, restore, prefill, decode, release) and joins it with the client records (arrival = t_send), then samples the
window every 50 ms:

  seat states  decode | prefill cold (> half the prompt uncached) | prefill delta | restore (copies RAM -> VRAM
               after admission) | idle, head waits on a disk promotion | idle, head waits (other: kv/park/sync) |
               idle, no request waiting | extra (more tasks than seats: small sessions)
  GPU phases   cold prefill running | delta prefill only | decode only | restore only | idle with queue | idle
               with the output tokens produced in each (decode progress lines, interpolated)

Usage: python bench/seat_timeline.py RUN_DIR [--log SERVER_LOG] [--seats N] [--from S] [--csv OUT.csv]
  RUN_DIR: replay output (requests.jsonl, telemetry-end.json); SERVER_LOG default RUN_DIR/../<name>.server.log
  --from: start of the window in client seconds (default: RUN_DIR/warmup_s, else 180), the end is the last t_end
"""

import argparse
import json
import re
import statistics
import sys
from bisect import bisect_right
from pathlib import Path

TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")
SLOT = re.compile(r"id +(\d+) \| task (-?\d+) \|")
PAT = {
    "launch": re.compile(r"launch_slot_: id +(\d+) \| task (\d+)"),
    "rest": re.compile(r"session_rest: id +(\d+) \| task -?\d+ \| (restoring|prefetching) session (\d+): (\d+) tokens"),
    "unpark": re.compile(r"unpark session: (\d+) tokens"),
    "pe": re.compile(r"prompt eval time = +([\d.]+) ms / +(\d+) tokens"),
    "ev": re.compile(r"\| +eval time = +([\d.]+) ms / +(\d+) tokens"),
    "release": re.compile(r"release: id +(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+)"),
    "ngen": re.compile(r"n_gen = +(\d+),"),
    "promo": re.compile(r"session (\d+) promoted from disk: ([\d.]+) MiB in ([\d.]+) s"),
    "demo": re.compile(r"session (\d+) demoted to disk: ([\d.]+) MiB in ([\d.]+) s"),
    "park": re.compile(r"parked session (\d+): (\d+) tokens, ([\d.]+) MiB copied, ([\d.]+) ms"),
}


def ts(line):
    m = TS.match(line)
    if not m:
        return None
    mi, s, ms, us = map(int, m.groups())
    return mi * 60 + s + ms / 1e3 + us / 1e6


def parse_log(path):
    tasks = {}  # task id -> dict
    promos, demos, parks = [], [], []
    last_rest = {}  # slot -> (t, session, prefetch)
    for line in open(path, errors="replace"):
        t = ts(line)
        if t is None:
            continue
        if m := PAT["promo"].search(line):
            promos.append((t - float(m[3]), t, int(m[1]), float(m[2])))
            continue
        if m := PAT["demo"].search(line):
            demos.append((t - float(m[3]), t, int(m[1]), float(m[2])))
            continue
        if m := PAT["rest"].search(line):
            last_rest[int(m[1])] = (t, int(m[3]), m[2] == "prefetching")
            continue
        if m := PAT["park"].search(line):
            parks.append((t - float(m[4]) / 1e3, t, int(m[1]), float(m[3])))
            continue
        if m := PAT["launch"].search(line):
            slot, tid = int(m[1]), int(m[2])
            r = last_rest.get(slot)
            sess = r[1] if r and t - r[0] < 2.0 else None
            tasks[tid] = {"id": tid, "slot": slot, "t_launch": t, "session": sess, "t_rest_done": None,
                          "ngen": [], "t_release": None}
            continue
        sm = SLOT.search(line)
        if not sm:
            continue
        tid = int(sm[2])
        if PAT["unpark"].search(line):
            if tid in tasks:
                tasks[tid]["t_rest_done"] = t
            continue
        if tid not in tasks:
            continue
        k = tasks[tid]
        if m := PAT["pe"].search(line):
            k["pf_ms"], k["pf_n"] = float(m[1]), int(m[2])
        elif m := PAT["ev"].search(line):
            k["dec_ms"], k["dec_n"] = float(m[1]), int(m[2])
        elif m := PAT["ngen"].search(line):
            k["ngen"].append((t, int(m[1])))
        elif m := PAT["release"].search(line):
            k["t_release"], k["n_ctx"] = t, int(m[3])
    return tasks, promos, demos, parks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--seats", type=int)
    ap.add_argument("--from", dest="t_from", type=float)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()

    rd = a.run_dir
    log = a.log or rd.parent / f"{rd.name}.server.log"
    reqs = sorted((json.loads(x) for x in open(rd / "requests.jsonl")), key=lambda r: r["t_send"])
    tel = json.loads((rd / "telemetry-end.json").read_text()) if (rd / "telemetry-end.json").exists() else {}
    kv = (tel.get("lovelace") or {}).get("kv") or (tel.get("memory") or {}).get("kv") or {}  # ext schema 2, or schema 1
    cells = kv.get("cells_total") or 376832
    t_from = a.t_from if a.t_from is not None else (
        float((rd / "warmup_s").read_text()) if (rd / "warmup_s").exists() else 180.0)

    tasks, promos, demos, parks = parse_log(log)
    done = sorted((k for k in tasks.values() if k["t_release"] is not None and "pf_n" in k and "dec_n" in k),
                  key=lambda k: k["t_release"])

    # join client records with server tasks on (prompt_n, predicted_n), in completion order
    ok = [r for r in reqs if r.get("status") == "ok" and r.get("timings")]
    used = set()
    pairs = []
    for r in sorted(ok, key=lambda r: r["t_end"]):
        key = (r["timings"]["prompt_n"], r["timings"]["predicted_n"])
        for k in done:
            if k["id"] not in used and (k["pf_n"], k["dec_n"]) == key:
                used.add(k["id"])
                pairs.append((r, k))
                break
    if not pairs:
        sys.exit("no client request matched a server task")
    off = statistics.median(k["t_release"] - r["t_end"] for r, k in pairs)
    for r, k in pairs:
        r["_task"] = k
        k["_req"] = r

    t0, t1 = off + t_from, off + max((r.get("t_end") or 0.0) for r in reqs)
    depth = statistics.median(r["timings"]["prompt_n"] + r["timings"]["cache_n"] for r in ok if r["t_end"] >= t_from)
    seats = a.seats or max(1, int(cells // (depth + 8192 + 256)))

    # per task: intervals of its states
    ivs = []  # (start, end, state, task)
    for k in tasks.values():
        tl = k["t_launch"]
        tr = k["t_release"] if k["t_release"] is not None else t1
        if tr < t0 or tl > t1:
            continue
        if "dec_ms" in k:
            t_pf_end = tr - k["dec_ms"] / 1e3
            t_pf_start = t_pf_end - k["pf_ms"] / 1e3
            cold = k["pf_n"] > 0.5 * max(1, k.get("n_ctx", k["pf_n"]) - k["dec_n"])
        else:  # cut off by the end of the run
            t_pf_end = t_pf_start = tr
            cold = False
        t_rest = k["t_rest_done"] or tl
        k["cold"] = cold
        k["pf_end"] = t_pf_end
        ivs.append((tl, max(tl, min(t_rest, tr)), "restore", k))
        ivs.append((max(tl, t_rest), max(tl, t_rest, t_pf_end), "prefill_cold" if cold else "prefill_delta", k))
        ivs.append((max(tl, t_rest, t_pf_end), tr, "decode", k))

    # queue: client arrival -> launch (unmatched requests wait until the end)
    waits = []
    for r in reqs:
        arr = off + r["t_send"]
        k = r.get("_task")
        waits.append((arr, k["t_launch"] if k else t1, k["session"] if k else None))
    waits.sort()

    # output tokens: piecewise linear n_gen per task
    def gen_curve(k):
        pts = [(k["pf_end"], 0)] + [p for p in k["ngen"] if p[0] >= k["pf_end"]]
        if "dec_n" in k:
            pts.append((k["t_release"], k["dec_n"]))
        return pts

    curves = [gen_curve(k) for k in tasks.values() if "pf_end" in k]

    def tokens_at(pts, t):
        if t <= pts[0][0]:
            return 0.0
        if t >= pts[-1][0]:
            return float(pts[-1][1])
        i = bisect_right([p[0] for p in pts], t)
        (ta, na), (tb, nb) = pts[i - 1], pts[i]
        return na + (nb - na) * (t - ta) / max(1e-9, tb - ta)

    seat_names = ["decode", "prefill_cold", "prefill_delta", "restore", "idle_q_disk", "idle_q_other", "idle_no_demand"]
    gpu_names = ["cold_prefill", "delta_prefill", "decode_only", "restore_only", "idle_queue", "idle"]
    seat_s = {n: 0.0 for n in seat_names}
    extra_s = 0.0
    gpu_s = {n: 0.0 for n in gpu_names}
    gpu_tok = {n: 0.0 for n in gpu_names}
    busy_decode = []
    rows = []

    ivs.sort(key=lambda x: x[0])
    t = t0
    dt = a.dt
    prev_tok = sum(tokens_at(c, t) for c in curves)
    while t < t1:
        tm = t + dt / 2
        states = [st for (s, e, st, _) in ivs if s <= tm < e and e > s]
        q = [w for w in waits if w[0] <= tm < w[1]]
        promoting = {p[2] for p in promos if p[0] <= tm < p[1]}
        head_disk = bool(q) and min(q)[2] in promoting
        order = ["decode", "prefill_cold", "prefill_delta", "restore"]
        filled = sorted(states, key=order.index)
        for i in range(seats):
            if i < len(filled):
                seat_s[filled[i]] += dt
            elif q:
                seat_s["idle_q_disk" if head_disk else "idle_q_other"] += dt
            else:
                seat_s["idle_no_demand"] += dt
        extra_s += max(0, len(filled) - seats) * dt
        if "prefill_cold" in states:
            g = "cold_prefill"
        elif "prefill_delta" in states:
            g = "delta_prefill"
        elif "decode" in states:
            g = "decode_only"
            busy_decode.append(states.count("decode"))
        elif "restore" in states:
            g = "restore_only"
        else:
            g = "idle_queue" if q else "idle"
        tok = sum(tokens_at(c, t + dt) for c in curves)
        gpu_s[g] += dt
        gpu_tok[g] += tok - prev_tok
        prev_tok = tok
        if a.csv is not None:
            rows.append((round(t - off, 2), states.count("decode"), states.count("prefill_cold"),
                         states.count("prefill_delta"), states.count("restore"), len(q), int(head_disk), g))
        t += dt

    win = t1 - t0
    steady = [r for r in ok if r["t_end"] >= t_from]
    out_tok = sum(r["timings"]["predicted_n"] for r in steady)
    print(f"# Seat timeline: {rd.name}\n")
    print(f"window {t_from:.0f}-{t1 - off:.0f} s ({win:.0f} s), {len(steady)} turns ending in it, {out_tok} tokens "
          f"({out_tok / win:.1f} t/s), median depth {depth / 1e3:.1f}k, {cells} cells -> {seats} seats; "
          f"client/server clock offset {off:.2f} s from {len(pairs)} matched requests\n")
    print("| seat state | seat-s | share |")
    print("|---|---:|---:|")
    tot = seats * win
    for n in seat_names:
        print(f"| {n} | {seat_s[n]:.0f} | {100 * seat_s[n] / tot:.1f}% |")
    print(f"| (extra tasks beyond {seats} seats) | {extra_s:.0f} | |\n")
    print("| GPU phase | s | share | output tokens | t/s in phase |")
    print("|---|---:|---:|---:|---:|")
    for n in gpu_names:
        print(f"| {n} | {gpu_s[n]:.0f} | {100 * gpu_s[n] / win:.1f}% | {gpu_tok[n]:.0f} | "
              f"{gpu_tok[n] / gpu_s[n] if gpu_s[n] else 0:.1f} |")
    if busy_decode:
        print(f"\ndecode-only steps: {statistics.mean(busy_decode):.2f} decoding tasks on average")

    # per-request views
    cold = [k for k in tasks.values() if k.get("cold") and t0 <= k["t_launch"] <= t1]
    pw = [p for p in promos if t0 <= p[1] <= t1]
    pk = [p for p in parks if t0 <= p[1] <= t1 and p[3] > 0]
    qw = sorted(w[1] - w[0] for w in waits if w[0] >= t0)
    print(f"\ncold prefills launched in the window: {len(cold)} "
          f"({sum(k.get('pf_n', 0) for k in cold)} tokens, {sum(k.get('pf_ms', 0) for k in cold) / 1e3:.0f} s)")
    print(f"disk promotions: {len(pw)} ({sum(p[3] for p in pw) / 1024:.0f} GiB, {sum(p[1] - p[0] for p in pw):.0f} s of reads)"
          f"; demotions (written): {len([d for d in demos if t0 <= d[1] <= t1])}; parks that synced first: {len(pk)} "
          f"({sum(p[1] - p[0] for p in pk):.1f} s)")
    if qw:
        print(f"queue wait (arrival -> admission): p50 {qw[len(qw) // 2]:.1f} s, p90 {qw[int(len(qw) * 0.9)]:.1f} s, "
              f"max {qw[-1]:.1f} s")
    rest = [k["t_rest_done"] - k["t_launch"] for k in tasks.values()
            if k["t_rest_done"] and t0 <= k["t_launch"] <= t1]
    if rest:
        print(f"restore after admission: {len(rest)}, mean {1e3 * statistics.mean(rest):.0f} ms, "
              f"total {sum(rest):.1f} s")

    if a.csv is not None:
        with open(a.csv, "w") as f:
            f.write("t,decode,prefill_cold,prefill_delta,restore,queue,head_disk,gpu_phase\n")
            for r in rows:
                f.write(",".join(map(str, r)) + "\n")


if __name__ == "__main__":
    main()

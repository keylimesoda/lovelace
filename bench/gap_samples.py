"""Where the decode thread spends the host gap before each target decode (a CPU_SAMPLE=1 bench/step_profile.sh capture).

Window per step: from the end of the last GPU activity before a target decode's first kernel (normally the last
draft step's logits copy) to that first kernel. Prints the decode thread's CUDA runtime calls in the window (time
and offset of the first one from the window start) and its CPU samples: inclusive counts per function (a function
counts once per sample however deep it recurs) and self (leaf) counts.

Usage: python3 bench/gap_samples.py CAPTURE.sqlite [--top 40]
"""

import argparse
import sqlite3
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    q = lambda sql: db.execute(sql).fetchall()  # noqa: E731
    strings = dict(q("SELECT id, value FROM StringIds"))

    rt = q("SELECT start, end, correlationId, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")
    corr_tid = {c: tid for _, _, c, _, tid in rt}
    ker = q("SELECT start, end, correlationId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    launches = Counter(corr_tid[k[2]] for k in ker if k[2] in corr_tid)
    main_tid = launches.most_common(1)[0][0]

    acts = sorted([(s, e) for s, e, *_ in ker] +
                  [(s, e) for s, e in q("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY")])
    # a target decode starts at its first Gated-DeltaNet-graph kernel: the first kernel after a gap that is followed
    # (within the same launch burst) by a gated_delta kernel; approximated by the first kernel of each runtime
    # segment containing gated_delta, found through the kernels' correlation ids -> launching runtime call
    rt_by_corr = {c: (s, e, strings.get(n, "")) for s, e, c, n, tid in rt if tid == main_tid}
    gdn = [k for k in ker if "gated_delta" in strings.get(k[3], "") and k[2] in rt_by_corr]
    # group gated_delta kernels by their launch call (a graph launch has one correlation id for all its nodes)
    first_kernel_of_launch = {}
    for s, e, c, sn in ker:
        if c in rt_by_corr and c not in first_kernel_of_launch:
            first_kernel_of_launch[c] = s
    target_starts = sorted({first_kernel_of_launch[c] for _, _, c, _ in gdn})
    # keep the first launch of each target decode (a decode may span several graph launches / eager launches)
    starts = []
    for t in target_starts:
        prev = max((e for s, e in acts if e <= t), default=None)
        if prev is not None and t - prev > 200_000:   # > 0.2 ms idle before it: a step boundary
            starts.append((prev, t))

    samples, chains = [], defaultdict(list)
    try:  # CPU_SAMPLE=1 captures only
        samples = q("SELECT id, start FROM COMPOSITE_EVENTS WHERE globalTid = %d ORDER BY start" % main_tid)
        for sid, sym, depth in q("SELECT id, symbol, stackDepth FROM SAMPLING_CALLCHAINS ORDER BY id, stackDepth"):
            chains[sid].append(strings.get(sym, "?"))
    except sqlite3.OperationalError:
        pass

    api = defaultdict(lambda: [0.0, 0])
    first_api = []
    incl, leaf = Counter(), Counter()
    n_samples, win_ms = 0, 0.0
    si = 0
    rts = [(s, e, strings.get(n, "")) for s, e, c, n, tid in rt if tid == main_tid]
    for w0, w1 in starts:
        win_ms += (w1 - w0) / 1e6
        calls = [(s, e, n) for s, e, n in rts if w0 <= s < w1]
        for s, e, n in calls:
            api[n][0] += (min(e, w1) - s) / 1e6
            api[n][1] += 1
        if calls:
            first_api.append((calls[0][0] - w0) / 1e6)
        while si < len(samples) and samples[si][1] < w0:
            si += 1
        j = si
        while j < len(samples) and samples[j][1] < w1:
            ch = chains.get(samples[j][0], [])
            if ch:
                leaf[ch[0]] += 1
                for f in set(ch):
                    incl[f] += 1
                n_samples += 1
            j += 1

    n = len(starts)
    print(f"{a.db}: {n} target-decode starts, mean gap before the first target kernel {win_ms / n:.3f} ms, "
          f"{n_samples} decode-thread samples in the windows")
    if first_api:
        print(f"first CUDA runtime call after the window start: mean {sum(first_api) / len(first_api):.3f} ms")
    print(f"\nCUDA runtime calls in the window (per step): {'':30s} {'ms':>7s} {'calls':>6s}")
    for k, (ms, cnt) in sorted(api.items(), key=lambda kv: -kv[1][0])[:15]:
        print(f"  {k[:70]:70s} {ms / n:7.3f} {cnt / n:6.1f}")
    for title, c in (("inclusive", incl), ("self", leaf)):
        print(f"\ndecode-thread samples, {title} (% of {n_samples}):")
        for f, v in c.most_common(a.top):
            print(f"  {100 * v / n_samples:5.1f}%  {f[:150]}")


if __name__ == "__main__":
    main()

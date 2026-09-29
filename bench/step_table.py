"""Per-step kernel-class table from an nsys sqlite export of a llama-server decode capture (bench/step_profile.sh).

Segmentation: host-side runtime API calls are split at synchronizing calls; every kernel/memcpy belongs to the
segment of the API call that launched it (graph launches included, via --cuda-graph-trace=node). A segment that
runs a Gated-DeltaNet kernel is a target decode (the verify batch); other kernel segments are draft-context decodes
(MTP catch-up, then the draft steps). A step runs from the first GPU activity of one target decode to the first of
the next. Host gap = step wall time during which the GPU runs nothing.

Usage: python3 bench/step_table.py CAPTURE.sqlite [--steps 20] [--detail 25]
"""

import argparse
import re
import sqlite3
import statistics
from collections import defaultdict

GGML_TYPES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1", 10: "q2_K",
              11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K", 15: "q8_K", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
              19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 29: "iq1_m", 30: "bf16",
              39: "mxfp4", 40: "nvfp4", 41: "q1_0", 42: "q2_0"}

SYNC_CALLS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy_v", "cudaMemcpy(")


def tmpl_args(name):
    """Top-level template arguments of a demangled kernel name."""
    i = name.find("<")
    if i < 0:
        return []
    depth, cur, out = 0, "", []
    for ch in name[i + 1:]:
        if ch == "<":
            depth += 1
        elif ch == ">":
            if depth == 0:
                out.append(cur.strip())
                break
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        cur += ch
    return out


def gtype(arg):
    m = re.search(r"\)\s*(\d+)$", arg) or re.fullmatch(r"(\d+)", arg)
    if m:
        return GGML_TYPES.get(int(m.group(1)), arg)
    return arg.replace("GGML_TYPE_", "").lower()


def classify(short, dem):
    """(class, detail) for one kernel."""
    s, d = short.lower(), dem
    if s.startswith("mul_mat_vec_q"):
        a = tmpl_args(d)
        return "matmul", f"mmvq {gtype(a[0]) if a else '?'} r{a[1] if len(a) > 1 else '?'}"
    if s.startswith("mul_mat_q_stream_k_fixup"):
        a = tmpl_args(d)
        return "matmul", f"mmq-fixup {gtype(a[0]) if a else '?'} x{a[1] if len(a) > 1 else '?'}"
    if s.startswith("mul_mat_q"):
        a = tmpl_args(d)
        return "matmul", f"mmq {gtype(a[0]) if a else '?'} x{a[1] if len(a) > 1 else '?'}"
    if s.startswith("mul_mat_vec_f"):
        a = tmpl_args(d)
        return "matmul", f"mmvf {a[0] if a else ''} r{a[2] if len(a) > 2 else '?'}"
    if "quantize" in s:
        return "matmul", "quantize_q8_1 (activations)"
    if "gemm" in s or "xmma" in s or "cutlass" in s or s.startswith("sm80_") or s.startswith("sm89_") or "mul_mat_f" in s:
        return "matmul", f"other {short[:40]}"
    if "flash_attn" in s or "fattn" in s:
        return "FA", short[:60]
    if "gated_delta" in s or "gdn" in s or "delta" in s or "ssm_conv" in s or "ssm" in s or "replay" in s:
        return "DeltaNet", short[:60]
    if "norm" in s:
        return "norms", short[:60]
    if "argsort" in s or "top_k" in s or "topk" in s or "sampl" in s or "argmax" in s or "cumsum" in s:
        return "sampling", short[:60]
    return "misc", short[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--steps", type=int, default=20, help="steady steps to report (middle of the capture)")
    ap.add_argument("--detail", type=int, default=30, help="rows of the per-kernel detail table")
    ap.add_argument("--gaps", action="store_true", help="also split the host gap by the GPU work around each idle span")
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
            extra = ", copyKind" if "copyKind" in cols else ", 0"
            mem += [(s, e, c, kind, b, ck) for s, e, c, b, ck in q(f"SELECT start, end, correlationId, bytes{extra} FROM {tbl}")]
        except sqlite3.OperationalError:
            pass

    # the server's decode thread: the thread issuing the most kernel launches
    launches = defaultdict(int)
    corr_tid = {}
    for s, e, c, n, tid in rt:
        corr_tid[c] = tid
    for k in ker:
        if k[2] in corr_tid:
            launches[corr_tid[k[2]]] += 1
    main_tid = max(launches, key=launches.get)

    # segments: runtime calls of the main thread, split after each synchronizing call
    seg_of_corr, seg_bounds = {}, []
    seg = 0
    seg_start = None
    for s, e, c, n, tid in rt:
        if tid != main_tid:
            continue
        name = strings.get(n, "")
        if seg_start is None:
            seg_start = s
        seg_of_corr[c] = seg
        if name.startswith(SYNC_CALLS):
            seg_bounds.append((seg_start, e))
            seg += 1
            seg_start = None

    segs = defaultdict(lambda: {"k": [], "m": []})
    for s, e, c, sn, dn in ker:
        sg = seg_of_corr.get(c)
        if sg is not None:
            segs[sg]["k"].append((s, e, strings.get(sn, "?"), strings.get(dn, "?")))
    for s, e, c, kind, b, ck in mem:
        sg = seg_of_corr.get(c)
        if sg is not None:
            segs[sg]["m"].append((s, e, kind, b, ck))

    # label segments
    order = sorted(i for i in segs if segs[i]["k"])
    labels = {}
    for i in order:
        names = [x[2].lower() for x in segs[i]["k"]]
        labels[i] = "target" if any(("gated_delta" in n or "gdn" in n or "ssm_conv" in n) for n in names) else "draft"

    targets = [i for i in order if labels[i] == "target"]
    steps = []
    for j in range(len(targets) - 1):
        members = [i for i in order if targets[j] <= i < targets[j + 1]]
        steps.append(members)
    if len(steps) > a.steps:
        off = (len(steps) - a.steps) // 2
        steps = steps[off:off + a.steps]
    if not steps:
        print("no complete steps found")
        return

    per_step = []   # dicts class -> ms
    detail = defaultdict(lambda: [0.0, 0])
    walls = []
    gaps = defaultdict(float)   # (work before, work after) -> idle ms summed over steps
    for members in steps:
        acts = []
        cls = defaultdict(float)
        d_idx = 0
        for i in members:
            lab = labels[i]
            if lab == "draft":
                phase = "MTP catch-up" if d_idx == 0 else f"MTP draft {d_idx}"
                d_idx += 1
            else:
                phase = "target"
            for s, e, sn, dn in segs[i]["k"]:
                c, det = classify(sn, dn)
                key = f"target {c}" if lab == "target" else phase
                cls[key] += (e - s) / 1e6
                dk = f"{c:9s} {det}" if lab == "target" else f"{phase if d_idx == 1 else 'MTP draft':12s} {c:9s} {det}"
                detail[dk][0] += (e - s) / 1e6
                detail[dk][1] += 1
                acts.append((s, e, f"{phase} kernel"))
            for s, e, kind, b, ck in segs[i]["m"]:
                cls[f"{kind} (copyKind {ck})"] += (e - s) / 1e6
                acts.append((s, e, f"{phase} {kind} {ck}"))
        acts.sort()
        t0 = acts[0][0]
        # the step ends where the next target decode begins
        nxt_first = min(x[0] for x in segs[[t for t in targets if t > members[0]][0]]["k"])
        busy, cur_s, cur_e, cur_lab = 0.0, None, None, None
        for s, e, lab in acts:
            e = min(e, nxt_first)
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                    gaps[(cur_lab, lab)] += (s - cur_e) / 1e6
                cur_s, cur_e, cur_lab = s, e, lab
            elif e >= cur_e:
                cur_e, cur_lab = e, lab
        busy += cur_e - cur_s
        if nxt_first > cur_e:
            gaps[(cur_lab, "next target")] += (nxt_first - cur_e) / 1e6
        wall = (nxt_first - t0) / 1e6
        cls["host gap (GPU idle)"] = wall - busy / 1e6
        walls.append(wall)
        per_step.append(cls)

    n = len(per_step)
    keys = sorted({k for c in per_step for k in c}, key=lambda k: -sum(c.get(k, 0) for c in per_step))
    wall = statistics.mean(walls)
    print(f"{a.db}: {n} steps, step wall mean {wall:.2f} ms (median {statistics.median(walls):.2f}, "
          f"min {min(walls):.2f}, max {max(walls):.2f})")
    print(f"{'class':34s} {'ms/step':>8s} {'share':>6s}")
    for k in keys:
        v = sum(c.get(k, 0) for c in per_step) / n
        print(f"{k:34s} {v:8.3f} {100 * v / wall:5.1f}%")
    for title, sel in (("target-decode kernels", lambda k: not k.startswith("MTP")),
                       ("MTP kernels (draft = all draft steps)", lambda k: k.startswith("MTP"))):
        print(f"\n{title} (per step): {'':30s} {'ms':>7s} {'calls':>6s} {'us/call':>8s}")
        rows = sorted(((k, v) for k, v in detail.items() if sel(k)), key=lambda kv: -kv[1][0])
        for k, (ms, cnt) in rows[:a.detail]:
            print(f"  {k:78s} {ms / n:7.3f} {cnt / n:6.1f} {1000 * ms / cnt:8.1f}")
    if a.gaps:
        print(f"\nhost gap by the GPU work before -> after the idle span (per step): {'':12s} {'ms':>7s}")
        for (b, f), ms in sorted(gaps.items(), key=lambda kv: -kv[1]):
            if ms / n >= 0.01:
                print(f"  {b:38s} -> {f:38s} {ms / n:7.3f}")


if __name__ == "__main__":
    main()

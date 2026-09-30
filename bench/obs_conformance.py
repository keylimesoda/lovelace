"""Check that lovelace's observability payloads are a superset of tensorninja/ninfer-4090's (docs/observability-parity.md).

Compares a ninfer fixture set (bench/obs_capture_ninfer.sh -> bench/obs-fixtures/ninfer/) with a lovelace one (same file
names, bench/obs_capture.py -> e.g. bench/obs-fixtures/lovelace/, or captured live with --base-url):

  telemetry-{idle,busy,end}.json, slots-{busy,end}.json, models.json, model.json, health.json   JSON field walk
  events.sse + request-log.jsonl                                                                one schema per record type
  metrics-{busy,end}.txt                                                                        series names and TYPE

Every field path present in ninfer must exist in lovelace with the same JSON type (int and float are both "number";
null in ninfer matches anything; null in lovelace where ninfer has a value is a mismatch unless the path is nullable
per the contract's type column or NULLABLE_EXTRA below; array elements are compared as the union of their elements'
schemas). A ninfer leaf is "N/A" when the contract lists it as N/A (lovelace column starting with "N/A", or
PARTIAL_NA below) and every lovelace value there is the documented placeholder. Lovelace-only leaves under `lovelace`,
the hybrid widget keys under /telemetry `cache` and `lovelace:*` series are counted as extensions.

Also checked: SSE framing (a `retry:` line first, `event: <name>` equal to the record's `event`), JSONL lines
byte-identical to the /events `data:` payloads, the record envelope (artifact_type, schema_version 20),
`lovelace.schema_version` == 2 on every lovelace record and in /telemetry, monotonic `lovelace.seq` in the JSONL, the
`lovelace_sched` fields, and every `llamacpp:*` / `ninfer:*` series of ninfer's /metrics in lovelace's.

Usage:
  python3 bench/obs_conformance.py [--ninfer DIR] --lovelace DIR [--out FILE]
  python3 bench/obs_conformance.py [--ninfer DIR] --base-url http://127.0.0.1:8080 [--model ID]
                                   [--lovelace-jsonl PATH] [--out FILE]
  python3 bench/obs_conformance.py --ninfer DIR --lovelace DIR --structural-only   # self-test: ninfer vs ninfer
Exit status 1 on any missing field, type mismatch or failed check.
"""

import argparse
import fnmatch
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONTRACT = ROOT / "docs/observability-parity.md"

ARTIFACT_TYPE = "ninfer_serve_request_log"
SCHEMA_VERSION = 20
LOVELACE_SCHEMA_VERSION = 2
NINFER_EVENTS = {"server_start", "request_start", "request_done", "request_error", "throughput"}
LOVELACE_EVENTS = NINFER_EVENTS | {"lovelace_sched"}
JSON_FILES = ["health.json", "models.json", "model.json", "telemetry-idle.json", "telemetry-busy.json",
              "telemetry-end.json", "slots-busy.json", "slots-end.json"]
METRICS_FILES = ["metrics-busy.txt", "metrics-end.txt"]

# Placeholders for "N/A" rows whose lovelace column does not say which constant (compare with == after number
# normalization); ANY_ZERO accepts any of them.
ANY_ZERO = object()
ZERO_LIKE = (0, False, "", "none", None)

# Nullable paths the contract's type column does not spell out as `|null`.
NULLABLE_EXTRA = {
    # gpu_json() upstream: null when the board has no cumulative energy counter / before the energy sampler runs;
    # the contract says "same NVML fields", so lovelace follows the same rule.
    "telemetry": ["gpu.energy_joules_total", "gpu.server_energy_joules", "gpu.idle_watts"],
}

# Rows whose lovelace column is "mapped: ...; X N/A 0" or "as in <other table>" where the other table says N/A: the
# N/A part is listed here by hand (path patterns, placeholder).
PARTIAL_NA = {
    "server_start": [("adapters.rank", 0), ("adapters.slots", 0), ("adapters.device_bytes", 0),
                     ("adapters.file_bytes", 0), ("engine.continuation_cache.l1_idle_ttl_seconds", 0),
                     ("memory.request_transient", 0)],
    "throughput": [("continuation_cache.persistence_total.coalesced", 0),
                   ("continuation_cache.persistence_total.failures", 0),
                   ("continuation_cache.persistence_delta.coalesced", 0),
                   ("continuation_cache.persistence_delta.failures", 0)],
    "telemetry": [("scheduler.rejected_overloaded", 0), ("scheduler.rejected_queue_timeout", 0),
                  ("scheduler.max_pending_requests", 0), ("cache.restore_failures", 0),
                  ("cache.restore_deferrals", 0), ("cache.persistence_failures", 0),
                  ("cache.kv_growth.curtailed", 0), ("cache.kv_restore_reclaimed_lanes", 0),
                  ("cache.l3.evictions", 0), ("cache.l3.evicted_bytes", 0), ("memory.request_transient", 0),
                  ("adapters.rank", 0), ("adapters.slots", 0), ("adapters.device_bytes", 0),
                  ("adapters.file_bytes", 0)],
}

# Arrays that may legitimately be empty in lovelace's capture while ninfer's has elements (element schema then
# unchecked and reported as such).
EMPTY_OK = {
    "slots": ["[].checkpoints"],        # context checkpoints of a slot: none until -ctxcp takes one
    "telemetry": ["gpu.throttle_reasons", "slots[].checkpoints"],   # board state at capture time
}

# /telemetry `cache` keys of the hybrid runtime that the omarchy-ninfer widget reads (contract: served by lovelace).
HYBRID_CACHE_KEYS = ["kv_capacity", "kv_capacity_page_groups", "kv_capacity_max_page_groups",
                     "device_main_kv_occupied_pages", "device_backend_kv_occupied_pages",
                     "device_state_occupied_slots", "host_state_occupied_slots", "host_state_capacity_slots",
                     "host_kv_occupied_bytes", "host_kv_capacity_bytes", "pressure.checkpoints_dropped",
                     "pressure.private_owners_degraded", "pressure.shared_owners_degraded",
                     "pressure.private_owners_evicted", "pressure.shared_owners_evicted", "pressure.spill_pages"]

LOVELACE_SCHED_FIELDS = ["schema_version", "seq", "action", "id_slot", "id_task", "n_tokens", "bytes", "ms",
                         "host_ms", "gbps"]


# ---------------------------------------------------------------------------------------------------------------------
# contract parsing

def _cells(line):
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def _expand_braces(name):
    m = re.search(r"\{([^{}]*)\}", name)
    if not m:
        return [name]
    out = []
    for alt in m.group(1).split(","):
        alt = alt.strip()
        if alt and alt != "…":
            out += _expand_braces(name[:m.start()] + alt + name[m.end():])
    return out


def _row_paths(first_cell, prefix):
    """Field paths named by a contract table row's first cell (brace groups expanded, bare names = siblings of the
    first path, `(+ `delta_`)` adds the delta_ variants)."""
    names = re.findall(r"`([^`]+)`", first_cell)
    delta = "(+ `delta_`)" in first_cell
    names = [n for n in names if n not in ("delta_", "_delta", "_operations_*")]
    paths, parent = [], ""
    for i, n in enumerate(names):
        if n.startswith("{"):          # `cache.l1`, `cache.l2` `{entries, ...}`: children of the preceding names
            continue
        if i == 0:
            full = n
            parent = n.rsplit(".", 1)[0] if "." in n.replace("[]", "") else ""
        elif n.startswith("."):
            full = (parent + n) if parent else n[1:]
        elif "." in n and parent and n.split(".")[0] == names[0].split(".")[0]:
            full = n
        else:
            full = f"{parent}.{n}" if parent else n
        for p in _expand_braces(full):
            p = (prefix + p) if prefix else p
            paths.append(p)
            if delta:
                head, _, leaf = p.rpartition(".")
                paths.append(f"{head}.delta_{leaf}" if head else f"delta_{leaf}")
    return paths


def _placeholder(text):
    rest = text[3:].strip()
    m = re.match(r"`([^`]*)`", rest)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            return m.group(1)
    m = re.match(r"(0|false|true|null)\b", rest)
    if m:
        return json.loads(m.group(1))
    if rest.startswith("(all null)"):
        return None
    return ANY_ZERO


def parse_contract(path=CONTRACT):
    """{scope: {"na": [(pattern, placeholder)], "listed": [pattern], "nullable": [pattern]}} with scope one of
    server_start, request (prefix `request.`, applied to request_start/done/error), request_done, throughput,
    telemetry."""
    scopes = {s: {"na": [], "listed": [], "nullable": []} for s in
              ("envelope", "server_start", "request", "request_done", "throughput", "telemetry")}
    scope = None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            h = line[3:]
            scope = ("request" if "RequestContext" in h else "server_start" if h == "`server_start`"
                     else "request_done" if h == "`request_done`" else "throughput" if h.startswith("`throughput`")
                     else "telemetry" if h == "`GET /telemetry`" else "envelope" if h.startswith("Record envelope")
                     else None)
            continue
        if scope is None or not line.startswith("| `"):
            continue
        cells = _cells(line)
        if len(cells) < 3:
            continue
        prefix = "request." if scope == "request" else ""
        paths = _row_paths(cells[0], prefix)
        lov, typ = cells[-1], cells[1]
        s = scopes[scope]
        if lov.startswith("N/A"):
            ph = _placeholder(lov)
            s["na"] += [(p, ph) for p in paths]
            if ph is None:             # "N/A null" / "N/A (all null)": the path and everything below it
                s["nullable"] += paths + [p + ".*" for p in paths]
        else:
            s["listed"] += paths
        types = [t.strip() for t in typ.split(",")]
        if len(types) == len(paths) and len(paths) > 1:
            s["nullable"] += [p for p, t in zip(paths, types) if "null" in t]
        elif "null" in typ:
            s["nullable"] += paths
    return scopes


# ---------------------------------------------------------------------------------------------------------------------
# schema trees

def jtype(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


class Node:
    __slots__ = ("n", "types", "children", "items", "values")

    def __init__(self):
        self.n, self.types, self.children, self.items, self.values = 0, Counter(), {}, None, []

    def add(self, v):
        self.n += 1
        t = jtype(v)
        self.types[t] += 1
        if t == "object":
            for k, x in v.items():
                self.children.setdefault(k, Node()).add(x)
        elif t == "array":
            for x in v:
                if self.items is None:
                    self.items = Node()
                self.items.add(x)
        else:
            self.values.append(v)
        return self


def tree(values):
    n = Node()
    for v in values:
        n.add(v)
    return n


def leaves(node):
    if node is None:
        return 0
    k = sum(leaves(c) for c in node.children.values()) + (leaves(node.items) if node.items else 0)
    return k or 1


def join(path, key):
    return f"{path}.{key}" if path else key


# ---------------------------------------------------------------------------------------------------------------------
# rules for one payload

class Rules:
    def __init__(self, contract, scopes, kind):
        self.na, self.listed, self.nullable = [], [], []
        for s in scopes:
            c = contract.get(s) or {}
            self.na += c.get("na", [])
            self.listed += c.get("listed", [])
            self.nullable += c.get("nullable", [])
            self.na += PARTIAL_NA.get(s, [])
        self.nullable += NULLABLE_EXTRA.get(kind, [])
        self.empty_ok = EMPTY_OK.get(kind, [])

    @staticmethod
    def _match(pattern, path):
        p = path.replace("[]", "")
        pat = pattern.replace("[]", "")
        if "*" in pat:
            return fnmatch.fnmatchcase(p, pat) or fnmatch.fnmatchcase(p, pat + ".*")
        return p == pat or p.startswith(pat + ".")

    @staticmethod
    def _score(pattern):
        return len(pattern.replace("*", "")) - (1000 if "*" in pattern else 0)

    def na_placeholder(self, path):
        """Placeholder of the most specific matching N/A row, or None when a more specific mapped row names the
        path (or no N/A row does). Returns (True, placeholder) / (False, None)."""
        best_na = max(((self._score(p), ph) for p, ph in self.na if self._match(p, path)), default=None,
                      key=lambda x: x[0])
        if best_na is None:
            return False, None
        best_listed = max((self._score(p) for p in self.listed if self._match(p, path)), default=None)
        if best_listed is not None and best_listed > best_na[0]:
            return False, None
        return True, best_na[1]

    def is_nullable(self, path):
        """Exact (or wildcard) match only: an object that may be null does not make its fields nullable."""
        p = path.replace("[]", "")
        return any(fnmatch.fnmatchcase(p, pat.replace("[]", "")) for pat in self.nullable)

    def is_empty_ok(self, path):
        return any(self._match(p, path) for p in self.empty_ok)


def is_placeholder(value, ph):
    if ph is ANY_ZERO:
        return any(is_placeholder(value, z) for z in ZERO_LIKE)
    if ph is None or value is None:
        return value is None and ph is None
    if isinstance(ph, bool) or isinstance(value, bool):
        return isinstance(ph, bool) and isinstance(value, bool) and ph == value
    if isinstance(ph, (int, float)) and isinstance(value, (int, float)):
        return float(ph) == float(value)
    return ph == value


class Result:
    def __init__(self, name):
        self.name = name
        self.c = Counter()
        self.failures = []
        self.notes = []

    def fail(self, kind, path, msg, count=1):
        self.c[kind] += count
        self.failures.append(f"{self.name}: {kind} `{path or '<root>'}`: {msg}")


def compare(n, l, path, rules, res):
    if l is None:
        res.fail("missing", path, "absent in lovelace", leaves(n))
        return
    nt, lt = set(n.types), set(l.types)
    nt_v = nt - {"null"}
    if not nt_v:                       # ninfer only ever null: anything matches
        res.c["matched"] += 1
        return
    bad = (lt - {"null"}) - nt_v
    if bad:
        res.fail("type mismatch", path, f"ninfer {'/'.join(sorted(nt))}, lovelace {'/'.join(sorted(lt))}", leaves(n))
        return
    na, ph = rules.na_placeholder(path)
    if "null" in lt and "null" not in nt and not rules.is_nullable(path) and not (na and ph is None):
        res.fail("type mismatch", path, f"null in lovelace ({l.types['null']} of {l.n}), ninfer "
                 f"{'/'.join(sorted(nt))} and the contract does not list it as nullable", leaves(n))
        return
    if not (lt - {"null"}):            # lovelace always null where the contract allows it
        res.c["n/a" if na else "matched"] += leaves(n)
        return
    if "object" in nt_v and n.children:
        n_obj, l_obj = n.types["object"], l.types["object"]
        for k, nc in n.children.items():
            lc = l.children.get(k)
            if lc is not None and nc.n == n_obj and lc.n < l_obj:
                res.fail("missing", join(path, k), f"absent in {l_obj - lc.n} of {l_obj} lovelace objects",
                         leaves(nc))
                continue
            compare(nc, lc, join(path, k), rules, res)
        return
    if "array" in nt_v and n.items is not None:
        p = path + "[]"
        if l.items is None:
            if rules.is_empty_ok(p):
                res.c["unchecked"] += leaves(n.items)
                res.notes.append(f"{res.name}: `{path}` empty in lovelace, element schema unchecked (allowed)")
            else:
                res.fail("missing", p, "lovelace array always empty; ninfer's has elements", leaves(n.items))
            return
        compare(n.items, l.items, p, rules, res)
        return
    # leaf (scalar, or empty container on the ninfer side)
    if na and l.values and all(is_placeholder(v, ph) for v in l.values):
        res.c["n/a"] += 1
    else:
        res.c["matched"] += 1


def count_extensions(n, l, path, res, ext_roots):
    """Lovelace-only leaves: under an extension root -> extension, elsewhere -> other extra."""
    if l is None:
        return
    if n is None:
        k = leaves(l)
        if any(path == r or path.startswith(r + ".") or path.startswith(r + "[]") for r in ext_roots):
            res.c["extension"] += k
        else:
            res.c["other extra"] += k
        return
    for key, lc in l.children.items():
        count_extensions(n.children.get(key), lc, join(path, key), res, ext_roots)
    if l.items is not None:
        count_extensions(n.items, l.items, path + "[]", res, ext_roots)


def check_payload(name, nval, lval, rules, ext_roots):
    res = Result(name)
    n, l = tree(nval), tree(lval)
    res.c["ninfer fields"] = leaves(n)
    compare(n, l, "", rules, res)
    count_extensions(n, l, "", res, ext_roots)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# inputs

def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def parse_sse(raw):
    """-> (retry value or None, [(event name or None, data string)], [framing problems])"""
    problems, frames, retry = [], [], None
    torn = not raw.endswith("\n\n")
    blocks = raw.split("\n\n")
    if torn:
        blocks = blocks[:-1]
    for i, block in enumerate(blocks):
        lines = [ln for ln in block.split("\n") if ln != ""]
        if not lines:
            continue
        if all(ln.startswith(":") for ln in lines):
            continue
        fields = {}
        for ln in lines:
            if ln.startswith(":"):
                continue
            k, _, v = ln.partition(":")
            v = v[1:] if v.startswith(" ") else v
            fields.setdefault(k, []).append(v)
        if "retry" in fields:
            if i != 0 or frames:
                problems.append("`retry:` is not the first frame")
            retry = fields["retry"][0]
            if set(fields) - {"retry"}:
                problems.append(f"retry frame carries other fields: {sorted(set(fields) - {'retry'})}")
            continue
        if "data" not in fields:
            problems.append(f"frame without data: {block[:80]!r}")
            continue
        ev = fields.get("event", [None])[0]
        frames.append((ev, "\n".join(fields["data"])))
    return retry, frames, problems


def load_records(d, side, checks):
    """Records of a fixture dir: from events.sse (framing checked) and request-log.jsonl (identity checked)."""
    recs, sse_payloads = [], []
    sse = d / "events.sse"
    if sse.exists():
        retry, frames, problems = parse_sse(sse.read_text())
        if retry is None:
            problems.append("no `retry:` line")
        checks.append((f"{side} events.sse framing", problems, retry))
        for ev, data in frames:
            try:
                r = json.loads(data)
            except ValueError:
                problems.append(f"data is not JSON: {data[:80]!r}")
                continue
            if ev != r.get("event"):
                problems.append(f"`event: {ev}` but record.event = {r.get('event')!r}")
            recs.append(r)
            sse_payloads.append(data)
    jl = d / "request-log.jsonl"
    if jl.exists():
        lines = [ln for ln in jl.read_text().split("\n") if ln.strip()]
        jrecs = []
        for ln in lines:
            try:
                jrecs.append(json.loads(ln))
            except ValueError:
                pass
        problems = []
        if sse_payloads:
            have = set(lines)
            miss = [p for p in sse_payloads if p not in have]
            if miss:
                problems.append(f"{len(miss)} of {len(sse_payloads)} /events payloads not byte-identical to a JSONL "
                                f"line, e.g. {miss[0][:120]!r}")
        checks.append((f"{side} request-log.jsonl == /events data", problems, None))
        recs += jrecs
    if not recs:
        recs = [r for p in sorted(d.glob("*.json")) if p.stem in LOVELACE_EVENTS
                for r in [load_json(p)] if isinstance(r, dict)]
    return recs


def parse_metrics(text):
    names, types = set(), {}
    for ln in text.splitlines():
        if ln.startswith("# TYPE "):
            parts = ln.split()
            if len(parts) >= 4:
                types[parts[2]] = parts[3]
        elif ln and not ln.startswith("#"):
            names.add(re.split(r"[{\s]", ln, maxsplit=1)[0])
    return names, types


# ---------------------------------------------------------------------------------------------------------------------

def envelope_problems(recs, side):
    out = []
    for r in recs:
        ev = r.get("event")
        where = f"{ev} ({r.get('timestamp_unix_ms')})"
        if r.get("artifact_type") != ARTIFACT_TYPE:
            out.append(f"{where}: artifact_type {r.get('artifact_type')!r}")
        if r.get("schema_version") != SCHEMA_VERSION:
            out.append(f"{where}: schema_version {r.get('schema_version')!r}")
        if ev not in (LOVELACE_EVENTS if side == "lovelace" else NINFER_EVENTS):
            out.append(f"{where}: unknown event")
        if side == "lovelace":
            lv = r.get("lovelace")
            if not isinstance(lv, dict) or lv.get("schema_version") != LOVELACE_SCHEMA_VERSION:
                out.append(f"{where}: lovelace.schema_version "
                           f"{lv.get('schema_version') if isinstance(lv, dict) else 'absent'!r} (want 2)")
            if ev == "lovelace_sched" and isinstance(lv, dict):
                miss = [f for f in LOVELACE_SCHED_FIELDS if f not in lv]
                if miss:
                    out.append(f"{where}: lovelace_sched lacks lovelace.{', lovelace.'.join(miss)}")
    return list(dict.fromkeys(out))    # records seen on both /events and the JSONL: report each problem once


def seq_problems(d):
    jl = d / "request-log.jsonl"
    if not jl.exists():
        return None
    last, out = {}, []
    for ln in jl.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        inst, seq = r.get("server_instance_id"), (r.get("lovelace") or {}).get("seq")
        if not isinstance(seq, int):
            out.append(f"{r.get('event')}: lovelace.seq {seq!r}")
            continue
        if inst in last and seq <= last[inst]:
            out.append(f"{r.get('event')}: lovelace.seq {seq} after {last[inst]} (instance {inst})")
        last[inst] = seq
    return out


def run(ndir, ldir, contract, lovelace_checks=True):
    """lovelace_checks=False (--structural-only): only the ninfer-field walk, framing, JSONL identity, the schema-20
    envelope and series names; used to self-test the checker on ninfer's own fixtures."""
    results, checks, missing_payloads = [], [], []

    # JSON endpoints
    for fname in JSON_FILES:
        nval = load_json(ndir / fname)
        if nval is None:
            continue
        lval = load_json(ldir / fname)
        if lval is None:
            missing_payloads.append(fname)
            continue
        kind = "telemetry" if fname.startswith("telemetry") else "slots" if fname.startswith("slots") else fname
        rules = Rules(contract, ["telemetry"] if kind == "telemetry" else [], kind)
        ext = ["lovelace"] + ([f"cache.{k.split('.')[0]}" for k in HYBRID_CACHE_KEYS] if kind == "telemetry" else [])
        res = check_payload(fname, [nval], [lval], rules, ext)
        if kind == "telemetry" and lovelace_checks:
            lv = lval.get("lovelace") if isinstance(lval, dict) else None
            if not isinstance(lv, dict) or lv.get("schema_version") != LOVELACE_SCHEMA_VERSION:
                res.fail("check", "lovelace.schema_version", f"{(lv or {}).get('schema_version')!r}, want 2")
            cache = (lval.get("cache") or {}) if isinstance(lval, dict) else {}
            for k in HYBRID_CACHE_KEYS:
                v = cache
                for part in k.split("."):
                    v = v.get(part) if isinstance(v, dict) else None
                if jtype(v) != "number":
                    res.fail("check", f"cache.{k}", f"hybrid widget key {'absent' if v is None else jtype(v)}")
        results.append(res)

    # records
    nrecs = load_records(ndir, "ninfer", checks)
    lrecs = load_records(ldir, "lovelace", checks)
    checks.append(("ninfer record envelope", envelope_problems(nrecs, "ninfer"), None))
    if lrecs:
        checks.append(("lovelace record envelope" + (" + lovelace.schema_version" if lovelace_checks else ""),
                       envelope_problems(lrecs, "lovelace" if lovelace_checks else "ninfer"), None))
        sp = seq_problems(ldir)
        if sp is not None and lovelace_checks:
            checks.append(("lovelace JSONL lovelace.seq monotonic per instance", sp, None))
    nretry = next((c[2] for c in checks if c[0] == "ninfer events.sse framing"), None)
    lretry = next((c[2] for c in checks if c[0] == "lovelace events.sse framing"), None)
    if nretry is not None and lretry is not None and nretry != lretry:
        checks.append(("SSE retry value", [f"ninfer `retry: {nretry}`, lovelace `retry: {lretry}`"], None))
    by_ev = lambda recs: {e: [r for r in recs if r.get("event") == e] for e in {r.get("event") for r in recs}}
    nby, lby = by_ev(nrecs), by_ev(lrecs)
    for ev in sorted(nby):
        name = f"record {ev}"
        if not lby.get(ev):
            missing_payloads.append(name)
            continue
        scopes = {"server_start": ["envelope", "server_start"], "request_start": ["envelope", "request"],
                  "request_done": ["envelope", "request", "request_done"], "request_error": ["envelope", "request"],
                  "throughput": ["envelope", "throughput"]}.get(ev, ["envelope"])
        results.append(check_payload(name, nby[ev], lby[ev], Rules(contract, scopes, ev), ["lovelace"]))
    for ev in sorted(set(lby) - set(nby)):
        res = Result(f"record {ev} (lovelace only)")
        res.c["extension"] = leaves(tree(lby[ev]))
        results.append(res)

    # metrics
    for fname in METRICS_FILES:
        if not (ndir / fname).exists():
            continue
        if not (ldir / fname).exists():
            missing_payloads.append(fname)
            continue
        nn, nty = parse_metrics((ndir / fname).read_text())
        ln, lty = parse_metrics((ldir / fname).read_text())
        res = Result(fname)
        want = sorted(x for x in nn if x.startswith(("ninfer:", "llamacpp:")))
        res.c["ninfer fields"] = len(want)
        for m in want:
            if m not in ln:
                res.fail("missing", m, "series absent in lovelace")
            elif nty.get(m) and lty.get(m) and nty[m] != lty[m]:
                res.fail("type mismatch", m, f"TYPE ninfer {nty[m]}, lovelace {lty[m]}")
            else:
                res.c["matched"] += 1
        res.c["extension"] = sum(1 for x in ln if x.startswith("lovelace:"))
        res.c["other extra"] = len(ln - nn) - res.c["extension"]
        results.append(res)
    return results, checks, missing_payloads


def report(ndir, ldir, results, checks, missing_payloads):
    cols = ["ninfer fields", "matched", "n/a", "missing", "type mismatch", "unchecked", "extension", "other extra",
            "check"]
    out = ["# Observability conformance: lovelace vs tensorninja/ninfer-4090", "",
           f"ninfer: `{ndir}`  ", f"lovelace: `{ldir}`  ",
           f"contract: `docs/observability-parity.md`  ", "",
           "| payload | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    tot = Counter()
    for r in results:
        out.append(f"| {r.name} | " + " | ".join(str(r.c[c]) for c in cols) + " |")
        tot.update({c: r.c[c] for c in cols})
    out.append("| **total** | " + " | ".join(f"**{tot[c]}**" for c in cols) + " |")
    out += ["", "`n/a`: contract N/A and the lovelace value is the documented placeholder. `unchecked`: element schema "
                "of an array that is empty in lovelace's capture (allowed list). `extension`: lovelace-only leaves "
                "under `lovelace`, hybrid `cache` keys, lovelace-only records and `lovelace:*` series. `other extra`: "
                "other lovelace-only leaves (llama.cpp's own fields). `check`: failed lovelace-side checks of the "
                "payload (`lovelace.schema_version`, hybrid widget keys).", "", "## Checks", ""]
    failures = [f for r in results for f in r.failures]
    failures += [f"payload missing in lovelace: {m}" for m in missing_payloads]
    for name, problems, _ in checks:
        out.append(f"- {'PASS' if not problems else 'FAIL'}: {name}")
        failures += [f"{name}: {p}" for p in problems]
    notes = [n for r in results for n in r.notes]
    if notes:
        out += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    out += ["", f"## Failures ({len(failures)})", ""] + ([f"- {f}" for f in failures] or ["none"])
    return "\n".join(out) + "\n", failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ninfer", type=Path, default=HERE / "obs-fixtures/ninfer")
    ap.add_argument("--lovelace", type=Path, help="fixture dir captured from lovelace (bench/obs_capture.py)")
    ap.add_argument("--base-url", help="capture lovelace live instead (into ~/.cache/lovelace-bench/obs-conformance/)")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--lovelace-jsonl", type=Path, help="with --base-url: the server's --request-log-jsonl file")
    ap.add_argument("--contract", type=Path, default=CONTRACT)
    ap.add_argument("--structural-only", action="store_true",
                    help="skip the lovelace-only checks (self-test of the checker on ninfer's own fixtures)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if not a.ninfer.is_dir() or not any((a.ninfer / f).exists() for f in
                                        JSON_FILES + METRICS_FILES + ["events.sse", "request-log.jsonl"]):
        sys.exit(f"no ninfer fixtures in {a.ninfer} (bench/obs_capture_ninfer.sh)")
    if a.base_url:
        sys.path.insert(0, str(HERE))
        from obs_capture import capture  # noqa: E402
        ldir = Path.home() / ".cache/lovelace-bench/obs-conformance" / time.strftime("%Y%m%d-%H%M%S")
        capture(a.base_url.rstrip("/"), a.model, ldir)
        if a.lovelace_jsonl:
            shutil.copyfile(a.lovelace_jsonl, ldir / "request-log.jsonl")
    elif a.lovelace:
        ldir = a.lovelace
    else:
        sys.exit("give --lovelace DIR or --base-url URL")
    if not ldir.is_dir():
        sys.exit(f"lovelace fixtures not found: {ldir}")
    results, checks, missing = run(a.ninfer, ldir, parse_contract(a.contract), not a.structural_only)
    text, failures = report(a.ninfer, ldir, results, checks, missing)
    print(text, end="")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

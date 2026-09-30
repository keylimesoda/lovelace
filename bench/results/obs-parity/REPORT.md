# Observability parity with tensorninja/ninfer-4090 (f408990)

Branch `feat/obs-parity`. Contract and field-by-field audit: [docs/observability-parity.md](../../../docs/observability-parity.md);
lovelace extension: [docs/observability.md](../../../docs/observability.md); dashboard: [docs/dashboard.md](../../../docs/dashboard.md).

## Audit summary

- Upstream pinned at tensorninja/ninfer-4090 `f408990` (request-log schema 20). lovelace now serves its `/telemetry`,
  `/events` (named SSE frames, `retry: 2000`), `/metrics` (`ninfer:*` series, same names/order), `/slots` (ninfer keys),
  `/health`, `/v1/models` (+ `context_window`, `modalities.vision`) and `GET /v1/models/{id}`, and writes the same
  schema-20 JSONL records (`server_start`, `request_start`, `request_done`, `request_error`, `throughput`).
- Meaningful equivalents are mapped: continuation cache l1/l2/l3 = lovelace session store GPU/RAM/NVMe; worker
  seconds = server-loop phase clock; scheduler occupancy, queue seconds, decode batch, energy (exact board counter,
  phase split, residual 0 by construction).
- N/A fields (preflight/stable-prefix/gates/supersession, adapter bank, arena reservations, tool-call counters, queue
  bounds, PCIe throughput) keep their JSON type with 0/false/""/"none"/null and are listed in the parity doc.
- lovelace-only data under `lovelace` (`lovelace.schema_version` 2): KV budget, sessions (tiers, sizes, promotions,
  demotions, prefetch), per-slot state, VRAM breakdown, loop window, energy totals; `lovelace_sched` records;
  `lovelace:*` Prometheus series. The omarchy-ninfer widget's hybrid `/telemetry` `cache` keys are served too.
- Cost model (E1): the server loop only does relaxed atomic stores plus one short snapshot lock per step (containers
  at most every 250 ms, preallocated); `/telemetry`, `/metrics`, `/slots` are built on HTTP threads from the snapshot
  and never post to, wake or wait for the loop (previously `/telemetry` and `/metrics` posted a task to the loop).

## Conformance (`bench/obs_conformance.py` → [conformance.md](conformance.md))

Run against lovelace live payloads (production profile `balanced.args`, container `ninfer-qwen` on :8080, one
gpu-lock hold, 2026-09-29 22:31). **0 missing, 0 type mismatches** over 256 ninfer fields: 202 matched, 54 N/A
(documented placeholders), 560 extension leaves. All SSE/JSONL/envelope checks pass.

Limitation: the first tensorninja capture (`bench/obs_capture_ninfer.sh`, real `ninfer-serve` f408990 built from its
Dockerfile, official artifact sha256 `eec39564…`) stopped at the first chat request (ninfer rejects
`chat_template_kwargs.enable_thinking`; fixed in `obs_capture.py`), so only health/models/model/telemetry-idle and
`server_start` were real fixtures at conformance time; request/throughput/metrics/slots records were not compared
against ninfer. A full capture is queued on the GPU lock and writes `bench/obs-fixtures/ninfer/`; then rerun
`python3 bench/obs_conformance.py --out bench/results/obs-parity/conformance.md` (no GPU needed; lovelace capture is in
`bench/obs-fixtures/lovelace/`).

## omarchy-ninfer widget ([widget/labels.md](widget/labels.md)): PASS

The unmodified helper `ninfer-qwen info` (omarchy-ninfer HEAD e72d47c, clean checkout) ran against lovelace, and
its output was passed through the unmodified `Model.js` with bun. Every live label is filled: GPU `RTX 4090 · 23.6/24 GB`,
prefix cache 83%, draft accept 45%, generated 1.5k, device KV `1.7k / 376.8k tok`, host spill `0.6 / 24 GiB`, state
`dev 4 · host 4`, pressure `Stable`, generation 129 tok/s, 357 W. Model, KV and spec labels (`RK4V4 E8`, `DFLASH2`) are
the helper's hard-coded ninfer settings, not server data.

## Dashboard ([dashboard.png](dashboard.png))

`apps/web` is now tensorninja f408990's dashboard verbatim (41 files unchanged), plus 4 modified files (with
Apache §4(b) headers) and 7 new ones: Sessions, Queue & waits and KV pool panels. `bun run build`, `bun test` (22)
and `format:check` pass. The screenshot is headless Chromium against the running server after the capture traffic.

## Overhead A/B (E1)

**PASS** ([overhead/RESULTS.md](overhead/RESULTS.md)). `BIN=engine/build-sm89/bin bench/obs_overhead.sh`: telemetry
OFF/ON/OFF/ON, same binary; (a) single stream at 64k via depth_sweep, (b) six sequences via step_drive; ON runs had a
1 Hz /telemetry poller, a 4 s /metrics scraper and an /events subscriber (202 polls, 51 scrapes, 0 errors). The desktop
was in normal use (a browser game rendering on the CPU), so throughput figures swing 2-10% run to run.

| figure | OFF median | ON median | ON vs OFF | spread (OFF / ON) |
|---|---:|---:|---:|---:|
| 6-sequence step ms | 50.80 | 50.99 | +0.39% | 0.24% / 3.81% |
| single decode tok/s | 100.91 | 103.75 | +2.81% | 1.87% / 2.65% |
| 6-sequence aggregate tok/s | 309.75 | 297.45 | −3.97% | 10.23% / 4.67% |

The step time, the least noisy figure, moves +0.39% (≤ 0.5%), inside the ON spread; the other differences are inside
their spreads, in both directions. Also: conformance against the real ninfer fixtures (captured from the hybrid image
at f408990) has 0 failures ([conformance-ninfer-fixtures.md](conformance-ninfer-fixtures.md)).

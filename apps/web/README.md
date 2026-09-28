# lovelace-web

Observability dashboard for a running lovelace `llama-server`. It shows throughput, the server
loop's wall-clock split, slot and queue occupancy, TTFT decomposition, KV and prompt-cache
occupancy, the VRAM budget, board telemetry and energy, and per-request detail, either live or
replayed from a request log.

[`docs/dashboard.md`](../../docs/dashboard.md) describes what each panel means and how to read
it. This file covers building and developing the app itself.

## Build and serve

```bash
cd apps/web
bun install
bun run build
```

Then point the server at the build output. `--web-dir` is an alias of llama.cpp's `--path`; the
dashboard is served same-origin with the API, so no CORS configuration is involved:

```bash
engine/build-sm89/bin/llama-server -m models/model.gguf \
  --web-dir apps/web/dist \
  --request-log-jsonl /var/log/lovelace/requests.jsonl
```

The dashboard is then at the server's root. `--request-log-jsonl` is optional for live use — the
same records reach the browser over `/events` either way — but it is what produces a file the
dashboard can replay later.

## Development

```bash
LOVELACE_BASE_URL=http://127.0.0.1:8080 bun run dev
```

Vite serves on `127.0.0.1:5180` and proxies `/telemetry`, `/events`, `/metrics`, `/slots`,
`/health`, `/v1`, and `/props` to the server, so the development build sees the same same-origin
layout as production. `LOVELACE_BASE_URL` defaults to `http://127.0.0.1:8080` and is validated: a
malformed or path-bearing origin fails the config instead of quietly pointing at the wrong
server. Proxy timeouts are disabled because `/events` is an open-ended stream that a long
generation can outlast.

| Script                 | Purpose                                        |
| ---------------------- | ---------------------------------------------- |
| `bun run dev`          | Vite dev server with the server proxy          |
| `bun run build`        | Typecheck, then production bundle into `dist/` |
| `bun run typecheck`    | `tsc -b` only                                  |
| `bun test`             | Derivation, replay, and stream-ingestion tests |
| `bun run format`       | Prettier write                                 |
| `bun run format:check` | Prettier check, for CI or a pre-commit hook    |

## Where the data comes from

Two server endpoints (schema_version 1):

- `GET /telemetry` — a complete snapshot, not a delta: NVML board sensors, board energy split by
  loop phase since start, slot and queue gauges, the server loop's wall-clock split over a
  rolling 10 s window, KV cell and prompt-cache occupancy, the VRAM budget, per-slot state, and
  event-stream counters. Polled at 1 Hz. Because it is complete, the client resynchronizes by
  simply fetching it again.
- `GET /events` — Server-Sent Events whose `data:` payloads are the same records
  `--request-log-jsonl` writes, byte for byte: `server_start`, `request_start`, `request_done`,
  `request_rejected`, and `throughput`. A new subscriber first receives the retained
  `server_start`, then a bounded backlog, then live records, so a browser opened mid-run still
  knows the server configuration.

The stream is bounded and lossy by design: a slow subscriber drops its oldest records and the
server counts them in `/telemetry.events.records_dropped`, which the top bar shows. On reconnect
the backlog is replayed again; the client ignores any record at or below the last `seq` it applied
for the same `server_instance_id`, and a new instance id resets all history. The dashboard never
reads `/metrics`.

## Replay

`load jsonl` reads a `--request-log-jsonl` file and re-derives every record-backed panel offline:
throughput, scheduler loop split (summed over intervals), latency, energy, requests, VRAM (from
`server_start.vram`), and cache reuse. Board sensors, slot state, and KV / prompt-cache occupancy
are levels that are never written to the log, so those panels say so instead of rendering zeros.
The log is opened in append mode, so a file can hold several server instances; only the last one
is kept. Blank lines and a torn trailing line are skipped.

`fixtures/sample.jsonl` is a small hand-written, contract-conformant log used by the replay tests
and handy for trying the dashboard without a server.

## Layout

| Path                         | Contents                                                         |
| ---------------------------- | ---------------------------------------------------------------- |
| `src/lib/records.ts`         | Record types and a tolerant line parser                          |
| `src/lib/telemetry.ts`       | `/telemetry` payload types and the live/replay cache view        |
| `src/lib/derive.ts`          | Percentiles, TTFT shares, cache reuse, energy and loop summaries |
| `src/lib/engine-client.ts`   | Poll + stream + file replay, and the state they produce          |
| `src/lib/series.ts`          | Records to time-accurate bands and snapshot points               |
| `src/lib/glossary.ts`        | Definitions behind every tooltip                                 |
| `src/lib/palette.ts`         | Chart colours, mirroring the CSS custom properties               |
| `src/components/echart.tsx`  | ECharts registration, shared theme, React binding                |
| `src/components/charts.tsx`  | Band, sample, line, and stacked-bar primitives                   |
| `src/components/tooltip.tsx` | Portal-positioned tooltips                                       |
| `src/panels/`                | One file per panel                                               |
| `fixtures/sample.jsonl`      | Replay fixture                                                   |

Records are discriminated on `event`, never on `schema_version`.

## Conventions worth preserving

**`null` is not zero.** The payloads use `null` for "not measured / not applicable" (energy on a
board without a counter, TTFT for a request that produced no token, per-token energy with no
tokens). Those render as `—` or as an unavailable panel, never as 0.

**Interval averages and snapshots are drawn differently.** A throughput sample is an average over
its whole `interval_seconds`, so it is drawn as a band filling that interval. The scheduler
counts on the same record are read at the instant it is emitted, so they are drawn as points.

**Reporting intervals are irregular.** The server skips intervals with no activity and folds the
skipped time into the next sample, so series are plotted against real wall-clock time.

**`derive.ts` has an oracle.** Its aggregations are pinned by hand-computed fixtures in
`derive.test.ts`, and `replay.test.ts` runs `fixtures/sample.jsonl` through the same function the
file loader uses.

**Terminology lives in `glossary.ts`.** Panels reference entries by key rather than inlining prose,
so a reading and its explanation cannot drift apart.

**The palette is duplicated deliberately.** ECharts renders to canvas and cannot resolve `var()`,
so `palette.ts` mirrors the CSS custom properties in `styles/foundation.css`.

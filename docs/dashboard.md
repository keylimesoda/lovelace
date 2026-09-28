# Dashboard

An optional single-page dashboard for one running lovelace `llama-server`. It answers, at a
glance, what the server is doing: current prefill and decode rates, whether requests are waiting
for a slot, how much of each prompt the caches served, where the server loop spends its time, how
the VRAM budget is spent, what the work costs in joules, and whether the board is the limit. It
also loads a `--request-log-jsonl` file to analyze a past session offline.

The application lives in [`apps/web/`](../apps/web/) and is built with Bun, Vite, React, and
ECharts. Payload shapes are schema_version 1 of the server's observability layer.

## Build and run

```bash
cd apps/web
bun install
bun run build
```

Then serve it from the server itself, same-origin with the API (`--web-dir` is an alias of
llama.cpp's `--path`):

```bash
engine/build-sm89/bin/llama-server -m models/model.gguf --web-dir apps/web/dist
```

Open `http://127.0.0.1:8080/`. Add `--request-log-jsonl PATH` to also write the record stream to
a file that can be replayed later.

For development against a running server:

```bash
cd apps/web
LOVELACE_BASE_URL=http://127.0.0.1:8080 bun run dev
```

The dev server proxies `/telemetry`, `/events`, `/metrics`, `/slots`, `/health`, `/v1`, and
`/props` to that origin, so CORS is not required. `LOVELACE_BASE_URL` is validated as an origin
and defaults to `http://127.0.0.1:8080`.

| Command                | Purpose                                            |
| ---------------------- | -------------------------------------------------- |
| `bun run dev`          | development server on `127.0.0.1:5180`             |
| `bun run build`        | typecheck and emit `dist/`                         |
| `bun run typecheck`    | types only                                         |
| `bun test`             | derivation, replay, and stream-ingestion tests     |
| `bun run format`       | Prettier write over the app sources                |
| `bun run format:check` | Prettier check                                     |

## Data sources

The dashboard reads two channels because they answer different questions.

| Channel          | Kind           | Carries                                                                  |
| ---------------- | -------------- | ------------------------------------------------------------------------ |
| `GET /telemetry` | polled at 1 Hz | levels: board sensors, slots, queue, loop split, KV / prompt cache, VRAM |
| `GET /events`    | SSE            | history: `server_start`, request lifecycle, `throughput` intervals       |

Each SSE frame is `data: <record JSON>`; keepalives are SSE comments. A new subscriber receives
the retained `server_start`, then up to 512 recent records, then live records. Levels cannot be
reconstructed by replaying deltas, and the event stream is bounded and lossy under backpressure,
so the poll is always authoritative for current state. After a reconnect the client drops records
at or below the last `seq` it applied for the same `server_instance_id`, and a new instance id
(a server restart) clears all history. `/metrics` is not read by the dashboard.

Board energy is carried in the `throughput` record as an interval total, so the Energy panel works
on a replayed log; the GPU panel, which reports levels, does not.

## Panels

- **Headline** — decode and prefill tok/s (latest interval), busy / total slots, deferred tasks,
  queue-wait and TTFT p50, cache reuse, draft acceptance, served J/token, board watts over the
  last second, GPU utilization and temperature.
- **Throughput** — prefill and decode tok/s as interval bands on independent axes over one time
  axis, mean busy slots per `llama_decode` call, and the scheduler snapshot (busy slots, slots
  processing prompt, deferred tasks) as points. Captions give the interval-weighted means, the
  per-sequence decode rate (aggregate ÷ busy slots per call), and each phase's own rate (tokens ÷
  loop seconds in that phase).
- **Scheduler** — current loop phase, slots busy / processing prompt / generating, deferred tasks,
  the loop wall-clock split (prefill / decode / host / idle) over the server's rolling 10 s
  window with the since-start split beneath it, and the cumulative totals (requests started,
  done, rejected, `llama_decode` calls, busy slots per call, prompt / cached / predicted tokens,
  drafted / accepted). In replay the split is summed over the logged intervals and the totals
  are the summed interval deltas.
- **Latency** — TTFT p50/p90/p99, queue-wait p50/p90, per-request decode tok/s p50, the TTFT
  decomposition, the finish-reason partition, token spreads (prompt / cached / computed prefill /
  generated), and draft acceptance with the per-position accepted counts when speculative decoding
  ran.
- **Energy** — served / active / prefill / decode J per token, idle draw, Wh per 1M tokens, and a
  stacked band chart of prefill / decode / idle joules per interval. Unavailable (not zero) when
  the board has no energy counter.
- **GPU** — live only: utilization, temperature, power against limit, SM clock, memory controller
  utilization, board memory, PCIe rx/tx, P-state, active throttle reasons, and a utilization
  history with throttled samples marked.
- **Cache** — KV cells used / total (unified or per-slot), prompt cache (`--cache-ram`) bytes used
  against its limit (0 = unlimited), entries and tokens, and cache reuse over completed requests
  as cached vs computed prompt tokens. Occupancy is live only; replay shows the configured
  capacity and `--cache-ram` from `server_start`.
- **VRAM** — the ggml backend memory breakdown captured after model load: per device, model /
  context / compute / unaccounted / free as a stacked bar, plus host buffers. Static for the life
  of the process; replay reads it from `server_start.vram`.
- **Slots** — live only: per slot state, prompt tokens, cached share, generated tokens, and
  `n_ctx_used / n_ctx` as a bar.
- **Requests** — the most recent `request_done` and `request_rejected` records interleaved by
  `seq`: endpoint, prompt, cached share, generated, finish reason, TTFT, decode tok/s, attributed
  joules, and a queue / prefill / decode phase bar. Rejections show stage and error type, with the
  server's message on hover.

## Energy

A watt-second is a joule, so the panel reports joules per token: energy composes additively across
phases while a rate does not.

Every `throughput` record carries the exact board counter delta for its interval
(`board_joules`). The server splits it by the loop's phase time share over the same interval:

$$
J_{prefill} = B\,\frac{t_{prefill} + t_{host}\,\frac{t_{prefill}}{t_{prefill}+t_{decode}}}{T},\quad
J_{decode} = B\,\frac{t_{decode} + t_{host}\,\frac{t_{decode}}{t_{prefill}+t_{decode}}}{T},\quad
J_{idle} = B\,\frac{t_{idle}}{T}
$$

where $B$ is the board delta, $t_\ast$ the loop seconds in each phase, and
$T = t_{prefill}+t_{decode}+t_{host}+t_{idle}$. Host time — loop work outside `llama_decode` —
is split pro rata between prefill and decode. The three parts sum to the board delta exactly, so
there is no residual.

Over the retained window the dashboard sums the interval deltas (never differencing cumulative
endpoints, so a counter reset cannot produce a negative reading) and prices each figure against
tokens from the same intervals:

| Figure  | Definition                                                     |
| ------- | -------------------------------------------------------------- |
| served  | $\sum B \,/\, \sum (\text{prefill} + \text{decode tokens})$    |
| active  | $\sum (J_{prefill} + J_{decode}) \,/\, \sum (\text{prefill} + \text{decode tokens})$ |
| prefill | $\sum J_{prefill} \,/\, \sum \text{computed prompt tokens}$    |
| decode  | $\sum J_{decode} \,/\, \sum \text{committed generated tokens}$ |

A figure whose token denominator is zero is null and renders as `—`. **Served** prices idle loop
time too, so it degrades on a mostly idle server even when nothing about the model changed;
**active** tracks the schedule rather than the duty cycle and is the figure to compare between
builds. Prefill divides by computed prompt tokens, so cache hits do not flatter it; decode divides
by committed tokens, so rejected drafts cost energy without adding to the denominator.

The idle draw (`idle_watts`) is calibrated from fully idle sampling periods and shown as a
reference; it is not subtracted from anything.

Per-request energy on `request_done` is an attribution, not a measurement: the request's computed
prompt tokens times the server's prefill J/token over the request's lifetime, plus the same for
decode tokens. Concurrent requests share the same mean rate.

**Per 1M tokens** restates the per-token figures in watt-hours (× 1e6 / 3600), the denominator
inference is priced in. Board energy excludes the CPU and the rest of the platform, so none of
these are wall-socket figures.

## Reading the charts

- **Throughput** is an average over each reporting interval (5 s by default), so a sample fills
  its whole interval as a band. The server skips intervals with no activity and folds the skipped
  time into the next record's `interval_seconds`, so a wide band is idle time folded forward, not
  a long sustained rate.
- **Scheduler counts** on a throughput record are a snapshot taken when it was emitted, so they
  are drawn as points at their own timestamps.

Decode throughput is the sum across every busy slot, so it rises with concurrency even when no
individual request gets faster; the per-sequence rate divides by the mean busy slots per
`llama_decode` call. Both decode means are taken over intervals that generated tokens, so idle
time does not deflate them. Per-request decode rates in the Requests table are llama.cpp's own
per-slot figures and are not a partition of the aggregate.

The loop wall-clock split is what explains stalls: one loop drives every slot, so a second of host
work (tokenization, batch building, sampling, prompt-cache save / restore) is a second no slot
advanced.

TTFT is decomposed into queue wait and prompt processing over the requests that produced a first
token. A queue-dominated TTFT means requests are waiting for a slot, which more slots or shorter
generations address before any kernel work does; a prefill-dominated TTFT is prompts being
computed, and the cache reuse figure says how much the slot KV and prompt cache already saved.

Every reading carries an explanatory tooltip sourced from one glossary module, so a reading and its
explanation cannot drift apart.

## Cache

Two mechanisms produce cached prompt tokens, and both are counted in `n_cached_tokens`:

- **Slot KV reuse.** A slot keeps its KV after a request finishes. A follow-up prompt that shares
  its prefix is assigned back to that slot and reuses those cells without compute. This is why KV
  cells used stays high after traffic stops: it is retained prefixes, not leaked state. With
  `--kv-unified` every slot draws from one pool of `n_ctx` cells; otherwise each slot owns
  `n_ctx_slot` cells.
- **Prompt cache** (`--cache-ram`). Saved slot states in host RAM. A new prompt that shares a
  prefix with an entry restores it instead of recomputing. The limit is in MiB; -1 is unlimited, 0
  disables it, and a `bytes_limit` of 0 in the live snapshot reads as unlimited.

Cache reuse is $\sum n_{cached} / \sum n_{prompt}$ over the retained completed requests, so it is
available in replay; occupancy is live only.

## Replay

`load jsonl` reads a `--request-log-jsonl` file and renders it through the same components. When a
file contains records from more than one server instance — it is opened in append mode — only the
last instance is kept, because mixing two configurations on one axis would misattribute every
derived figure. Blank lines and a torn trailing line are skipped.

A log carries request and throughput history plus the `server_start` configuration and VRAM budget,
so throughput, the loop split, latency, energy, VRAM, cache reuse, and per-request analysis are all
available offline. Board sensors, slot state, and KV / prompt-cache occupancy are sampled levels,
never recorded, and those panels say so rather than showing a stale or zero reading.

## Derived analytics

The request aggregations — percentiles, finish-reason partitioning, TTFT decomposition, queue-wait
percentiles, cache reuse, speculative acceptance, and the energy and loop summaries — follow
definitions pinned by hand-computed fixture tests in `apps/web/src/lib/derive.test.ts`, including
the truncating nearest-rank percentile rule. `replay.test.ts` runs `apps/web/fixtures/sample.jsonl`
through the file loader's code path.

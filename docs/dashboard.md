<!-- Modified from tensorninja/ninfer-4090 (f408990) by lovelace. -->

# Dashboard

An optional single-page dashboard for one running lovelace `llama-server`. It answers, at a glance,
what the server is doing: current prefill and decode rates, whether requests are queueing for a
slot, where prompts are being served from, how the VRAM budget is spent, and whether the board is
the limit. It also loads a `--request-log-jsonl` file to analyze a past session offline.

The application lives in [`apps/web/`](../apps/web/) and is built with Bun, Vite, and React. It is
tensorninja/ninfer-4090's dashboard at `f408990`: lovelace serves ninfer's observability API as a
superset ([observability-parity.md](observability-parity.md)), so every upstream panel reads the
same fields at the same paths, and three lovelace panels read the lovelace-only data under the
`lovelace` key (see [lovelace panels](#lovelace-panels)).

## Build and run

```bash
cd apps/web
bun install
bun run build
```

Then serve it from the server itself, same-origin with the API (`--web-dir` is an alias of
llama.cpp's `--path`):

```bash
engine/build-sm89/bin/llama-server -m models/model.gguf \
  --telemetry --metrics --slots --web-dir apps/web/dist
```

`--telemetry` enables `/telemetry` and `/events`, `--metrics` enables `/metrics`, `--slots` enables
`/slots`. Open `http://127.0.0.1:8080/`. Add `--request-log-jsonl PATH` to also write the record
stream to a file that can be replayed later. The `lovelace-serve` image copies a prebuilt
`apps/web/dist` to `/opt/lovelace/web` (`deploy/serve.Dockerfile`), so build it before building the
image.

For development against a running server:

```bash
cd apps/web
NINFER_BASE_URL=http://127.0.0.1:8080 bun run dev
```

The dev server proxies `/telemetry`, `/events`, `/metrics`, `/slots`, `/health`, and `/v1` to that
origin, so CORS is not required. `NINFER_BASE_URL` (the upstream name, kept) is validated as an
origin and defaults to `http://127.0.0.1:8080`.

| Command | Purpose |
|---|---|
| `bun run dev` | development server on `127.0.0.1:5180` |
| `bun run build` | typecheck and emit `dist/` |
| `bun run typecheck` | types only |
| `bun test` | derivation-layer and replay tests |
| `bun run format` | Prettier write over the app sources |
| `bun run format:check` | Prettier check |

## Upstream panels on a lovelace server

Every upstream panel renders from the ninfer fields lovelace serves, with the mappings of
[observability-parity.md](observability-parity.md): a lane is a slot (`-np`), L1 / L2 / L3 are the
GPU-resident sessions, the RAM mirrors (`--session-ram`) and the NVMe files (`--session-dir`) of
the session store, and the execution-thread split is the server loop's time in `llama_decode`
(prefill / decode), in parks and restores (admission) and in the rest of the loop (upkeep). Fields
with no lovelace equivalent are constant placeholders (0, `false`, `""`, `"none"`, `null`), so some
readings are structurally empty:

- **Adapters** — lovelace serves `--lora` adapters through llama.cpp, without a resident adapter
  bank: rank, slots and bank bytes are 0, and requests carry no resolved adapter, so the panel
  shows the pool (if any) with no traffic.
- **Cache churn** — restore deferrals, publication superseded and preflight rejections are 0 by
  design, and recomputed coverage has no preflight agreement to measure (always null). Imported
  reuse, demotions and lost sessions (parks without a RAM mirror) are real.
- **Scheduler** — `max_pending_requests` is 0 (the queue is unbounded and never times out), so the
  ingress-queue meter reads empty, the header says "0 pending slots", the waiting count is drawn in
  the at-limit colour even when it is 0, and rejections are 0; publish seconds and the admission
  plan / commit split are 0. The Queue & waits panel is the meaningful queue view.
- **VRAM** — only weights / KV + state / workspace come from the ggml backend breakdown; CUDA graph,
  replay-record, DFlash and LoRA bank bytes are 0. The per-device breakdown is in the KV pool panel.
- **Requests / Latency** — tool-call counts, prepare / vision / publish timings are 0.

## Adapters

The engine reports its adapter pool, and usage is derived from completed request records, so the
panel distinguishes three states that a traffic-only view would conflate: an adapter serving
requests, an adapter in the pool but idle (servable, carrying no traffic), and an adapter that
appears in a replayed log but is not in the pool of the server now reporting. When a source reports
no inventory at all, the panel says pool membership is unknown rather than claiming an adapter is
absent.

On lovelace the pool is the `--lora` list and there is no bank (see above).

## Data sources

The dashboard reads two channels because they answer different questions.

| Channel | Kind | Carries |
|---|---|---|
| `GET /telemetry` | polled at 1 Hz | levels: board sensors, scheduler occupancy, VRAM, cache fill, adapter inventory; `lovelace`: KV budget, KV pool, sessions, loop split, ggml VRAM |
| `GET /events` | SSE | history: throughput samples, board energy, and completed requests, each with its `lovelace` block |

Levels cannot be reconstructed by replaying deltas, and the event stream is bounded and lossy
under backpressure, so the poll is always authoritative for current state. `/metrics` is not read
by the dashboard: every counter it needs is already in the two channels above, in a form that does
not require differencing scrapes. lovelace also streams `lovelace_sched` records (one per state
transfer between VRAM and host memory); the dashboard does not subscribe to them, the KV budget
totals and the throughput `lovelace.parking` deltas carry the same information aggregated.

Board energy is the one sensor reading that is carried in the record stream rather than only in the
poll, because it is an interval total rather than a level. The Energy panel therefore works on a
replayed log; the GPU panel, which reports levels, does not.

## Energy

The Energy panel reports what the work costs, in joules. A watt-second is a joule, so tokens per
watt-second and tokens per joule are the same figure; the panel reports joules per token, because
energy composes additively across phases while a rate does not.

Two denominators are shown because they answer different questions and neither substitutes for the
other. **Served** divides all board energy by all tokens, including the draw while idle between
requests; it is what the work actually costs and it degrades on a mostly idle server even when
nothing about the model changed. **Active** removes the idle share and tracks the schedule rather
than the duty cycle, which is the figure to compare between two builds.

Prefill divides by computed prefill tokens, so prefix-cache hits do not flatter the kernels, and
decode divides by committed decode tokens, which under MTP counts accepted tokens only —
speculation burns compute on drafts that may be rejected, so it can raise tokens per second and
joules per token at the same time.

The board total is the exact counter delta. lovelace splits it by the server loop's phase time over
the interval (sampled at 10 Hz): prefill and decode get their `llama_decode` time, host time is
shared pro rata between them, and idle gets the rest. The parts sum to the board total exactly, so
the residual is always 0 and the residual marker never fires. Boards with no cumulative energy
counter report the panel as unavailable rather than as zero.

The **per 1M tokens** stat restates the served figure in watt-hours, which is the denominator
inference is priced in and the unit electricity is billed in; one multiplication by a local rate
makes it comparable to a published $/1M-token price. It is an exact rescale of joules per token by
`1e6/3600` and carries no extra information.

## Reading the charts

Two sampling semantics are deliberately drawn differently, because conflating them would assert
measurements that were never taken.

- **Throughput** is an average over each reporting interval (5 s), so a sample fills its whole
  interval as a band. The server skips intervals with no activity and folds the skipped time into
  the next sample's `interval_seconds`, so a wide band is idle time folded forward, not a long
  sustained rate.
- **Scheduler occupancy** (`running`, `waiting`, and the lovelace queue, preemption and session
  tier counts) is a snapshot taken when the report was emitted, not an interval average, so it is
  drawn as points at their own timestamps.

Decode throughput is server-wide: the server sums committed tokens over every slot in a
`llama_decode` call, so the figure rises with concurrency even when no individual request gets
faster. The panel pairs it with the mean decode batch and the per-sequence rate (aggregate divided
by batch), which is what a single client experiences. Both decode means are taken over intervals
that ran a decode call rather than the whole window, so idle time does not deflate them.
Per-request rates in the Requests table are not a partition of the aggregate, so they do not sum to
it.

Prefill and decode are plotted on separate rate scales over one shared time axis. Prefill runs
roughly an order of magnitude faster than decode, so a single linear axis renders decode as a flat
sliver; each series is therefore read against its own axis, and the interval tooltip reports both
rates so they remain directly comparable at any instant.

Every reading carries an explanatory tooltip. Panel headings, stat captions, legend entries,
stacked-bar segments, and table headers are hover and focus targets that define the term, name the
units, and say what it means when the value moves. The wording lives in one glossary module
(`src/lib/glossary.ts`, with the lovelace terms in `src/lib/glossary-lovelace.ts`) rather than
beside each panel, so a reading and its explanation cannot drift apart.

The execution-thread wall-clock split is the panel that explains contention: one server loop
drives every slot, so a second spent in any unit is a second in which no other slot advances. The
lovelace loop split in the Queue & waits panel shows the same time as prefill / decode / host /
idle over the rolling 10 s window.

TTFT is decomposed into queue, restore, and prefill. A queue-dominated TTFT means requests are
waiting for a slot or for KV cells rather than being computed; the Queue & waits panel says which.

## Cache churn

The occupancy panel answers what the cache is holding. The churn panel answers whether holding it
is still paying, which is a different question and is not answered by the hit rate: a working set
that has outgrown VRAM keeps hitting and simply starts paying a RAM or NVMe restore on every turn.

Eviction on its own is a cache working normally, so no single count here is pathological. The
readings that carry meaning on lovelace are:

- **imported reuse** — share of restores served from RAM (L2) or NVMe (L3) rather than a session
  still resident in its slot (L1). Drift down the tiers is the churn signal, and the restore chart
  shows it directly as the tier mix per interval.
- **lost sessions** — sessions that left VRAM without a RAM mirror (`l1_evictions` minus
  `l1_demotions`). The next turn of that conversation has no state to restore and prefills from
  zero.
- **deferrals**, **superseded**, **recomputed coverage** — 0 / not measured on lovelace, see above.

RAM evictions are demotions to NVMe (`l2_evictions`, bytes written); lovelace deletes sessions only
by TTL or the disk budget, so NVMe evictions are 0.

Both charts are interval counts, so they are drawn as bands over the window each sample covers,
and the tier series are stacked because they partition one total.

## lovelace panels

Three panels after the upstream ones read only the `lovelace` extension
(`lovelace.schema_version` 2, [observability-parity.md](observability-parity.md)). Every field is
declared optional in `src/lib/lovelace.ts`, so a pure ninfer server or log still renders: each panel
then says "not reported" instead of showing zeros.

### Sessions

lovelace's per-agent session store: every conversation keeps a write-back mirror of its KV in a
pinned host arena, so a slot can be freed at no cost and the session restored at PCIe speed; mirrors
are demoted to NVMe files when the arena runs short.

- **Tiers** — GPU / RAM / NVMe session counts (`sessions.n_gpu`, `n_ram`, `n_disk`), with the GPU
  sessions' resident cells against the KV pool, `ram_bytes` against `arena_bytes`, and
  `disk_bytes` against `disk_limit_bytes` (0 = unlimited).
- **Moves** — `promotions` (NVMe → RAM), `demotions` (RAM → NVMe), `prefetches` / `prefetch_hits`
  (restores started while the request still queued), NVMe IO bytes and seconds (`io_bytes_read`,
  `io_read_s`, `io_bytes_written`, `io_write_s`), mirror syncs and stale syncs.
- **Sessions table** (live only) — per session: `id`, `tier` (gpu, ram, nvme, or ram → nvme /
  nvme → ram while a file job runs), `slot`, `resident_cells`, `mirror_tokens` / `mirror_bytes`,
  `file_bytes`, `restores` with `last_restore_ms`, and `stale` (cells the mirror has not synced).
- **History** — the tier counts per throughput report (points) and promotions + demotions per
  interval (stacked bands), from `throughput.lovelace.sessions`. In replay the counters are the sums
  of those interval deltas and the tiers the last report's gauges, against the configured
  `--session-ram` / disk limit of `server_start.engine.continuation_cache`.

Live source: `/telemetry` `lovelace.sessions`; null (panel says so) without `--session-ram`.

### Queue & waits

The KV budget (`--kv-budget`) admits a request only when a slot is free and the pool can hold its
reservation; otherwise it waits, idle sessions are parked to host memory, and a generating slot can
be preempted at a token boundary.

- **Now** — `scheduler.kv_budget.queue_waiting`, `slots_preempted`, and whether the budget is on.
- **Totals since start** (`kv_budget.totals`) — `waited_slot` / `waited_kv`, parks / unparks with
  `park_bytes`, `park_seconds`, `park_gbps` (and the unpark equivalents), preempts / resumes,
  `recompute_tokens`, `kv_overflows` (should stay 0), park drops, grows, rollback fallbacks and the
  loop time spent issuing transfers.
- **Wait reasons** — completed requests partitioned by `request_done.lovelace.wait_reason`
  (none / slot / kv) with `queue_wait_ms` p50 / p90 / max per reason.
- **Queue history** — `throughput.lovelace.scheduler.queue_waiting`, `queue_deferred` and
  `slots_preempted` as points.
- **Loop split** — `lovelace.loop` (prefill / decode / host / idle over the rolling window, and the
  current phase).

In replay the totals are the summed `throughput.lovelace.parking` and `scheduler.waited_*`
deltas (GB/s = bytes / device seconds) and the loop split is the summed `loop_seconds`.

### KV pool

- **Cells** — `lovelace.kv.cells_used` / `cells_reserved` / `cells_total` as a used / reserved /
  free bar, with `unified`, `n_ctx_slot` and the K / V cache types. Reserved is what the KV budget
  has promised to running requests beyond their current use.
- **Prompt cache** — `lovelace.prompt_cache` bytes against its limit (0 = unlimited), entries,
  tokens.
- **ggml VRAM breakdown** — `lovelace.vram`: per device, model / context / compute / unaccounted /
  free as a stacked bar, plus host buffers. Static for the life of the process.

In replay the occupancy is the last `throughput.lovelace.kv` snapshot, the configuration and the
VRAM breakdown come from `server_start.lovelace` (`config`, `vram`), and the prompt cache shows
only its configured `--cache-ram`.

## Replay

`load jsonl` reads a `--request-log-jsonl` file and renders it through the same components. When a
file contains records from more than one server instance — it is opened in append mode — only the
last instance is kept, because mixing two configurations on one axis would misattribute every
derived figure.

A log carries request and throughput history plus the `server_start` configuration, so throughput,
latency, cache occupancy against configured capacity, per-request analysis, and the lovelace
session, queue and KV pool history are all available offline. Board telemetry, live slot occupancy
and the per-session list are sampled, never recorded, and those panels say so rather than showing a
stale or zero reading.

## Derived analytics

The request aggregations — percentiles, prompt-source distribution, miss-reason and
restore-failure partitioning, TTFT decomposition, and restore statistics — follow the definitions
of upstream's maintainer script `cache_health.py`, including its truncating nearest-rank percentile
rule; `src/lib/derive.test.ts` (upstream, unchanged) pins them.

Churn, recomputed coverage and the lovelace session and parking totals are summed from the interval
deltas the throughput record already carries, rather than by differencing the cumulative endpoints
of the window, so a server restart inside the window cannot turn a counter reset into a negative
or absurd reading. `src/lib/lovelace.test.ts` pins the lovelace derivations by hand on
`apps/web/fixtures/sample.jsonl` (a schema-20 log with the lovelace extension), loaded through the
file loader's code path.

# Observability: request log, /events, /telemetry, /metrics, /slots, board energy

lovelace serves **tensorninja/ninfer-4090's observability API** (pinned at `f408990`, request-log schema 20) as a
superset: the same paths, field names, JSON types and units, so tensorninja's dashboard (`apps/web`) and clients such
as the omarchy-ninfer widget work unchanged. lovelace-only data is under a `lovelace` key (**`lovelace.schema_version`
2**) in every record and in `/telemetry`; Prometheus additions use the `lovelace:` prefix. The field-by-field mapping
(same / mapped / N/A) is in [observability-parity.md](observability-parity.md); this page describes how the data is
produced and the lovelace extension.

`llama-server` flags (engine/tools/server, implemented in `server-observe.{h,cpp}`):

| Flag | Effect |
|---|---|
| `--telemetry` | enables `GET /telemetry`, `GET /events`, the 10 Hz energy sampler, and the snapshot-built `/metrics` and `/slots` |
| `--request-log-jsonl PATH` | appends every record below to PATH (append mode); implies `--telemetry` |
| `--metrics`, `--slots` | llama.cpp's switches for `/metrics` and `/slots` (both needed by tensorninja's clients) |
| `--web-dir DIR` | alias of llama.cpp's `--path`: serves the dashboard (`apps/web/dist`) at `/` on the API port; API routes win |

## Cost model (CHARTER E1: nothing measurable on the decode path)

- The server loop only performs relaxed atomic stores (phase clock, token counters), writes fixed-size fields into the
  observer **snapshot** under one short mutex per `llama_decode` step (per-slot view, scheduler gauges, KV budget and
  llama.cpp metric totals; the prompt-cache and session-store tables at most every 250 ms, into preallocated
  storage), and enqueues POD records at request start and end.
- `GET /telemetry`, `/metrics` and `/slots` are built on the HTTP threads from that snapshot. They never post a task
  to, wake, or wait for the server loop (a sleeping server is not woken either). `/slots` keeps llama.cpp's fields;
  `params` and `prompt` come from the launched task, which is immutable; `generated` is not reported.
- A writer thread formats records, writes the file and fans them out to `/events` subscribers; a sampler thread reads
  the NVML energy counter at 10 Hz. NVML is `dlopen`ed (`libnvidia-ml.so.1`); live sensors are read on the HTTP thread
  when `/telemetry` is requested. `nvmlDeviceGetPcieThroughput` is never called (it stalls ~20 ms in the driver and
  slowed decode by 1-2.5 % at 1 Hz); the PCIe fields are 0.

Measured: `bench/results/obs-parity/REPORT.md` (telemetry off vs on with a 1 Hz `/telemetry` poller, a 4 s
`/metrics` scraper and an `/events` subscriber).

## Conventions

- Records are tensorninja's: envelope `artifact_type: "ninfer_serve_request_log"`, `schema_version: 20`, `event`,
  `timestamp_unix_ms`, `server_instance_id` (16 hex chars, new per process); then `lovelace: {schema_version: 2,
  seq, …}`, `seq` monotonic per instance (0 = `server_start`).
- ninfer's units: `timings_seconds.*` in seconds, `*_microseconds` in µs, bytes, joules. In `lovelace` objects:
  `*_ms` milliseconds, `*_seconds` / `*_s` seconds, `*_bytes` bytes, `*_joules` joules.
- `null` means "not measured / not applicable" and must never render as 0. tensorninja fields with no lovelace
  equivalent keep their JSON type with 0 / false / "" / "none" (listed in the parity doc).

## Record stream (`--request-log-jsonl` lines == `GET /events` `data:` payloads, byte-identical)

SSE framing as tensorninja's: `retry: 2000`, then per record `event: <record.event>\ndata: <json>\n\n`; keepalives are
SSE comments (`: keepalive`). A new subscriber first receives the retained `server_start`, then up to 512 recent
records, then live records. Records may be dropped for a slow subscriber (bounded queue, drop-oldest);
`/telemetry.lovelace.events.records_dropped` counts it.

Events: `server_start`, `request_start` (when a slot launches the request, after its queue wait), `request_done`,
`request_error` (every request that produced no result: HTTP parse / tokenize failure, rejection at launch, failure
before or during compute), `throughput` (every 5 s with activity; idle time folds into the next `interval_seconds`),
and lovelace's `lovelace_sched`.

### lovelace extension per record

```
server_start.lovelace:   { build_info, model: {path, name, ftype, n_params, size_bytes, n_ctx_train},
                           config: {n_ctx, n_ctx_slot, n_parallel, n_batch, n_ubatch, kv_unified, cache_type_k, cache_type_v,
                                    flash_attn, cache_ram_mib /* -1 unlimited, 0 off */, n_gpu_layers,
                                    speculative: {types /* "none" when off */, n_max, n_min},
                                    throughput_interval_ms, energy_sample_hz, request_log_jsonl: string|null},
                           gpu: {available, error?, name, uuid, driver_version, memory_total_bytes, sm_clock_max_mhz,
                                 power_limit_watts, energy_available},
                           vram: VramBudget|null }
request_start.lovelace:  { id_task, id_slot, id_parent, task_type /* completion|infill|embedding|rerank */,
                           endpoint /* native|oai_chat|oai_cmpl|oai_resp|oai_asr|oai_embd|anthropic */,
                           n_prompt_tokens, n_predict /* -1 unlimited */, n_ctx_slot,
                           queue_wait_ms,                      // arrival at the HTTP handler -> slot launch
                           wait_reason: "none"|"slot"|"kv",    // KV budget: first reason the request waited
                           n_unparked_tokens }                 // tokens of a parked session restored at launch
request_done.lovelace:   { id_task, id_slot, id_parent, task_type, endpoint, n_predict, n_ctx_slot,
                           n_ctx_used,                         // slot sequence length at release
                           truncated, wait_reason, n_unparked_tokens, queue_wait_ms,
                           session: { id, tier: "gpu"|"ram"|"disk"|"new",   // where the session was at admission
                                      restore_bytes, restore_ms,             // bytes and device time of the restore copies
                                      park_ms, restore_issue_ms,             // host time parking others / issuing the restore
                                      restore_ready_ms|null,                 // copies complete, ms after admission (0 = before)
                                      prompt_start_ms|null,                  // first prompt batch queued, ms after admission
                                      prefetch_hit } | null,                 // null without --session-ram
                           rates: {prefill_tps|null, decode_tps|null},      // llama.cpp's per-request definitions
                           speculative: {n_verif_steps},
                           energy: {prefill_joules, decode_joules, total_joules} | null }
request_error.lovelace:  { id_task|null, id_slot|null, task_type, endpoint,
                           stage: "http"|"launch"|"prompt"|"generation", error_type, n_prompt_tokens, queue_wait_ms|null }
throughput.lovelace:     { tokens: {drafted, accepted},
                           phase_tokens_per_second: {prefill|null, decode|null},   // tokens / loop seconds in that phase
                           requests: {started, done, rejected},
                           scheduler: {slots_total, slots_busy, slots_processing_prompt, queue_deferred,   // snapshot
                                       queue_waiting, slots_preempted,                                     // snapshot
                                       waited_slot, waited_kv},                                            // interval
                           kv: {cells_total, cells_used, cells_reserved},                                  // snapshot
                           parking: {sessions_parked /* snapshot */, parks, unparks, preempts, resumes, park_bytes,
                                     unpark_bytes, park_seconds, unpark_seconds, recompute_tokens},       // interval
                           loop_seconds: {prefill, decode, host, idle},   // server loop wall clock, sums to ~interval_seconds
                           batch: {decode_calls, mean_busy_slots|null},
                           sessions: {n_gpu, n_ram, n_disk, n_io, ram_bytes, disk_bytes,              // snapshot
                                      promotions, demotions, prefetches, prefetch_hits,
                                      io_bytes_read, io_bytes_written} | null }                     // interval
lovelace_sched.lovelace: { action: "park"|"unpark"|"preempt"|"resume"|"sync",
                           id_slot, id_task|null, n_tokens, bytes,
                           ms,        // device time of the copies
                           host_ms,   // server loop time issuing and waiting for them
                           gbps }
```

VramBudget (ggml backend memory breakdown, captured once after model load):
`{devices: [{name, description, total_bytes, free_bytes, model_bytes, context_bytes, compute_bytes,
unaccounted_bytes}], host: {model_bytes, context_bytes, compute_bytes}}`; `context_bytes` is KV cache + recurrent state,
`unaccounted = total - free - (model + context + compute)`.

## GET /telemetry

tensorninja's snapshot (`gpu`, `scheduler`, `cache`, `slots`, `adapters`, `memory`, `events`; mapping in the parity
doc), plus the keys of keylimesoda/ninfer-4090-lin-hybrid's `cache` that the omarchy-ninfer widget reads, plus:

```
lovelace: {
  schema_version: 2, build_info, sleeping,
  energy: { available, joules: {board, prefill, decode, idle} | null,   // since server start
            idle_watts|null,                                             // calibrated from fully idle periods
            board_watts_1s|null },                                       // mean board power over the last second
  gpu: { pstate|null },
  loop: { window_seconds /* 10 */, phase: "idle"|"host"|"prefill"|"decode", seconds: {prefill, decode, host, idle},
          fraction: {…} },
  loop_totals_seconds: { prefill, decode, host, idle },
  scheduler: {
    slots_total, slots_busy, slots_idle, slots_processing_prompt, slots_generating, queue_deferred,
    kv_budget: { enabled, queue_waiting, slots_preempted,
                 gen_reserve, headroom, prefill_chunk_gen, timeslice,  // --kv-budget-gen, --kv-budget-headroom, --prefill-chunk-gen, --kv-timeslice
                 totals: { parks, park_drops, unparks, preempts, resumes, grows, park_bytes, unpark_bytes,
                           park_seconds, unpark_seconds,              // device time of the copies
                           park_host_seconds, unpark_host_seconds,    // server loop time issuing and waiting for them
                           park_gbps, unpark_gbps, waited_slot, waited_kv,
                           rollback_fallbacks,                        // truncations the memory refused, recovered
                           recompute_tokens, kv_overflows } },        // kv_overflows should stay 0
    totals: { prompt_tokens, prompt_tokens_cached, predicted_tokens, prompt_seconds, predicted_seconds, n_decode,
              n_busy_slots_per_decode, drafted, accepted, requests_started, requests_done, requests_rejected } },
  kv: { cells_total, cells_used, cells_reserved, unified, n_ctx_slot, cache_type_k, cache_type_v },
      // cells_used = sum of slot sequence lengths (idle slots keep their KV until parked)
      // cells_reserved = cells_used plus the unused part of running requests' reservations (KV budget)
  prompt_cache: { enabled, bytes_used, bytes_limit /* 0 = unlimited */, tokens, tokens_limit, entries },
  sessions: { arena_bytes, arena_free_bytes, ram_bytes, disk_bytes, disk_limit_bytes /* 0 = unlimited */,
              n_sessions, n_gpu, n_ram, n_disk, n_io, demotions, promotions, deleted, stale_syncs, prefetches,
              prefetch_hits, syncs, sync_bytes, io_bytes_written, io_bytes_read, io_write_s, io_read_s,
              sessions: [{ id, tier: "gpu"|"ram"|"demoting"|"disk"|"promoting", slot, resident_cells, mirror_tokens,
                           mirror_bytes, mirror_segments, checkpoints, file_bytes, stale, sync_ms|null,
                           since_sync_s|null, syncs, truncations, restores, last_restore_bytes, last_restore_ms,
                           last_restore_wall_ms, prefetches, prefetch_hits, demotions, promotions }] } | null,
  vram: VramBudget | null,
  slots: [{ id, state: "idle"|"wait_other"|"started"|"processing_prompt"|"done_prompt"|"generating"|"preempted"|"recompute",
            is_processing, n_ctx, n_ctx_used, n_reserved, id_task|null, wait_reason|null, n_unparked_tokens,
            n_prompt_tokens, n_cached_tokens, n_prefill_tokens, n_decoded, speculative, session_id|null }],
  events: { jsonl_path|null, records_emitted, records_dropped }
}
```

The session table and prompt-cache figures are refreshed at most every 250 ms (they walk containers); everything else
is as of the last `llama_decode` step.

## Board energy

The sampler reads the NVML cumulative energy counter at 10 Hz and splits each delta by the server loop's phase time
over the same period: `prefill` / `decode` = `llama_decode()` + sync (split by the batch's prompt-row share, the draft
model counts as decode), `host` = loop work while a request is in flight (split pro rata between prefill and decode),
`idle` = everything else. The parts sum to the board delta exactly, so ninfer's `residual_joules` is 0 by
construction. `idle_watts` is calibrated from fully idle sampling periods (0 until the first calibration, as
upstream). Board energy is the whole board: other processes on the GPU are included. Per-request energy
(`request_done.lovelace.energy`) is an attribution: the request's computed tokens times the server's J/token over its
lifetime.

## GET /metrics

llama.cpp's series (rate gauges averaged since the previous scrape), then tensorninja's `ninfer:*` series with the
same names and order as `serve_metrics.cpp` (the widget reads `ninfer:requests_total`,
`ninfer:prefix_cache_hit_tokens_total`, `ninfer:draft_tokens_total`, `ninfer:draft_accepted_tokens_total`), then:

```
lovelace:loop_seconds_total{phase="prefill|decode|host|idle"}   counter
lovelace:energy_joules_total{phase="prefill|decode|idle"}       counter   (with an energy counter)
lovelace:board_power_watts                                      gauge     (mean over the last 1 s)
lovelace:kv_cells_total | kv_cells_used | kv_cells_reserved     gauge
lovelace:sched_waiting | sched_preempted | sessions_parked      gauge
lovelace:sched_{parks,park_drops,unparks,preempts,resumes,grows}_total                counter
lovelace:sched_{park,unpark}_bytes_total, sched_{park,unpark}_seconds_total           counter
lovelace:sched_wait_{slot,kv}_total                                                   counter
lovelace:sched_{rollback_fallbacks,recompute_tokens,kv_overflows}_total               counter
lovelace:sessions | sessions_gpu | sessions_ram | sessions_disk | sessions_{ram,disk}_bytes   gauge (--session-ram)
lovelace:sessions_{demotions,promotions,prefetches,prefetch_hits}_total                      counter (--session-ram)
```

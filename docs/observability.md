# Observability: request log, /events, /telemetry, board energy (schema_version 1)

`llama-server` flags (engine/tools/server, implemented in `server-observe.{h,cpp}`):

| Flag | Effect |
|---|---|
| `--telemetry` | enables `GET /telemetry`, `GET /events`, the 10 Hz energy sampler and the `/metrics` additions |
| `--request-log-jsonl PATH` | appends every record below to PATH (append mode); implies `--telemetry` |
| `--web-dir DIR` | alias of llama.cpp's `--path`: serves the dashboard (`apps/web/dist`) at `/` on the API port; API routes win |

Threading: the server loop only performs relaxed atomic stores (phase clock, token counters) and enqueues POD
records. A writer thread formats records and writes the file / fans out to subscribers; a sampler thread reads the
NVML energy counter at 10 Hz and attributes each delta to prefill / decode / idle by the loop's phase time share
over the same period. NVML is `dlopen`ed (`libnvidia-ml.so.1`); without it every GPU/energy field is absent or `null`.
NVML live sensors are read on the HTTP thread only when `/telemetry` is requested.

Loop phases: `prefill` / `decode` = `llama_decode()` + sync (and the draft model), split by the batch's prompt-row
share; `host` = everything else the loop does while a request is in flight or a task is handled; `idle` = waiting
for tasks. Host time is attributed to prefill / decode pro rata when a request was in flight during the sampling
period, to idle otherwise. Board energy is the whole board: other processes on the GPU are included.

All numbers are JSON numbers. `null` means "not measured / not applicable" and must never render as 0.
Times named `*_ms` are milliseconds, `*_seconds` seconds, `*_joules` joules, `*_bytes` bytes.

## Record stream (`--request-log-jsonl PATH` lines == `GET /events` SSE `data:` payloads, byte-identical)

SSE framing: each record is sent as `data: <json>\n\n`. Keepalives are SSE comments (`: keepalive\n\n`).
A new subscriber first receives the retained `server_start`, then up to 512 most recent records (replay backlog), then live records.
Records may be dropped for a slow subscriber (bounded queue, drop-oldest); `/telemetry.events.records_dropped` counts it.

Envelope on every record:
```
schema_version: 1
event: "server_start" | "request_start" | "request_done" | "request_rejected" | "throughput"
server_instance_id: string      // 16 hex chars, new per process
seq: number                     // monotonic per instance, starts at 0 (server_start)
timestamp_unix_ms: number
```

### server_start
```
build_info: string
model: { path: string, name: string, ftype: string, n_params: number, size_bytes: number, n_ctx_train: number }
config: {
  n_ctx: number, n_ctx_slot: number, n_parallel: number, n_batch: number, n_ubatch: number,
  kv_unified: boolean, cache_type_k: string, cache_type_v: string, flash_attn: string,
  cache_ram_mib: number,            // --cache-ram; -1 = unlimited, 0 = disabled
  n_gpu_layers: number,
  speculative: { types: string, n_max: number, n_min: number },   // types "none" when off
  throughput_interval_ms: number,   // reporting interval of `throughput` records (5000)
  energy_sample_hz: number,         // 10
  request_log_jsonl: string | null
}
gpu: GpuStatic                     // see /telemetry gpu; only the static fields: available, error?, name, uuid, driver_version, memory_total_bytes, sm_clock_max_mhz, power_limit_watts, energy_available
vram: VramBudget | null
```
VramBudget (from ggml backend memory breakdown, captured once after model load):
```
{ devices: [{ name, description, total_bytes, free_bytes, model_bytes, context_bytes, compute_bytes, unaccounted_bytes }],
  host: { model_bytes, context_bytes, compute_bytes } }
```
`context_bytes` is KV cache + recurrent state; `compute_bytes` is the compute graph buffers; `unaccounted = total - free - (model+context+compute)` (other processes, CUDA context, fragmentation).

### request_start  (emitted when a slot picks the task up)
```
request: RequestInfo
queue_wait_ms: number              // arrival at HTTP handler -> slot launch
wait_reason: "none" | "slot" | "kv"   // KV budget: first reason the request waited for admission (no idle slot / no KV cells)
n_unparked_tokens: number          // tokens of a parked session restored from host memory at launch (0 = none)
```
RequestInfo:
```
{ id_task: number, id_slot: number | null, id_parent: number | null,
  task_type: "completion" | "infill" | "embedding" | "rerank",
  endpoint: "native" | "oai_chat" | "oai_cmpl" | "oai_resp" | "oai_asr" | "oai_embd" | "anthropic",
  stream: boolean, n_prompt_tokens: number, n_predict: number /* -1 unlimited */, n_ctx_slot: number }
```

### request_done  (emitted when the slot is released: finished, cancelled, or failed mid-flight)
```
request: RequestInfo
result: {
  finish_reason: "eos" | "word" | "limit" | "cancelled" | "error" | "done",   // done = embedding/rerank task
  truncated: boolean,
  n_prompt_tokens: number,        // prompt length
  n_cached_tokens: number,        // reused from slot KV / prompt cache (no compute)
  n_prefill_tokens: number,       // prompt tokens actually computed
  n_completion_tokens: number,    // generated tokens committed
  n_ctx_used: number,             // slot sequence length at release
  n_unparked_tokens: number,      // as in request_start
  wait_reason: "none" | "slot" | "kv",
  error: string | null
}
timings_ms: {
  queue_wait: number,
  prefill: number,                // slot prompt processing wall time
  decode: number,                 // first token -> last token
  ttft: number | null,            // arrival -> first token (queue_wait + prefill, approximately)
  total: number                   // arrival -> release
}
rates: { prefill_tps: number | null, decode_tps: number | null }   // per request (llama.cpp's own definitions)
speculative: { n_drafted, n_accepted, n_verif_steps, accept_rate: number|null, accepted_per_pos: number[] } | null   // null when no drafting happened
energy: { prefill_joules: number, decode_joules: number, total_joules: number } | null   // null without a board energy counter
```
Per-request energy = request's computed prefill tokens x (server prefill J / server prefill token) over the request's lifetime, plus the same for decode tokens. It is an attribution, not a measurement of the request alone: concurrent requests share the mean rate.

### request_rejected  (the request never produced a result)
```
request: { id_task: number | null, id_slot: number | null, task_type, endpoint, n_prompt_tokens: number, n_ctx_slot: number }
stage: "http" | "launch" | "prompt"      // http: parse/tokenize failure; launch: rejected at slot launch; prompt: rejected before any compute (e.g. context exceeded)
error: { type: string, message: string }  // type e.g. "invalid_request", "exceed_context_size", "server", "not_supported", "unavailable"
queue_wait_ms: number | null
```

### sched  (KV budget, one record per state transfer between VRAM and host memory)
```
action: "park" | "unpark" | "preempt" | "resume"   // park: idle session VRAM -> host; unpark: host -> VRAM at launch;
                                                   // preempt / resume: a generating slot paused at a token boundary and resumed
id_slot: number, id_task: number | null, n_tokens: number,
bytes: number,                     // state bytes copied (target + draft context; 0 = no copy, e.g. resume by re-prefill)
ms: number, gbps: number           // wall time of the copy (host allocation included) and bytes / time
```

### throughput  (every throughput_interval_ms, only when the interval had activity; idle time is folded into the next record's interval_seconds)
```
interval_seconds: number
tokens: { prefill: number, decode: number, drafted: number, accepted: number }   // prefill = computed prompt tokens, decode = committed generated tokens
throughput_tokens_per_second: { prefill: number, decode: number }                // tokens / interval_seconds (aggregate over all slots)
phase_tokens_per_second: { prefill: number | null, decode: number | null }       // tokens / loop seconds spent in that phase
requests: { started: number, done: number, rejected: number }
scheduler: { slots_total, slots_busy, slots_processing_prompt, queue_deferred,   // snapshot at emit time (drawn as points, not bands)
             queue_waiting, slots_preempted,                                     // KV budget: admission queue length, paused slots
             waited_slot, waited_kv }                                            // requests that started waiting in the interval, by reason
kv: { cells_total, cells_used, cells_reserved }   // snapshot; reserved = sum over slots of max(used, reservation) of running requests + cells of idle slots
parking: { sessions_parked,                       // prompt cache entries (snapshot)
           parks, unparks, preempts, resumes, park_bytes, unpark_bytes, park_seconds, unpark_seconds,
           recompute_tokens }                     // interval deltas; recompute = tokens re-prefilled to recover a lost sequence
loop_seconds: { prefill: number, decode: number, host: number, idle: number }    // server loop wall clock split, sums to ~interval_seconds
batch: { decode_calls: number, mean_busy_slots: number | null }                  // llama_decode calls and mean busy slots per call
energy: {
  board_joules: number,           // exact board counter delta
  prefill_joules: number, decode_joules: number, idle_joules: number,   // board joules split by loop phase time share (host time is split pro rata between prefill and decode); sums to board_joules
  idle_watts: number | null,      // board draw calibrated from fully idle sampling periods
  joules_per_token: { served: number|null, active: number|null, prefill: number|null, decode: number|null }
    // served = board / (prefill+decode tokens); active = (prefill+decode joules) / (prefill+decode tokens);
    // prefill = prefill_joules / prefill tokens; decode = decode_joules / decode tokens
} | null
```

## GET /telemetry (complete snapshot, poll at 1 Hz; `sleeping: true` and null loop-side sections while the server sleeps)
```
timestamp_unix_ms, server_instance_id: string, uptime_seconds: number, model_id: string, build_info: string
gpu: {
  available: boolean, error?: string,
  name?, uuid?, driver_version?, temperature_c?, fan_percent?,
  power_watts?,                    // NVML trailing-average board power
  power_limit_watts?, energy_available?: boolean,
  energy_joules_total?: number | null,   // board counter since driver load
  utilization_gpu_percent?, utilization_memory_percent?,
  sm_clock_mhz?, sm_clock_max_mhz?, memory_clock_mhz?,
  memory_used_bytes?, memory_total_bytes?,
  pstate?: number,
  throttle_reasons?: string[]      // gpu_idle, applications_clocks_setting, sw_power_cap, hw_slowdown, sync_boost, sw_thermal_slowdown, hw_thermal_slowdown, hw_power_brake_slowdown, display_clock_setting
}
energy: {
  available: boolean,
  joules: { board: number, prefill: number, decode: number, idle: number } | null,   // since server start
  idle_watts: number | null,
  board_watts_1s: number | null    // mean board power over the last second, from counter deltas
}
scheduler: {
  slots_total, slots_busy, slots_idle, slots_processing_prompt, slots_generating, queue_deferred,
  kv_budget: {
    enabled: boolean, queue_waiting: number, slots_preempted: number,
    gen_reserve: number, headroom: number, prefill_chunk_gen: number,   // --kv-budget-gen, --kv-budget-headroom, --prefill-chunk-gen
    totals: { parks, park_drops, unparks, preempts, resumes, grows, park_bytes, unpark_bytes, park_seconds, unpark_seconds,
              park_gbps, unpark_gbps, waited_slot, waited_kv, rollback_fallbacks, recompute_tokens, kv_overflows }
      // park_drops: idle sessions evicted without a new host copy (already cached, or no cache room)
      // rollback_fallbacks: truncations the memory refused, recovered from a checkpoint or by re-prefill
      // kv_overflows: llama_decode() found no free cells despite the budget (should stay 0)
  },
  loop: {
    window_seconds: number,        // rolling window (10 s)
    phase: "idle" | "host" | "prefill" | "decode",   // current
    seconds:  { prefill, decode, host, idle },       // over the window
    fraction: { prefill, decode, host, idle }        // seconds / window_seconds
  },
  loop_totals_seconds: { prefill, decode, host, idle },   // since start
  totals: {
    prompt_tokens, prompt_tokens_cached, predicted_tokens,
    prompt_seconds, predicted_seconds, n_decode, n_busy_slots_per_decode,
    drafted, accepted, requests_started, requests_done, requests_rejected
  }
}
memory: {
  kv: { cells_total: number, cells_used: number, cells_reserved: number, unified: boolean, n_ctx_slot: number, cache_type_k: string, cache_type_v: string },
      // cells_used = sum of slot sequence lengths (idle slots keep their KV until parked or evicted)
      // cells_reserved = cells_used plus the unused part of the reservations of running requests (KV budget)
  prompt_cache: { enabled: boolean, bytes_used: number, bytes_limit: number /* 0 = unlimited */, tokens: number, tokens_limit: number, entries: number },
  vram: VramBudget | null
}
slots: [{ id, state: "idle"|"started"|"processing_prompt"|"done_prompt"|"generating"|"wait_other"|"preempted"|"recompute",
          is_processing: boolean, n_ctx: number, n_ctx_used: number, n_reserved: number, id_task: number | null,
          wait_reason: "none"|"slot"|"kv"|null, n_unparked_tokens: number,
          n_prompt_tokens: number, n_cached_tokens: number, n_prefill_tokens: number, n_decoded: number, speculative: boolean }]
events: { jsonl_enabled: boolean, jsonl_path: string | null, subscribers: number, records_emitted: number, records_dropped: number }
```

## /metrics additions (Prometheus, with --metrics and telemetry on)
```
llamacpp:board_energy_joules_total                      counter
llamacpp:energy_joules_total{phase="prefill|decode|idle"} counter
llamacpp:loop_seconds_total{phase="prefill|decode|host|idle"} counter
llamacpp:board_power_watts                              gauge (mean over last 1 s)
llamacpp:idle_power_watts                               gauge
llamacpp:kv_cells_total | kv_cells_used | kv_cells_reserved   gauge
llamacpp:sched_waiting | sched_preempted | sessions_parked    gauge
llamacpp:sched_{parks,park_drops,unparks,preempts,resumes,grows}_total              counter
llamacpp:sched_{park,unpark}_bytes_total, sched_{park,unpark}_seconds_total         counter
llamacpp:sched_wait_{slot,kv}_total                                                 counter
llamacpp:sched_{rollback_fallbacks,recompute_tokens,kv_overflows}_total             counter
```

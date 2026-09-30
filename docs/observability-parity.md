# Observability parity with tensorninja/ninfer-4090

lovelace's observability API is **tensorninja/ninfer-4090's API, as a superset**. Pinned upstream:
[tensorninja/ninfer-4090](https://github.com/tensorninja/ninfer-4090) at `f408990` ("feat(serve): stream tool calls
as the model writes them", 2026-09-23), request-log schema **20** (`artifact_type: "ninfer_serve_request_log"`).

- Every endpoint the tensorninja dashboard (`apps/web`) or the omarchy-ninfer widget reads is served at the same path
  with the same field names, JSON types and units.
- Fields with a meaningful lovelace equivalent carry it (mapping below). Fields with no equivalent are present with
  `0`, `false`, `""`, `"none"` or `null` (whatever keeps the upstream JSON type) and are listed as **N/A**.
- lovelace-only data lives under one namespaced key, `lovelace`, in every record and in `/telemetry`, versioned by
  `lovelace.schema_version` (**2**; version 1 was lovelace's previous standalone schema, which this replaces).
  Prometheus additions use the `lovelace:` prefix.
- The omarchy-ninfer widget additionally reads the `/telemetry` `cache` keys of
  [keylimesoda/ninfer-4090-lin-hybrid](https://github.com/keylimesoda/ninfer-4090-lin-hybrid) (the "hybrid" runtime,
  image `ninfer-4090:hybrid`); those keys are served too (they do not collide with tensorninja's).

Evidence: real payloads of tensorninja `f408990` (built from its Dockerfile's build stage, official artifact
`sha256 eec39564…bf3e`) in `bench/obs-fixtures/ninfer/`, captured by `bench/obs_capture_ninfer.sh`; lovelace's in
`bench/obs-fixtures/lovelace/`; `bench/obs_conformance.py` checks every ninfer field is present in lovelace's payload
with the same JSON type. Results: `bench/results/obs-parity/REPORT.md`.

Legend: **same** = same meaning; **mapped** = lovelace's nearest equivalent, semantics noted; **N/A** = no
equivalent, constant placeholder; **ext** = lovelace-only, under `lovelace`.

## Performance contract

Observability must cost nothing measurable on the decode/prefill path (CHARTER E1):

- the server loop only performs relaxed atomic stores and fixed-size field writes into the observer snapshot (one
  short mutex per step at most, no allocation, no JSON, no formatting, no syscalls);
- `/telemetry`, `/metrics`, `/slots` and `/events` are built on HTTP threads from that snapshot and never post a task
  to, wake, or wait for the server loop;
- request records are POD copies taken at request start/end, formatted on the observer's writer thread.

## Endpoints

| path | tensorninja | lovelace |
|---|---|---|
| `GET /health` | `{"status":"ok"}` | same (llama.cpp: 503 while loading) |
| `GET /v1/models` | `{object:"list", data:[{id, object, created, owned_by, context_window, modalities:{vision}}]}` | same fields; `owned_by` "llamacpp"; `context_window` = per-slot context; `vision` = mmproj loaded; llama.cpp's own extras (`aliases`, `tags`, `meta`, top-level `models`) kept |
| `GET /v1/models/{id}` | one model object | same |
| `GET /telemetry` | live snapshot | superset, below; requires `--telemetry` (or `--request-log-jsonl`) |
| `GET /events` | SSE: `retry: 2000`, then `event: <record.event>\ndata: <record>\n\n` frames; keepalive `: keepalive` | same framing; lovelace also sends `lovelace_sched` frames (ignored by ninfer consumers) |
| `GET /metrics` | Prometheus text: `llamacpp:*` + `ninfer:*` | llama.cpp's `llamacpp:*` + every `ninfer:*` name + `lovelace:*`; requires `--metrics` |
| `GET /slots` | array of `{id, is_processing, retained, session_digest, checkpoints:[{frontier, session_digest}], n_ctx, n_prompt_tokens, n_prompt_tokens_cache, speculative}` | the ninfer keys (`n_prompt_tokens` = cells the slot holds, `checkpoints` = context checkpoints, `frontier` = tokens covered) plus llama.cpp's `id_task`, `n_prompt_tokens_processed`, `params`, `next_token`, `prompt` for a slot that ran a task; llama.cpp's `generated` is not reported (built from the snapshot, off the loop); requires `--slots` |
| `POST /slots/{id}?action=save\|restore\|erase` | session persistence | llama.cpp's (same actions) |
| `--request-log-jsonl` | the `/events` records, one per line | same (byte-identical to the `data:` payloads) |

## Record envelope (every record)

| field | type | ninfer | lovelace |
|---|---|---|---|
| `artifact_type` | string | `"ninfer_serve_request_log"` | same |
| `schema_version` | number | 20 | 20 |
| `event` | string | `server_start` `request_start` `request_done` `request_error` `throughput` | same, plus `lovelace_sched` |
| `timestamp_unix_ms` | number (ms) | emit time | same |
| `server_instance_id` | string | `serve-<pid>-<us>` | 16 hex chars, new per process (same role) |
| `lovelace` | object | — | ext: `{schema_version: 2, seq, …}`; `seq` is monotonic per instance, 0 = `server_start` |

## `server_start`

| path | type | lovelace |
|---|---|---|
| `server.host`, `server.port` | string, number | same |
| `server.public_model_id` | string | mapped: `--alias` (else model file name) |
| `server.api_key_configured` | bool | same |
| `server.cors_enabled` | bool | mapped: `true` (llama-server always answers CORS) |
| `server.max_request_bytes` | number | N/A 0 |
| `server.request_log_jsonl` | string | same (`""` when off) |
| `server.slot_save_path` | string | same |
| `server.default_output_tokens` | number | mapped: `-n` (-1 = unlimited) |
| `server.default_thinking` | bool | mapped: reasoning enabled by default |
| `server.default_preserve_thinking` | bool | mapped: `--reasoning-preserve` |
| `artifact.path` | string | mapped: GGUF path |
| `artifact.size_bytes` | number\|null | mapped: GGUF file size |
| `artifact.target` | string | mapped: model architecture (`general.architecture`) |
| `artifact.weights_id` | string | mapped: model description (name, size, ftype) |
| `artifact.bytes_read`, `artifact.host_to_device_bytes` | number | mapped: model tensor bytes |
| `artifact.peak_staging_bytes`, `artifact.resource_count` | number | N/A 0 |
| `artifact.tensor_count` | number | mapped: GGUF tensor count |
| `artifact.load_seconds` | number (s) | mapped: model + context load wall time |
| `artifact.upload_seconds` | number (s) | N/A 0 (included in `load_seconds`) |
| `adapters.{count, names, rank, slots, device_bytes, file_bytes}` | | mapped: `--lora` adapters (count, file names); rank, slots, bytes N/A 0 |
| `engine.device` | number | same (0) |
| `engine.max_context` | number | mapped: per-slot context (`n_ctx / np` with `--kv-unified`: the per-agent limit) |
| `engine.kv_capacity_mode` | string | mapped: `"explicit"` |
| `engine.kv_capacity` | number (tokens) | mapped: KV cells of the pool (`-c`) |
| `engine.kv_capacity_page_groups`, `engine.kv_capacity_max_page_groups` | number | mapped: = `kv_capacity` (a lovelace "page" is one cell) |
| `engine.max_concurrency` | number | mapped: slots (`-np`) |
| `engine.max_pending_requests`, `engine.pending_timeout_ms` | number | N/A 0 (unbounded queue, no timeout) |
| `engine.prefill_chunk` | number | mapped: `-b` |
| `engine.log_stats_interval_ms` | number | mapped: throughput interval (5000) |
| `engine.kv_cache` | string | mapped: `-ctk` type (`"q5_0"`; `"k/v"` when they differ) |
| `engine.vision` | bool | mapped: mmproj loaded |
| `engine.cuda_graph` | bool | mapped: ggml CUDA graphs enabled (`GGML_CUDA_DISABLE_GRAPHS` unset) |
| `engine.prefix_reuse` | bool | mapped: prompt caching (on) |
| `engine.prefix_checkpoint_policy` | string | mapped: `"ctx-checkpoints:<ctxcp>"` |
| `engine.speculative_backend` | string | mapped: `"mtp"` (`draft-mtp`), `"draft"`, `"none"` |
| `engine.speculative_draft_window` | number | mapped: `--spec-draft-n-max` |
| `engine.proposal_head` | string | N/A `"full"` |
| `engine.continuation_cache.tiers` | string | mapped: session store: `"l1-l2-l3"` with `--session-dir`, `"l1-l2"` without, `"l1"` with no `--session-ram` (slot reuse only) |
| `engine.continuation_cache.policy` | string | mapped: `"session-store"` |
| `engine.continuation_cache.l1_capacity_mib` | number | mapped: KV pool size (VRAM holding resident sessions) |
| `engine.continuation_cache.l2_capacity_mib` | number | mapped: `--session-ram` |
| `engine.continuation_cache.l3_capacity_mib` | number | mapped: session disk limit (0 = unlimited) |
| `engine.continuation_cache.directory` | string | mapped: `--session-dir` |
| `engine.continuation_cache.namespace` | string | N/A `""` |
| `engine.continuation_cache.l{1,2,3}_idle_ttl_seconds` | number | mapped: session TTL for l2/l3; l1 N/A 0 |
| `engine.continuation_cache.persist_interval_seconds`, `persist_min_tokens`, `filesystem_reserve_mib` | number | N/A 0 |
| `engine.continuation_cache.prefix_checkpoint_history` | number | mapped: `-ctxcp` |
| `sampling_defaults.{thinking, non_thinking}` | `{temperature, top_p, top_k, min_p, presence_penalty, frequency_penalty}` | mapped: the server's sampling defaults (one preset, both entries) |
| `sampling_defaults.server_overrides` | object of nulls | N/A (all null) |
| `sampling_defaults.omitted_seed`, `sampling_defaults.greedy` | string, bool | `"random"`, false |
| `memory.weights`, `.sequence`, `.workspace`, `.request_transient` | `{capacity_bytes, used_bytes, peak_used_bytes}` | mapped from the ggml backend breakdown: weights = model buffers, sequence = KV + recurrent state (context buffers), workspace = compute buffers; request_transient N/A 0 |
| `memory.kv_payload_bytes` | number | mapped: context (KV + recurrent) bytes |
| `memory.available_after_weights_bytes`, `memory.available_after_startup_bytes` | number | mapped: device total − model; device free after load |
| `memory.*` (other reservation/graph/lora byte counts) | number | N/A 0 |
| `environment.{device, gpu_name, gpu_uuid, total_device_memory_bytes, compute_capability_major/minor, cuda_*_version}` | | mapped from NVML (compile version from the CUDA build) |
| `argv` | string[] | same (API key redacted) |
| `lovelace` | object | ext: `{schema_version, seq, build_info, model:{path, name, ftype, n_params, size_bytes, n_ctx_train}, config:{n_ctx, n_ctx_slot, n_parallel, n_batch, n_ubatch, kv_unified, cache_type_k, cache_type_v, flash_attn, cache_ram_mib, n_gpu_layers, speculative:{types, n_max, n_min}, throughput_interval_ms, energy_sample_hz, request_log_jsonl}, gpu: GpuStatic, vram: VramBudget\|null}` |

VramBudget: `{devices:[{name, description, total_bytes, free_bytes, model_bytes, context_bytes, compute_bytes,
unaccounted_bytes}], host:{model_bytes, context_bytes, compute_bytes}}`.

## `request_start` / `request_done` / `request_error`: `request` (RequestContext)

ninfer emits `request_start` when the request is submitted; lovelace when a slot launches it (after the queue wait,
which it reports).

| field | type | lovelace |
|---|---|---|
| `request_id` | number | mapped: task id |
| `x_request_id` | string | mapped: the task id as a string (lovelace does not issue an `x-request-id` header) |
| `protocol` | string | same names where the endpoint exists: `openai_chat_completions`, `openai_responses`, `anthropic_messages`; lovelace-only: `openai_completions`, `openai_embeddings`, `openai_transcriptions`, `native`, `infill`, `embeddings`, `rerank` |
| `model` | string | mapped: the served model id |
| `stream` | bool | same |
| `message_count`, `tool_count` | number | N/A 0 (the loop sees the rendered prompt only) |
| `requested_output_tokens` | number | same (`n_predict`, -1 = unlimited) |
| `requested_output_tokens_source` | string | `"client"` when the request set it, else `"server_default"` |
| `tool_choice` | string | N/A `"auto"` |
| `has_tool_history`, `preserve_thinking_semantic_change` | bool | N/A false |
| `enable_thinking`, `preserve_thinking` | bool | mapped: reasoning on for the request; `--reasoning-preserve` |
| `adapter` | string | N/A `""` |
| `prompt_cache_key_digest` | string | mapped: the lovelace session id as 16 hex chars (the agent session the prompt was routed to); `""` without a session store |
| `sampling` | `{temperature, top_p, top_k, min_p, presence_penalty, frequency_penalty, seed}` | same; `seed` is the requested seed (4294967295 = random), not the resolved one |

`request_start.lovelace`: `{schema_version, seq, id_task, id_slot, id_parent, task_type, endpoint, n_prompt_tokens,
n_predict, n_ctx_slot, queue_wait_ms, wait_reason: "none"|"slot"|"kv", n_unparked_tokens}`.

## `request_done`

| path | type | lovelace |
|---|---|---|
| `result.finish_reason` | string | mapped: eos → `stop_token`, stop word → `stop_string`, `n_predict` → `output_limit`, context full → `context_capacity`, cancelled → `cancelled`, embedding/rerank → `none` (a request that failed mid-flight is a `request_error`) |
| `result.prompt_tokens`, `result.completion_tokens` | number | same |
| `result.computed_prefill_tokens` | number | same (prompt tokens actually computed) |
| `result.prefix_cache_hit_tokens` | number | same (reused from slot KV, session restore or prompt cache) |
| `result.prefix_reuse_path` | string | mapped: `full_reset` (nothing reused), `append_frontier` (slot KV reused in place), `restore_turn_checkpoint` (session restored from RAM/NVMe or a context checkpoint) |
| `result.tool_call_count`, `tool_calls_incomplete`, `tool_calls_invalid_arguments`, `tool_call_discarded_bytes` | number | N/A 0 |
| `timings_seconds.{prepare, vision, publish}` | number (s) | N/A 0 |
| `timings_seconds.ttft` | number (s) | same (arrival → first token; 0 when no token) |
| `timings_seconds.queue` | number (s) | same (arrival → slot launch) |
| `timings_seconds.restore` | number (s) | mapped: session restore wall time after launch |
| `timings_seconds.prefill`, `.decode`, `.total` | number (s) | same |
| `speculative.backend`, `.draft_window` | string, number | mapped as `engine.speculative_*` |
| `speculative.rounds` | number | mapped: verification steps |
| `speculative.drafted_tokens`, `.accepted_tokens`, `.accepted_per_position` | number, number[] | same |
| `speculative.fallback_steps` | number | N/A 0 |
| `continuation_cache.source` | string | mapped: where the session was at admission: GPU-resident → `l1`, RAM mirror → `l2`, NVMe file → `l3`, new → `none`; without a session store `l1` when slot KV was reused |
| `continuation_cache.alias_kind` | string | mapped: `routed_session` for a known session, `none` for a new one |
| `continuation_cache.final_miss_reason` | string | mapped: `none` on reuse, `no_alias` for a new session, `not_attempted` without a session store and no reuse |
| `continuation_cache.restore_failure` | string | N/A `"none"` (restores do not fail; a lost sequence is re-prefilled, see `lovelace:sched_recompute_tokens_total`) |
| `continuation_cache.lookup_microseconds`, `preflight_microseconds` | number | N/A 0 |
| `continuation_cache.restore_microseconds` | number (µs) | mapped: device time of the restore copies |
| `continuation_cache.restored_tokens`, `restored_bytes` | number | mapped: tokens / bytes restored from host memory or disk |
| `continuation_cache.cache_write_tokens`, `restore_gate_checks`, `restore_gate_passes` | number | N/A 0 |
| `continuation_cache.restore_target_lane`, `admitted_lane` | number | mapped: slot id |
| `continuation_cache.destructive_rollback` | bool | N/A false |
| `continuation_cache.completion_publication_queued` | bool | mapped: the session mirror sync was queued at release |
| `continuation_cache.deepest_candidate_agreement` | number\|null | N/A null |
| `lovelace` | object | ext: `{schema_version, seq, id_task, id_slot, id_parent, task_type, endpoint, n_predict, n_ctx_slot, n_ctx_used, truncated, wait_reason, n_unparked_tokens, queue_wait_ms, session:{id, tier, restore_bytes, restore_ms, park_ms, restore_issue_ms, restore_ready_ms, prompt_start_ms, prefetch_hit}\|null, rates:{prefill_tps, decode_tps}, speculative:{n_verif_steps}, energy:{prefill_joules, decode_joules, total_joules}\|null}` |

## `request_error`

`request` (above), `error: {message}`. ninfer logs only generation errors; lovelace logs every request that did not
produce a result: HTTP parse/tokenize failures, rejection at launch or before any compute, and failure mid-flight.
`lovelace`: `{schema_version, seq, id_task|null, id_slot|null, task_type, endpoint, stage: "http"|"launch"|"prompt"|"generation",
error_type, n_prompt_tokens, queue_wait_ms|null}`.

## `throughput` (every 5 s with activity; idle time folds into the next `interval_seconds`)

| path | type | lovelace |
|---|---|---|
| `interval_seconds` | number (s) | same |
| `tokens.computed_prefill`, `tokens.committed_decode` | number | same |
| `throughput_tokens_per_second.prefill`, `.decode` | number | same |
| `energy` | object\|null | same shape; null without an NVML energy counter |
| `energy.board_joules` | number (J) | same (exact counter delta) |
| `energy.prefill_joules`, `decode_joules`, `idle_joules` | number (J) | mapped: board delta split by the server loop's phase time (10 Hz), host time pro rata; sums exactly to `board_joules` |
| `energy.idle_watts` | number (W) | same (calibrated from idle periods; 0 until calibrated) |
| `energy.accounted_seconds` | number (s) | mapped: loop seconds in prefill + decode + host while serving |
| `energy.residual_joules`, `residual_fraction` | number | mapped: 0 (the time split has no residual by construction) |
| `energy.joules_per_token.{served, active, prefill, decode}` | number\|null | same definitions (`active` = (board − idle) / tokens) |
| `scheduler.running` | number | mapped: busy slots |
| `scheduler.prefilling` | number | mapped: slots processing their prompt |
| `scheduler.decode_ready` | number | mapped: slots generating |
| `scheduler.waiting` | number | mapped: requests not yet on a slot (deferred + KV-budget admission queue) |
| `scheduler.admitted_requests`, `delta_admitted_requests` | number | mapped: requests launched on a slot (cumulative, interval) |
| `scheduler.queue_seconds_total`, `delta_queue_seconds` | number (s) | mapped: sum of arrival → launch waits |
| `scheduler.rejected_overloaded`, `rejected_queue_timeout` | number | N/A 0 (no queue bound or timeout) |
| `continuation_cache.lookup_hits` / `lookup_misses` (+ `delta_`) | number | mapped: requests that continued a known session / started a new one |
| `continuation_cache.preflight_candidate_rejections` (+ `delta_`) | number | N/A 0 |
| `continuation_cache.restore_successes` (+ `delta_`) | number | mapped: requests served from a session (l1 + l2 + l3) |
| `continuation_cache.restore_failures`, `restore_deferrals` (+ `delta_`) | number | N/A 0 |
| `continuation_cache.restored_tokens`, `restored_bytes` (+ `delta_`) | number | mapped: tokens reused / bytes restored, summed over those requests |
| `continuation_cache.publication_successes` (+ `delta_`) | number | mapped: session mirror syncs |
| `continuation_cache.publication_failures` (+ `delta_`) | number | mapped: syncs deferred for lack of arena room (stale) |
| `continuation_cache.publication_superseded` (+ `delta_`) | number | N/A 0 |
| `continuation_cache.tiers.{l1,l2,l3}_restore_successes`, `_restored_tokens`, `_restored_bytes` (+ `delta_`) | number | mapped: per `source` above |
| `continuation_cache.aliases.routed_session_restores` (+ `delta_`) | number | mapped: = restore_successes |
| `continuation_cache.aliases.stable_prefix_restores` (+ `delta_`) | number | N/A 0 |
| `continuation_cache.miss_reasons.no_alias` (+ `delta_`) | number | mapped: new sessions |
| `continuation_cache.miss_reasons.*` (others) | number | N/A 0 |
| `continuation_cache.latency_microseconds.l3_restore_total` (+ `_delta`, `_operations_*`) | number | mapped: restore device time of requests whose session came from NVMe (the file read is in `lovelace.sessions.io_read_s`) |
| `continuation_cache.latency_microseconds.l2_restore_total` (+ …) | number | mapped: restore device time of requests whose session came from RAM |
| `continuation_cache.latency_microseconds.l3_persistence_total` (+ …) | number | mapped: NVMe write time / demotions |
| `continuation_cache.latency_microseconds.*` (lookup, preflight, admission, export) | number | N/A 0 |
| `continuation_cache.occupancy.l1_entries`, `l1_bytes` | number | mapped: GPU-resident sessions, their KV cells × bytes per cell |
| `continuation_cache.occupancy.l2_entries`, `l2_bytes` | number | mapped: sessions with a RAM mirror, arena bytes |
| `continuation_cache.occupancy.l3_entries`, `l3_bytes` | number | mapped: sessions only on NVMe, file bytes |
| `continuation_cache.occupancy.l1_evictions`, `l1_demotions` (+ `delta_`) | number | mapped: parks (sessions leaving VRAM) / parks that kept a mirror |
| `continuation_cache.occupancy.l2_evictions`, `l2_evicted_bytes` (+ `delta_`) | number | mapped: demotions RAM → NVMe, bytes written |
| `continuation_cache.occupancy.l3_evictions`, `l3_evicted_bytes` (+ `delta_`) | number | N/A 0 (lovelace deletes only by TTL, which ninfer does not count either) |
| `continuation_cache.occupancy.kv_growth_attempts` | number | mapped: in-flight KV reservation grows |
| `continuation_cache.occupancy.kv_growth_forced_spills` | number | mapped: preemptions |
| `continuation_cache.occupancy.kv_growth_curtailed`, `kv_restore_reclaimed_lanes` | number | N/A 0 |
| `continuation_cache.persistence_total`, `persistence_delta` `{queued, coalesced, successes, failures}` | number | mapped: demotion writes queued / completed; coalesced, failures N/A 0 |
| `decode_batch.rounds`, `row_rounds`, `average_size` | number, number, number\|null | mapped: `llama_decode` calls that generated, generating slots summed over them, their ratio |
| `lovelace` | object | ext: `{schema_version, seq, tokens:{drafted, accepted}, phase_tokens_per_second:{prefill, decode}, requests:{started, done, rejected}, scheduler:{slots_total, slots_busy, slots_processing_prompt, queue_deferred, queue_waiting, slots_preempted, waited_slot, waited_kv}, kv:{cells_total, cells_used, cells_reserved}, parking:{sessions_parked, parks, unparks, preempts, resumes, park_bytes, unpark_bytes, park_seconds, unpark_seconds, recompute_tokens}, loop_seconds:{prefill, decode, host, idle}, batch:{decode_calls, mean_busy_slots}, sessions:{n_gpu, n_ram, n_disk, ram_bytes, disk_bytes, promotions, demotions, prefetches, prefetch_hits, io_bytes_read, io_bytes_written}\|null}` (all `lovelace` counters are interval deltas except the gauges `slots_*`, `queue_*`, `kv`, `sessions_parked`, `n_*`, `*_bytes` of sessions) |

## `lovelace_sched` (lovelace-only record, one per state transfer between VRAM and host memory)

Envelope plus `lovelace: {schema_version, seq, action: "park"|"unpark"|"preempt"|"resume"|"sync", id_slot, id_task|null,
n_tokens, bytes, ms, host_ms, gbps}` (`sync`: a session mirror update after a request).

## `GET /telemetry`

| path | type | lovelace |
|---|---|---|
| `timestamp_unix_ms`, `server_instance_id`, `uptime_seconds`, `model_id` | | same |
| `attached` | bool | mapped: model loaded (always true once the endpoint answers) |
| `gpu.*` | | same NVML fields: `available, error?, name, uuid, driver_version, temperature_c, fan_percent, power_watts, power_limit_watts, energy_available, energy_joules_total, server_energy_joules, idle_watts, utilization_gpu_percent, utilization_memory_percent, sm_clock_mhz, sm_clock_max_mhz, memory_clock_mhz, memory_used_bytes, memory_total_bytes, throttle_reasons`; `pcie_rx_bytes_per_second`, `pcie_tx_bytes_per_second` N/A 0 (NVML's PCIe throughput query stalls ~20 ms and slowed decode; never called) |
| `scheduler.running`, `prefilling`, `decode_ready`, `waiting`, `admitted_requests`, `queue_seconds_total`, `rejected_*` | | as in `throughput.scheduler` |
| `scheduler.max_concurrency`, `max_pending_requests` | number | as in `server_start.engine` |
| `scheduler.decode_rounds`, `decode_row_rounds` | number | as in `throughput.decode_batch` (cumulative) |
| `scheduler.decode_rounds_abandoned` | number | N/A 0 |
| `scheduler.worker_seconds.decode`, `.prefill` | number (s) | mapped: server loop seconds in `llama_decode` for generation / prompt rows |
| `scheduler.worker_seconds.admission` | number (s) | mapped: loop seconds issuing and waiting for parks / restores |
| `scheduler.worker_seconds.publish` | number (s) | N/A 0 (session syncs are asynchronous copies) |
| `scheduler.worker_seconds.upkeep` | number (s) | mapped: remaining host time of the loop (batching, sampling, detokenizing) |
| `scheduler.worker_admission.calls` | number | mapped: slot launches |
| `scheduler.worker_admission.restore_seconds` | number (s) | mapped: = `worker_seconds.admission` |
| `scheduler.worker_admission.plan_seconds`, `commit_seconds` | number (s) | N/A 0 |
| `scheduler.worker_decode_rounds`, `worker_prefill_steps` | number | mapped: `llama_decode` calls with generation / prompt rows |
| `cache.l1`, `cache.l2`, `cache.l3` `{entries, bytes, capacity_bytes, evictions, demotions / evicted_bytes}` | | as `throughput.continuation_cache.occupancy` + capacities of `server_start` |
| `cache.kv_growth.{attempts, forced_spills, curtailed}`, `cache.kv_restore_reclaimed_lanes` | | as in `throughput` |
| `cache.restore_successes`, `restore_failures`, `restore_deferrals`, `lookup_hits`, `lookup_misses`, `publication_successes`, `publication_failures`, `persistence_successes`, `persistence_failures` | | as in `throughput` (cumulative) |
| `cache.kv_capacity`, `kv_capacity_page_groups`, `kv_capacity_max_page_groups` | number | hybrid keys (widget): KV cells of the pool (page = cell) |
| `cache.device_main_kv_occupied_pages` | number | hybrid (widget): KV cells held by all slots |
| `cache.device_backend_kv_occupied_pages` | number | hybrid: N/A 0 |
| `cache.device_state_occupied_slots` | number | hybrid (widget): GPU-resident sessions (slots holding a sequence without a session store) |
| `cache.host_state_occupied_slots`, `host_state_capacity_slots` | number | hybrid (widget): sessions with a RAM mirror; capacity N/A 0 |
| `cache.host_kv_occupied_bytes`, `host_kv_capacity_bytes` | number | hybrid (widget): session arena bytes used / `--session-ram` |
| `cache.pressure.{checkpoints_dropped, private_owners_degraded, shared_owners_degraded, private_owners_evicted, shared_owners_evicted, spill_pages, …}` | number | hybrid (widget "Pressure"): `private_owners_evicted` = sessions evicted without a mirror (`park_drops`); `spill_pages` = in-flight preemptions to host (`slots_preempted`); others N/A 0 |
| `slots[]` `{processing, retained, prompt_tokens, cached_tokens, session_digest, checkpoints}` | | mapped per slot: busy; idle holding a sequence; prompt tokens; reused tokens; session id (16 hex, `""` = none); context checkpoints |
| `memory.device`, `max_context`, `kv_capacity`, `kv_capacity_page_groups`, `kv_capacity_max_page_groups` | | as `server_start.engine` |
| `memory.weights`, `sequence`, `workspace`, `request_transient` | arena | as `server_start.memory` |
| `memory.kv_payload_bytes`, `text_kv_bytes` | number | mapped: context bytes of the target model |
| `memory.mtp_kv_bytes` | number | mapped: context bytes of the draft (MTP) context |
| `memory.gdn_state_bytes`, `dflash_kv_bytes`, `replay_records_bytes`, `cuda_graph_*`, `lora_bank_bytes` | number | N/A 0 |
| `memory.available_after_startup_bytes`, `available_after_weights_bytes` | number | as `server_start.memory` |
| `adapters.{count, names, model_ids, rank, slots, device_bytes, file_bytes}` | | as `server_start.adapters` (`model_ids` = names) |
| `events.jsonl_enabled`, `events.subscribers` | bool, number | same |
| `lovelace` | object | ext, below |

`/telemetry` `lovelace` (ext): `{schema_version, build_info, sleeping, energy:{available, joules:{board, prefill,
decode, idle}|null, idle_watts|null, board_watts_1s|null}, gpu:{pstate|null}, loop:{window_seconds, phase, seconds:{prefill,
decode, host, idle}, fraction:{…}}, loop_totals_seconds:{…}, scheduler:{slots_total, slots_busy, slots_idle,
slots_processing_prompt, slots_generating, queue_deferred, kv_budget:{enabled, queue_waiting, slots_preempted,
gen_reserve, headroom, prefill_chunk_gen, timeslice, totals:{parks, park_drops, unparks, preempts, resumes, grows,
park_bytes, unpark_bytes, park_seconds, unpark_seconds, park_host_seconds, unpark_host_seconds, park_gbps, unpark_gbps,
waited_slot, waited_kv, rollback_fallbacks, recompute_tokens, kv_overflows}}, totals:{prompt_tokens,
prompt_tokens_cached, predicted_tokens, prompt_seconds, predicted_seconds, n_decode, n_busy_slots_per_decode, drafted,
accepted, requests_started, requests_done, requests_rejected}}, kv:{cells_total, cells_used, cells_reserved, unified,
n_ctx_slot, cache_type_k, cache_type_v}, prompt_cache:{enabled, bytes_used, bytes_limit, tokens, tokens_limit,
entries}, sessions:{arena_bytes, arena_free_bytes, ram_bytes, disk_bytes, disk_limit_bytes, n_sessions, n_gpu, n_ram,
n_disk, n_io, demotions, promotions, deleted, stale_syncs, prefetches, prefetch_hits, syncs, sync_bytes,
io_bytes_written, io_bytes_read, io_write_s, io_read_s, sessions:[{id, tier, slot, resident_cells, mirror_tokens,
mirror_bytes, mirror_segments, checkpoints, file_bytes, stale, sync_ms|null, since_sync_s|null, syncs, truncations,
restores, last_restore_bytes, last_restore_ms, last_restore_wall_ms, prefetches, prefetch_hits, demotions,
promotions}]}|null, vram: VramBudget|null, slots:[{id, state, is_processing, n_ctx, n_ctx_used, n_reserved, id_task|null,
wait_reason|null, n_unparked_tokens, n_prompt_tokens, n_cached_tokens, n_prefill_tokens, n_decoded, speculative,
session_id|null}], events:{jsonl_path|null, records_emitted, records_dropped}}`.

These are lovelace's previous (schema 1) `/telemetry` sections, moved under `lovelace` unchanged.

## `GET /metrics`

llama.cpp's own series are unchanged (`llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`,
`llamacpp:prompt_seconds_total`, `llamacpp:tokens_predicted_seconds_total`, `llamacpp:requests_processing`,
`llamacpp:requests_deferred`, …). Note: ninfer's `llamacpp:prompt_tokens_total` counts computed prefill tokens, as
llama.cpp's does. Added, with ninfer's names and the mappings above:

- `ninfer:requests_total` (completed requests), `ninfer:prefix_cache_hit_tokens_total`, `ninfer:draft_tokens_total`,
  `ninfer:draft_accepted_tokens_total` — the four the widget reads;
- every other `ninfer:*` series of `serve_metrics.cpp` (continuation, admission, worker, energy), N/A ones at 0;
- `ninfer:board_energy_joules_total`, `ninfer:prefill_energy_joules_total`, `ninfer:decode_energy_joules_total`,
  `ninfer:energy_accounted_seconds_total`, `ninfer:energy_samples_total`, `ninfer:board_idle_watts` only with an
  energy counter, as upstream;
- lovelace-only: `lovelace:loop_seconds_total{phase}`, `lovelace:energy_joules_total{phase}`,
  `lovelace:board_power_watts`, `lovelace:kv_cells_{total,used,reserved}`, `lovelace:sched_*`,
  `lovelace:sessions_*` (the previous `llamacpp:` additions of lovelace schema 1, renamed to their own prefix).

## Not applicable, by design

- **Continuation cache internals** (lookup/preflight latencies, stable-prefix aliases, restore gates, publication
  supersession, persistence coalescing): lovelace's session store identifies a prompt's session directly and keeps
  every session (never drops), so these steps do not exist; the counters are 0.
- **Anchors / checkpoints ring** (`deepest_candidate_agreement`, `destructive_rollback`): no equivalent; null / false.
- **Adapter bank**: lovelace serves `--lora` adapters through llama.cpp, without a resident bank; bank sizes are 0.
- **Arena budget details** (reservations, CUDA graph allowances, replay records, DFlash KV): ggml allocates per
  backend buffer; only weights / KV+state / compute are known.
- **Tool-call quality counters and message/tool counts**: the server loop sees the rendered prompt, not the request
  structure; 0.
- **Queue bounds** (`max_pending_requests`, `pending_timeout_ms`, overload / timeout rejections): lovelace's queue is
  unbounded and never times out; 0.

// lovelace's extension of the ninfer observability API: everything under the `lovelace` key of a
// record or of GET /telemetry (lovelace.schema_version 2). The contract is
// docs/observability-parity.md; the sections are lovelace's previous standalone schema, moved.
//
// Every use is optional: a pure ninfer server or log carries no `lovelace` key, and the panels that
// read these say so instead of rendering zeros.

import { percentile } from './derive'
import { seconds } from './format'
import type { RequestDoneRecord, ThroughputRecord } from './records'

export type LovelaceWaitReason = 'none' | 'slot' | 'kv'

export interface LovelaceLoopSeconds {
  prefill: number
  decode: number
  host: number
  idle: number
}

export type LovelaceLoopPhase = keyof LovelaceLoopSeconds

// --- records -------------------------------------------------------------------------------------

/** Per-device ggml backend buffers, captured once after model load. */
export interface VramDevice {
  name: string
  description: string
  total_bytes: number
  free_bytes: number
  model_bytes: number
  context_bytes: number
  compute_bytes: number
  /** total - free - (model + context + compute): other processes, CUDA context, fragmentation. */
  unaccounted_bytes: number
}

export interface VramBudget {
  devices: VramDevice[]
  host: { model_bytes: number; context_bytes: number; compute_bytes: number }
}

/** The static subset of the /telemetry `gpu` block. */
export interface LovelaceGpuStatic {
  available: boolean
  error?: string
  name?: string
  uuid?: string
  driver_version?: string
  memory_total_bytes?: number
  sm_clock_max_mhz?: number
  power_limit_watts?: number
  energy_available?: boolean
}

interface LovelaceRecordExt {
  schema_version: number
  /** Monotonic per server instance; 0 is `server_start`. */
  seq: number
}

export interface LovelaceServerStart extends LovelaceRecordExt {
  build_info: string
  model: {
    path: string
    name: string
    ftype: string
    n_params: number
    size_bytes: number
    n_ctx_train: number
  }
  config: {
    n_ctx: number
    n_ctx_slot: number
    n_parallel: number
    n_batch: number
    n_ubatch: number
    kv_unified: boolean
    cache_type_k: string
    cache_type_v: string
    flash_attn: string
    /** --cache-ram in MiB: -1 unlimited, 0 disabled. */
    cache_ram_mib: number
    n_gpu_layers: number
    speculative: { types: string; n_max: number; n_min: number }
    throughput_interval_ms: number
    energy_sample_hz: number
    request_log_jsonl: string | null
  }
  gpu: LovelaceGpuStatic
  vram: VramBudget | null
}

export interface LovelaceRequestStart extends LovelaceRecordExt {
  id_task: number
  id_slot: number
  id_parent: number | null
  task_type: string
  endpoint: string
  n_prompt_tokens: number
  n_predict: number
  n_ctx_slot: number
  queue_wait_ms: number
  wait_reason: LovelaceWaitReason
  n_unparked_tokens: number
}

/** Where the request's agent session was at admission. */
export type LovelaceSessionOrigin = 'gpu' | 'ram' | 'disk' | 'new'

export interface LovelaceRequestDone extends LovelaceRecordExt {
  id_task: number
  id_slot: number
  id_parent: number | null
  task_type: string
  endpoint: string
  n_predict: number
  n_ctx_slot: number
  n_ctx_used: number
  truncated: boolean
  /** First reason the request waited for admission under the KV budget. */
  wait_reason: LovelaceWaitReason
  n_unparked_tokens: number
  queue_wait_ms: number
  session: {
    id: number | string
    tier: LovelaceSessionOrigin
    restore_bytes: number
    restore_ms: number
    park_ms: number
    restore_issue_ms: number
    restore_ready_ms: number | null
    prompt_start_ms: number | null
    prefetch_hit: boolean
  } | null
  rates: { prefill_tps: number | null; decode_tps: number | null }
  speculative: { n_verif_steps: number }
  energy: { prefill_joules: number; decode_joules: number; total_joules: number } | null
}

export interface LovelaceRequestError extends LovelaceRecordExt {
  id_task: number | null
  id_slot: number | null
  task_type: string
  endpoint: string
  stage: 'http' | 'launch' | 'prompt' | 'generation'
  error_type: string
  n_prompt_tokens: number
  queue_wait_ms: number | null
}

/** Session store counters of one interval. `n_*` and `*_bytes` are gauges, the rest deltas. */
export interface LovelaceIntervalSessions {
  n_gpu: number
  n_ram: number
  n_disk: number
  ram_bytes: number
  disk_bytes: number
  promotions: number
  demotions: number
  prefetches: number
  prefetch_hits: number
  io_bytes_read: number
  io_bytes_written: number
}

/**
 * Every counter is an interval delta except the gauges `slots_*`, `queue_*`, `kv`,
 * `sessions_parked` and the session `n_*` / `*_bytes`, which are snapshots at emit time.
 */
export interface LovelaceThroughput extends LovelaceRecordExt {
  tokens: { drafted: number; accepted: number }
  phase_tokens_per_second: { prefill: number | null; decode: number | null }
  requests: { started: number; done: number; rejected: number }
  scheduler: {
    slots_total: number
    slots_busy: number
    slots_processing_prompt: number
    queue_deferred: number
    queue_waiting: number
    slots_preempted: number
    waited_slot: number
    waited_kv: number
  }
  kv: { cells_total: number; cells_used: number; cells_reserved: number }
  parking: {
    sessions_parked: number
    parks: number
    unparks: number
    preempts: number
    resumes: number
    park_bytes: number
    unpark_bytes: number
    park_seconds: number
    unpark_seconds: number
    recompute_tokens: number
  }
  loop_seconds: LovelaceLoopSeconds
  batch: { decode_calls: number; mean_busy_slots: number | null }
  /** Null without a session store. */
  sessions: LovelaceIntervalSessions | null
}

// --- GET /telemetry ------------------------------------------------------------------------------

export interface LovelaceKvBudgetTotals {
  parks: number
  park_drops: number
  unparks: number
  preempts: number
  resumes: number
  grows: number
  park_bytes: number
  unpark_bytes: number
  park_seconds: number
  unpark_seconds: number
  park_host_seconds: number
  unpark_host_seconds: number
  park_gbps: number
  unpark_gbps: number
  waited_slot: number
  waited_kv: number
  rollback_fallbacks: number
  recompute_tokens: number
  kv_overflows: number
}

export interface LovelaceKvBudget {
  enabled: boolean
  queue_waiting: number
  slots_preempted: number
  gen_reserve: number
  headroom: number
  prefill_chunk_gen: number
  timeslice: number
  totals: LovelaceKvBudgetTotals
}

export interface LovelaceKv {
  cells_total: number
  cells_used: number
  cells_reserved: number
  unified: boolean
  n_ctx_slot: number
  cache_type_k: string
  cache_type_v: string
}

export interface LovelacePromptCache {
  enabled: boolean
  bytes_used: number
  /** 0 = unlimited. */
  bytes_limit: number
  tokens: number
  tokens_limit: number
  entries: number
}

/** Live tier of a session: resident in a slot, mirrored in RAM, on NVMe, or moving between. */
export type LovelaceSessionTier = 'gpu' | 'ram' | 'demoting' | 'disk' | 'promoting'

export interface LovelaceSession {
  id: number | string
  tier: LovelaceSessionTier
  slot: number
  resident_cells: number
  mirror_tokens: number
  mirror_bytes: number
  mirror_segments: number
  checkpoints: number
  file_bytes: number
  stale: boolean
  sync_ms: number | null
  since_sync_s: number | null
  syncs: number
  truncations: number
  restores: number
  last_restore_bytes: number
  last_restore_ms: number
  last_restore_wall_ms: number
  prefetches: number
  prefetch_hits: number
  demotions: number
  promotions: number
}

export interface LovelaceSessions {
  arena_bytes: number
  arena_free_bytes: number
  ram_bytes: number
  disk_bytes: number
  /** 0 = unlimited. */
  disk_limit_bytes: number
  n_sessions: number
  n_gpu: number
  n_ram: number
  n_disk: number
  n_io: number
  demotions: number
  promotions: number
  deleted: number
  stale_syncs: number
  prefetches: number
  prefetch_hits: number
  syncs: number
  sync_bytes: number
  io_bytes_written: number
  io_bytes_read: number
  io_write_s: number
  io_read_s: number
  sessions: LovelaceSession[]
}

export interface LovelaceSlot {
  id: number
  state: string
  is_processing: boolean
  n_ctx: number
  n_ctx_used: number
  n_reserved: number
  id_task: number | null
  wait_reason: LovelaceWaitReason | null
  n_unparked_tokens: number
  n_prompt_tokens: number
  n_cached_tokens: number
  n_prefill_tokens: number
  n_decoded: number
  speculative: boolean
  session_id: number | string | null
}

export interface LovelaceTelemetry {
  schema_version: number
  build_info: string
  sleeping: boolean
  energy: {
    available: boolean
    joules: { board: number; prefill: number; decode: number; idle: number } | null
    idle_watts: number | null
    board_watts_1s: number | null
  }
  gpu: { pstate: number | null }
  loop: {
    window_seconds: number
    phase: LovelaceLoopPhase
    seconds: LovelaceLoopSeconds
    fraction: LovelaceLoopSeconds
  }
  loop_totals_seconds: LovelaceLoopSeconds
  scheduler: {
    slots_total: number
    slots_busy: number
    slots_idle: number
    slots_processing_prompt: number
    slots_generating: number
    queue_deferred: number
    kv_budget: LovelaceKvBudget
    totals: {
      prompt_tokens: number
      prompt_tokens_cached: number
      predicted_tokens: number
      prompt_seconds: number
      predicted_seconds: number
      n_decode: number
      n_busy_slots_per_decode: number
      drafted: number
      accepted: number
      requests_started: number
      requests_done: number
      requests_rejected: number
    }
  }
  kv: LovelaceKv
  prompt_cache: LovelacePromptCache
  /** Null without a session store. */
  sessions: LovelaceSessions | null
  vram: VramBudget | null
  slots: LovelaceSlot[]
  events: { jsonl_path: string | null; records_emitted: number; records_dropped: number }
}

/**
 * A duration given in ms, kept in ms below a second. lovelace reports queue waits, restores and
 * transfer times in ms at µs precision, and format.seconds would round 1.5 ms to "2ms".
 */
export function milliseconds(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 ms'
  if (value >= 1000) return seconds(value / 1000)
  return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ms`
}

// --- derivations ---------------------------------------------------------------------------------

/**
 * Session store activity summed over the interval deltas of the retained throughput records, with
 * the gauges of the latest one. Null when no record carries a session block: a ninfer log, or a
 * lovelace server without a session store.
 */
export interface SessionHistory {
  intervals: number
  latest: LovelaceIntervalSessions
  promotions: number
  demotions: number
  prefetches: number
  prefetchHits: number
  ioBytesRead: number
  ioBytesWritten: number
}

export function summarizeSessionHistory(
  records: readonly ThroughputRecord[],
): SessionHistory | null {
  let summary: SessionHistory | null = null
  for (const record of records) {
    const sessions = record.lovelace?.sessions
    if (!sessions) continue
    summary ??= {
      intervals: 0,
      latest: sessions,
      promotions: 0,
      demotions: 0,
      prefetches: 0,
      prefetchHits: 0,
      ioBytesRead: 0,
      ioBytesWritten: 0,
    }
    summary.intervals += 1
    summary.latest = sessions
    summary.promotions += sessions.promotions
    summary.demotions += sessions.demotions
    summary.prefetches += sessions.prefetches
    summary.prefetchHits += sessions.prefetch_hits
    summary.ioBytesRead += sessions.io_bytes_read
    summary.ioBytesWritten += sessions.io_bytes_written
  }
  return summary
}

/**
 * KV-budget transfers and admission waits, summed over the interval deltas of the retained
 * throughput records. This is what replay has in place of the live `kv_budget.totals`, which are
 * cumulative since start. GB/s are null when no transfer time was spent.
 */
export interface ParkingSummary {
  parks: number
  unparks: number
  preempts: number
  resumes: number
  parkBytes: number
  unparkBytes: number
  parkSeconds: number
  unparkSeconds: number
  parkGbps: number | null
  unparkGbps: number | null
  recomputeTokens: number
  waitedSlot: number
  waitedKv: number
}

export function summarizeParking(records: readonly ThroughputRecord[]): ParkingSummary | null {
  let seen = false
  const out: ParkingSummary = {
    parks: 0,
    unparks: 0,
    preempts: 0,
    resumes: 0,
    parkBytes: 0,
    unparkBytes: 0,
    parkSeconds: 0,
    unparkSeconds: 0,
    parkGbps: null,
    unparkGbps: null,
    recomputeTokens: 0,
    waitedSlot: 0,
    waitedKv: 0,
  }
  for (const record of records) {
    const ext = record.lovelace
    if (!ext) continue
    seen = true
    const parking = ext.parking
    out.parks += parking.parks
    out.unparks += parking.unparks
    out.preempts += parking.preempts
    out.resumes += parking.resumes
    out.parkBytes += parking.park_bytes
    out.unparkBytes += parking.unpark_bytes
    out.parkSeconds += parking.park_seconds
    out.unparkSeconds += parking.unpark_seconds
    out.recomputeTokens += parking.recompute_tokens
    out.waitedSlot += ext.scheduler.waited_slot
    out.waitedKv += ext.scheduler.waited_kv
  }
  if (!seen) return null
  out.parkGbps = out.parkSeconds > 0 ? out.parkBytes / out.parkSeconds / 1e9 : null
  out.unparkGbps = out.unparkSeconds > 0 ? out.unparkBytes / out.unparkSeconds / 1e9 : null
  return out
}

/** Server loop seconds summed over the retained intervals; null without any lovelace record. */
export function loopSecondsFromRecords(
  records: readonly ThroughputRecord[],
): LovelaceLoopSeconds | null {
  let out: LovelaceLoopSeconds | null = null
  for (const record of records) {
    const loop = record.lovelace?.loop_seconds
    if (!loop) continue
    out ??= { prefill: 0, decode: 0, host: 0, idle: 0 }
    out.prefill += loop.prefill
    out.decode += loop.decode
    out.host += loop.host
    out.idle += loop.idle
  }
  return out
}

export const WAIT_REASONS = ['none', 'slot', 'kv'] as const satisfies readonly LovelaceWaitReason[]

export interface WaitGroup {
  count: number
  /** Queue wait in ms, truncating nearest-rank like every other percentile here. */
  p50: number
  p90: number
  max: number
}

/**
 * Completed requests partitioned by why they waited for admission, with their queue waits.
 * Null when no retained request carries the lovelace block.
 */
export interface WaitSummary {
  reported: number
  byReason: Record<LovelaceWaitReason, WaitGroup>
}

export function summarizeWaits(requests: readonly RequestDoneRecord[]): WaitSummary | null {
  const waits: Record<LovelaceWaitReason, number[]> = { none: [], slot: [], kv: [] }
  let reported = 0
  for (const request of requests) {
    const ext = request.lovelace
    if (!ext) continue
    reported += 1
    waits[ext.wait_reason].push(ext.queue_wait_ms)
  }
  if (reported === 0) return null
  const group = (values: number[]): WaitGroup => ({
    count: values.length,
    p50: percentile(values, 0.5),
    p90: percentile(values, 0.9),
    max: values.length === 0 ? 0 : Math.max(...values),
  })
  return {
    reported,
    byReason: { none: group(waits.none), slot: group(waits.slot), kv: group(waits.kv) },
  }
}

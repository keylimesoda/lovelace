// GET /telemetry payload: a complete snapshot rather than a delta, which is what lets the
// dashboard resynchronize after dropped stream records without replaying anything.

import type { LoopSeconds, ServerStartRecord, VramBudget } from './records'

export interface GpuTelemetry {
  available: boolean
  error?: string
  name?: string
  uuid?: string
  driver_version?: string
  temperature_c?: number
  fan_percent?: number
  /** NVML trailing-average board power. */
  power_watts?: number
  power_limit_watts?: number
  /** False on boards with no cumulative energy counter. */
  energy_available?: boolean
  /** Board counter since driver load; only meaningful as a difference. */
  energy_joules_total?: number | null
  utilization_gpu_percent?: number
  utilization_memory_percent?: number
  sm_clock_mhz?: number
  sm_clock_max_mhz?: number
  memory_clock_mhz?: number
  memory_used_bytes?: number
  memory_total_bytes?: number
  pstate?: number
  throttle_reasons?: string[]
}

// Boards report throttle reasons that are not faults. `gpu_idle` means there is nothing to run,
// and `applications_clocks_setting` is a deliberate cap, so neither indicates a limit on work the
// server wanted to do.
const BENIGN_THROTTLES: Record<string, true> = {
  gpu_idle: true,
  applications_clocks_setting: true,
}

/** Throttle reasons that actually constrain achievable throughput. */
export function activeThrottles(gpu: GpuTelemetry | undefined): string[] {
  return (gpu?.throttle_reasons ?? []).filter((reason) => !BENIGN_THROTTLES[reason])
}

export interface EnergyTelemetry {
  available: boolean
  /** Since server start, split by loop phase time share; sums to `board`. */
  joules: { board: number; prefill: number; decode: number; idle: number } | null
  idle_watts: number | null
  /** Mean board power over the last second, from counter deltas. */
  board_watts_1s: number | null
}

export type LoopPhase = 'idle' | 'host' | 'prefill' | 'decode'

export interface SchedulerTelemetry {
  slots_total: number
  slots_busy: number
  slots_idle: number
  slots_processing_prompt: number
  slots_generating: number
  queue_deferred: number
  loop: {
    /** Rolling window length, 10 s. */
    window_seconds: number
    /** Phase the server loop is in right now. */
    phase: LoopPhase
    seconds: LoopSeconds
    /** seconds / window_seconds. */
    fraction: LoopSeconds
  }
  loop_totals_seconds: LoopSeconds
  /** Cumulative since start; llama.cpp's own server metrics. */
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

export interface KvTelemetry {
  cells_total: number
  /** Sum of slot sequence lengths; idle slots keep their KV until evicted. */
  cells_used: number
  unified: boolean
  n_ctx_slot: number
  cache_type_k: string
  cache_type_v: string
}

export interface PromptCacheTelemetry {
  enabled: boolean
  bytes_used: number
  /** 0 means unlimited. */
  bytes_limit: number
  tokens: number
  tokens_limit: number
  entries: number
}

export interface MemoryTelemetry {
  kv: KvTelemetry
  prompt_cache: PromptCacheTelemetry
  vram: VramBudget | null
}

export type SlotState =
  'idle' | 'started' | 'processing_prompt' | 'done_prompt' | 'generating' | 'wait_other'

export interface SlotTelemetry {
  id: number
  state: SlotState
  is_processing: boolean
  n_ctx: number
  n_ctx_used: number
  id_task: number | null
  n_prompt_tokens: number
  n_cached_tokens: number
  n_prefill_tokens: number
  n_decoded: number
  speculative: boolean
}

export interface Telemetry {
  timestamp_unix_ms: number
  server_instance_id: string
  uptime_seconds: number
  model_id: string
  build_info: string
  gpu: GpuTelemetry
  energy: EnergyTelemetry
  scheduler: SchedulerTelemetry
  memory: MemoryTelemetry
  slots: SlotTelemetry[]
  events: {
    jsonl_enabled: boolean
    jsonl_path: string | null
    subscribers: number
    records_emitted: number
    records_dropped: number
  }
}

/**
 * What the Cache panel reads. Live, it is the `/telemetry` memory snapshot. In replay only the
 * configuration survives - occupancy is a level the log never records - so the KV capacity comes
 * from `server_start.config` and the occupancy fields read as null rather than as zero.
 */
export interface CacheView {
  kv: Omit<KvTelemetry, 'cells_used'> & { cells_used: number | null }
  promptCache: PromptCacheTelemetry | null
  /** --cache-ram as configured: -1 unlimited, 0 disabled. Null when no server_start was seen. */
  cacheRamMib: number | null
}

export function cacheView(
  telemetry: Telemetry | null,
  start: ServerStartRecord | null,
): CacheView | undefined {
  if (telemetry !== null) {
    return {
      kv: telemetry.memory.kv,
      promptCache: telemetry.memory.prompt_cache,
      cacheRamMib: start?.config.cache_ram_mib ?? null,
    }
  }
  if (start === null) return undefined
  const config = start.config
  return {
    // llama.cpp sizes the KV cache at n_ctx cells in total: one stream of n_ctx when unified,
    // otherwise n_parallel streams of n_ctx_slot each.
    kv: {
      cells_total: config.n_ctx,
      cells_used: null,
      unified: config.kv_unified,
      n_ctx_slot: config.n_ctx_slot,
      cache_type_k: config.cache_type_k,
      cache_type_v: config.cache_type_v,
    },
    promptCache: null,
    cacheRamMib: config.cache_ram_mib,
  }
}

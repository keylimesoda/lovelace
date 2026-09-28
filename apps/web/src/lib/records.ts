// Record shapes, as emitted by llama-server's observability layer (schema_version 1).
//
// GET /events streams these live (one record per SSE `data:` frame) and `--request-log-jsonl`
// appends the byte-identical lines, so one set of types serves both the live dashboard and file
// replay. Every field the contract defines is declared here; `null` always means "not measured /
// not applicable" and must never be rendered as 0.
//
// Records are discriminated on `event`, never on `schema_version`.

/** Envelope carried by every record. */
interface RecordEnvelope {
  schema_version: number
  /** 16 hex chars, new per process. A change means the server restarted. */
  server_instance_id: string
  /** Monotonic per instance, starting at 0 on `server_start`. */
  seq: number
  timestamp_unix_ms: number
}

export type TaskType = 'completion' | 'infill' | 'embedding' | 'rerank'

export type Endpoint =
  'native' | 'oai_chat' | 'oai_cmpl' | 'oai_resp' | 'oai_asr' | 'oai_embd' | 'anthropic'

/** The request as the slot saw it when it picked the task up. */
export interface RequestInfo {
  id_task: number
  id_slot: number | null
  id_parent: number | null
  task_type: TaskType
  endpoint: Endpoint
  stream: boolean
  n_prompt_tokens: number
  /** -1 means unlimited. */
  n_predict: number
  n_ctx_slot: number
}

/**
 * Board identity captured once at startup. Only the static fields of the live `/telemetry` gpu
 * object; every field but `available` is absent when NVML is unavailable.
 */
export interface GpuStatic {
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

/** One backend device in the ggml memory breakdown. */
export interface VramDevice {
  name: string
  description: string
  total_bytes: number
  free_bytes: number
  model_bytes: number
  /** KV cache plus recurrent state. */
  context_bytes: number
  /** Compute graph buffers. */
  compute_bytes: number
  /** total - free - (model + context + compute): other processes, CUDA context, fragmentation. */
  unaccounted_bytes: number
}

/** ggml backend memory breakdown, captured once after model load. */
export interface VramBudget {
  devices: VramDevice[]
  host: { model_bytes: number; context_bytes: number; compute_bytes: number }
}

export interface ServerStartRecord extends RecordEnvelope {
  event: 'server_start'
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
    /** --cache-ram: -1 unlimited, 0 disabled. */
    cache_ram_mib: number
    n_gpu_layers: number
    /** `types` is "none" when speculative decoding is off. */
    speculative: { types: string; n_max: number; n_min: number }
    /** Reporting interval of `throughput` records. */
    throughput_interval_ms: number
    energy_sample_hz: number
    request_log_jsonl: string | null
  }
  gpu: GpuStatic
  vram: VramBudget | null
}

/** Emitted when a slot picks the task up. */
export interface RequestStartRecord extends RecordEnvelope {
  event: 'request_start'
  request: RequestInfo
  /** Arrival at the HTTP handler to slot launch. */
  queue_wait_ms: number
}

export type FinishReason = 'eos' | 'word' | 'limit' | 'cancelled' | 'error' | 'done'

export interface RequestResult {
  finish_reason: FinishReason
  truncated: boolean
  n_prompt_tokens: number
  /** Reused from slot KV or the prompt cache: no compute. */
  n_cached_tokens: number
  /** Prompt tokens actually computed. */
  n_prefill_tokens: number
  /** Generated tokens committed. */
  n_completion_tokens: number
  /** Slot sequence length at release. */
  n_ctx_used: number
  error: string | null
}

export interface RequestTimingsMs {
  queue_wait: number
  /** Slot prompt processing wall time. */
  prefill: number
  /** First token to last token. */
  decode: number
  /** Arrival to first token (≈ queue_wait + prefill); null when no token was produced. */
  ttft: number | null
  /** Arrival to release. */
  total: number
}

export interface RequestSpeculative {
  n_drafted: number
  n_accepted: number
  n_verif_steps: number
  accept_rate: number | null
  /** Acceptance count per draft position; its decay is the shape of the accept curve. */
  accepted_per_pos: number[]
}

/**
 * Attributed, not measured: the request's computed prefill and decode tokens priced at the
 * server-wide per-phase J/token over the request's lifetime.
 */
export interface RequestEnergy {
  prefill_joules: number
  decode_joules: number
  total_joules: number
}

/** Emitted when the slot is released: finished, cancelled, or failed mid-flight. */
export interface RequestDoneRecord extends RecordEnvelope {
  event: 'request_done'
  request: RequestInfo
  result: RequestResult
  timings_ms: RequestTimingsMs
  rates: { prefill_tps: number | null; decode_tps: number | null }
  /** Null when no drafting happened. */
  speculative: RequestSpeculative | null
  /** Null without a board energy counter. */
  energy: RequestEnergy | null
}

export type RejectStage = 'http' | 'launch' | 'prompt'

/** The request never produced a result. */
export interface RequestRejectedRecord extends RecordEnvelope {
  event: 'request_rejected'
  request: {
    id_task: number | null
    id_slot: number | null
    task_type: TaskType
    endpoint: Endpoint
    n_prompt_tokens: number
    n_ctx_slot: number
  }
  /** http: parse/tokenize failure; launch: rejected at slot launch; prompt: before any compute. */
  stage: RejectStage
  /** `type` e.g. invalid_request, exceed_context_size, server, not_supported, unavailable. */
  error: { type: string; message: string }
  queue_wait_ms: number | null
}

/** Loop wall-clock split by phase; sums to about the covered interval. */
export interface LoopSeconds {
  prefill: number
  decode: number
  host: number
  idle: number
}

/**
 * Board energy for one interval, split by loop phase time share (host time pro rata between
 * prefill and decode). prefill + decode + idle sums to board_joules exactly.
 */
export interface IntervalEnergy {
  /** Exact board counter delta. */
  board_joules: number
  prefill_joules: number
  decode_joules: number
  idle_joules: number
  /** Board draw calibrated from fully idle sampling periods. */
  idle_watts: number | null
  /** Null per figure when its token denominator is zero: no tokens is not zero joules each. */
  joules_per_token: {
    served: number | null
    active: number | null
    prefill: number | null
    decode: number | null
  }
}

/**
 * One reporting interval. Only emitted when the interval had activity; idle time is folded into
 * the next record, so `interval_seconds` - not the gap between timestamps - is the authority on
 * the window a sample covers.
 */
export interface ThroughputRecord extends RecordEnvelope {
  event: 'throughput'
  interval_seconds: number
  /** prefill = computed prompt tokens, decode = committed generated tokens. */
  tokens: { prefill: number; decode: number; drafted: number; accepted: number }
  /** tokens / interval_seconds, aggregate over all slots. */
  throughput_tokens_per_second: { prefill: number; decode: number }
  /** tokens / loop seconds spent in that phase. */
  phase_tokens_per_second: { prefill: number | null; decode: number | null }
  requests: { started: number; done: number; rejected: number }
  /** Snapshot at emit time: drawn as points, never as bands. */
  scheduler: {
    slots_total: number
    slots_busy: number
    slots_processing_prompt: number
    queue_deferred: number
  }
  loop_seconds: LoopSeconds
  /** llama_decode calls and mean busy slots per call. */
  batch: { decode_calls: number; mean_busy_slots: number | null }
  /** Null on boards without a cumulative energy counter. */
  energy: IntervalEnergy | null
}

export type EngineRecord =
  | ServerStartRecord
  | RequestStartRecord
  | RequestDoneRecord
  | RequestRejectedRecord
  | ThroughputRecord

export type RecordEvent = EngineRecord['event']

const RECORD_EVENTS: Record<RecordEvent, true> = {
  server_start: true,
  request_start: true,
  request_done: true,
  request_rejected: true,
  throughput: true,
}

export function isEngineRecord(value: unknown): value is EngineRecord {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  if (typeof candidate.event !== 'string') return false
  if (typeof candidate.server_instance_id !== 'string') return false
  if (typeof candidate.seq !== 'number') return false
  if (typeof candidate.timestamp_unix_ms !== 'number') return false
  return Object.hasOwn(RECORD_EVENTS, candidate.event)
}

/**
 * Parses one JSONL line. Returns null for blank lines and for a torn trailing record, which a
 * reader tailing a file the server is still appending to will encounter.
 */
export function parseRecordLine(line: string): EngineRecord | null {
  const trimmed = line.trim()
  if (trimmed.length === 0) return null
  try {
    const parsed: unknown = JSON.parse(trimmed)
    return isEngineRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}

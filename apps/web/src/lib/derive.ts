// Derived request, energy, and loop analytics over the retained records.
//
// Durations in the records are milliseconds; every duration this module returns is in seconds,
// the unit the formatters take.

import type {
  LoopSeconds,
  RequestDoneRecord,
  RequestRejectedRecord,
  ThroughputRecord,
} from './records'

/**
 * Nearest-rank percentile with truncating index selection, not an interpolating definition: the
 * two disagree on small samples, and one rule everywhere keeps the dashboard's numbers
 * reproducible by hand from the raw log.
 */
export function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const index = Math.min(sorted.length - 1, Math.trunc(quantile * sorted.length))
  return sorted[index]!
}

export function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0)
}

function tally(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1
  return counts
}

export interface Spread {
  p50: number
  p90: number
  p99: number
  max: number
  mean: number
}

function spread(values: readonly number[]): Spread {
  return {
    p50: percentile(values, 0.5),
    p90: percentile(values, 0.9),
    p99: percentile(values, 0.99),
    max: values.length === 0 ? 0 : Math.max(...values),
    mean: values.length === 0 ? 0 : sum(values) / values.length,
  }
}

export interface SpeculativeSummary {
  drafted: number
  accepted: number
  verifSteps: number
  /** Accepted over drafted; null when nothing was drafted, which is not a 0% acceptance. */
  acceptRate: number | null
  /** Acceptance count per draft position, summed across requests. */
  perPosition: number[]
}

export interface RequestSummary {
  count: number
  promptTokens: Spread
  cachedTokens: Spread
  prefillTokens: Spread
  generated: Spread
  totalPromptTokens: number
  totalCachedTokens: number
  totalPrefillTokens: number
  /** n_cached_tokens / n_prompt_tokens over the window: prompt served without compute. */
  cacheReuse: number
  truncated: number
  byFinishReason: Record<string, number>
  /** Over requests that produced a first token (ttft non-null), in seconds. */
  ttft: Spread
  /** Arrival to slot launch, over every completed request, in seconds. */
  queueWait: Spread
  /**
   * Share of summed TTFT attributable to queue wait and to prefill, over the requests that have a
   * TTFT. The two do not have to sum to exactly 1: TTFT is measured end to end and also covers
   * the slot's first sampling step.
   */
  ttftShare: { queue: number; prefill: number }
  speculative: SpeculativeSummary
  /** llama.cpp's per-request decode rate, over requests that decoded for at least 10 ms. */
  decodeTokensPerSecond: Spread
  /** Per-request attributed energy: how many requests carried it, and its sum. */
  energy: { requests: number; joules: number }
}

// Below this a request's decode rate is one token over timer noise and would dominate the spread.
const MIN_DECODE_MS = 10

export function summarizeRequests(records: readonly RequestDoneRecord[]): RequestSummary {
  const prompt = records.map((record) => record.result.n_prompt_tokens)
  const cached = records.map((record) => record.result.n_cached_tokens)
  const prefill = records.map((record) => record.result.n_prefill_tokens)
  const generated = records.map((record) => record.result.n_completion_tokens)
  const queue = records.map((record) => record.timings_ms.queue_wait / 1000)

  // Queue and prefill shares use the same population as their TTFT denominator; an embedding or
  // a request cancelled before its first token has no TTFT and must not inflate the numerators.
  const firstToken = records.filter((record) => record.timings_ms.ttft !== null)
  const ttftMs = sum(firstToken.map((record) => record.timings_ms.ttft!))
  const queueMs = sum(firstToken.map((record) => record.timings_ms.queue_wait))
  const prefillMs = sum(firstToken.map((record) => record.timings_ms.prefill))

  const perPosition: number[] = []
  let drafted = 0
  let accepted = 0
  let verifSteps = 0
  let energyRequests = 0
  let energyJoules = 0
  for (const record of records) {
    const speculative = record.speculative
    if (speculative) {
      drafted += speculative.n_drafted
      accepted += speculative.n_accepted
      verifSteps += speculative.n_verif_steps
      speculative.accepted_per_pos.forEach((value, index) => {
        perPosition[index] = (perPosition[index] ?? 0) + value
      })
    }
    if (record.energy) {
      energyRequests += 1
      energyJoules += record.energy.total_joules
    }
  }

  const decodeRates = records
    .filter(
      (record) => record.rates.decode_tps !== null && record.timings_ms.decode >= MIN_DECODE_MS,
    )
    .map((record) => record.rates.decode_tps!)

  const totalPrompt = sum(prompt)
  const totalCached = sum(cached)

  return {
    count: records.length,
    promptTokens: spread(prompt),
    cachedTokens: spread(cached),
    prefillTokens: spread(prefill),
    generated: spread(generated),
    totalPromptTokens: totalPrompt,
    totalCachedTokens: totalCached,
    totalPrefillTokens: sum(prefill),
    cacheReuse: totalPrompt === 0 ? 0 : totalCached / totalPrompt,
    truncated: records.filter((record) => record.result.truncated).length,
    byFinishReason: tally(records.map((record) => record.result.finish_reason)),
    ttft: spread(firstToken.map((record) => record.timings_ms.ttft! / 1000)),
    queueWait: spread(queue),
    ttftShare: {
      queue: ttftMs === 0 ? 0 : queueMs / ttftMs,
      prefill: ttftMs === 0 ? 0 : prefillMs / ttftMs,
    },
    speculative: {
      drafted,
      accepted,
      verifSteps,
      acceptRate: drafted === 0 ? null : accepted / drafted,
      perPosition,
    },
    decodeTokensPerSecond: spread(decodeRates),
    energy: { requests: energyRequests, joules: energyJoules },
  }
}

export interface RejectionSummary {
  count: number
  byStage: Record<string, number>
  byErrorType: Record<string, number>
}

export function summarizeRejections(records: readonly RequestRejectedRecord[]): RejectionSummary {
  return {
    count: records.length,
    byStage: tally(records.map((record) => record.stage)),
    byErrorType: tally(records.map((record) => record.error.type)),
  }
}

export interface EnergySummary {
  /** False when no record in the window carried energy, i.e. the board has no counter. */
  available: boolean
  boardJoules: number
  prefillJoules: number
  decodeJoules: number
  idleJoules: number
  /** Latest calibrated idle draw, or null until the server has observed a fully idle period. */
  idleWatts: number | null
  prefillTokens: number
  decodeTokens: number
  /**
   * Joules per token, or null where no tokens of that kind were produced.
   *
   * `served` prices every board joule in the window, idle loop time included: what the work
   * actually cost. `active` prices only the prefill and decode shares, so it tracks the schedule
   * rather than the duty cycle. Because the phase split sums exactly to the board counter, the
   * difference between the two is precisely the idle share.
   */
  servedJoulesPerToken: number | null
  activeJoulesPerToken: number | null
  prefillJoulesPerToken: number | null
  decodeJoulesPerToken: number | null
}

/** Energy per token, or null when the denominator is zero: no tokens is not zero joules each. */
function perToken(joules: number, tokens: number): number | null {
  return tokens === 0 ? null : joules / tokens
}

/**
 * Sums interval deltas rather than differencing cumulative endpoints, so a counter reset inside
 * the window cannot produce a negative reading. Tokens are only counted from intervals that
 * carried energy, so each ratio's numerator and denominator cover the same time.
 */
export function summarizeEnergy(records: readonly ThroughputRecord[]): EnergySummary {
  let boardJoules = 0
  let prefillJoules = 0
  let decodeJoules = 0
  let idleJoules = 0
  let prefillTokens = 0
  let decodeTokens = 0
  let idleWatts: number | null = null
  let available = false
  for (const record of records) {
    const energy = record.energy
    if (!energy) continue
    available = true
    boardJoules += energy.board_joules
    prefillJoules += energy.prefill_joules
    decodeJoules += energy.decode_joules
    idleJoules += energy.idle_joules
    prefillTokens += record.tokens.prefill
    decodeTokens += record.tokens.decode
    // The baseline is a running calibration, so the most recent reading is the current one.
    if (energy.idle_watts !== null) idleWatts = energy.idle_watts
  }
  const tokens = prefillTokens + decodeTokens
  return {
    available,
    boardJoules,
    prefillJoules,
    decodeJoules,
    idleJoules,
    idleWatts,
    prefillTokens,
    decodeTokens,
    servedJoulesPerToken: perToken(boardJoules, tokens),
    activeJoulesPerToken: perToken(prefillJoules + decodeJoules, tokens),
    prefillJoulesPerToken: perToken(prefillJoules, prefillTokens),
    decodeJoulesPerToken: perToken(decodeJoules, decodeTokens),
  }
}

export interface LoopSummary extends LoopSeconds {
  total: number
}

/** Server loop wall clock by phase, summed over the retained throughput intervals. */
export function summarizeLoop(records: readonly ThroughputRecord[]): LoopSummary {
  const out = { prefill: 0, decode: 0, host: 0, idle: 0, total: 0 }
  for (const record of records) {
    const loop = record.loop_seconds
    out.prefill += loop.prefill
    out.decode += loop.decode
    out.host += loop.host
    out.idle += loop.idle
  }
  out.total = out.prefill + out.decode + out.host + out.idle
  return out
}

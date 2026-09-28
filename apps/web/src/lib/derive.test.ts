import { expect, test } from 'bun:test'

import {
  percentile,
  summarizeEnergy,
  summarizeLoop,
  summarizeRejections,
  summarizeRequests,
} from './derive'
import {
  parseRecordLine,
  type FinishReason,
  type RequestDoneRecord,
  type RequestRejectedRecord,
  type ThroughputRecord,
} from './records'

test('percentile matches the truncating nearest-rank rule', () => {
  const values = [5, 1, 3, 2, 4]
  expect(percentile(values, 0.5)).toBe(3)
  expect(percentile(values, 0.9)).toBe(5)
  expect(percentile(values, 0)).toBe(1)
  expect(percentile(values, 1)).toBe(5)
  expect(percentile([1, 2, 3, 4], 0.5)).toBe(3)
  expect(percentile([], 0.5)).toBe(0)
})

function done(
  seq: number,
  values: {
    prompt: number
    cached: number
    generated: number
    queue: number
    prefill: number
    decode: number
    ttft: number | null
    decodeTps?: number | null
    finish?: FinishReason
    truncated?: boolean
    drafted?: number
    accepted?: number
    perPosition?: number[]
    energy?: number
  },
): RequestDoneRecord {
  return {
    schema_version: 1,
    event: 'request_done',
    server_instance_id: '0123456789abcdef',
    seq,
    timestamp_unix_ms: 1_790_000_000_000 + seq * 1000,
    request: {
      id_task: seq,
      id_slot: 0,
      id_parent: null,
      task_type: 'completion',
      endpoint: 'oai_chat',
      stream: true,
      n_prompt_tokens: values.prompt,
      n_predict: -1,
      n_ctx_slot: 8192,
    },
    result: {
      finish_reason: values.finish ?? 'eos',
      truncated: values.truncated ?? false,
      n_prompt_tokens: values.prompt,
      n_cached_tokens: values.cached,
      n_prefill_tokens: values.prompt - values.cached,
      n_completion_tokens: values.generated,
      n_ctx_used: values.prompt + values.generated,
      error: null,
    },
    timings_ms: {
      queue_wait: values.queue,
      prefill: values.prefill,
      decode: values.decode,
      ttft: values.ttft,
      total: values.queue + values.prefill + values.decode,
    },
    rates: { prefill_tps: null, decode_tps: values.decodeTps ?? null },
    speculative:
      values.drafted === undefined
        ? null
        : {
            n_drafted: values.drafted,
            n_accepted: values.accepted ?? 0,
            n_verif_steps: 10,
            accept_rate: values.drafted === 0 ? null : (values.accepted ?? 0) / values.drafted,
            accepted_per_pos: values.perPosition ?? [],
          },
    energy:
      values.energy === undefined
        ? null
        : { prefill_joules: 0, decode_joules: values.energy, total_joules: values.energy },
  }
}

test('request summary decomposes TTFT over the requests that have one', () => {
  const records = [
    done(1, {
      prompt: 2000,
      cached: 0,
      generated: 100,
      queue: 2000,
      prefill: 1000,
      decode: 1000,
      ttft: 3100,
      decodeTps: 100,
    }),
    done(2, {
      prompt: 4000,
      cached: 3000,
      generated: 500,
      queue: 4000,
      prefill: 500,
      decode: 2000,
      ttft: 4600,
      decodeTps: 250,
      finish: 'limit',
      truncated: true,
    }),
    done(3, {
      prompt: 1000,
      cached: 0,
      generated: 300,
      queue: 500,
      prefill: 1500,
      decode: 3000,
      ttft: 2100,
      decodeTps: 100,
      drafted: 150,
      accepted: 100,
      perPosition: [60, 30, 10],
      energy: 12.5,
    }),
    // Near-zero decode: its rate is timer noise and must not enter the spread.
    done(4, {
      prompt: 1500,
      cached: 500,
      generated: 1,
      queue: 1000,
      prefill: 200,
      decode: 5,
      ttft: 1300,
      decodeTps: 200,
    }),
    // No first token (e.g. cancelled in the queue): contributes to queue wait, never to TTFT.
    done(5, {
      prompt: 500,
      cached: 0,
      generated: 0,
      queue: 9000,
      prefill: 0,
      decode: 0,
      ttft: null,
      finish: 'cancelled',
      energy: 2.5,
    }),
  ]
  const summary = summarizeRequests(records)
  expect(summary.count).toBe(5)
  expect(summary.totalPromptTokens).toBe(9000)
  expect(summary.totalCachedTokens).toBe(3500)
  expect(summary.totalPrefillTokens).toBe(5500)
  expect(summary.cacheReuse).toBeCloseTo(3500 / 9000, 12)
  expect(summary.truncated).toBe(1)
  expect(summary.byFinishReason).toEqual({ eos: 3, limit: 1, cancelled: 1 })

  // Over requests 1-4 only: TTFT sum 11100 ms, queue 7500 ms, prefill 3200 ms.
  expect(summary.ttftShare.queue).toBeCloseTo(7500 / 11100, 12)
  expect(summary.ttftShare.prefill).toBeCloseTo(3200 / 11100, 12)
  // Sorted TTFT seconds [1.3, 2.1, 3.1, 4.6]: index trunc(0.5*4)=2.
  expect(summary.ttft.p50).toBeCloseTo(3.1, 12)
  expect(summary.ttft.max).toBeCloseTo(4.6, 12)
  // Queue wait covers all five: sorted [0.5, 1, 2, 4, 9] seconds.
  expect(summary.queueWait.p50).toBeCloseTo(2, 12)
  expect(summary.queueWait.p90).toBeCloseTo(9, 12)
  expect(summary.queueWait.mean).toBeCloseTo(16.5 / 5, 12)

  // Request 4 (5 ms decode) and 5 (no rate) are excluded: [100, 250, 100].
  expect(summary.decodeTokensPerSecond.max).toBe(250)
  expect(summary.decodeTokensPerSecond.p50).toBe(100)

  expect(summary.speculative.drafted).toBe(150)
  expect(summary.speculative.accepted).toBe(100)
  expect(summary.speculative.acceptRate).toBeCloseTo(100 / 150, 12)
  expect(summary.speculative.perPosition).toEqual([60, 30, 10])

  expect(summary.energy).toEqual({ requests: 2, joules: 15 })
})

test('acceptance reads as absent, not zero, when nothing was drafted', () => {
  const summary = summarizeRequests([
    done(1, { prompt: 10, cached: 0, generated: 5, queue: 1, prefill: 1, decode: 50, ttft: 3 }),
  ])
  expect(summary.speculative.acceptRate).toBeNull()
  expect(summarizeRequests([]).cacheReuse).toBe(0)
})

function throughput(
  seq: number,
  values: {
    prefillTokens?: number
    decodeTokens?: number
    loop?: ThroughputRecord['loop_seconds']
    energy?: Partial<NonNullable<ThroughputRecord['energy']>> | null
  },
): ThroughputRecord {
  const energy =
    values.energy === null || values.energy === undefined
      ? null
      : {
          board_joules: 0,
          prefill_joules: 0,
          decode_joules: 0,
          idle_joules: 0,
          idle_watts: null,
          joules_per_token: { served: null, active: null, prefill: null, decode: null },
          ...values.energy,
        }
  return {
    schema_version: 1,
    event: 'throughput',
    server_instance_id: '0123456789abcdef',
    seq,
    timestamp_unix_ms: 1_790_000_000_000 + seq * 5000,
    interval_seconds: 5,
    tokens: {
      prefill: values.prefillTokens ?? 0,
      decode: values.decodeTokens ?? 0,
      drafted: 0,
      accepted: 0,
    },
    throughput_tokens_per_second: {
      prefill: (values.prefillTokens ?? 0) / 5,
      decode: (values.decodeTokens ?? 0) / 5,
    },
    phase_tokens_per_second: { prefill: null, decode: null },
    requests: { started: 0, done: 0, rejected: 0 },
    scheduler: { slots_total: 4, slots_busy: 0, slots_processing_prompt: 0, queue_deferred: 0 },
    loop_seconds: values.loop ?? { prefill: 0, decode: 0, host: 0, idle: 5 },
    batch: { decode_calls: 0, mean_busy_slots: null },
    energy,
  }
}

test('energy summary sums interval deltas and prices each phase against its own tokens', () => {
  const records = [
    throughput(1, {
      prefillTokens: 4400,
      decodeTokens: 400,
      energy: { board_joules: 1500, prefill_joules: 330, decode_joules: 990, idle_joules: 180 },
    }),
    throughput(2, {
      decodeTokens: 368,
      energy: { board_joules: 1000, decode_joules: 500, idle_joules: 500, idle_watts: 31 },
    }),
    // No counter this interval: its tokens must not dilute the priced denominators.
    throughput(3, { prefillTokens: 9999, decodeTokens: 9999, energy: null }),
    throughput(4, {
      prefillTokens: 160,
      decodeTokens: 120,
      energy: {
        board_joules: 2000,
        prefill_joules: 2.4,
        decode_joules: 57.6,
        idle_joules: 1940,
        idle_watts: 29.5,
      },
    }),
  ]
  const summary = summarizeEnergy(records)
  expect(summary.available).toBe(true)
  expect(summary.boardJoules).toBe(4500)
  expect(summary.prefillJoules).toBeCloseTo(332.4, 12)
  expect(summary.decodeJoules).toBeCloseTo(1547.6, 12)
  expect(summary.idleJoules).toBe(2620)
  // The split is exact: the parts sum to the board counter.
  expect(summary.prefillJoules + summary.decodeJoules + summary.idleJoules).toBeCloseTo(4500, 9)
  expect(summary.prefillTokens).toBe(4560)
  expect(summary.decodeTokens).toBe(888)
  expect(summary.idleWatts).toBe(29.5)
  expect(summary.servedJoulesPerToken).toBeCloseTo(4500 / 5448, 12)
  expect(summary.activeJoulesPerToken).toBeCloseTo(1880 / 5448, 12)
  expect(summary.prefillJoulesPerToken).toBeCloseTo(332.4 / 4560, 12)
  expect(summary.decodeJoulesPerToken).toBeCloseTo(1547.6 / 888, 12)
})

test('energy per token is null, not zero, without tokens; unavailable without a counter', () => {
  const idle = summarizeEnergy([
    throughput(1, { energy: { board_joules: 150, idle_joules: 150, idle_watts: 30 } }),
  ])
  expect(idle.available).toBe(true)
  expect(idle.servedJoulesPerToken).toBeNull()
  expect(idle.activeJoulesPerToken).toBeNull()
  expect(idle.prefillJoulesPerToken).toBeNull()
  expect(idle.decodeJoulesPerToken).toBeNull()

  const decodeOnly = summarizeEnergy([
    throughput(1, {
      decodeTokens: 100,
      energy: { board_joules: 300, decode_joules: 250, idle_joules: 50 },
    }),
  ])
  expect(decodeOnly.prefillJoulesPerToken).toBeNull()
  expect(decodeOnly.decodeJoulesPerToken).toBe(2.5)
  expect(decodeOnly.idleWatts).toBeNull()

  const none = summarizeEnergy([throughput(1, { decodeTokens: 100, energy: null })])
  expect(none.available).toBe(false)
  expect(none.servedJoulesPerToken).toBeNull()
})

test('loop summary sums the phase split across intervals', () => {
  const loop = summarizeLoop([
    throughput(1, { loop: { prefill: 1, decode: 3, host: 0.4, idle: 0.6 } }),
    throughput(2, { loop: { prefill: 0, decode: 2.3, host: 0.2, idle: 2.5 } }),
  ])
  expect(loop.prefill).toBe(1)
  expect(loop.decode).toBeCloseTo(5.3, 12)
  expect(loop.host).toBeCloseTo(0.6, 12)
  expect(loop.idle).toBeCloseTo(3.1, 12)
  expect(loop.total).toBeCloseTo(10, 12)
})

test('rejections partition by stage and error type', () => {
  const rejected = (seq: number, stage: RequestRejectedRecord['stage'], type: string) =>
    ({
      schema_version: 1,
      event: 'request_rejected',
      server_instance_id: '0123456789abcdef',
      seq,
      timestamp_unix_ms: 1_790_000_000_000,
      request: {
        id_task: null,
        id_slot: null,
        task_type: 'completion',
        endpoint: 'native',
        n_prompt_tokens: 0,
        n_ctx_slot: 8192,
      },
      stage,
      error: { type, message: '' },
      queue_wait_ms: null,
    }) satisfies RequestRejectedRecord
  const summary = summarizeRejections([
    rejected(1, 'http', 'invalid_request'),
    rejected(2, 'prompt', 'exceed_context_size'),
    rejected(3, 'prompt', 'exceed_context_size'),
  ])
  expect(summary.count).toBe(3)
  expect(summary.byStage).toEqual({ http: 1, prompt: 2 })
  expect(summary.byErrorType).toEqual({ invalid_request: 1, exceed_context_size: 2 })
})

test('record lines parse only well-formed records', () => {
  expect(parseRecordLine('')).toBeNull()
  expect(parseRecordLine('   ')).toBeNull()
  expect(parseRecordLine('not json')).toBeNull()
  // Torn trailing line of a file still being appended to.
  expect(parseRecordLine('{"schema_version":1,"event":"throughput","server_ins')).toBeNull()
  // Envelope incomplete: no seq.
  expect(
    parseRecordLine('{"event":"throughput","server_instance_id":"a","timestamp_unix_ms":1}'),
  ).toBeNull()
  // Unknown event.
  expect(
    parseRecordLine(
      '{"event":"request_error","server_instance_id":"a","seq":1,"timestamp_unix_ms":1}',
    ),
  ).toBeNull()
  const parsed = parseRecordLine(JSON.stringify(throughput(1, {})))
  expect(parsed?.event).toBe('throughput')
})

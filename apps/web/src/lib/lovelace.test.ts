import { expect, test } from 'bun:test'

import { readFileSync } from 'node:fs'
import { summarizeRequests } from './derive'
import { EngineClient, type EngineState } from './engine-client'
import {
  loopSecondsFromRecords,
  summarizeParking,
  summarizeSessionHistory,
  summarizeWaits,
} from './lovelace'

// The fixture is a hand-written schema-20 log with the lovelace extension (schema_version 2):
// one server_start, a cold request, a throughput interval, a request that waited for KV cells and
// continued its session from the RAM tier, and a second interval. Expected values below are
// computed by hand from the numbers in that file.
function replay(): EngineState {
  let state: EngineState | null = null
  const client = new EngineClient((next) => {
    state = next
  })
  client.loadFile(
    'sample.jsonl',
    readFileSync(new URL('../../fixtures/sample.jsonl', import.meta.url), 'utf8'),
  )
  return state!
}

test('replay loads the schema-20 + lovelace log through the file loader', () => {
  const state = replay()
  expect(state.error).toBeNull()
  expect(state.serverStart?.lovelace?.schema_version).toBe(2)
  expect(state.throughput.map((record) => record.lovelace?.seq)).toEqual([2, 4])
  expect(state.requests.map((record) => record.lovelace?.wait_reason)).toEqual(['none', 'kv'])

  // Upstream derivations read the ninfer fields unchanged: 4096 + 4400 prompt tokens, of which
  // 4096 + 48 were computed, one prompt restored from L2.
  const summary = summarizeRequests(state.requests)
  expect(summary.totalPromptTokens).toBe(8496)
  expect(summary.totalComputedPrefill).toBe(4144)
  expect(summary.bySource).toEqual({ none: 1, l2: 1 })

  // server_start carries the ggml VRAM breakdown the KV pool panel draws in replay.
  const device = state.serverStart!.lovelace!.vram!.devices[0]!
  expect(
    device.model_bytes +
      device.context_bytes +
      device.compute_bytes +
      device.unaccounted_bytes +
      device.free_bytes,
  ).toBe(device.total_bytes)
})

test('session history sums interval deltas and keeps the latest gauges', () => {
  const history = summarizeSessionHistory(replay().throughput)!
  expect(history.intervals).toBe(2)
  expect(history.promotions).toBe(1)
  expect(history.demotions).toBe(2)
  expect(history.prefetches).toBe(1)
  expect(history.prefetchHits).toBe(1)
  expect(history.ioBytesRead).toBe(134_217_728)
  expect(history.ioBytesWritten).toBe(268_435_456)
  expect(history.latest.n_disk).toBe(1)
  expect(history.latest.ram_bytes).toBe(536_870_912)
})

test('parking sums transfers and derives GB/s from bytes over device time', () => {
  const parking = summarizeParking(replay().throughput)!
  expect(parking.parks).toBe(1)
  expect(parking.unparks).toBe(1)
  expect(parking.preempts).toBe(1)
  expect(parking.resumes).toBe(1)
  expect(parking.recomputeTokens).toBe(32)
  expect(parking.waitedKv).toBe(1)
  expect(parking.waitedSlot).toBe(0)
  // 268435456 B / 0.02 s = 13.4217728 GB/s; 285212672 B / 0.0115 s = 24.80110191... GB/s
  expect(parking.parkGbps).toBeCloseTo(13.4217728, 9)
  expect(parking.unparkGbps).toBeCloseTo(24.801101913, 8)
})

test('loop seconds sum over intervals', () => {
  // 1.3 + 0.02, 6.4 + 3.2, 0.3 + 0.28, 2.0 + 4.0: the two intervals of 10 s and 7.5 s.
  const loop = loopSecondsFromRecords(replay().throughput)!
  expect(loop.prefill).toBeCloseTo(1.32, 12)
  expect(loop.decode).toBeCloseTo(9.6, 12)
  expect(loop.host).toBeCloseTo(0.58, 12)
  expect(loop.idle).toBeCloseTo(6.0, 12)
})

test('waits partition requests by reason with truncating nearest-rank percentiles', () => {
  const waits = summarizeWaits(replay().requests)!
  expect(waits.reported).toBe(2)
  expect(waits.byReason.none).toEqual({ count: 1, p50: 1.5, p90: 1.5, max: 1.5 })
  expect(waits.byReason.kv).toEqual({ count: 1, p50: 850.25, p90: 850.25, max: 850.25 })
  expect(waits.byReason.slot.count).toBe(0)

  // Four slot waits of 10, 20, 30, 40 ms: p50 index trunc(0.5 * 4) = 2 -> 30, p90 index
  // trunc(0.9 * 4) = 3 -> 40.
  const base = replay().requests[1]!
  const slotWaits = [40, 10, 30, 20].map((queue_wait_ms) => ({
    ...base,
    lovelace: { ...base.lovelace!, wait_reason: 'slot' as const, queue_wait_ms },
  }))
  expect(summarizeWaits(slotWaits)!.byReason.slot).toEqual({ count: 4, p50: 30, p90: 40, max: 40 })
})

test('a ninfer log without the extension reads as not reported, not as zero', () => {
  const state = replay()
  const bare = state.throughput.map(({ lovelace: _, ...record }) => record)
  const requests = state.requests.map(({ lovelace: _, ...record }) => record)
  expect(summarizeSessionHistory(bare)).toBeNull()
  expect(summarizeParking(bare)).toBeNull()
  expect(loopSecondsFromRecords(bare)).toBeNull()
  expect(summarizeWaits(requests)).toBeNull()
})

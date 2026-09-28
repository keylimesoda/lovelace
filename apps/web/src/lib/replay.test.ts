import { afterEach, expect, test } from 'bun:test'

import { summarizeEnergy, summarizeLoop, summarizeRequests } from './derive'
import { EngineClient, replayState, type EngineState } from './engine-client'
import type { EngineRecord } from './records'
import { cacheView } from './telemetry'

const fixture = await Bun.file(new URL('../../fixtures/sample.jsonl', import.meta.url)).text()

test('the sample log replays through the file loader path', () => {
  const state = replayState('sample.jsonl', fixture)
  expect(state.connection).toBe('replay')
  expect(state.error).toBeNull()
  expect(state.droppedInstances).toBe(0)
  expect(state.serverStart?.config.n_parallel).toBe(4)
  expect(state.throughput.map((record) => record.seq)).toEqual([5, 7, 9])
  expect(state.requests.map((record) => record.request.id_task)).toEqual([0, 1, 3])
  expect(state.rejected).toHaveLength(1)
  expect(state.rejected[0]!.error.type).toBe('exceed_context_size')

  const summary = summarizeRequests(state.requests)
  expect(summary.cacheReuse).toBeCloseTo(7648 / 12208, 12)
  expect(summary.ttftShare.queue).toBeCloseTo(47 / 830, 12)
  expect(summary.ttftShare.prefill).toBeCloseTo(750 / 830, 12)
  expect(summary.byFinishReason).toEqual({ eos: 2, limit: 1 })
  expect(summary.energy.joules).toBeCloseTo(723.6 + 1096.4 + 98.4, 9)

  const energy = summarizeEnergy(state.throughput)
  expect(energy.boardJoules).toBe(4500)
  expect(energy.servedJoulesPerToken).toBeCloseTo(4500 / 5448, 12)
  expect(energy.activeJoulesPerToken).toBeCloseTo(1880 / 5448, 12)
  expect(energy.idleWatts).toBe(29.5)

  expect(summarizeLoop(state.throughput).total).toBeCloseTo(60, 9)

  // Replay has no live snapshot: capacities come from server_start, occupancy reads as absent.
  const cache = cacheView(null, state.serverStart)
  expect(cache?.kv.cells_total).toBe(32768)
  expect(cache?.kv.cells_used).toBeNull()
  expect(cache?.promptCache).toBeNull()
  expect(cache?.cacheRamMib).toBe(8192)
  expect(state.serverStart?.vram?.devices[0]?.unaccounted_bytes).toBe(1329594368)
})

test('replay keeps only the last server instance and skips a torn tail', () => {
  const earlier = fixture.replaceAll('5f3a9c0e7b21d468', '00000000deadbeef')
  const torn = '{"schema_version":1,"event":"throughput","server_instance_id":"5f3a'
  const state = replayState('multi.jsonl', `${earlier}\n\n${fixture}${torn}`)
  expect(state.droppedInstances).toBe(1)
  expect(state.serverStart?.server_instance_id).toBe('5f3a9c0e7b21d468')
  expect(state.throughput).toHaveLength(3)
  expect(state.requests).toHaveLength(3)

  expect(replayState('empty.jsonl', '\n\n').error).toBe('empty.jsonl contains no server records')
})

// --- Live stream ingestion ------------------------------------------------

class FakeEventSource {
  static last: FakeEventSource | null = null
  onmessage: ((message: MessageEvent<string>) => void) | null = null
  onerror: (() => void) | null = null
  constructor() {
    FakeEventSource.last = this
  }
  close(): void {}
  emit(record: EngineRecord): void {
    this.onmessage?.({ data: JSON.stringify(record) } as MessageEvent<string>)
  }
}

const realEventSource = globalThis.EventSource
const realFetch = globalThis.fetch
afterEach(() => {
  globalThis.EventSource = realEventSource
  globalThis.fetch = realFetch
})

test('a reconnect backlog is not double counted, and a restart resets history', () => {
  globalThis.EventSource = FakeEventSource as unknown as typeof EventSource
  globalThis.fetch = (() => Promise.withResolvers<Response>().promise) as unknown as typeof fetch
  const records = fixture
    .split('\n')
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as EngineRecord)

  let state: EngineState | null = null
  const client = new EngineClient((next) => {
    state = next
  })
  client.start()
  const stream = FakeEventSource.last!
  for (const record of records.slice(0, 8)) stream.emit(record)
  const partial = state as unknown as EngineState
  expect(partial.throughput).toHaveLength(2)
  expect(partial.requests).toHaveLength(2)
  expect(partial.rejected).toHaveLength(1)
  // Task 0 and 1 finished, task 2 was rejected after its start; nothing is in flight.
  expect(partial.active).toHaveLength(0)

  // Reconnect: the server resends server_start and its backlog before live records.
  for (const record of records) stream.emit(record)
  const resumed = state as unknown as EngineState
  expect(resumed.throughput.map((record) => record.seq)).toEqual([5, 7, 9])
  expect(resumed.requests.map((record) => record.seq)).toEqual([4, 6, 10])

  // A new instance id is a restart: the previous process's history is dropped.
  stream.emit({ ...records[0]!, server_instance_id: '1111222233334444' })
  const restarted = state as unknown as EngineState
  expect(restarted.serverStart?.server_instance_id).toBe('1111222233334444')
  expect(restarted.throughput).toHaveLength(0)
  expect(restarted.requests).toHaveLength(0)
  client.stop()
})

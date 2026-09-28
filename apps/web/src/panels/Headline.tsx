import { Stat, type Tone } from '../components/ui'
import { summarizeEnergy, type RequestSummary } from '../lib/derive'
import { count, joulesPerToken, percent, rate, seconds } from '../lib/format'
import type { ThroughputRecord } from '../lib/records'
import { latest } from '../lib/series'
import type { Telemetry } from '../lib/telemetry'

/**
 * The glance row. Every reading here is one a maintainer would otherwise reconstruct by hand:
 * current rates, whether requests are queueing, whether the cache is doing its job, and whether
 * the board is the limit.
 */
export function Headline({
  telemetry,
  records,
  summary,
  slotsConfigured,
}: {
  telemetry: Telemetry | null
  records: ThroughputRecord[]
  summary: RequestSummary
  /** n_parallel from server_start, for a replay with no live snapshot. */
  slotsConfigured: number | undefined
}) {
  const sample = latest(records)
  const scheduler = telemetry?.scheduler
  const gpu = telemetry?.gpu

  // A replayed log has no live snapshot, but each throughput record carries the scheduler
  // snapshot taken when it was emitted.
  const busy = scheduler?.slots_busy ?? sample?.scheduler.slots_busy ?? 0
  const slots = scheduler?.slots_total ?? sample?.scheduler.slots_total ?? slotsConfigured ?? 0
  const deferred = scheduler?.queue_deferred ?? sample?.scheduler.queue_deferred ?? 0
  const deferredTone: Tone = deferred > 0 ? 'warning' : 'neutral'

  // Board readings are live-only. Reporting 0 for an unavailable sensor would be a measurement
  // that was never taken, so they read as absent instead.
  const board = gpu?.available === true
  const watts = telemetry?.energy.board_watts_1s ?? null

  // Energy is carried in the throughput record, so it survives a replay. Served rather than
  // active, because the glance row should show what the work costs.
  const energy = summarizeEnergy(records)
  const accept = summary.speculative.acceptRate

  return (
    <div className="headline">
      <Stat
        value={rate(sample?.throughput_tokens_per_second.decode ?? 0)}
        unit="tok/s"
        label="decode · all slots"
        hint="decodeRate"
        tone="accent"
      />
      <Stat
        value={rate(sample?.throughput_tokens_per_second.prefill ?? 0)}
        unit="tok/s"
        label="prefill"
        hint="prefillRate"
      />
      <Stat
        value={`${busy}/${slots}`}
        label="slots busy"
        hint="slots"
        tone={slots > 0 && busy >= slots ? 'warning' : 'neutral'}
      />
      <Stat value={count(deferred)} label="deferred" hint="queueDeferred" tone={deferredTone} />
      <Stat
        value={seconds(summary.queueWait.p50)}
        label="queue p50"
        hint="queueWait"
        tone={summary.queueWait.p50 > 1 ? 'warning' : 'neutral'}
      />
      <Stat value={seconds(summary.ttft.p50)} label="ttft p50" hint="ttft" />
      <Stat
        value={percent(summary.cacheReuse)}
        label="cache reuse"
        hint="cacheReuse"
        tone={summary.cacheReuse > 0.5 ? 'accent' : 'neutral'}
      />
      <Stat
        value={accept === null ? '—' : percent(accept)}
        label="draft accept"
        hint="draftAccept"
        tone={accept !== null && accept > 0.5 ? 'accent' : 'neutral'}
      />
      <Stat
        value={energy.available ? joulesPerToken(energy.servedJoulesPerToken) : '—'}
        unit={energy.available && energy.servedJoulesPerToken !== null ? 'J/tok' : undefined}
        label="energy"
        hint="energyServed"
      />
      <Stat
        value={watts === null ? '—' : watts.toFixed(0)}
        unit={watts === null ? undefined : 'W'}
        label="board"
        hint="boardPower"
      />
      <Stat
        value={board ? `${gpu.utilization_gpu_percent ?? 0}` : '—'}
        unit={board ? '%' : undefined}
        label="gpu"
        hint="gpuUtil"
        tone={board && (gpu.utilization_gpu_percent ?? 0) > 90 ? 'accent' : 'neutral'}
      />
      <Stat
        value={board ? `${gpu.temperature_c ?? 0}` : '—'}
        unit={board ? '°C' : undefined}
        label="temp"
        hint="gpuTemp"
        tone={
          !board
            ? 'neutral'
            : (gpu.temperature_c ?? 0) >= 83
              ? 'danger'
              : (gpu.temperature_c ?? 0) >= 75
                ? 'warning'
                : 'neutral'
        }
      />
    </div>
  )
}

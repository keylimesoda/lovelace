import { BandChart, Legend } from '../components/charts'
import { CHART } from '../components/echart'
import { Info, Term } from '../components/tooltip'
import { Empty, Panel, Stat } from '../components/ui'
import { summarizeEnergy, type RequestSummary } from '../lib/derive'
import { energyPerMillionTokens, joules, joulesPerToken, percent } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { ThroughputRecord } from '../lib/records'
import { bands } from '../lib/series'
import type { GpuTelemetry } from '../lib/telemetry'

/**
 * What the work costs, in joules.
 *
 * Board energy is the exact delta of the GPU's cumulative counter per interval; the server splits
 * it by loop phase time share, so prefill + decode + idle sums to the board total with no
 * residual. Energy is part of the record stream, so this panel works on a replayed log.
 */
export function EnergyPanel({
  records,
  summary,
  gpu,
  energyAvailable,
  replay,
}: {
  records: ThroughputRecord[]
  summary: RequestSummary
  gpu: GpuTelemetry | undefined
  /** The board's energy counter, from /telemetry live or server_start.gpu in replay. */
  energyAvailable: boolean | undefined
  replay: boolean
}) {
  const energy = summarizeEnergy(records)

  if (!energy.available) {
    return (
      <Panel title="Energy" hint="energySplit" className="panel--wide" note="NVML">
        {/* Many boards do not implement nvmlDeviceGetTotalEnergyConsumption. That is a missing
            counter, not a zero, so nothing is drawn rather than a flat line at zero. */}
        <Empty>
          {energyAvailable === false
            ? 'unavailable — this board exposes no cumulative energy counter'
            : records.length === 0
              ? replay
                ? 'no throughput records in the loaded log'
                : 'no samples yet'
              : 'unavailable — no interval in the window carried energy'}
        </Empty>
      </Panel>
    )
  }

  const first = records[0]!
  const last = records[records.length - 1]!
  const domain: [number, number] = [
    first.timestamp_unix_ms - first.interval_seconds * 1000,
    last.timestamp_unix_ms,
  ]
  const idleShare = energy.boardJoules === 0 ? 0 : energy.idleJoules / energy.boardJoules

  return (
    <Panel
      title="Energy"
      hint="energySplit"
      className="panel--wide"
      note={`${joules(energy.boardJoules)} measured`}
    >
      <div className="stat-row">
        <Stat
          value={joulesPerToken(energy.servedJoulesPerToken)}
          unit="J/tok"
          label="served"
          hint="energyServed"
          tone="accent"
        />
        <Stat
          value={joulesPerToken(energy.activeJoulesPerToken)}
          unit="J/tok"
          label="active"
          hint="energyActive"
        />
        <Stat
          value={joulesPerToken(energy.prefillJoulesPerToken)}
          unit="J/tok"
          label="prefill"
          hint="energyPrefill"
        />
        <Stat
          value={joulesPerToken(energy.decodeJoulesPerToken)}
          unit="J/tok"
          label="decode"
          hint="energyDecode"
        />
        <Stat
          value={energy.idleWatts === null ? '—' : energy.idleWatts.toFixed(0)}
          unit={energy.idleWatts === null ? undefined : 'W'}
          label="idle draw"
          hint="idleWatts"
        />
        <Stat
          value={energyPerMillionTokens(energy.servedJoulesPerToken)}
          label="per 1M tokens"
          hint="energyPerMillion"
        />
      </div>

      <BandChart
        label="Board joules per interval, split by loop phase"
        domain={domain}
        series={[
          {
            name: 'prefill',
            bands: bands(records, (r) => r.energy?.prefill_joules ?? 0),
            color: CHART.violet,
          },
          {
            name: 'decode',
            bands: bands(records, (r) => r.energy?.decode_joules ?? 0),
            color: CHART.accent,
          },
          {
            name: 'idle',
            bands: bands(records, (r) => r.energy?.idle_joules ?? 0),
            color: CHART.dim,
          },
        ]}
        unit="J"
        stack
        legend={
          <Legend
            items={[
              { label: 'prefill', color: CHART.violet, hint: GLOSSARY.energyPrefill.body },
              { label: 'decode', color: CHART.accent, hint: GLOSSARY.energyDecode.body },
              { label: 'idle', color: CHART.dim, hint: GLOSSARY.energyIdle.body },
            ]}
          />
        }
        caption={
          <>
            interval bands, stacked to the board total <Info k="intervalBands" />
          </>
        }
      />

      <p className="panel__footnote">
        <Term k="energySplit">Attribution</Term>: each interval&apos;s exact board delta is divided
        by loop phase time share, with host time split between prefill and decode in proportion to
        their loop seconds, so prefill + decode + idle is the board total with no residual.{' '}
        <Term k="energyServed">Served</Term> prices every joule, including the {percent(idleShare)}{' '}
        of this window attributed to idle loop time; <Term k="energyActive">active</Term> prices
        only the prefill and decode shares. Board energy excludes the CPU and the rest of the
        platform, so it is not a wall-socket figure.
        {gpu?.power_limit_watts
          ? ` Energy per token is a function of the ${gpu.power_limit_watts.toFixed(0)} W board limit, not a fixed property of the model.`
          : null}
      </p>
      <p className="panel__footnote">
        <Term k="energyPerMillion">Per million tokens</Term>:{' '}
        {energyPerMillionTokens(energy.servedJoulesPerToken)} served,{' '}
        {energyPerMillionTokens(energy.activeJoulesPerToken)} active
        {energy.prefillJoulesPerToken === null
          ? null
          : `, ${energyPerMillionTokens(energy.prefillJoulesPerToken)} per million prefilled`}
        {energy.decodeJoulesPerToken === null
          ? null
          : `, ${energyPerMillionTokens(energy.decodeJoulesPerToken)} per million decoded`}
        .
        {summary.energy.requests > 0 ? (
          <>
            {' '}
            <Term k="requestEnergy">Attributed to requests</Term>: {joules(summary.energy.joules)}{' '}
            over {summary.energy.requests} completed, mean{' '}
            {joules(summary.energy.joules / summary.energy.requests)} each.
          </>
        ) : null}
      </p>
    </Panel>
  )
}

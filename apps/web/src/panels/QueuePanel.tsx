import { Legend, SampleChart, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Info, Term } from '../components/tooltip'
import { Empty, Panel, Pill, Stat } from '../components/ui'
import { bytes, count, percent, seconds } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import {
  loopSecondsFromRecords,
  milliseconds,
  summarizeParking,
  summarizeWaits,
  WAIT_REASONS,
  type LovelaceLoopSeconds,
  type LovelaceTelemetry,
} from '../lib/lovelace'
import type { RequestDoneRecord, ThroughputRecord } from '../lib/records'
import { latest, samples } from '../lib/series'

const LOOP_PHASES = [
  { key: 'prefill', label: 'prefill', color: CHART.violet, hint: GLOSSARY.lvLoopPrefill.body },
  { key: 'decode', label: 'decode', color: CHART.accent, hint: GLOSSARY.lvLoopDecode.body },
  { key: 'host', label: 'host', color: CHART.warning, hint: GLOSSARY.lvLoopHost.body },
  { key: 'idle', label: 'idle', color: CHART.line, hint: GLOSSARY.lvLoopIdle.body },
] as const

const WAIT_COLOR = { none: CHART.accent, slot: CHART.blue, kv: CHART.warning } as const

/**
 * lovelace's KV-budget admission: what is waiting and why, what parking and preemption moved
 * between VRAM and host memory, and where the server loop spent its time. Reads only the
 * `lovelace` extension of /telemetry, throughput and request_done.
 */
export function QueuePanel({
  ext,
  records,
  requests,
  replay,
}: {
  ext: LovelaceTelemetry | undefined
  records: ThroughputRecord[]
  requests: RequestDoneRecord[]
  replay: boolean
}) {
  const budget = ext?.scheduler?.kv_budget
  const parking = summarizeParking(records)
  const waits = summarizeWaits(requests)
  const lovelaceRecords = records.filter((record) => record.lovelace !== undefined)
  const sample = latest(lovelaceRecords)?.lovelace

  if (!budget && !parking && !waits) {
    return (
      <Panel title="Queue & waits" hint="lvKvBudget">
        <Empty>
          not reported {replay ? 'in this log' : 'by this server'} <Info k="lvNotReported" />
        </Empty>
      </Panel>
    )
  }

  // Live totals are cumulative since start; in replay the interval deltas summed over the log.
  const totals = budget?.totals
  const transfers = [
    {
      label: 'parks',
      n: totals ? totals.parks : (parking?.parks ?? 0),
      moved: totals ? totals.park_bytes : (parking?.parkBytes ?? 0),
      time: totals ? totals.park_seconds : (parking?.parkSeconds ?? 0),
      gbps: totals ? totals.park_gbps : (parking?.parkGbps ?? null),
    },
    {
      label: 'unparks',
      n: totals ? totals.unparks : (parking?.unparks ?? 0),
      moved: totals ? totals.unpark_bytes : (parking?.unparkBytes ?? 0),
      time: totals ? totals.unpark_seconds : (parking?.unparkSeconds ?? 0),
      gbps: totals ? totals.unpark_gbps : (parking?.unparkGbps ?? null),
    },
  ]
  const preempts = totals ? totals.preempts : (parking?.preempts ?? 0)
  const resumes = totals ? totals.resumes : (parking?.resumes ?? 0)
  const waitedSlot = totals ? totals.waited_slot : (parking?.waitedSlot ?? 0)
  const waitedKv = totals ? totals.waited_kv : (parking?.waitedKv ?? 0)
  const recompute = totals ? totals.recompute_tokens : (parking?.recomputeTokens ?? 0)
  const queueWaiting = budget?.queue_waiting ?? sample?.scheduler.queue_waiting ?? 0
  const preempted = budget?.slots_preempted ?? sample?.scheduler.slots_preempted ?? 0

  const loop: LovelaceLoopSeconds | null = ext?.loop?.seconds ?? loopSecondsFromRecords(records)
  const loopTotal = loop ? loop.prefill + loop.decode + loop.host + loop.idle : 0

  return (
    <Panel
      title="Queue & waits"
      hint="lvKvBudget"
      action={
        budget ? (
          <Pill tone={budget.enabled ? 'accent' : 'neutral'}>
            kv budget {budget.enabled ? 'on' : 'off'}
          </Pill>
        ) : undefined
      }
    >
      <div className="stat-row">
        <Stat
          value={count(queueWaiting)}
          label="queue waiting"
          hint="lvQueueWaiting"
          tone={queueWaiting > 0 ? 'warning' : 'neutral'}
        />
        <Stat
          value={count(preempted)}
          label="preempted"
          hint="lvPreempted"
          tone={preempted > 0 ? 'warning' : 'neutral'}
        />
        <Stat value={count(waitedSlot)} label="waited slot" hint="lvWaited" />
        <Stat value={count(waitedKv)} label="waited kv" hint="lvWaited" />
        <Stat
          value={count(recompute)}
          label="recomputed"
          hint="lvRecompute"
          tone={recompute > 0 ? 'warning' : 'neutral'}
        />
        {totals ? (
          <Stat
            value={count(totals.kv_overflows)}
            label="kv overflows"
            hint="lvKvOverflows"
            tone={totals.kv_overflows > 0 ? 'danger' : 'neutral'}
          />
        ) : null}
      </div>

      <table className="table">
        <thead>
          <tr>
            <th>
              <Term k="lvParks">transfer</Term>
            </th>
            <th className="numeric">n</th>
            <th className="numeric">bytes</th>
            <th className="numeric">device time</th>
            <th className="numeric">
              <Term k="lvTransferRate">rate</Term>
            </th>
          </tr>
        </thead>
        <tbody>
          {transfers.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              <td className="numeric emphasis">{count(row.n)}</td>
              <td className="numeric">{bytes(row.moved)}</td>
              <td className="numeric">{milliseconds(row.time * 1000)}</td>
              <td className="numeric">
                {row.gbps === null || row.gbps <= 0 ? '—' : `${row.gbps.toFixed(1)} GB/s`}
              </td>
            </tr>
          ))}
          <tr>
            <td>
              <Term k="lvPreemptions">preempts / resumes</Term>
            </td>
            <td className="numeric emphasis">
              {count(preempts)} / {count(resumes)}
            </td>
            <td />
            <td />
            <td />
          </tr>
        </tbody>
      </table>
      {totals ? (
        <p className="panel__footnote">
          Since start: {count(totals.park_drops)} parks dropped without a host copy ·{' '}
          {count(totals.grows)} reservation grows · {count(totals.rollback_fallbacks)} rollback
          fallbacks · loop time on transfers {milliseconds(totals.park_host_seconds * 1000)} park /{' '}
          {milliseconds(totals.unpark_host_seconds * 1000)} unpark.
        </p>
      ) : (
        <p className="panel__footnote">Summed over the logged intervals.</p>
      )}

      {waits ? (
        <div className="latency__split">
          <div className="eyebrow">
            <Term k="lvWaitReason">wait reason</Term> · {count(waits.reported)} completed requests
          </div>
          <StackedBar
            segments={WAIT_REASONS.map((reason) => ({
              label: reason,
              value: waits.byReason[reason].count,
              color: WAIT_COLOR[reason],
              display: `${count(waits.byReason[reason].count)} requests · ${percent(
                waits.byReason[reason].count / waits.reported,
              )}`,
            }))}
          />
          <table className="table">
            <thead>
              <tr>
                <th>reason</th>
                <th className="numeric">n</th>
                <th className="numeric">queue p50</th>
                <th className="numeric">p90</th>
                <th className="numeric">max</th>
              </tr>
            </thead>
            <tbody>
              {WAIT_REASONS.map((reason) => {
                const group = waits.byReason[reason]
                return (
                  <tr key={reason}>
                    <td>{reason}</td>
                    <td className="numeric emphasis">{count(group.count)}</td>
                    <td className="numeric">{group.count ? milliseconds(group.p50) : '—'}</td>
                    <td className="numeric">{group.count ? milliseconds(group.p90) : '—'}</td>
                    <td className="numeric">{group.count ? milliseconds(group.max) : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      {lovelaceRecords.length > 0 ? (
        <SampleChart
          label="Admission queue, deferred tasks and preempted slots at each report"
          series={[
            {
              name: 'queue',
              samples: samples(lovelaceRecords, (r) => r.lovelace!.scheduler.queue_waiting),
              color: CHART.warning,
            },
            {
              name: 'deferred',
              samples: samples(lovelaceRecords, (r) => r.lovelace!.scheduler.queue_deferred),
              color: CHART.blue,
            },
            {
              name: 'preempted',
              samples: samples(lovelaceRecords, (r) => r.lovelace!.scheduler.slots_preempted),
              color: CHART.danger,
            },
          ]}
          legend={
            <Legend
              items={[
                { label: 'queue', color: CHART.warning, hint: GLOSSARY.lvQueueWaiting.body },
                { label: 'deferred', color: CHART.blue, hint: GLOSSARY.lvQueueHistory.body },
                { label: 'preempted', color: CHART.danger, hint: GLOSSARY.lvPreempted.body },
              ]}
            />
          }
          caption={
            <>
              snapshots, not interval means <Info k="lvQueueHistory" />
            </>
          }
        />
      ) : null}

      {loop && loopTotal > 0 ? (
        <div className="scheduler__worker">
          <div className="eyebrow">
            <Term k="lvLoopSplit">server loop wall clock</Term>
            {ext?.loop
              ? ` · ${seconds(ext.loop.window_seconds)} window · now ${ext.loop.phase}`
              : ` · ${seconds(loopTotal)} logged`}
          </div>
          <StackedBar
            segments={LOOP_PHASES.map((phase) => ({
              label: phase.label,
              value: loop[phase.key],
              color: phase.color,
              hint: phase.hint,
              display: `${seconds(loop[phase.key])} · ${percent(loop[phase.key] / loopTotal)} of loop time`,
            }))}
          />
          <Legend
            items={LOOP_PHASES.map((phase) => ({
              label: phase.label,
              color: phase.color,
              hint: phase.hint,
            }))}
          />
        </div>
      ) : null}
    </Panel>
  )
}

import { Legend, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term } from '../components/tooltip'
import { Empty, Meter, Panel, Pill, Stat, type Tone } from '../components/ui'
import { summarizeLoop } from '../lib/derive'
import { count, percent, seconds } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { LoopSeconds, ThroughputRecord } from '../lib/records'
import { latest } from '../lib/series'
import type { LoopPhase, Telemetry } from '../lib/telemetry'

const LOOP_PHASES = [
  { key: 'prefill', color: CHART.violet, hint: GLOSSARY.loopPrefill.body },
  { key: 'decode', color: CHART.accent, hint: GLOSSARY.loopDecode.body },
  { key: 'host', color: CHART.blue, hint: GLOSSARY.loopHost.body },
  { key: 'idle', color: CHART.line, hint: GLOSSARY.loopIdle.body },
] as const

const PHASE_TONE: Record<LoopPhase, Tone> = {
  prefill: 'accent',
  decode: 'accent',
  host: 'warning',
  idle: 'neutral',
}

function LoopBar({ loop, total }: { loop: LoopSeconds; total: number }) {
  return (
    <>
      <StackedBar
        height={10}
        segments={LOOP_PHASES.map((phase) => ({
          label: phase.key,
          value: loop[phase.key],
          color: phase.color,
          hint: phase.hint,
          display: `${seconds(loop[phase.key])} · ${percent(loop[phase.key] / total, 1)} of loop time`,
        }))}
      />
      <Legend
        items={LOOP_PHASES.map((phase) => ({
          label: `${phase.key} ${percent(loop[phase.key] / total)}`,
          color: phase.color,
          hint: phase.hint,
        }))}
      />
    </>
  )
}

export function SchedulerPanel({
  telemetry,
  records,
  slotsConfigured,
}: {
  telemetry: Telemetry | null
  records: ThroughputRecord[]
  /** n_parallel from server_start, used when no live telemetry is attached. */
  slotsConfigured: number | undefined
}) {
  const scheduler = telemetry?.scheduler
  const sample = latest(records)
  const snapshot = scheduler ?? sample?.scheduler
  const slots = snapshot?.slots_total ?? slotsConfigured ?? 0
  const busy = snapshot?.slots_busy ?? 0
  const processingPrompt = snapshot?.slots_processing_prompt ?? 0
  const deferred = snapshot?.queue_deferred ?? 0

  // Live, the split is the server's own rolling window. In replay the same quantity is summed
  // from the throughput intervals, which cover the whole retained log rather than 10 s.
  const windowLoop = scheduler?.loop.seconds ?? summarizeLoop(records)
  const windowTotal = scheduler
    ? scheduler.loop.window_seconds
    : windowLoop.prefill + windowLoop.decode + windowLoop.host + windowLoop.idle
  const windowLabel = scheduler
    ? `last ${seconds(scheduler.loop.window_seconds)}`
    : `${records.length} intervals · ${seconds(windowTotal)}`

  const lifetime = scheduler?.loop_totals_seconds
  const lifetimeTotal = lifetime
    ? lifetime.prefill + lifetime.decode + lifetime.host + lifetime.idle
    : 0

  // Totals: the cumulative server counters when live, the interval deltas summed in replay.
  let started = 0
  let done = 0
  let rejected = 0
  let decodeCalls = 0
  for (const record of records) {
    started += record.requests.started
    done += record.requests.done
    rejected += record.requests.rejected
    decodeCalls += record.batch.decode_calls
  }
  const totals = scheduler?.totals
  const rows: Array<[string, string]> = totals
    ? [
        ['requests started', count(totals.requests_started)],
        ['requests done', count(totals.requests_done)],
        ['requests rejected', count(totals.requests_rejected)],
        ['llama_decode calls', count(totals.n_decode)],
        ['busy slots / decode', totals.n_busy_slots_per_decode.toFixed(2)],
        [
          'prompt tokens',
          `${count(totals.prompt_tokens)} (${count(totals.prompt_tokens_cached)} cached)`,
        ],
        ['predicted tokens', count(totals.predicted_tokens)],
        ['drafted / accepted', `${count(totals.drafted)} / ${count(totals.accepted)}`],
      ]
    : [
        ['requests started', count(started)],
        ['requests done', count(done)],
        ['requests rejected', count(rejected)],
        ['llama_decode calls', count(decodeCalls)],
      ]

  if (!snapshot && records.length === 0) {
    return (
      <Panel title="Scheduler" hint="loopSplit">
        <Empty>no scheduler data yet</Empty>
      </Panel>
    )
  }

  return (
    <Panel
      title="Scheduler"
      hint="loopSplit"
      note={
        scheduler ? (
          <Term k="loopPhase">
            <Pill tone={PHASE_TONE[scheduler.loop.phase]}>{scheduler.loop.phase}</Pill>
          </Term>
        ) : (
          `${slots} slot${slots === 1 ? '' : 's'} · last report`
        )
      }
    >
      <div className="stat-row">
        <Stat
          value={`${busy}/${slots}`}
          label="busy"
          hint="slots"
          tone={slots > 0 && busy >= slots ? 'warning' : 'accent'}
        />
        <Stat value={count(processingPrompt)} label="prompt" hint="processingPrompt" />
        {scheduler ? (
          <Stat value={count(scheduler.slots_generating)} label="generating" hint="slots" />
        ) : null}
        <Stat
          value={count(deferred)}
          label="deferred"
          hint="queueDeferred"
          tone={deferred > 0 ? 'warning' : 'neutral'}
        />
      </div>

      <div className="scheduler__occupancy">
        <div className="eyebrow">
          <Term k="slots">slot occupancy</Term>
        </div>
        <Meter fraction={slots === 0 ? 0 : busy / slots} />
      </div>

      <div className="scheduler__worker">
        <div className="eyebrow">
          <Term k="loopSplit">server loop wall clock</Term> · {windowLabel}
        </div>
        {windowTotal > 0 ? (
          <LoopBar loop={windowLoop} total={windowTotal} />
        ) : (
          <Empty>no loop time recorded</Empty>
        )}
        {lifetime && lifetimeTotal > 0 ? (
          <p className="panel__footnote">
            Since start ({seconds(lifetimeTotal)}):{' '}
            {LOOP_PHASES.map(
              (phase) => `${phase.key} ${percent(lifetime[phase.key] / lifetimeTotal, 1)}`,
            ).join(' · ')}
            . Host time is the loop busy outside llama_decode; a rising host share with steady
            compute is bookkeeping, sampling, or cache save/restore, not model speed.
          </p>
        ) : null}
      </div>

      <table className="table">
        <thead>
          <tr>
            <th>
              <Term k="schedulerTotals">{totals ? 'since start' : 'over the log'}</Term>
            </th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <td>{label}</td>
              <td className="numeric emphasis">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  )
}

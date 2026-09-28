import { Legend, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term } from '../components/tooltip'
import { Empty, Panel, Stat } from '../components/ui'
import type { RequestSummary } from '../lib/derive'
import { count, percent, seconds } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'

const PHASES = [
  {
    key: 'queue',
    label: 'queue',
    color: CHART.warning,
    hint: 'Waiting for a free slot, from arrival at the HTTP handler to slot launch. Responds to slot count and generation length, not to kernel speed.',
  },
  {
    key: 'prefill',
    label: 'prefill',
    color: CHART.violet,
    hint: 'Slot prompt processing: computing the prompt tokens that neither the slot KV nor the prompt cache covered.',
  },
] as const

const FINISH_COLOR: Record<string, string> = {
  eos: CHART.accent,
  word: CHART.blue,
  limit: CHART.warning,
  cancelled: CHART.muted,
  error: CHART.danger,
}

export function LatencyPanel({ summary }: { summary: RequestSummary }) {
  if (summary.count === 0) {
    return (
      <Panel title="Latency" hint="ttft">
        <Empty>no completed requests in the retained window</Empty>
      </Panel>
    )
  }

  const share = summary.ttftShare
  const dominant = PHASES.reduce((best, phase) =>
    share[phase.key] > share[best.key] ? phase : best,
  )
  const accept = summary.speculative.acceptRate

  const finishSegments = Object.entries(summary.byFinishReason)
    .sort((a, b) => b[1] - a[1])
    .map(([reason, requests]) => ({
      label: reason,
      value: requests,
      color: FINISH_COLOR[reason] ?? CHART.dim,
      hint: GLOSSARY.finishReason.body,
      display: `${requests} of ${summary.count} requests`,
    }))

  return (
    <Panel title="Latency" hint="ttft" note={`${summary.count} completed requests`}>
      <div className="stat-row">
        {/* No fixed threshold makes a TTFT "bad" - acceptable latency depends on prompt size and
            slot count - so these report without an alarm colour. */}
        <Stat value={seconds(summary.ttft.p50)} label="ttft p50" hint="ttft" />
        <Stat value={seconds(summary.ttft.p90)} label="ttft p90" hint="percentiles" />
        <Stat value={seconds(summary.ttft.p99)} label="ttft p99" hint="percentiles" />
        <Stat value={seconds(summary.queueWait.p50)} label="queue p50" hint="queueWait" />
        <Stat value={seconds(summary.queueWait.p90)} label="queue p90" hint="queueWait" />
        <Stat
          value={`${summary.decodeTokensPerSecond.p50.toFixed(1)}`}
          unit="tok/s"
          label="decode p50"
          hint="perSequenceRate"
          tone="accent"
        />
      </div>

      <div className="latency__split">
        <div className="eyebrow">
          <Term k="ttftSplit">time to first token, decomposed</Term>
        </div>
        <StackedBar
          segments={PHASES.map((phase) => ({
            label: phase.label,
            value: share[phase.key],
            color: phase.color,
            hint: phase.hint,
            display: `${percent(share[phase.key], 1)} of summed TTFT`,
          }))}
          height={10}
        />
        <Legend
          items={PHASES.map((phase) => ({
            label: phase.label,
            color: phase.color,
            hint: phase.hint,
          }))}
        />
        <p className="panel__footnote">
          {PHASES.map((phase) => `${phase.label} ${percent(share[phase.key], 1)}`).join(' · ')}.
          TTFT is dominated by <strong>{dominant.label}</strong>
          {dominant.key === 'queue'
            ? ' — requests are waiting for a slot, not being computed, so more slots (-np) or shorter generations move this before any kernel work does.'
            : ' — prompts are genuinely being computed; check cache reuse and whether prompts share a prefix the slot KV or the prompt cache could have kept.'}
        </p>
      </div>

      <div className="latency__split">
        <div className="eyebrow">
          <Term k="finishReason">finish reasons</Term>
          {summary.truncated > 0 ? ` · ${summary.truncated} truncated` : ''}
        </div>
        <StackedBar segments={finishSegments} />
        <Legend
          items={finishSegments.map((segment) => ({
            label: `${segment.label} ${segment.value}`,
            color: segment.color,
            hint: segment.hint,
          }))}
        />
      </div>

      <table className="table">
        <thead>
          <tr>
            <th>tokens</th>
            <th className="numeric">
              <Term k="percentiles">p50</Term>
            </th>
            <th className="numeric">
              <Term k="percentiles">p90</Term>
            </th>
            <th className="numeric">max</th>
          </tr>
        </thead>
        <tbody>
          {(
            [
              ['prompt', summary.promptTokens],
              ['cached', summary.cachedTokens],
              ['computed prefill', summary.prefillTokens],
              ['generated', summary.generated],
            ] as const
          ).map(([label, values]) => (
            <tr key={label}>
              <td>{label === 'cached' ? <Term k="cacheReuse">cached</Term> : label}</td>
              <td className="numeric emphasis">{count(values.p50)}</td>
              <td className="numeric emphasis">{count(values.p90)}</td>
              <td className="numeric emphasis">{count(values.max)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {accept === null ? null : (
        <p className="panel__footnote">
          <Term k="draftAccept">Draft acceptance</Term> {percent(accept, 1)} ·{' '}
          {count(summary.speculative.accepted)} of {count(summary.speculative.drafted)} drafted over{' '}
          {count(summary.speculative.verifSteps)} verification steps
          {summary.speculative.perPosition.length > 0
            ? ` · accepted per draft position ${summary.speculative.perPosition
                .map((value) => count(value))
                .join(' / ')}`
            : null}
        </p>
      )}
    </Panel>
  )
}

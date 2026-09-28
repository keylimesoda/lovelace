import { StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term, Tooltip } from '../components/tooltip'
import { Empty, Panel, Pill } from '../components/ui'
import type { RejectionSummary } from '../lib/derive'
import { clock, count, joules, percent, seconds } from '../lib/format'
import type { RequestDoneRecord, RequestRejectedRecord, RequestStartRecord } from '../lib/records'

const PHASES = [
  { key: 'queue', color: CHART.warning, hint: 'Waiting for a free slot.' },
  {
    key: 'prefill',
    color: CHART.violet,
    hint: 'Slot prompt processing: prompt tokens no cache covered.',
  },
  {
    key: 'decode',
    color: CHART.accent,
    hint: 'Generating tokens, first token to last.',
  },
] as const

const FINISH_COLOR: Record<string, string> = {
  eos: CHART.muted,
  word: CHART.muted,
  limit: CHART.warning,
  cancelled: CHART.dim,
  error: CHART.danger,
}

type Row =
  { kind: 'done'; record: RequestDoneRecord } | { kind: 'rejected'; record: RequestRejectedRecord }

/** Per-request time split, drawn on the same phase colors the latency panel uses. */
function Waterfall({ record }: { record: RequestDoneRecord }) {
  const t = record.timings_ms
  const values: Record<string, number> = {
    queue: t.queue_wait / 1000,
    prefill: t.prefill / 1000,
    decode: t.decode / 1000,
  }
  return (
    <StackedBar
      height={5}
      segments={PHASES.map((phase) => ({
        label: phase.key,
        value: values[phase.key]!,
        color: phase.color,
        hint: phase.hint,
        display: `${seconds(values[phase.key]!)} of ${seconds(t.total / 1000)} total`,
      }))}
    />
  )
}

/** Everything the record knows about one request, shown on its timestamp. */
function detail(record: RequestDoneRecord): string {
  const { request, result, speculative, energy } = record
  const lines: string[] = [
    `task ${request.id_task} · slot ${request.id_slot ?? '—'} · ${request.task_type} · ${request.endpoint}${request.stream ? ' · stream' : ''}`,
    `finish: ${result.finish_reason}${result.truncated ? ' (truncated)' : ''}`,
    `cached ${count(result.n_cached_tokens)} of ${count(result.n_prompt_tokens)} prompt tokens, computed ${count(result.n_prefill_tokens)}`,
    `ctx used ${count(result.n_ctx_used)} of ${count(request.n_ctx_slot)}`,
  ]
  if (record.rates.prefill_tps !== null) {
    lines.push(`prefill ${record.rates.prefill_tps.toFixed(0)} tok/s`)
  }
  if (speculative) {
    lines.push(
      `draft: ${speculative.n_accepted}/${speculative.n_drafted} accepted (${
        speculative.accept_rate === null ? '—' : percent(speculative.accept_rate)
      })`,
    )
  }
  if (energy) {
    lines.push(
      `energy: ${joules(energy.total_joules)} (prefill ${joules(energy.prefill_joules)}, decode ${joules(energy.decode_joules)}), attributed`,
    )
  }
  if (result.error) lines.push(`error: ${result.error}`)
  return lines.join('\n')
}

export function RequestsPanel({
  requests,
  rejected,
  rejections,
  active,
}: {
  requests: RequestDoneRecord[]
  rejected: RequestRejectedRecord[]
  rejections: RejectionSummary
  active: RequestStartRecord[]
}) {
  // Completed and rejected requests interleave in arrival order; seq is the server's own order.
  const rows: Row[] = [
    ...requests.map((record) => ({ kind: 'done' as const, record })),
    ...rejected.map((record) => ({ kind: 'rejected' as const, record })),
  ]
    .sort((a, b) => b.record.seq - a.record.seq)
    .slice(0, 14)

  return (
    <Panel
      title="Requests"
      hint="ttftSplit"
      className="panel--wide"
      note={
        <>
          {active.length > 0 ? <Pill tone="accent">{active.length} in flight</Pill> : null}{' '}
          {rejections.count > 0 ? (
            <Term k="rejected">
              <Pill tone="danger">
                {rejections.count} rejected ·{' '}
                {Object.entries(rejections.byErrorType)
                  .map(([type, n]) => `${type} ${n}`)
                  .join(' · ')}
              </Pill>
            </Term>
          ) : null}{' '}
          {`${requests.length} retained`}
        </>
      }
    >
      {rows.length === 0 ? (
        <Empty>no completed requests yet</Empty>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>at</th>
              <th>
                <Term k="endpoint">api</Term>
              </th>
              <th className="numeric">prompt</th>
              <th className="numeric">
                <Term k="cacheReuse">cached</Term>
              </th>
              <th className="numeric">gen</th>
              <th>
                <Term k="finishReason">end</Term>
              </th>
              <th className="numeric">
                <Term k="ttft">ttft</Term>
              </th>
              <th className="numeric">
                <Term k="perSequenceRate">tok/s</Term>
              </th>
              <th className="numeric">
                <Term k="requestEnergy">J</Term>
              </th>
              <th style={{ width: '20%' }}>
                <Term k="ttftSplit">phases</Term>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const record = row.record
              const key = `${record.server_instance_id}-${record.seq}`
              if (row.kind === 'rejected') {
                const rejection = row.record
                return (
                  <tr key={key}>
                    <td>{clock(rejection.timestamp_unix_ms)}</td>
                    <td>{rejection.request.endpoint}</td>
                    <td className="numeric emphasis">{count(rejection.request.n_prompt_tokens)}</td>
                    <td colSpan={7} style={{ color: CHART.danger }}>
                      <Tooltip
                        title={`rejected at ${rejection.stage}`}
                        body={<span className="tipwrap">{rejection.error.message}</span>}
                        className="tip--term"
                      >
                        rejected · {rejection.stage} · {rejection.error.type}
                      </Tooltip>
                      {rejection.queue_wait_ms === null
                        ? null
                        : ` · waited ${seconds(rejection.queue_wait_ms / 1000)}`}
                    </td>
                  </tr>
                )
              }
              const done = row.record
              const result = done.result
              return (
                <tr key={key}>
                  <td>
                    <Tooltip
                      title={clock(done.timestamp_unix_ms)}
                      body={<span className="tipwrap">{detail(done)}</span>}
                      className="tip--term"
                    >
                      {clock(done.timestamp_unix_ms)}
                    </Tooltip>
                  </td>
                  <td>{done.request.endpoint}</td>
                  <td className="numeric emphasis">{count(result.n_prompt_tokens)}</td>
                  <td className="numeric">
                    {result.n_prompt_tokens === 0
                      ? '—'
                      : percent(result.n_cached_tokens / result.n_prompt_tokens)}
                  </td>
                  <td className="numeric emphasis">{count(result.n_completion_tokens)}</td>
                  <td style={{ color: FINISH_COLOR[result.finish_reason] ?? CHART.muted }}>
                    {result.finish_reason}
                  </td>
                  <td className="numeric emphasis">
                    {done.timings_ms.ttft === null ? '—' : seconds(done.timings_ms.ttft / 1000)}
                  </td>
                  <td className="numeric">
                    {done.rates.decode_tps === null ? '—' : done.rates.decode_tps.toFixed(0)}
                  </td>
                  <td className="numeric">
                    {done.energy === null ? '—' : done.energy.total_joules.toFixed(1)}
                  </td>
                  <td>
                    <Waterfall record={done} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
      <p className="panel__footnote">
        Phase bars are proportional to each request&apos;s own total, coloured queue / prefill /
        decode. Hover a timestamp for slot, cache, draft, and energy detail, and a rejection for the
        server&apos;s error message. Per-request energy is an attribution at the server-wide rate,
        not a measurement of the request alone.
      </p>
    </Panel>
  )
}

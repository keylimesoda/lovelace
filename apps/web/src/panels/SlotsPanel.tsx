import { Term, Tooltip } from '../components/tooltip'
import { Empty, Meter, Panel, StatusDot, type Tone } from '../components/ui'
import { count, percent } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { SlotState, SlotTelemetry } from '../lib/telemetry'

const STATE_TONE: Record<SlotState, Tone> = {
  idle: 'neutral',
  started: 'warning',
  processing_prompt: 'warning',
  done_prompt: 'accent',
  generating: 'accent',
  wait_other: 'warning',
}

export function SlotsPanel({
  slots,
  replay,
}: {
  slots: SlotTelemetry[] | undefined
  replay: boolean
}) {
  if (!slots || slots.length === 0) {
    return (
      <Panel title="Slots" hint={replay ? 'replayMode' : 'slots'}>
        <Empty>{replay ? 'live only — slot state is not recorded' : 'server not attached'}</Empty>
      </Panel>
    )
  }

  return (
    <Panel
      title="Slots"
      hint="slots"
      note={`${slots.filter((slot) => slot.is_processing).length} busy of ${slots.length}`}
    >
      <table className="table">
        <thead>
          <tr>
            <th />
            <th>
              <Term k="slotState">slot</Term>
            </th>
            <th className="numeric">prompt</th>
            <th className="numeric">
              <Term k="slotCached">cached</Term>
            </th>
            <th className="numeric">gen</th>
            <th className="numeric">used</th>
            <th className="numeric" style={{ width: 72 }}>
              <Term k="slotCtx">ctx</Term>
            </th>
          </tr>
        </thead>
        <tbody>
          {slots.map((slot) => {
            const fill = slot.n_ctx === 0 ? 0 : slot.n_ctx_used / slot.n_ctx
            return (
              <tr key={slot.id}>
                <td>
                  <StatusDot tone={STATE_TONE[slot.state] ?? 'neutral'} />
                </td>
                <td className="emphasis">
                  {slot.id}
                  <Tooltip
                    title={slot.state}
                    body={
                      slot.speculative
                        ? `Speculative decoding is active on this slot. ${GLOSSARY.slotState.body}`
                        : GLOSSARY.slotState.body
                    }
                    className="slots__state tip--term"
                  >
                    {slot.state.replace('_', ' ')}
                    {slot.speculative ? ' · spec' : ''}
                  </Tooltip>
                </td>
                <td className="numeric emphasis">{count(slot.n_prompt_tokens)}</td>
                <td className="numeric">
                  {slot.n_prompt_tokens === 0
                    ? '—'
                    : percent(slot.n_cached_tokens / slot.n_prompt_tokens)}
                </td>
                <td className="numeric">{count(slot.n_decoded)}</td>
                <td className="numeric">{count(slot.n_ctx_used)}</td>
                <td className="numeric">
                  <Tooltip
                    title={`slot ${slot.id}${slot.id_task === null ? '' : ` · task ${slot.id_task}`}`}
                    body={`${count(slot.n_ctx_used)} of ${count(slot.n_ctx)} cells (${percent(fill)}) · ${count(slot.n_prefill_tokens)} prompt tokens computed`}
                    className="tip--term"
                  >
                    <Meter fraction={fill} color={fill > 0.9 ? 'var(--warning)' : 'var(--blue)'} />
                  </Tooltip>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="panel__footnote">
        An idle slot keeps its KV, so <Term k="slotCtx">ctx</Term> stays filled after a request
        finishes: a follow-up prompt that shares the prefix is assigned back to it and reuses those
        cells without compute. <Term k="slotCached">cached</Term> is the share of the current prompt
        served from that reuse or from the prompt cache.
      </p>
    </Panel>
  )
}

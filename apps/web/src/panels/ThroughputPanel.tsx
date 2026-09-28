import { BandChart, Legend, SampleChart } from '../components/charts'
import { CHART } from '../components/echart'
import { Info, Term } from '../components/tooltip'
import { Empty, Panel } from '../components/ui'
import { rate, seconds } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { ThroughputRecord } from '../lib/records'
import { bands, samples, span, weightedRate } from '../lib/series'

export function ThroughputPanel({
  records,
  slots,
}: {
  records: ThroughputRecord[]
  /** Slot count, used as the ceiling for the batch and occupancy charts. */
  slots: number | undefined
}) {
  const prefill = bands(records, (record) => record.throughput_tokens_per_second.prefill)
  const decode = bands(records, (record) => record.throughput_tokens_per_second.decode)
  const window = span(records)

  // Prefill runs an order of magnitude faster than decode, so a shared linear axis renders decode
  // as a flat sliver. Each series gets its own scale and the time axis is pinned to a common
  // domain, which keeps them comparable in time without one flattening the other.
  const domain: [number, number] | undefined =
    records.length === 0
      ? undefined
      : [
          Math.min(...prefill.map((band) => band.start)),
          Math.max(...prefill.map((band) => band.end)),
        ]

  // Mean busy slots per llama_decode call: an interval mean like the rates, so a band too, and
  // what explains a change in aggregate decode.
  const batch = bands(records, (record) => record.batch.mean_busy_slots ?? 0)

  const report = [
    { name: 'prefill', bands: prefill, color: CHART.violet },
    { name: 'decode (all slots)', bands: decode, color: CHART.accent },
    { name: 'busy slots / decode', bands: batch, color: CHART.blue },
  ]

  // Decode means are taken over intervals that actually generated. A mean across the whole window
  // would divide by idle time and report a rate the server never sustained.
  const decoding = records.filter((record) => record.tokens.decode > 0)
  const meanDecode = weightedRate(decoding, (r) => r.throughput_tokens_per_second.decode)
  const meanBatch = weightedRate(decoding, (r) => r.batch.mean_busy_slots ?? 0)
  // Aggregate ÷ mean busy slots is the rate one sequence sees.
  const perSequence = meanBatch > 0 ? meanDecode / meanBatch : 0

  let prefillTokens = 0
  let decodeTokens = 0
  let prefillLoop = 0
  let decodeLoop = 0
  for (const record of records) {
    prefillTokens += record.tokens.prefill
    decodeTokens += record.tokens.decode
    prefillLoop += record.loop_seconds.prefill
    decodeLoop += record.loop_seconds.decode
  }
  // Tokens over loop seconds spent in that phase; null when the phase never ran in the window.
  const prefillPhase = prefillLoop > 0 ? prefillTokens / prefillLoop : null
  const decodePhase = decodeLoop > 0 ? decodeTokens / decodeLoop : null

  return (
    <Panel
      title="Throughput"
      hint="intervalBands"
      className="panel--wide"
      note={
        records.length === 0 ? undefined : `${seconds(window)} window · ${records.length} samples`
      }
    >
      {records.length === 0 ? (
        <Empty>waiting for a reporting interval with activity</Empty>
      ) : (
        <>
          <div className="throughput__row">
            <span className="eyebrow" style={{ color: CHART.violet }}>
              <Term k="prefillRate">prefill</Term>
            </span>
            <span className="chart__caption">
              mean {rate(weightedRate(records, (r) => r.throughput_tokens_per_second.prefill))}{' '}
              tok/s
              {prefillPhase === null ? null : (
                <>
                  {' · '}
                  <Term k="phaseRate">{rate(prefillPhase)} while prefilling</Term>
                </>
              )}
            </span>
          </div>
          <BandChart
            label="Prefill tokens per second"
            unit="tok/s"
            height={86}
            domain={domain}
            report={report}
            series={[{ name: 'prefill', bands: prefill, color: CHART.violet, opacity: 0.85 }]}
          />

          <div className="throughput__row">
            <span className="eyebrow" style={{ color: CHART.accent }}>
              <Term k="decodeRate">decode · all slots</Term>
            </span>
            <span className="chart__caption">
              {decoding.length === 0 ? (
                'no generated tokens yet'
              ) : (
                <>
                  {rate(meanDecode)} tok/s ·{' '}
                  <Term k="perSequenceRate">{rate(perSequence)} per seq</Term>
                  {decodePhase === null ? null : (
                    <>
                      {' · '}
                      <Term k="phaseRate">{rate(decodePhase)} while decoding</Term>
                    </>
                  )}
                </>
              )}
            </span>
          </div>
          <BandChart
            label="Decode tokens per second, summed across slots"
            unit="tok/s"
            height={86}
            domain={domain}
            report={report}
            series={[
              { name: 'decode (all slots)', bands: decode, color: CHART.accent, opacity: 0.9 },
            ]}
          />

          <div className="throughput__row">
            <span className="eyebrow" style={{ color: CHART.blue }}>
              <Term k="busySlotsPerDecode">busy slots / decode</Term>
            </span>
            <span className="chart__caption">
              {decoding.length === 0
                ? 'no generated tokens yet'
                : `${meanBatch.toFixed(2)} slots per llama_decode while generating`}
            </span>
          </div>
          <BandChart
            label="Mean busy slots per llama_decode call"
            unit="slots"
            height={64}
            domain={domain}
            report={report}
            ceiling={slots}
            series={[
              { name: 'busy slots / decode', bands: batch, color: CHART.blue, opacity: 0.8 },
            ]}
          />

          <div className="throughput__row">
            <span className="eyebrow">
              <Term k="snapshotSeries">slots and queue</Term>
            </span>
          </div>
          <SampleChart
            label="Busy slots, slots processing prompt, and deferred tasks at each report"
            height={80}
            domain={domain}
            series={[
              {
                name: 'deferred',
                samples: samples(records, (record) => record.scheduler.queue_deferred),
                color: CHART.warning,
              },
              {
                name: 'processing prompt',
                samples: samples(records, (record) => record.scheduler.slots_processing_prompt),
                color: CHART.violet,
              },
              {
                name: 'busy',
                samples: samples(records, (record) => record.scheduler.slots_busy),
                color: CHART.blue,
              },
            ]}
            legend={
              <Legend
                items={[
                  { label: 'busy', color: CHART.blue, hint: GLOSSARY.slots.body },
                  {
                    label: 'processing prompt',
                    color: CHART.violet,
                    hint: GLOSSARY.processingPrompt.body,
                  },
                  { label: 'deferred', color: CHART.warning, hint: GLOSSARY.queueDeferred.body },
                ]}
              />
            }
            caption={
              <>
                shared time axis; bands are interval means, points are snapshots{' '}
                <Info k="snapshotSeries" />
              </>
            }
          />
          <p className="panel__footnote">
            Decode is the sum across every busy slot, so it rises with concurrency even when each
            request gets no faster; divide by the busy slots per llama_decode call for the rate one
            sequence sees. Prefill is scaled separately because it runs roughly an order of
            magnitude faster. Bands cover each record&apos;s own reporting interval, and the server
            skips intervals with no activity, so a wide band is idle time folded forward rather than
            a long sustained rate. The &quot;while&quot; rates divide by loop seconds spent in that
            phase, so they are the phase&apos;s own speed independent of idle and host time.
          </p>
        </>
      )}
    </Panel>
  )
}

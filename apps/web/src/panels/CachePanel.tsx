import { Legend, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term } from '../components/tooltip'
import { Empty, Meter, Panel, Pill, Stat } from '../components/ui'
import type { RequestSummary } from '../lib/derive'
import { bytes, count, percent } from '../lib/format'
import { GLOSSARY, type GlossaryKey } from '../lib/glossary'
import type { CacheView } from '../lib/telemetry'

function Occupancy({
  name,
  hint,
  used,
  capacity,
  display,
  color,
}: {
  name: string
  hint: GlossaryKey
  used: number
  /** 0 means unbounded: the meter stays empty rather than claiming a fill it cannot know. */
  capacity: number
  display: string
  color: string
}) {
  const fraction = capacity === 0 ? 0 : used / capacity
  const tone = fraction > 0.95 ? 'danger' : fraction > 0.85 ? 'warning' : 'neutral'
  return (
    <div className="cache__tier">
      <div className="cache__tier-head">
        <span className="cache__tier-name">
          <Term k={hint}>{name}</Term>
        </span>
        <span className="cache__tier-fill">{display}</span>
      </div>
      <Meter
        fraction={fraction}
        color={tone === 'danger' ? 'var(--danger)' : tone === 'warning' ? 'var(--warning)' : color}
      />
    </div>
  )
}

function cacheRamLabel(mib: number | null): string {
  if (mib === null) return 'unknown'
  if (mib < 0) return 'unlimited'
  if (mib === 0) return 'disabled'
  return `${count(mib)} MiB`
}

export function CachePanel({
  cache,
  summary,
  replay,
}: {
  cache: CacheView | undefined
  summary: RequestSummary
  replay: boolean
}) {
  if (!cache) {
    return (
      <Panel title="Cache" hint="kvCells">
        <Empty>{replay ? 'no server_start record in the loaded log' : 'server not attached'}</Empty>
      </Panel>
    )
  }

  const kv = cache.kv
  const prompt = cache.promptCache
  const cellsFraction = kv.cells_used === null ? 0 : kv.cells_used / Math.max(1, kv.cells_total)

  const reuseSegments = [
    {
      label: 'cached',
      value: summary.totalCachedTokens,
      color: CHART.accent,
      hint: GLOSSARY.cacheReuse.body,
      display: `${count(summary.totalCachedTokens)} of ${count(summary.totalPromptTokens)} prompt tokens`,
    },
    {
      label: 'computed',
      value: summary.totalPrefillTokens,
      color: CHART.violet,
      hint: GLOSSARY.energyPrefill.body,
      display: `${count(summary.totalPrefillTokens)} of ${count(summary.totalPromptTokens)} prompt tokens`,
    },
  ]

  return (
    <Panel
      title="Cache"
      hint="kvCells"
      note={
        kv.cells_used !== null && cellsFraction > 0.95 ? (
          <Pill tone="warning">kv cells at capacity</Pill>
        ) : (
          `${kv.cache_type_k}/${kv.cache_type_v}${kv.unified ? ' · unified' : ''}`
        )
      }
    >
      <div className="stat-row">
        <Stat
          value={percent(summary.cacheReuse)}
          label="reuse"
          hint="cacheReuse"
          tone={summary.cacheReuse > 0.5 ? 'accent' : 'neutral'}
        />
        <Stat
          value={kv.cells_used === null ? '—' : percent(cellsFraction)}
          label="kv cells"
          hint="kvCells"
          tone={cellsFraction > 0.95 ? 'danger' : 'neutral'}
        />
        <Stat
          value={prompt === null ? '—' : prompt.enabled ? bytes(prompt.bytes_used) : 'off'}
          label="prompt cache"
          hint="promptCache"
        />
        <Stat
          value={prompt === null || !prompt.enabled ? '—' : count(prompt.entries)}
          label="entries"
          hint="promptCache"
        />
      </div>

      <div className="cache__tiers">
        {kv.cells_used === null ? (
          <div className="cache__tier">
            <div className="cache__tier-head">
              <span className="cache__tier-name">
                <Term k="kvCells">kv cells</Term>
              </span>
              <span className="cache__tier-fill">
                {count(kv.cells_total)} total · occupancy live only
              </span>
            </div>
          </div>
        ) : (
          <Occupancy
            name="kv cells"
            hint="kvCells"
            used={kv.cells_used}
            capacity={kv.cells_total}
            display={`${count(kv.cells_used)} / ${count(kv.cells_total)} · ${count(kv.n_ctx_slot)} per slot`}
            color={CHART.accent}
          />
        )}
        {prompt === null ? (
          <div className="cache__tier">
            <div className="cache__tier-head">
              <span className="cache__tier-name">
                <Term k="promptCache">prompt cache</Term>
              </span>
              <span className="cache__tier-fill">
                <Term k="cacheRam">--cache-ram</Term> {cacheRamLabel(cache.cacheRamMib)} · occupancy
                live only
              </span>
            </div>
          </div>
        ) : !prompt.enabled ? (
          <div className="cache__tier">
            <div className="cache__tier-head">
              <span className="cache__tier-name">
                <Term k="promptCache">prompt cache</Term>
              </span>
              <span className="cache__tier-fill">disabled</span>
            </div>
          </div>
        ) : (
          <>
            <Occupancy
              name="prompt cache"
              hint="promptCache"
              used={prompt.bytes_used}
              capacity={prompt.bytes_limit}
              display={`${bytes(prompt.bytes_used)} / ${
                prompt.bytes_limit === 0 ? 'unlimited' : bytes(prompt.bytes_limit)
              } · ${count(prompt.entries)} entries`}
              color={CHART.blue}
            />
            <Occupancy
              name="cached tokens"
              hint="promptCache"
              used={prompt.tokens}
              capacity={prompt.tokens_limit}
              display={`${count(prompt.tokens)} / ${
                prompt.tokens_limit === 0 ? 'unlimited' : count(prompt.tokens_limit)
              }`}
              color={CHART.blue}
            />
          </>
        )}
      </div>

      {summary.count === 0 ? (
        <Empty>no completed requests in the retained window</Empty>
      ) : (
        <div className="cache__sources">
          <div className="eyebrow">
            <Term k="cacheReuse">prompt tokens by source</Term> ({summary.count} requests)
          </div>
          <StackedBar segments={reuseSegments} />
          <Legend
            items={reuseSegments.map((s) => ({ label: s.label, color: s.color, hint: s.hint }))}
          />
        </div>
      )}

      <p className="panel__footnote">
        <Term k="kvUnified">{kv.unified ? 'Unified KV' : 'Per-slot KV'}</Term>:{' '}
        {count(kv.cells_total)} cells
        {kv.unified
          ? ' shared by every slot, so one long sequence can use cells another slot is not using.'
          : `, ${count(kv.n_ctx_slot)} per slot.`}{' '}
        Idle slots keep their KV until another prompt needs the cells, so a high fill after traffic
        stops is retained prefixes, not leaked state. The <Term k="promptCache">prompt cache</Term>{' '}
        holds saved slot states in host RAM and restores one when a new prompt shares its prefix;
        both paths show up as cached tokens.
      </p>
    </Panel>
  )
}

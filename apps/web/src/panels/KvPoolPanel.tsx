import { Legend, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Info, Term } from '../components/tooltip'
import { Empty, Meter, Panel, Stat } from '../components/ui'
import { bytes, count, percent } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { LovelaceTelemetry, VramBudget } from '../lib/lovelace'
import type { ServerStartRecord, ThroughputRecord } from '../lib/records'
import { latest } from '../lib/series'

const VRAM_PARTS = [
  { key: 'model_bytes', label: 'model', color: CHART.blue, hint: GLOSSARY.lvVramModel.body },
  {
    key: 'context_bytes',
    label: 'context',
    color: CHART.accent,
    hint: GLOSSARY.lvVramContext.body,
  },
  {
    key: 'compute_bytes',
    label: 'compute',
    color: CHART.violet,
    hint: GLOSSARY.lvVramCompute.body,
  },
  {
    key: 'unaccounted_bytes',
    label: 'unaccounted',
    color: CHART.warning,
    hint: GLOSSARY.lvVramUnaccounted.body,
  },
  { key: 'free_bytes', label: 'free', color: CHART.line, hint: GLOSSARY.lvVramFree.body },
] as const

function Vram({ vram }: { vram: VramBudget }) {
  return (
    <>
      <div className="eyebrow">
        <Term k="lvVram">ggml vram breakdown</Term>
      </div>
      {vram.devices.map((device) => {
        const total = Math.max(1, device.total_bytes)
        return (
          <div className="memory__kv" key={device.name}>
            <div className="eyebrow">
              {device.name}
              {device.description ? ` · ${device.description}` : ''} · {bytes(device.total_bytes)}
            </div>
            <StackedBar
              height={10}
              segments={VRAM_PARTS.map((part) => ({
                label: part.label,
                value: device[part.key],
                color: part.color,
                hint: part.hint,
                display: `${bytes(device[part.key])} · ${percent(device[part.key] / total, 1)} of device`,
              }))}
            />
          </div>
        )
      })}
      <Legend
        items={VRAM_PARTS.map((part) => ({
          label: part.label,
          color: part.color,
          hint: part.hint,
        }))}
      />
      <table className="table">
        <tbody>
          <tr>
            <td>
              <Term k="lvVramHost">host buffers</Term>
            </td>
            <td className="numeric emphasis">
              model {bytes(vram.host.model_bytes)} · context {bytes(vram.host.context_bytes)} ·
              compute {bytes(vram.host.compute_bytes)}
            </td>
          </tr>
        </tbody>
      </table>
    </>
  )
}

/**
 * lovelace's KV pool in cells (used, reserved by the KV budget, free), the prompt cache, and the
 * ggml per-device VRAM breakdown. Reads only the `lovelace` extension.
 */
export function KvPoolPanel({
  ext,
  records,
  serverStart,
  replay,
}: {
  ext: LovelaceTelemetry | undefined
  records: ThroughputRecord[]
  serverStart: ServerStartRecord | null
  replay: boolean
}) {
  const config = serverStart?.lovelace?.config
  // Occupancy: live from /telemetry, else the latest throughput snapshot of a replayed log.
  const snapshot = latest(records.filter((record) => record.lovelace !== undefined))?.lovelace?.kv
  const occupancy = ext?.kv ?? snapshot
  const vram = ext?.vram ?? serverStart?.lovelace?.vram ?? null
  const promptCache = ext?.prompt_cache

  if (!occupancy && !config && !vram) {
    return (
      <Panel title="KV pool" hint="lvKvPool">
        <Empty>
          not reported {replay ? 'in this log' : 'by this server'} <Info k="lvNotReported" />
        </Empty>
      </Panel>
    )
  }

  const unified = ext?.kv?.unified ?? config?.kv_unified
  const perSlot = ext?.kv?.n_ctx_slot ?? config?.n_ctx_slot
  const typeK = ext?.kv?.cache_type_k ?? config?.cache_type_k
  const typeV = ext?.kv?.cache_type_v ?? config?.cache_type_v
  const cellsTotal = occupancy?.cells_total ?? config?.n_ctx ?? 0
  const used = occupancy?.cells_used ?? 0
  const reserved = Math.max(used, occupancy?.cells_reserved ?? 0)
  const total = Math.max(1, cellsTotal)
  const free = Math.max(0, cellsTotal - reserved)

  return (
    <Panel
      title="KV pool"
      hint="lvKvPool"
      note={
        typeK
          ? `${typeK === typeV ? typeK : `${typeK}/${typeV}`} · ${unified ? 'unified' : 'per slot'}`
          : undefined
      }
    >
      <div className="stat-row">
        <Stat
          value={occupancy ? count(used) : '—'}
          label="used cells"
          hint="lvKvUsed"
          tone="accent"
        />
        <Stat
          value={occupancy ? count(reserved) : '—'}
          label="reserved"
          hint="lvKvReserved"
          tone={reserved / total > 0.95 ? 'warning' : 'neutral'}
        />
        <Stat value={count(cellsTotal)} label="total cells" hint="lvKvPool" />
        <Stat
          value={perSlot === undefined ? '—' : count(perSlot)}
          label="per slot"
          hint="lvKvUnified"
        />
      </div>

      {occupancy ? (
        <>
          <StackedBar
            height={10}
            segments={[
              {
                label: 'used',
                value: used,
                color: CHART.accent,
                hint: GLOSSARY.lvKvUsed.body,
                display: `${count(used)} cells · ${percent(used / total)} of pool`,
              },
              {
                label: 'reserved',
                value: reserved - used,
                color: CHART.blue,
                hint: GLOSSARY.lvKvReserved.body,
                display: `${count(reserved - used)} cells reserved beyond use · ${percent(
                  (reserved - used) / total,
                )} of pool`,
              },
              {
                label: 'free',
                value: free,
                color: CHART.line,
                display: `${count(free)} cells · ${percent(free / total)} of pool`,
              },
            ]}
          />
          <Legend
            items={[
              { label: 'used', color: CHART.accent, hint: GLOSSARY.lvKvUsed.body },
              { label: 'reserved', color: CHART.blue, hint: GLOSSARY.lvKvReserved.body },
              { label: 'free', color: CHART.line },
            ]}
          />
          {!ext?.kv ? (
            <p className="panel__footnote">Snapshot of the last logged throughput report.</p>
          ) : null}
        </>
      ) : null}

      {promptCache ? (
        <div className="memory__kv">
          <div className="eyebrow">
            <Term k="lvPromptCache">prompt cache</Term>{' '}
            {promptCache.enabled
              ? `${bytes(promptCache.bytes_used)} of ${
                  promptCache.bytes_limit > 0 ? bytes(promptCache.bytes_limit) : 'unlimited'
                } · ${count(promptCache.entries)} entries · ${count(promptCache.tokens)} tokens`
              : 'off'}
          </div>
          {promptCache.enabled && promptCache.bytes_limit > 0 ? (
            <Meter fraction={promptCache.bytes_used / promptCache.bytes_limit} />
          ) : null}
        </div>
      ) : config ? (
        <p className="panel__footnote">
          <Term k="lvPromptCache">Prompt cache</Term>:{' '}
          {config.cache_ram_mib === 0
            ? 'off'
            : config.cache_ram_mib < 0
              ? 'unlimited'
              : `${count(config.cache_ram_mib)} MiB limit`}{' '}
          (--cache-ram); occupancy is live only.
        </p>
      ) : null}

      {vram && vram.devices.length > 0 ? <Vram vram={vram} /> : null}
    </Panel>
  )
}

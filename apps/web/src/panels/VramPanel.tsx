import { Legend, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term } from '../components/tooltip'
import { Empty, Panel, Stat } from '../components/ui'
import { bytes, percent } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { VramBudget, VramDevice } from '../lib/records'
import type { GpuTelemetry } from '../lib/telemetry'

const PARTS = [
  { key: 'model_bytes', label: 'model', color: CHART.blue, hint: GLOSSARY.vramModel.body },
  { key: 'context_bytes', label: 'context', color: CHART.accent, hint: GLOSSARY.vramContext.body },
  { key: 'compute_bytes', label: 'compute', color: CHART.violet, hint: GLOSSARY.vramCompute.body },
  {
    key: 'unaccounted_bytes',
    label: 'unaccounted',
    color: CHART.warning,
    hint: GLOSSARY.vramUnaccounted.body,
  },
  { key: 'free_bytes', label: 'free', color: CHART.line, hint: GLOSSARY.vramFree.body },
] as const

function Device({ device }: { device: VramDevice }) {
  const total = Math.max(1, device.total_bytes)
  return (
    <div className="memory__kv">
      <div className="eyebrow">
        {device.name}
        {device.description ? ` · ${device.description}` : ''} · {bytes(device.total_bytes)}
      </div>
      <StackedBar
        height={10}
        segments={PARTS.map((part) => ({
          label: part.label,
          value: device[part.key],
          color: part.color,
          hint: part.hint,
          display: `${bytes(device[part.key])} · ${percent(device[part.key] / total, 1)} of device`,
        }))}
      />
    </div>
  )
}

export function VramPanel({
  vram,
  gpu,
  replay,
}: {
  /** Live from /telemetry memory.vram, or the log's server_start.vram in replay. */
  vram: VramBudget | null | undefined
  gpu: GpuTelemetry | undefined
  replay: boolean
}) {
  if (!vram || vram.devices.length === 0) {
    return (
      <Panel title="VRAM" hint="vramBudget">
        <Empty>
          {vram === undefined
            ? replay
              ? 'no server_start record in the loaded log'
              : 'server not attached'
            : 'no device memory breakdown reported'}
        </Empty>
      </Panel>
    )
  }

  let model = 0
  let context = 0
  let compute = 0
  let free = 0
  for (const device of vram.devices) {
    model += device.model_bytes
    context += device.context_bytes
    compute += device.compute_bytes
    free += device.free_bytes
  }
  const host = vram.host

  return (
    <Panel
      title="VRAM"
      hint="vramBudget"
      note={
        gpu?.available && gpu.memory_used_bytes !== undefined
          ? `${bytes(gpu.memory_used_bytes)} resident on board now`
          : 'captured after model load'
      }
    >
      <div className="stat-row">
        <Stat value={bytes(model)} label="model" hint="vramModel" />
        <Stat value={bytes(context)} label="context" hint="vramContext" tone="accent" />
        <Stat value={bytes(compute)} label="compute" hint="vramCompute" />
        <Stat
          value={bytes(free)}
          label="free"
          hint="vramFree"
          tone={free < 512e6 ? 'warning' : 'neutral'}
        />
      </div>

      {vram.devices.map((device) => (
        <Device key={device.name} device={device} />
      ))}
      <Legend
        items={PARTS.map((part) => ({ label: part.label, color: part.color, hint: part.hint }))}
      />

      <table className="table">
        <thead>
          <tr>
            <th>
              <Term k="vramHost">host buffers</Term>
            </th>
            <th />
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>model</td>
            <td className="numeric emphasis">{bytes(host.model_bytes)}</td>
          </tr>
          <tr>
            <td>context</td>
            <td className="numeric emphasis">{bytes(host.context_bytes)}</td>
          </tr>
          <tr>
            <td>compute</td>
            <td className="numeric emphasis">{bytes(host.compute_bytes)}</td>
          </tr>
        </tbody>
      </table>
      <p className="panel__footnote">
        A static budget: llama.cpp reserves the KV cache and compute buffers when the context is
        created, so none of this grows with traffic. Current KV use is in the Cache panel.
      </p>
    </Panel>
  )
}

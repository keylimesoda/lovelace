import { useMemo, useRef } from 'react'

import { Pill, StatusDot, type Tone } from './components/ui'
import { summarizeRejections, summarizeRequests } from './lib/derive'
import { duration } from './lib/format'
import { cacheView } from './lib/telemetry'
import { useEngine } from './lib/use-engine'
import { CachePanel } from './panels/CachePanel'
import { EnergyPanel } from './panels/EnergyPanel'
import { GpuPanel } from './panels/GpuPanel'
import { Headline } from './panels/Headline'
import { LatencyPanel } from './panels/LatencyPanel'
import { RequestsPanel } from './panels/RequestsPanel'
import { SchedulerPanel } from './panels/SchedulerPanel'
import { SlotsPanel } from './panels/SlotsPanel'
import { ThroughputPanel } from './panels/ThroughputPanel'
import { VramPanel } from './panels/VramPanel'

const CONNECTION_TONE: Record<string, Tone> = {
  live: 'accent',
  replay: 'warning',
  connecting: 'warning',
  offline: 'danger',
}

export function App() {
  const { state, loadFile, resumeLive } = useEngine()
  const filePicker = useRef<HTMLInputElement>(null)

  // Summaries are pure over the retained record window; recomputing them on every 1 Hz telemetry
  // poll would be wasted work on a list that only changes when a request completes.
  const summary = useMemo(() => summarizeRequests(state.requests), [state.requests])
  const rejections = useMemo(() => summarizeRejections(state.rejected), [state.rejected])

  const start = state.serverStart
  const config = start?.config
  const replay = state.source === 'file'
  const slots = state.telemetry?.scheduler.slots_total ?? config?.n_parallel

  // Live telemetry wins; a replayed log reconstructs the configured capacities from server_start.
  const cache = useMemo(
    () => cacheView(state.telemetry, state.serverStart),
    [state.telemetry, state.serverStart],
  )
  const vram = state.telemetry ? state.telemetry.memory.vram : start?.vram

  const modelName = state.telemetry?.model_id ?? start?.model.name ?? 'llama-server'
  const speculative =
    config && config.speculative.types !== 'none' ? config.speculative.types : null

  return (
    <div className="console">
      <header className="topbar">
        <div className="topbar__identity">
          <StatusDot tone={CONNECTION_TONE[state.connection]} />
          <strong>{modelName}</strong>
          {config ? (
            <span className="topbar__config">
              {start?.model.ftype} · {config.n_parallel} slots ·{' '}
              {config.n_ctx.toLocaleString('en-US')} ctx{config.kv_unified ? ' unified' : ''} · kv{' '}
              {config.cache_type_k}/{config.cache_type_v} · fa {config.flash_attn}
              {speculative ? ` · spec ${speculative}` : ''}
            </span>
          ) : null}
        </div>

        <div className="topbar__actions">
          {replay ? (
            <Pill tone="warning">
              replay · {state.fileName}
              {state.droppedInstances > 0
                ? ` · ${state.droppedInstances} earlier instance${state.droppedInstances === 1 ? '' : 's'} skipped`
                : ''}
            </Pill>
          ) : state.telemetry ? (
            <span className="topbar__config">
              up {duration(state.telemetry.uptime_seconds)}
              {state.telemetry.events.records_dropped > 0
                ? ` · ${state.telemetry.events.records_dropped} stream records dropped`
                : ''}
            </span>
          ) : null}
          {state.error ? <Pill tone="danger">{state.error}</Pill> : null}
          <input
            ref={filePicker}
            type="file"
            accept=".jsonl,.json,application/jsonl,text/plain"
            className="sr-only"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) void loadFile(file)
              event.target.value = ''
            }}
          />
          <button className="button" type="button" onClick={() => filePicker.current?.click()}>
            load jsonl
          </button>
          {replay ? (
            <button className="button" type="button" onClick={resumeLive}>
              go live
            </button>
          ) : null}
        </div>
      </header>

      <Headline
        telemetry={state.telemetry}
        records={state.throughput}
        summary={summary}
        slotsConfigured={config?.n_parallel}
      />

      <main className="grid">
        <ThroughputPanel records={state.throughput} slots={slots} />
        <SchedulerPanel
          telemetry={state.telemetry}
          records={state.throughput}
          slotsConfigured={config?.n_parallel}
        />
        <LatencyPanel summary={summary} />
        <EnergyPanel
          records={state.throughput}
          summary={summary}
          gpu={state.telemetry?.gpu}
          energyAvailable={state.telemetry?.energy.available ?? start?.gpu.energy_available}
          replay={replay}
        />
        <GpuPanel gpu={state.telemetry?.gpu} history={state.gpu} replay={replay} />
        <CachePanel cache={cache} summary={summary} replay={replay} />
        <VramPanel vram={vram} gpu={state.telemetry?.gpu} replay={replay} />
        <SlotsPanel slots={state.telemetry?.slots} replay={replay} />
        <RequestsPanel
          requests={state.requests}
          rejected={state.rejected}
          rejections={rejections}
          active={state.active}
        />
      </main>
    </div>
  )
}

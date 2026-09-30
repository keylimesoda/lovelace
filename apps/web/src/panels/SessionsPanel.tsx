import { BandChart, Legend, SampleChart } from '../components/charts'
import { CHART } from '../components/echart'
import { Info, Term } from '../components/tooltip'
import { Empty, Meter, Panel, Stat } from '../components/ui'
import { bytes, count, percent, seconds } from '../lib/format'
import { GLOSSARY, type GlossaryKey } from '../lib/glossary'
import {
  milliseconds,
  summarizeSessionHistory,
  type LovelaceSession,
  type LovelaceTelemetry,
} from '../lib/lovelace'
import type { ServerStartRecord, ThroughputRecord } from '../lib/records'
import { bands, samples } from '../lib/series'

const MIB = 1024 * 1024
const MAX_ROWS = 24

const TIER_COLOR = { gpu: CHART.accent, ram: CHART.blue, nvme: CHART.violet } as const

const TIER_LABEL: Record<LovelaceSession['tier'], string> = {
  gpu: 'gpu',
  ram: 'ram',
  demoting: 'ram → nvme',
  disk: 'nvme',
  promoting: 'nvme → ram',
}

function gbps(bytesMoved: number, secondsSpent: number): string {
  return secondsSpent > 0 ? `${(bytesMoved / secondsSpent / 1e9).toFixed(2)} GB/s` : '—'
}

function Tier({
  name,
  hint,
  sessions,
  used,
  capacity,
  unit,
  color,
}: {
  name: string
  hint: GlossaryKey
  sessions: number
  used: number | null
  /** 0 = unlimited; null = not known in this source. */
  capacity: number | null
  unit: (value: number) => string
  color: string
}) {
  const bounded = used !== null && capacity !== null && capacity > 0
  const fraction = bounded ? used / capacity : 0
  const tone = fraction > 0.95 ? 'var(--danger)' : fraction > 0.85 ? 'var(--warning)' : color
  return (
    <div className="cache__tier">
      <div className="cache__tier-head">
        <span className="cache__tier-name">
          <Term k={hint}>{name}</Term>
        </span>
        <span className="cache__tier-fill">
          {used === null ? '' : unit(used)}
          {used === null
            ? ''
            : capacity === null
              ? ''
              : capacity === 0
                ? ' / unlimited'
                : ` / ${unit(capacity)}`}
          {bounded ? <em>{percent(fraction)}</em> : null}
        </span>
      </div>
      <Meter fraction={fraction} color={tone} />
      <div className="cache__tier-foot">
        {count(sessions)} session{sessions === 1 ? '' : 's'}
      </div>
    </div>
  )
}

/**
 * lovelace's per-agent session store: where each session's KV lives (GPU slot, RAM mirror, NVMe
 * file) and what moving it between the tiers cost. Reads only the `lovelace` extension.
 */
export function SessionsPanel({
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
  const live = ext?.sessions ?? null
  const history = summarizeSessionHistory(records)

  if (!live && !history) {
    const reported = ext !== undefined || records.some((record) => record.lovelace !== undefined)
    return (
      <Panel title="Sessions" hint="lvSessions">
        <Empty>
          {reported ? (
            'no session store on this server (--session-ram not set)'
          ) : (
            <>
              not reported {replay ? 'in this log' : 'by this server'} <Info k="lvNotReported" />
            </>
          )}
        </Empty>
      </Panel>
    )
  }

  const configured = serverStart?.engine.continuation_cache
  const counts = live ?? history!.latest
  const gpuCells = live
    ? live.sessions.reduce((total, session) => total + session.resident_cells, 0)
    : null
  const promotions = live ? live.promotions : history!.promotions
  const demotions = live ? live.demotions : history!.demotions
  const prefetches = live ? live.prefetches : history!.prefetches
  const prefetchHits = live ? live.prefetch_hits : history!.prefetchHits
  const ioRead = live ? live.io_bytes_read : history!.ioBytesRead
  const ioWritten = live ? live.io_bytes_written : history!.ioBytesWritten
  const withSessions = records.filter((record) => record.lovelace?.sessions)

  return (
    <Panel
      title="Sessions"
      hint="lvSessions"
      className="panel--wide"
      note={
        live
          ? `${count(live.n_sessions)} sessions · ${count(live.n_io)} file jobs in flight`
          : `${count(history!.intervals)} intervals · totals over the window`
      }
    >
      <div className="stat-row">
        <Stat value={count(promotions)} label="promotions" hint="lvPromotions" />
        <Stat
          value={count(demotions)}
          label="demotions"
          hint="lvDemotions"
          tone={demotions > 0 ? 'warning' : 'neutral'}
        />
        <Stat
          value={`${count(prefetchHits)}/${count(prefetches)}`}
          label="prefetch hits"
          hint="lvPrefetch"
          tone="accent"
        />
        <Stat value={bytes(ioRead)} label="nvme read" hint="lvNvmeIo" />
        <Stat value={bytes(ioWritten)} label="nvme written" hint="lvNvmeIo" />
      </div>

      <div className="cache__tiers">
        <Tier
          name="gpu"
          hint="lvTierGpu"
          sessions={counts.n_gpu}
          used={gpuCells}
          capacity={ext?.kv?.cells_total ?? null}
          unit={(value) => `${count(value)} cells`}
          color={TIER_COLOR.gpu}
        />
        <Tier
          name="ram"
          hint="lvTierRam"
          sessions={counts.n_ram}
          used={counts.ram_bytes}
          capacity={live ? live.arena_bytes : configured ? configured.l2_capacity_mib * MIB : null}
          unit={bytes}
          color={TIER_COLOR.ram}
        />
        <Tier
          name="nvme"
          hint="lvTierNvme"
          sessions={counts.n_disk}
          used={counts.disk_bytes}
          capacity={
            live ? live.disk_limit_bytes : configured ? configured.l3_capacity_mib * MIB : null
          }
          unit={bytes}
          color={TIER_COLOR.nvme}
        />
      </div>
      {live ? (
        <p className="panel__footnote">
          <Term k="lvNvmeIo">NVMe IO</Term> since start: read {bytes(live.io_bytes_read)} in{' '}
          {seconds(live.io_read_s)} ({gbps(live.io_bytes_read, live.io_read_s)}) · written{' '}
          {bytes(live.io_bytes_written)} in {seconds(live.io_write_s)} (
          {gbps(live.io_bytes_written, live.io_write_s)}) · {count(live.syncs)} mirror syncs,{' '}
          {bytes(live.sync_bytes)} · {count(live.stale_syncs)} deferred for arena room ·{' '}
          {count(live.deleted)} deleted
        </p>
      ) : null}

      {withSessions.length > 0 ? (
        <>
          <SampleChart
            label="Sessions per tier at each report"
            series={[
              {
                name: 'gpu',
                samples: samples(withSessions, (r) => r.lovelace!.sessions!.n_gpu),
                color: TIER_COLOR.gpu,
              },
              {
                name: 'ram',
                samples: samples(withSessions, (r) => r.lovelace!.sessions!.n_ram),
                color: TIER_COLOR.ram,
              },
              {
                name: 'nvme',
                samples: samples(withSessions, (r) => r.lovelace!.sessions!.n_disk),
                color: TIER_COLOR.nvme,
              },
            ]}
            legend={
              <Legend
                items={[
                  { label: 'gpu', color: TIER_COLOR.gpu, hint: GLOSSARY.lvTierGpu.body },
                  { label: 'ram', color: TIER_COLOR.ram, hint: GLOSSARY.lvTierRam.body },
                  { label: 'nvme', color: TIER_COLOR.nvme, hint: GLOSSARY.lvTierNvme.body },
                ]}
              />
            }
            caption={
              <>
                snapshots, not interval means <Info k="lvSessionHistory" />
              </>
            }
          />
          <BandChart
            label="Session file moves per interval"
            series={[
              {
                name: 'promotions',
                bands: bands(withSessions, (r) => r.lovelace!.sessions!.promotions),
                color: CHART.violet,
              },
              {
                name: 'demotions',
                bands: bands(withSessions, (r) => r.lovelace!.sessions!.demotions),
                color: CHART.warning,
              },
            ]}
            unit="sessions"
            stack
            integer
            legend={
              <Legend
                items={[
                  { label: 'promotions', color: CHART.violet, hint: GLOSSARY.lvPromotions.body },
                  { label: 'demotions', color: CHART.warning, hint: GLOSSARY.lvDemotions.body },
                ]}
              />
            }
            caption={
              <>
                interval counts, not snapshots <Info k="intervalBands" />
              </>
            }
          />
        </>
      ) : null}

      {live ? (
        live.sessions.length === 0 ? (
          <Empty>no sessions yet</Empty>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>
                  <Term k="lvSessionTable">session</Term>
                </th>
                <th>tier</th>
                <th className="numeric">slot</th>
                <th className="numeric">
                  <Term k="lvResidentCells">cells</Term>
                </th>
                <th className="numeric">
                  <Term k="lvMirror">mirror</Term>
                </th>
                <th className="numeric">
                  <Term k="lvSessionFile">file</Term>
                </th>
                <th className="numeric">
                  <Term k="lvRestores">restores</Term>
                </th>
                <th>
                  <Term k="lvStale">stale</Term>
                </th>
              </tr>
            </thead>
            <tbody>
              {live.sessions.slice(0, MAX_ROWS).map((session) => (
                <tr key={String(session.id)}>
                  <td className="slots__digest">{String(session.id)}</td>
                  <td>{TIER_LABEL[session.tier] ?? session.tier}</td>
                  <td className="numeric">{session.slot >= 0 ? session.slot : '—'}</td>
                  <td className="numeric emphasis">
                    {session.resident_cells > 0 ? count(session.resident_cells) : '—'}
                  </td>
                  <td className="numeric">
                    {count(session.mirror_tokens)} tok · {bytes(session.mirror_bytes)}
                  </td>
                  <td className="numeric">
                    {session.file_bytes > 0 ? bytes(session.file_bytes) : '—'}
                  </td>
                  <td className="numeric">
                    {session.restores > 0
                      ? `${count(session.restores)} · ${milliseconds(session.last_restore_ms)}`
                      : '—'}
                  </td>
                  <td>{session.stale ? 'stale' : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      ) : (
        <p className="panel__footnote">
          The per-session list is live only; a log carries the tier counts and interval moves.
        </p>
      )}
      {live && live.sessions.length > MAX_ROWS ? (
        <p className="panel__footnote">
          showing {MAX_ROWS} of {count(live.sessions.length)} sessions
        </p>
      ) : null}
    </Panel>
  )
}

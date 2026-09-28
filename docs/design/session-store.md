# Per-agent session store: every agent keeps its own full context

Requirement (user): each of the 4-6 agents has its own full-size context (up to 262,144 tokens). None is ever
lost to eviction. Agents swap between GPU and host fast enough that the switch cost is a small fraction of a request.

## Facts this design rests on (measured on this box)

- One full session at q5_0 KV: 262,144 x 22.5 KB = 5.9 GB of attention KV, plus 151 MB of Gated DeltaNet state
  (48 layers x 48 heads x 128x128 fp32) and the conv state. Six full sessions come to ~36 GB.
- GPU working set: the unified pool fits 262,144 cells next to the weights, MTP, ReplaySSM arena and compute
  buffers (bench/results/main-gate). That is 1 full session, or about 3-4 at typical agent depth (60-100k).
- Host: 60 GB RAM, PCIe 4.0 x16 (~25 GB/s pinned). The LUKS-encrypted btrfs root is the overflow tier.
- Attention KV cells are append-only for a session: once written, a cell never changes (rollback only truncates
  the tail). The recurrent state changes every token, but it is small.
- The common prefix across OMP agents (system prompt + tool schemas) is ~7.4k tokens (bench/replay corpus;
  confirm on real captures).
- The KV bytes (q5_0: packed nibbles + high bits + fp16 scales) are near-maximum entropy. Lossless compression
  is only worth building if a measurement on a real session dump shows > 15% on the whole session.

## Design

1. **Session identity.** A session is keyed by the prefix hash of its token stream after the shared prefix
   (radix match on block hashes, block = 256 tokens). A request joins the session with the longest matching
   prefix; a new session starts when nothing matches beyond the shared prefix. No client change needed.
2. **Host mirror (write-back, append-only).** Every session has a mirror in a pinned host arena: KV cells in
   token order per layer (the same bytes as the GPU cells), plus the latest recurrent state and its checkpoints.
   After each request the server copies ONLY the cells appended since the last sync, plus the recurrent state,
   asynchronously on the copy stream. Parking a session is then free (drop its GPU cells), because the mirror
   is already current. A rollback past a synced point truncates the mirror's length.
3. **Restore.** To make a session resident, copy mirror -> cells (async, coalesced per layer), restore the
   recurrent state, and prefill only the suffix the mirror doesn't have. Prefetch: when the scheduler knows
   which session runs next (queue head) and the pool has room, start its restore while other slots decode.
4. **Tiering, never dropping.** The pinned arena is sized by `--session-ram MiB` (default ~30 GB). When full, the
   least-recently-used session's mirror is demoted to `--session-dir` (NVMe) in large sequential writes, and
   promoted back on use. A session is deleted only after `--session-ttl` of inactivity (default 24 h) or when
   the disk budget is exceeded, and that is logged.
5. **Shared prefix pinned in VRAM.** The longest prefix shared by at least 2 sessions (the system prompt + tools)
   is kept resident as cells shared by all sequences (llama_memory_seq_cp semantics). Its recurrent-state
   checkpoint sits at the prefix end. Mirrors store only cells after the shared prefix. A new subagent starts
   at the prefix checkpoint instead of prefilling ~7k tokens.
6. **Scheduling.** Request-granular fair queue, FIFO across agents. The KV budget (existing) admits a request
   when its session plus reservation fits. Resident sessions are preferred when that doesn't break FIFO fairness
   by more than one slot. Victims are the resident sessions not currently executing, LRU; their eviction costs
   nothing because the mirror is current. Any single agent can use the whole pool (262k).
7. **Observability.** Per-session: tier (gpu/ram/disk), resident cells, mirror bytes, last-sync lag, restore
   count, and bytes and ms per restore; arena and disk usage. Exposed in /telemetry and the request records.

## Acceptance

- 6-agent replay on the REAL depth distribution, with sessions driven past 200k: zero errors, zero sessions
  dropped, zero full re-prefills except first-contact prefills.
- Park cost ~0 (only async delta sync). Restore of a 200k+ session <= ~0.4 s wall on the critical path, or
  hidden by prefetch.
- Aggregate tok/s and TTFT p50/p90 reported against the current build (bench/results/main-gate/replay-6a).

## Parked capacity option: large-batch quantized-KV FA without the f16 copy (branch `feat/fattn-prefill`)

Measured (q5_0, -ub 1024): target compute buffer 1698 -> 782 MiB, and the production config starts at >= 360k
cells (vs 327k with `-ubd 32` alone). Cost: pp2048 -11% at d65536 and -15% at d196608, because K/V tiles are
dequantized once per Q tile. Not merged. Revisit if production telemetry shows residency (not swap) as the
bottleneck: either convert per KV head (speed-neutral, recovers ~3/4 of the buffer) or pipeline the quantized
tile loads with cp.async.

// Definitions for the lovelace panels: everything they show comes from the `lovelace` extension of
// the observability API (docs/observability-parity.md). Merged into GLOSSARY, so the same Term and
// Info components resolve them and the keys share one namespace; every key is prefixed `lv`.

export const LOVELACE_GLOSSARY = {
  // --- sessions ----------------------------------------------------------------------------
  lvSessions: {
    title: 'Agent sessions',
    body: 'lovelace keeps one session per agent conversation. Its KV lives in a slot while resident on the GPU, is mirrored write-back into a pinned host-RAM arena (--session-ram) so the slot can be freed at no cost, and is demoted to a file in --session-dir when the arena runs short. A returning turn restores it at PCIe speed instead of re-prefilling. A session is deleted only by its TTL or the disk budget. From /telemetry lovelace.sessions; null when no session store is configured.',
  },
  lvTierGpu: {
    title: 'GPU tier',
    body: 'Sessions bound to a slot: their KV cells are in VRAM and a follow-up turn continues without any copy. The cells are part of the KV pool and count toward its use.',
  },
  lvTierRam: {
    title: 'RAM tier',
    body: 'Sessions whose mirror is in the pinned host arena, set by --session-ram. Bytes are mirror bytes against the arena size; a full arena forces demotions to NVMe or leaves a resident session stale.',
  },
  lvTierNvme: {
    title: 'NVMe tier',
    body: 'Sessions only in their file under --session-dir. Restoring one first promotes it back into the arena (a file read), then copies it to VRAM. Bytes are file bytes against the disk limit; 0 means unlimited.',
  },
  lvPromotions: {
    title: 'Promotions',
    body: 'Session files read back from NVMe into the RAM arena, ahead of a restore. Each one is a disk read on the path of the request that needed it.',
  },
  lvDemotions: {
    title: 'Demotions',
    body: 'Session mirrors written from the RAM arena to NVMe to free arena room. Demotion happens off the request path; a rising rate means the RAM tier is too small for the working set.',
  },
  lvPrefetch: {
    title: 'Prefetches',
    body: 'Restores started for the session of the request at the head of the queue while it still waits for a slot or KV cells, so it starts without waiting for the copies. A hit is a prefetched session its request then used.',
  },
  lvNvmeIo: {
    title: 'NVMe IO',
    body: 'Bytes the session store read (promotions) and wrote (demotions), with the worker-thread seconds spent on them. The ratio is the effective disk throughput the tier gets.',
  },
  lvSessionTable: {
    title: 'Sessions',
    body: 'Every session the store holds, as of the latest /telemetry snapshot. Tier demoting and promoting mean a file write or read is in flight; the session is still in RAM while demoting.',
  },
  lvResidentCells: {
    title: 'Resident cells',
    body: 'KV cells the session holds in its slot right now. 0 for a session that is not on the GPU.',
  },
  lvMirror: {
    title: 'Mirror',
    body: 'Tokens and bytes of the session’s write-back copy in the RAM arena: the KV deltas of every sync, the recurrent-state snapshot and parked context checkpoints.',
  },
  lvSessionFile: {
    title: 'File',
    body: 'Size of the session’s file under --session-dir; empty when it was never demoted.',
  },
  lvRestores: {
    title: 'Restores',
    body: 'Times the session was copied back into a slot, and the device time of the copies of the last one.',
  },
  lvStale: {
    title: 'Stale',
    body: 'The slot holds cells the mirror has not synced because the arena had no room at the time. If the slot is taken before the next sync, those tokens are re-prefilled on the session’s next turn.',
  },
  lvSessionHistory: {
    title: 'Sessions per tier',
    body: 'Session counts per tier at each throughput report. They are snapshots taken when the report was emitted, drawn as points.',
  },

  // --- queue and waits ---------------------------------------------------------------------
  lvKvBudget: {
    title: 'KV budget',
    body: 'lovelace admits a request only when a slot is free and the KV pool can hold its reservation. Otherwise it waits in the admission queue; idle sessions are parked to host memory to make room, and a generating slot can be preempted at a token boundary and resumed later. From /telemetry lovelace.scheduler.kv_budget.',
  },
  lvQueueWaiting: {
    title: 'Queue waiting',
    body: 'Requests in the KV-budget admission queue right now, waiting for a slot or for KV cells.',
  },
  lvPreempted: {
    title: 'Preempted slots',
    body: 'Generating slots paused at a token boundary to free KV cells, whose state is in host memory until they resume.',
  },
  lvWaited: {
    title: 'Waited',
    body: 'Requests that had to wait for admission, by the first reason: no idle slot, or not enough free KV cells for their reservation. Cumulative since start live; summed interval counts in replay.',
  },
  lvParks: {
    title: 'Parks and unparks',
    body: 'A park copies an idle session’s KV from VRAM to host memory and frees its cells; an unpark copies it back when its next request launches. Parks overlap other slots’ decoding, so they cost bandwidth rather than loop time.',
  },
  lvPreemptions: {
    title: 'Preemptions and resumes',
    body: 'A preemption pauses a generating slot at a token boundary and moves its state to host memory so another request can run; a resume brings it back. Frequent preemption means the pool is oversubscribed for the concurrency.',
  },
  lvTransferRate: {
    title: 'Transfer rate',
    body: 'State bytes copied divided by the device time of the copies, in GB/s. Near the PCIe link rate means the copies are bandwidth-bound, as they should be.',
  },
  lvRecompute: {
    title: 'Recomputed tokens',
    body: 'Tokens re-prefilled to recover a sequence whose state was lost (dropped without a host copy, or a rollback the memory refused). Compute that a larger arena or pool would have saved.',
  },
  lvKvOverflows: {
    title: 'KV overflows',
    body: 'llama_decode found no free cells despite the budget. It should stay 0; anything else is a budget accounting bug.',
  },
  lvWaitReason: {
    title: 'Wait reason',
    body: 'Why each completed request waited for admission: none, slot (all slots busy) or kv (not enough free cells for its reservation), with its queue wait (arrival to slot launch) in ms. From request_done lovelace.wait_reason and queue_wait_ms.',
  },
  lvQueueHistory: {
    title: 'Queue and preemption',
    body: 'Admission queue length, deferred tasks and preempted slots at each throughput report. Snapshots at emit time, drawn as points.',
  },
  lvLoopSplit: {
    title: 'Server loop wall clock',
    body: 'The single server loop, split by what it was doing: computing prompt tokens, generating, host work between llama_decode calls, and waiting for tasks. The four sum to the window, so growth in host or idle is time no slot advanced. Live over the rolling window; in replay summed over the logged intervals.',
  },
  lvLoopPrefill: {
    title: 'Prefill',
    body: 'Loop time in llama_decode computing prompt tokens.',
  },
  lvLoopDecode: {
    title: 'Decode',
    body: 'Loop time in llama_decode generating tokens for slots past their prompt.',
  },
  lvLoopHost: {
    title: 'Host',
    body: 'Loop time with work pending but outside llama_decode: tokenization, batch building, sampling, speculative bookkeeping, and issuing or waiting for parks and restores.',
  },
  lvLoopIdle: {
    title: 'Idle',
    body: 'Loop time with no task to run.',
  },

  // --- KV pool -----------------------------------------------------------------------------
  lvKvPool: {
    title: 'KV pool',
    body: 'The llama.cpp KV cache, in cells (one cell holds the keys and values of one token). With --kv-unified every slot draws from one pool of n_ctx cells; otherwise each slot owns n_ctx_slot cells. From /telemetry lovelace.kv; in replay from the throughput snapshots and server_start config.',
  },
  lvKvUsed: {
    title: 'Used cells',
    body: 'Sum of the slots’ sequence lengths. Idle slots keep their KV until it is parked or evicted, so this does not drop when requests finish.',
  },
  lvKvReserved: {
    title: 'Reserved cells',
    body: 'Used cells plus the unused part of the reservations of running requests: what the KV budget has promised. Admission waits for kv when a new reservation would not fit beside it.',
  },
  lvKvUnified: {
    title: 'Unified KV',
    body: 'With --kv-unified every slot draws from one shared pool, so a slot may grow past n_ctx / n_parallel while others are short. Without it each slot owns a fixed stream of n_ctx_slot cells.',
  },
  lvPromptCache: {
    title: 'Prompt cache',
    body: 'Host-RAM store of saved slot states (--cache-ram). A request whose prompt shares a prefix with an entry restores it instead of recomputing. A limit of 0 reads as unlimited.',
  },
  lvVram: {
    title: 'ggml VRAM breakdown',
    body: 'The ggml backend memory breakdown captured once after model and context creation: per device, model weights, context (KV + recurrent state), compute buffers, memory not allocated by llama.cpp, and free. Static for the life of the process. From lovelace.vram; in replay from server_start lovelace.vram.',
  },
  lvVramModel: {
    title: 'Model',
    body: 'Weight buffers resident on the device: the offloaded layers at their quantized type.',
  },
  lvVramContext: {
    title: 'Context',
    body: 'The KV cache (n_ctx cells at the K/V cache types) plus any recurrent state, reserved in full at startup.',
  },
  lvVramCompute: {
    title: 'Compute',
    body: 'Compute graph buffers the backend scheduler reserved for the worst-case batch (n_ubatch).',
  },
  lvVramUnaccounted: {
    title: 'Unaccounted',
    body: 'Device memory in use that llama.cpp did not allocate as model, context or compute: other processes, the CUDA context, allocator fragmentation.',
  },
  lvVramFree: {
    title: 'Free',
    body: 'Device memory the driver reported free at capture time.',
  },
  lvVramHost: {
    title: 'Host buffers',
    body: 'Model, context and compute buffers held in host RAM: non-offloaded layers, host-side KV, CPU compute buffers.',
  },
  lvNotReported: {
    title: 'Not reported',
    body: 'This server or log carries no lovelace extension (a ninfer server, or a lovelace block without this section), so there is nothing to show rather than a zero.',
  },
} as const

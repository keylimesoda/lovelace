// Definitions for every term the dashboard shows.
//
// One place so a reading and its explanation cannot drift, and so the wording stays specific to
// llama-server rather than generic inference vocabulary. Each entry says what the number
// measures and, where it is useful, what to conclude when it moves.

export interface Definition {
  title: string
  body: string
}

export const GLOSSARY = {
  // --- Headline ------------------------------------------------------------

  decodeRate: {
    title: 'Decode rate, all slots',
    body:
      'Generated tokens committed per second over the reporting interval, summed across every ' +
      'slot. It rises with concurrency even when each request gets no faster; divide by the ' +
      'mean busy slots per llama_decode call for the rate one sequence sees.',
  },
  prefillRate: {
    title: 'Prefill rate',
    body:
      'Prompt tokens actually computed per second over the interval. Tokens reused from a ' +
      "slot's retained KV or restored from the prompt cache are not computed and do not count, " +
      'so a falling prefill rate with steady traffic is the cache working, not the server slowing.',
  },
  slots: {
    title: 'Slots',
    body:
      'A slot is one of the --parallel (-np) sequences the server loop interleaves; each owns a ' +
      'sequence id in the KV cache. Busy slots hold a task right now; the rest are idle but ' +
      'keep their KV until another prompt evicts it.',
  },
  queueDeferred: {
    title: 'Deferred tasks',
    body:
      'Tasks waiting in the server queue because no slot was free when they arrived. They are ' +
      'retried every time a slot is released; a persistently non-zero value means the server is ' +
      'saturated for its slot count.',
  },
  queueWait: {
    title: 'Queue wait',
    body:
      'Wall time from arrival at the HTTP handler to the slot launching the task. Long queue ' +
      'wait means every slot was busy, not that compute is slow: more slots or shorter ' +
      'generations move it, faster kernels do not.',
  },
  ttft: {
    title: 'Time to first token',
    body:
      'Arrival at the HTTP handler to the first generated token: queue wait, then slot prompt ' +
      'processing, then the first sampling step. Requests that never produced a token ' +
      '(embeddings, early cancellation) have no TTFT and are excluded, not counted as zero.',
  },
  percentiles: {
    title: 'Percentiles',
    body:
      'Truncating nearest-rank over the retained completed requests: p50 is the middle request, ' +
      'p99 the slowest 1% boundary. The retained window is at most a few hundred requests, so ' +
      'p99 is a statement about that sample, not about the population.',
  },
  cacheReuse: {
    title: 'Cache reuse',
    body:
      'Cached prompt tokens over prompt tokens across completed requests (n_cached_tokens / ' +
      "n_prompt_tokens). Cached tokens come from the slot's retained KV prefix or from a " +
      'prompt-cache restore and cost no compute. High on a multi-turn session that resends its ' +
      'history; near zero on fresh prompts.',
  },
  draftAccept: {
    title: 'Draft acceptance',
    body:
      'Accepted over drafted speculative tokens. Every rejected draft position is compute spent ' +
      'on a token that was thrown away; the per-position curve shows how quickly the draft ' +
      'diverges from the target model. Reads as absent when speculative decoding is off.',
  },
  boardPower: {
    title: 'Board power',
    body:
      'Mean board draw over the last second, from the NVML energy counter delta rather than the ' +
      'driver power estimate. Absent on boards without an energy counter.',
  },

  // --- Throughput panel ----------------------------------------------------

  busySlotsPerDecode: {
    title: 'Busy slots per decode call',
    body:
      'Mean number of busy slots per llama_decode call. One forward pass serves every slot in ' +
      'the batch, so the aggregate decode rate is roughly this times the per-sequence rate.',
  },
  perSequenceRate: {
    title: 'Per-sequence rate',
    body:
      'Aggregate decode rate divided by the mean busy slots per llama_decode call: the rate one ' +
      'sequence sees. This is the figure a client experiences; the aggregate is the figure the ' +
      'hardware pays for.',
  },
  phaseRate: {
    title: 'Phase rate',
    body:
      'Tokens divided by loop seconds spent in that phase, rather than by the whole interval. ' +
      'It is the speed of the phase while it runs, independent of how much of the interval the ' +
      'loop spent idle or on host work.',
  },
  intervalBands: {
    title: 'Interval bands',
    body:
      'Each throughput record is an average over its own reporting interval (5 s by default), ' +
      'so it is drawn as a band whose width is that interval. The server skips intervals with ' +
      'no activity and folds the idle time into the next record, so a wide band is idle time ' +
      'folded forward, not a long sustained rate.',
  },
  snapshotSeries: {
    title: 'Snapshot series',
    body:
      'Slot and queue counts are read once, when the throughput record is emitted, not averaged ' +
      'over its interval, so they are drawn as points. A spike between two reports is invisible ' +
      'to this chart.',
  },
  processingPrompt: {
    title: 'Processing prompt',
    body:
      'Busy slots still evaluating their prompt. Prompt batches share llama_decode calls with ' +
      'generating slots, so many of these at once is what stretches every other slot’s ' +
      'inter-token latency.',
  },

  // --- GPU panel -----------------------------------------------------------

  gpuUtil: {
    title: 'GPU utilization',
    body:
      'Share of the driver sample period in which at least one kernel was running. Decode is ' +
      'launch- and bandwidth-bound, so a healthy decoding server often reads below 100% without ' +
      'being idle.',
  },
  gpuTemp: {
    title: 'Board temperature',
    body:
      'The sensor the driver uses for its throttle curve. What matters is whether the thermal ' +
      'throttle reasons light up, not the reading itself.',
  },
  gpuPower: {
    title: 'Board power',
    body:
      'NVML trailing-average board draw against the active power limit. Holding at the limit ' +
      'means the work is power-limited and clocks are being traded for watts.',
  },
  gpuClock: {
    title: 'SM clock',
    body:
      'Current against maximum shader clock. A clock well below its ceiling while utilization ' +
      'is high means a throttle reason is holding the board back, not the workload.',
  },
  gpuMemBw: {
    title: 'Memory controller utilization',
    body:
      'Share of the sample period the memory controller was busy. Decode reads every weight ' +
      'once per llama_decode call, so this, not GPU utilization, is the ceiling decode runs into.',
  },
  gpuThrottle: {
    title: 'Active throttle reasons',
    body:
      'Driver-reported reasons the board is below maximum clocks. gpu_idle and a user-set ' +
      'application clock are filtered out because they do not limit work the server wanted to do.',
  },
  boardMemory: {
    title: 'Board memory',
    body:
      'Everything resident on the board as NVML reports it, including other processes. The VRAM ' +
      'panel decomposes what llama.cpp itself allocated.',
  },

  // --- VRAM panel ----------------------------------------------------------

  vramBudget: {
    title: 'VRAM budget',
    body:
      'The ggml backend memory breakdown, captured once after the model and context were ' +
      'created. It is static for the life of the process: llama.cpp reserves the KV cache and ' +
      'compute buffers up front, so this does not grow with traffic.',
  },
  vramModel: {
    title: 'Model',
    body: 'Weight buffers resident on the device: the offloaded layers (-ngl) at their quantized type.',
  },
  vramContext: {
    title: 'Context',
    body:
      'The KV cache (n_ctx cells at --cache-type-k/-v) plus any recurrent state. Sized from ' +
      '--ctx-size at startup, so it is the full reservation, not current use.',
  },
  vramCompute: {
    title: 'Compute',
    body:
      'Compute graph buffers the backend scheduler reserved for the worst-case batch (n_ubatch). ' +
      'Larger ubatch or flash attention off both grow it.',
  },
  vramUnaccounted: {
    title: 'Unaccounted',
    body:
      'Device memory in use that llama.cpp did not allocate as model, context, or compute: ' +
      'other processes, the CUDA context, and allocator fragmentation.',
  },
  vramFree: {
    title: 'Free',
    body:
      'Device memory the driver reported free at capture time. Headroom for a larger context, ' +
      'another model, or another process.',
  },
  vramHost: {
    title: 'Host buffers',
    body:
      'Model, context, and compute buffers held in host RAM: non-offloaded layers, host-side ' +
      'KV when --no-kv-offload is set, and CPU compute buffers.',
  },

  // --- Cache panel ---------------------------------------------------------

  kvCells: {
    title: 'KV cells',
    body:
      'One cell holds the keys and values of one token position. llama.cpp sizes the cache at ' +
      'n_ctx cells; used is the sum of slot sequence lengths. Idle slots keep their KV so a ' +
      'matching follow-up prompt can reuse it, so used does not drop when requests finish.',
  },
  kvUnified: {
    title: 'Unified KV',
    body:
      'With --kv-unified every slot draws from one shared pool of n_ctx cells, so a slot may ' +
      'grow past n_ctx / n_parallel while others are short. Without it each slot owns a fixed ' +
      'stream of n_ctx_slot cells.',
  },
  promptCache: {
    title: 'Prompt cache',
    body:
      'Host-RAM store of saved slot states (--cache-ram). When a slot is reassigned, its state ' +
      'can be saved here; a later request whose prompt shares a prefix restores it instead of ' +
      'recomputing. Distinct from the KV cells, which hold only what the slots currently own.',
  },
  cacheRam: {
    title: '--cache-ram',
    body:
      'Prompt cache size limit in MiB. -1 is unlimited and 0 disables the prompt cache; a ' +
      'limit of 0 bytes in the live snapshot also reads as unlimited.',
  },

  // --- Scheduler panel -----------------------------------------------------

  loopSplit: {
    title: 'Server loop wall clock',
    body:
      'The single server loop, split by what it was doing over the rolling window: computing ' +
      'prompt tokens, generating, host work between llama_decode calls, and waiting for tasks. ' +
      'The four sum to the window, so any growth in host or idle is time no slot advanced.',
  },
  loopPrefill: {
    title: 'Prefill',
    body: 'Loop time in llama_decode computing prompt tokens.',
  },
  loopDecode: {
    title: 'Decode',
    body: 'Loop time in llama_decode generating tokens for slots past their prompt.',
  },
  loopHost: {
    title: 'Host',
    body:
      'Loop time with work pending but outside llama_decode: tokenization, batch building, ' +
      'sampling, speculative bookkeeping, prompt-cache save and restore. Energy for this time is ' +
      'split pro rata between prefill and decode.',
  },
  loopIdle: {
    title: 'Idle',
    body: 'Loop time with no task to run. Its share of board energy is the idle joules.',
  },
  loopPhase: {
    title: 'Current phase',
    body: 'What the server loop is doing at the instant of the snapshot.',
  },
  schedulerTotals: {
    title: 'Totals since start',
    body:
      'Cumulative counters from the server process: requests started, completed, and rejected, ' +
      'llama_decode calls, and the mean busy slots per call.',
  },

  // --- Slots panel ---------------------------------------------------------

  slotState: {
    title: 'Slot state',
    body:
      'idle: no task. started: task assigned, prompt not yet loaded. processing_prompt: ' +
      'evaluating prompt tokens. done_prompt: prompt finished, about to sample. generating: ' +
      'producing tokens. wait_other: parked until a sibling slot (n > 1 completions) is ready.',
  },
  slotCtx: {
    title: 'Slot context',
    body:
      'Sequence length the slot holds in the KV cache against its per-slot context n_ctx. A ' +
      'full slot must shift or truncate its context before it can grow further.',
  },
  slotCached: {
    title: 'Cached',
    body: 'Share of the current prompt that was reused rather than computed.',
  },

  // --- Latency / requests --------------------------------------------------

  ttftSplit: {
    title: 'TTFT split',
    body:
      'Summed time to first token divided into queue wait and slot prompt processing. A ' +
      'queue-dominated TTFT is a slot-count problem; a prefill-dominated one is a prompt-size or ' +
      'cache-reuse problem.',
  },
  finishReason: {
    title: 'Finish reason',
    body:
      'eos: the model emitted end-of-generation. word: a stop string matched. limit: n_predict ' +
      'or the slot context was exhausted. cancelled: the client went away. error: the slot ' +
      'failed mid-flight. done: an embedding or rerank task completed (no sampling).',
  },
  endpoint: {
    title: 'Endpoint',
    body:
      'API surface the request arrived on: native (/completion), oai_chat, oai_cmpl, oai_resp, ' +
      'oai_asr, oai_embd, or anthropic.',
  },
  rejected: {
    title: 'Rejected',
    body:
      'Requests that never produced a result. http: parse or tokenize failure. launch: refused ' +
      'when a slot tried to start it. prompt: refused before any compute, for example a prompt ' +
      'longer than the slot context.',
  },

  // --- Energy panel --------------------------------------------------------

  energyServed: {
    title: 'Energy served',
    body:
      'Board joules per prefill-plus-decode token across the window, idle loop time included: ' +
      'what the work actually cost. It degrades when the server is mostly idle even though ' +
      'nothing about the model changed, which is why the active figure exists too.',
  },
  energyActive: {
    title: 'Energy active',
    body:
      'Prefill and decode joules per token, leaving out the idle share. It tracks the schedule ' +
      'rather than the duty cycle and is the figure to compare between two builds.',
  },
  energyPrefill: {
    title: 'Energy prefill',
    body:
      'Prefill joules per computed prompt token. Cached tokens are not in the denominator, so a ' +
      'good cache does not flatter this figure by hiding work.',
  },
  energyDecode: {
    title: 'Energy decode',
    body:
      'Decode joules per committed generated token. Rejected draft tokens are not committed, so ' +
      'speculative decoding can raise tokens per second and joules per token at the same time.',
  },
  energyIdle: {
    title: 'Idle energy',
    body:
      "Board joules attributed to the loop's idle share of each interval: the cost of being up " +
      'between requests.',
  },
  idleWatts: {
    title: 'Idle draw',
    body:
      'Board power calibrated from sampling periods in which the loop was fully idle. A ' +
      'reference, not a subtraction: the split attributes energy by time share, not by baseline.',
  },
  energySplit: {
    title: 'Energy attribution',
    body:
      'The exact board counter delta for each interval is divided by loop phase time share: ' +
      'prefill, decode, and idle get their share, and host time is split between prefill and ' +
      'decode in proportion to their loop seconds. The three parts sum to the board total ' +
      'exactly; there is no residual.',
  },
  requestEnergy: {
    title: 'Request energy',
    body:
      "An attribution, not a measurement: the request's computed prompt tokens and generated " +
      'tokens priced at the server-wide prefill and decode J/token over its lifetime. Concurrent ' +
      'requests share the same mean rate.',
  },
  energyPerMillion: {
    title: 'Energy per million tokens',
    body:
      'The per-token figure restated in watt-hours over a million tokens, the denominator ' +
      'inference is priced in. Multiply by a local rate for a $/1M figure comparable to ' +
      'published prices; it is the same measurement, not a new one.',
  },

  // --- Modes ---------------------------------------------------------------

  replayMode: {
    title: 'Replay mode',
    body:
      'The dashboard is reading a JSONL file instead of a live server. Records and the ' +
      'server_start configuration replay; board sensors, slot state, KV and prompt-cache ' +
      'occupancy, and anything only the running server knows read as absent.',
  },
} as const satisfies Record<string, Definition>

export type GlossaryKey = keyof typeof GLOSSARY

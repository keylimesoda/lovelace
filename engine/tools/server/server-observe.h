#pragma once

// Observability for llama-server: the request log (--request-log-jsonl), the GET /events record stream,
// GET /telemetry, the /metrics and /slots bodies, and board energy attribution.
//
// The API is tensorninja/ninfer-4090's (f408990, request-log schema 20) as a superset: lovelace-only data lives
// under a `lovelace` key (lovelace.schema_version 2). Field mapping: docs/observability-parity.md.
//
// Threading contract (nothing measurable on the decode path):
//  - the server loop thread is the only writer of the phase clock, the counters and the snapshot; it performs
//    relaxed atomic stores, fixed-size field writes into the snapshot under one short mutex per step, and enqueues
//    POD records at request start/end
//  - a 10 Hz sampler thread reads the NVML energy counter and the phase clock, attributes energy deltas to
//    prefill / decode / idle by the loop's phase time share, and emits the throughput records
//  - a writer thread formats the records, appends them to the JSONL file and fans them out to the /events
//    subscribers
//  - /telemetry, /metrics and /slots are built on the HTTP threads from the snapshot; they never post a task to,
//    wake, or wait for the server loop
// No NVML call, JSON formatting, allocation or file I/O happens on the server loop thread per step.
//
// Payload schema: docs/observability.md

#include "server-common.h"
#include "server-sessions.h"
#include "server-task.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

enum server_obs_phase : int {
    SERVER_OBS_PHASE_IDLE    = 0, // the loop is waiting for tasks
    SERVER_OBS_PHASE_HOST    = 1, // the loop runs host code: scheduling, batching, sampling, detokenizing
    SERVER_OBS_PHASE_PREFILL = 2, // llama_decode() + sync, prompt rows
    SERVER_OBS_PHASE_DECODE  = 3, // llama_decode() + sync, generation and draft rows, draft model
    SERVER_OBS_PHASE_COUNT   = 4,
};

const char * server_obs_phase_name(server_obs_phase phase);

// the process command line (API keys redacted), recorded by main() for server_start.argv
void server_obs_set_argv(int argc, char ** argv);
const std::vector<std::string> & server_obs_argv();

// Wall clock of the server loop split by phase: single writer (the loop thread), seqlock readers.
// A compute interval is split between prefill and decode by the prompt-row share of its batch.
struct server_obs_clock {
    struct snapshot {
        std::array<int64_t, SERVER_OBS_PHASE_COUNT> ns = {};
        int64_t          t_ns  = 0;
        server_obs_phase phase = SERVER_OBS_PHASE_IDLE;
    };

    // writer API (loop thread only)
    void mark(server_obs_phase phase) {
        if (w_phase == phase && !w_compute) {
            return;
        }
        commit(phase, false, 0);
    }
    // the batch has n_rows rows, n_prompt of them are prompt rows
    void compute_begin(int32_t n_prompt, int32_t n_rows) {
        const uint32_t permille = n_rows > 0 ? (uint32_t) ((int64_t) n_prompt * 1000 / n_rows) : 0;
        commit(permille >= 500 ? SERVER_OBS_PHASE_PREFILL : SERVER_OBS_PHASE_DECODE, true, permille);
    }
    void compute_end() {
        commit(SERVER_OBS_PHASE_HOST, false, 0);
    }

    // reader API (any thread); includes the in-progress interval
    snapshot read() const;

    void init();

private:
    void commit(server_obs_phase next, bool compute, uint32_t permille);

    // writer-private copies
    server_obs_phase w_phase    = SERVER_OBS_PHASE_IDLE;
    bool             w_compute  = false;
    uint32_t         w_permille = 0;
    int64_t          w_t_ns     = 0;
    std::array<int64_t, SERVER_OBS_PHASE_COUNT> w_ns = {};

    // published state
    std::atomic<uint32_t> seq { 0 };
    std::atomic<uint32_t> cur { 0 }; // phase | compute << 8 | permille << 16
    std::atomic<int64_t>  t_last { 0 };
    std::array<std::atomic<int64_t>, SERVER_OBS_PHASE_COUNT> acc = {};
};

// KV budget scheduling totals, owned by the server loop (server_context_impl::sched_stats)
struct server_obs_sched_totals {
    uint64_t n_park         = 0; // idle sessions saved to host memory and evicted from VRAM
    uint64_t n_park_drop    = 0; // idle sessions evicted without a copy (no cache room)
    uint64_t n_sync         = 0; // session mirrors synced after a request
    uint64_t bytes_sync     = 0;
    uint64_t n_unpark       = 0; // sessions restored from host memory
    uint64_t n_preempt      = 0; // generating slots paused to host memory
    uint64_t n_resume       = 0;
    uint64_t n_grow         = 0; // reservations extended in flight
    uint64_t n_wait_slot    = 0; // tasks that had to wait for a slot
    uint64_t n_wait_kv      = 0; // tasks that had to wait for KV cells
    uint64_t n_rollback_fallback = 0; // truncations the memory refused, recovered from a checkpoint or by re-prefill
    uint64_t n_recompute_tokens  = 0; // tokens re-prefilled to recover a generating slot
    uint64_t n_kv_overflow  = 0; // llama_decode() found no free cells despite the budget
    uint64_t bytes_park     = 0;
    uint64_t bytes_unpark   = 0;
    int64_t  us_park        = 0; // device time of the copies
    int64_t  us_unpark      = 0;
    int64_t  us_park_host   = 0; // time the server loop spent issuing and waiting for them
    int64_t  us_unpark_host = 0;
};

// counters written by the loop thread (relaxed stores), read by the sampler and the HTTP threads
struct server_obs_counters {
    std::atomic<uint64_t> prefill_tokens  { 0 }; // computed prompt tokens
    std::atomic<uint64_t> decode_tokens   { 0 }; // committed generated tokens
    std::atomic<uint64_t> drafted         { 0 };
    std::atomic<uint64_t> accepted        { 0 };
    std::atomic<uint64_t> decode_calls    { 0 };
    std::atomic<uint64_t> busy_slot_calls { 0 }; // sum over decode calls of busy slots
    std::atomic<uint64_t> decode_rounds   { 0 }; // decode calls with generating slots
    std::atomic<uint64_t> decode_rows     { 0 }; // sum over those calls of generating slots
    std::atomic<uint64_t> prefill_steps   { 0 }; // decode calls with prompt rows

    // gauges, snapshot of the loop's own view
    std::atomic<int32_t> slots_busy      { 0 };
    std::atomic<int32_t> slots_prompt    { 0 };
    std::atomic<int32_t> slots_gen       { 0 };
    std::atomic<int32_t> queue_deferred  { 0 };
    std::atomic<int32_t> kv_cells_total     { 0 };
    std::atomic<int32_t> kv_cells_used      { 0 }; // cells held by all slots
    std::atomic<int32_t> kv_cells_reserved  { 0 }; // cells held or promised (max(used, reservation) per slot)
    std::atomic<int32_t> sched_waiting      { 0 }; // requests in the admission queue
    std::atomic<int32_t> sched_preempted    { 0 }; // slots paused to host memory
    std::atomic<int32_t> sessions_parked    { 0 }; // prompt cache entries

    // written from several threads
    std::atomic<uint64_t> req_started  { 0 };
    std::atomic<uint64_t> req_done     { 0 };
    std::atomic<uint64_t> req_rejected { 0 };
};

// single-writer increment, no read-modify-write bus lock
static inline void server_obs_add(std::atomic<uint64_t> & a, uint64_t n) {
    a.store(a.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

// cumulative energy/token state published by the sampler, used for per-request attribution
struct server_obs_energy_mark {
    bool     valid          = false;
    double   prefill_joules = 0.0;
    double   decode_joules  = 0.0;
    uint64_t prefill_tokens = 0;
    uint64_t decode_tokens  = 0;
    // last interval rates, used when a request is too short to span a sample
    double   prefill_j_per_token = 0.0;
    double   decode_j_per_token  = 0.0;
};

enum server_obs_record_kind : uint8_t {
    SERVER_OBS_REC_REQUEST_START,
    SERVER_OBS_REC_REQUEST_DONE,
    SERVER_OBS_REC_REQUEST_ERROR,
    SERVER_OBS_REC_SCHED,         // lovelace_sched: a KV state transfer between VRAM and host memory
    SERVER_OBS_REC_RAW, // pre-formatted body (server_start, throughput), the envelope is added by the writer
};

// POD-like request record, filled on the loop (or HTTP) thread and formatted by the writer thread
// all `const char *` fields must point to string literals
struct server_obs_record {
    server_obs_record_kind kind = SERVER_OBS_REC_RAW;
    int64_t t_unix_ms = 0;

    // request info
    int32_t      id_task    = -1;
    int32_t      id_slot    = -1;
    uint64_t     session_id = 0;
    int32_t      id_parent  = -1;
    const char * task_type  = "completion";
    const char * endpoint   = "native";
    bool         stream     = false;
    int32_t      n_prompt_tokens = 0;
    int32_t      n_predict  = -1;
    bool         n_predict_client = false; // the request set n_predict / max_tokens
    int32_t      n_ctx_slot = 0;
    bool         thinking   = false;       // the prompt opens a reasoning block
    float        temperature = 0.0f, top_p = 0.0f, min_p = 0.0f, presence_penalty = 0.0f, frequency_penalty = 0.0f;
    int32_t      top_k = 0;
    uint32_t     seed  = 0;
    double       queue_wait_ms = -1.0; // < 0 = unknown
    const char * wait_reason   = "none"; // KV budget: "none", "slot" or "kv"
    int32_t      n_unparked_tokens = 0; // tokens restored from host memory at launch
    int32_t      sched_tokens = 0;      // SCHED: tokens of the transferred sequence (bytes: restore_bytes, device ms:
                                        // restore_ms, loop host ms: park_ms, action: event)

    // session store (--session-ram): where the session (session_id) was at admission (gpu/ram/disk/new, nullptr = none)
    const char * session_tier    = nullptr;
    size_t       restore_bytes   = 0;     // bytes restored from its mirror for the request
    double       restore_ms      = 0.0;   // device time of those copies
    // phases after the admission, ms from it (-1 = n/a)
    double       park_ms          = 0.0;  // host time parking other sessions
    double       restore_issue_ms = 0.0;  // host time issuing the restore
    double       restore_ready_ms = -1.0; // the restore copies completed (0 = before the admission)
    double       prompt_start_ms  = -1.0; // the first batch with prompt tokens was queued
    bool         prefetch_hit    = false; // restored while the request was queued
    bool         sync_queued     = false; // the session mirror sync was queued at release

    // done
    const char * finish_reason = "stop_token"; // ninfer names
    bool     truncated          = false;
    uint64_t n_cached_tokens     = 0;
    uint64_t n_prefill_tokens    = 0;
    uint64_t n_completion_tokens = 0;
    int32_t  n_ctx_used          = 0;
    double   t_prefill_ms = 0.0;
    double   t_decode_ms  = 0.0;
    double   t_ttft_ms    = -1.0; // < 0 = no token
    double   t_total_ms   = 0.0;
    double   prefill_tps  = 0.0;
    double   decode_tps   = 0.0;
    uint64_t n_drafted     = 0;
    uint64_t n_accepted    = 0;
    uint64_t n_verif_steps = 0;
    std::vector<uint64_t> accepted_per_pos;
    bool     energy_valid   = false;
    double   prefill_joules = 0.0;
    double   decode_joules  = 0.0;

    // error
    const char * stage      = "http"; // http, launch, prompt, generation
    const char * error_type = "server";
    std::string  error_msg;

    // SERVER_OBS_REC_RAW: `body` is merged after the envelope, `ext` becomes the `lovelace` object
    const char * event  = "";
    json         body;
    json         ext;
};

// fixed-size view of one slot, written by the loop into the snapshot
struct server_obs_slot_view {
    static constexpr int MAX_CKPT = 32;

    int32_t      id            = 0;
    const char * state         = "idle";
    bool         is_processing = false;
    bool         speculative   = false;
    bool         has_next_token = false;
    bool         has_new_line   = false;
    int32_t      n_ctx         = 0;
    int32_t      n_ctx_used    = 0;    // cells the slot holds
    int32_t      n_reserved    = 0;
    int32_t      id_task       = -1;   // running task, -1 = none
    const char * wait_reason   = nullptr;
    int32_t      n_unparked    = 0;
    int32_t      n_prompt_tokens = 0;  // prompt of the running task
    int32_t      n_cached      = 0;
    int32_t      n_prefill     = 0;
    int32_t      n_decoded     = 0;
    int32_t      n_remain      = 0;
    uint64_t     session_id    = 0;
    int32_t      n_ckpt        = 0;
    std::array<int32_t, MAX_CKPT> ckpt_pos = {};
    // the running (or last) task, replaced at launch only; immutable, read by /slots
    std::shared_ptr<const server_task> task;
};

// one session of the session store, copied into the snapshot at most every snapshot_full_us
struct server_obs_session_view {
    uint64_t     id = 0;
    const char * tier = "ram";
    int32_t      slot = -1;
    int32_t      resident_cells = 0;
    int32_t      mirror_tokens  = 0;
    size_t       mirror_bytes   = 0;
    size_t       mirror_segments = 0;
    size_t       checkpoints    = 0;
    size_t       file_bytes     = 0;
    bool         stale          = false;
    int64_t      t_sync_us      = 0;
    int64_t      t_synced_us    = 0;
    uint64_t     syncs = 0, truncations = 0, restores = 0;
    size_t       last_restore_bytes = 0;
    double       last_restore_ms = 0.0, last_restore_wall_ms = 0.0;
    uint64_t     prefetches = 0, prefetch_hits = 0, demotions = 0, promotions = 0;
};

// the loop's own view, published by the loop thread (one short lock per step), read by the HTTP and writer threads
struct server_obs_snapshot {
    std::mutex mtx;

    std::vector<server_obs_slot_view> slots; // sized once at init
    int32_t n_busy = 0, n_prompt = 0, n_gen = 0, n_preempted = 0, n_retained = 0;
    int32_t queue_deferred = 0, sched_waiting = 0;
    int64_t cells_used = 0, cells_reserved = 0;
    server_obs_sched_totals sched;

    // llama.cpp's server_metrics, cumulative
    uint64_t prompt_count = 0, prompt_steps = 0, prompt_time_us = 0;
    uint64_t predict_count = 0, predict_steps = 0, predict_time_us = 0;
    uint64_t n_prompt_cached = 0, n_tokens_max = 0, n_decode = 0, n_busy_slots = 0;
    uint64_t n_draft_tokens = 0, n_draft_accepted = 0, n_draft_verif_steps = 0;
    int64_t  metrics_t_start = 0;

    // refreshed at most every snapshot_full_us (walks containers)
    int64_t t_full_us = 0;
    bool    pc_enabled = false;
    size_t  pc_bytes = 0, pc_limit = 0, pc_tokens = 0, pc_tokens_limit = 0, pc_entries = 0;
    bool    sessions_on = false;
    server_session_store::stats_t sessions_stats;
    std::vector<server_obs_session_view> sessions; // capacity reserved at init, grows only with the session count
    size_t  sessions_resident_cells = 0;
};

// one /events reader: bounded, lossy (drop-oldest), never applies backpressure to the writer
struct server_obs_subscriber {
    explicit server_obs_subscriber(size_t capacity) : capacity(capacity) {}

    void publish(const std::string & line);
    // waits up to timeout_ms; false on timeout or close
    bool next(std::string & line, int timeout_ms);
    void close();
    bool is_closed() {
        std::lock_guard<std::mutex> lock(mtx);
        return closed;
    }

    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<std::string> queue;
    size_t   capacity = 0;
    uint64_t dropped  = 0;
    bool     closed   = false;
};

struct server_nvml; // dlopen'ed NVML, see server-observe.cpp

struct server_observer_config {
    std::string jsonl_path;          // empty = no file
    int64_t     throughput_interval_ms = 5000;
    int         sample_hz              = 10;
    size_t      replay_capacity        = 512;
    size_t      subscriber_capacity    = 1024;
    std::string gpu_pci_bus_id;      // from ggml backend props; empty = NVML index 0
    int32_t     n_slots = 0;
};

// configuration and load-time facts, immutable after the model is loaded (any thread)
struct server_obs_static {
    // server
    std::string model_id;          // the served id: --alias, else the model file name
    std::string build_info;
    std::string host;
    int32_t     port = 0;
    bool        api_key = false;
    std::string slot_save_path;
    int32_t     default_n_predict = -1;
    bool        default_thinking = true;
    bool        preserve_thinking = false;
    std::vector<std::string> argv; // API keys redacted

    // model
    std::string model_path;
    std::string model_desc;
    std::string model_arch;
    std::string model_ftype;
    uint64_t    model_n_params = 0;
    uint64_t    model_size = 0;     // tensor bytes
    int64_t     file_bytes = -1;    // < 0 = unknown
    int64_t     tensor_count = 0;
    int32_t     n_ctx_train = 0;
    double      load_seconds = 0.0;
    std::vector<std::string> lora_names;

    // engine
    int32_t     n_ctx = 0;          // KV cells of the pool
    int32_t     n_ctx_slot = 0;
    int32_t     n_slots = 0;
    int32_t     n_batch = 0, n_ubatch = 0, n_gpu_layers = 0;
    bool        kv_unified = false;
    std::string cache_type_k, cache_type_v, flash_attn;
    int32_t     cache_ram_mib = 0;
    std::string spec_types;         // "none" when off
    std::string spec_backend;       // ninfer name: mtp, draft, none
    int32_t     spec_n_max = 0, spec_n_min = 0;
    bool        vision = false;
    bool        cuda_graph = true;
    int32_t     ctx_checkpoints = 0;
    bool        kv_budget = false;
    int32_t     kv_budget_gen = 0, kv_budget_headroom = 0, prefill_chunk_gen = 0;
    double      kv_timeslice = 0.0;
    bool        sessions = false;
    size_t      session_ram_bytes = 0;
    std::string session_dir;
    size_t      session_disk_limit = 0;
    uint64_t    session_ttl_s = 0;

    // sampling defaults
    float       temperature = 0.0f, top_p = 0.0f, min_p = 0.0f, presence_penalty = 0.0f, frequency_penalty = 0.0f;
    int32_t     top_k = 0;

    // memory, from the ggml backend breakdown right after load
    json        vram;               // VramBudget
    uint64_t    model_bytes = 0;    // device buffers
    uint64_t    context_bytes = 0;  // KV + recurrent state, target
    uint64_t    draft_context_bytes = 0;
    uint64_t    compute_bytes = 0;
    uint64_t    device_total = 0, device_free = 0;
    double      bytes_per_cell = 0.0; // context bytes / n_ctx

    // /slots: llama.cpp's slot JSON detokenizes the prompt (read-only use of the model vocabulary)
    const llama_context * ctx = nullptr;
};

// the continuation-cache view of the loop's state: l1 = GPU-resident sessions, l2 = RAM mirrors, l3 = NVMe files
struct server_obs_occupancy {
    uint64_t l1_entries = 0, l1_bytes = 0, l2_entries = 0, l2_bytes = 0, l3_entries = 0, l3_bytes = 0;
    uint64_t l1_evictions = 0, l1_demotions = 0, l2_evictions = 0, l2_evicted_bytes = 0;
    uint64_t grows = 0, spills = 0;
    uint64_t publications = 0, publication_failures = 0, persisted = 0;
    double   persist_us = 0.0;
};

struct server_observer {
    explicit server_observer(const server_observer_config & cfg);
    ~server_observer();

    server_observer(const server_observer &) = delete;
    server_observer & operator=(const server_observer &) = delete;

    // starts the sampler and writer threads, emits server_start
    void start(server_obs_static st);
    void stop();

    // hot path (loop thread)
    server_obs_clock    clock;
    server_obs_counters counters;
    server_obs_snapshot snap;
    static constexpr int64_t snapshot_full_us = 250000;

    // any thread; the record is moved into the writer queue
    void emit(server_obs_record && rec);

    server_obs_energy_mark energy_mark() const;

    // energy attributed to a request that started at m0 and computed the given tokens
    // returns false when the board has no energy counter
    bool request_energy(const server_obs_energy_mark & m0, uint64_t n_prefill, uint64_t n_decode,
                        double & e_prefill, double & e_decode) const;

    // HTTP side
    std::shared_ptr<server_obs_subscriber> subscribe(std::vector<std::string> & backlog);
    void unsubscribe(const std::shared_ptr<server_obs_subscriber> & sub);

    // the GET /telemetry body (HTTP thread, from the snapshot)
    json telemetry_json(bool sleeping);
    // the GET /slots body: llama.cpp's slot objects plus ninfer's keys (HTTP thread, from the snapshot)
    json slots_json();
    // the GET /metrics body: llama.cpp's series (gauges averaged since the previous scrape), ninfer's, lovelace's
    std::string metrics_text(int64_t & t_start);

    const std::string & instance_id() const { return server_instance_id; }
    int64_t t_start_unix_ms() const { return t_start_ms; }
    const server_obs_static & config() const { return st; }

    // SSE event name of a formatted record (its `event` field)
    static std::string record_event(const std::string & line);

    static int64_t now_unix_ms();

private:
    void sampler_loop();
    void writer_loop();
    std::string format(server_obs_record & rec, uint64_t seq);
    void publish_line(std::string && line, bool retain);

    json energy_ext_json() const;
    json loop_json() const;     // {window_seconds, phase, seconds, fraction}
    json server_start_json() const;
    json request_json(const server_obs_record & rec) const;
    // writer thread: the scheduler and continuation_cache sections, and lovelace.sessions
    void throughput_body(json & body, json & ext);

    server_observer_config cfg;
    server_obs_static      st;
    std::string server_instance_id;
    int64_t     t_start_ms = 0;

    std::unique_ptr<server_nvml> nvml;

    // request-derived totals, accumulated by the writer thread from the records (continuation cache view)
    struct cc_totals {
        uint64_t admitted = 0;
        double   queue_s  = 0.0;
        uint64_t requests_done = 0;
        uint64_t prefix_hit_tokens = 0, drafted = 0, accepted = 0;
        uint64_t lookup_hits = 0, lookup_misses = 0, no_alias = 0;
        uint64_t restores = 0, restored_tokens = 0, restored_bytes = 0;
        uint64_t tier_restores[3] = {}, tier_tokens[3] = {}, tier_bytes[3] = {};
        uint64_t tier_restore_us[3] = {}, tier_restore_ops[3] = {}; // restores that copied state
    };
    mutable std::mutex mtx_cc;
    cc_totals cc;           // writer thread writes, HTTP threads read (mtx_cc)
    cc_totals cc_prev;      // writer thread only: at the previous throughput record
    server_obs_occupancy           occ_prev;      // writer thread only
    server_session_store::stats_t  sessions_prev; // writer thread only

    // /metrics: llama.cpp's gauges are averaged since the previous scrape
    std::mutex mtx_scrape;
    uint64_t   scrape_prompt_steps = 0, scrape_prompt_us = 0, scrape_predict_steps = 0, scrape_predict_us = 0;

    // writer
    std::thread             writer_thread;
    std::mutex              mtx_queue;
    std::condition_variable cv_queue;
    std::vector<server_obs_record> queue;
    bool                    stopping = false;
    FILE *                  jsonl = nullptr;
    uint64_t                seq_next = 0; // writer thread only

    // subscribers and replay (writer + HTTP threads)
    mutable std::mutex mtx_subs;
    std::string        server_start_line;
    std::deque<std::string> replay;
    std::vector<std::shared_ptr<server_obs_subscriber>> subs;
    std::atomic<uint64_t> n_emitted { 0 };
    std::atomic<uint64_t> n_dropped_closed { 0 };

    // sampler
    std::thread             sampler_thread;
    std::mutex              mtx_sampler_stop;
    std::condition_variable cv_sampler_stop;
    bool                    sampler_stop = false;

    struct sample {
        int64_t t_ns = 0;
        std::array<int64_t, SERVER_OBS_PHASE_COUNT> ns = {};
        double  board_joules = 0.0; // cumulative since start
    };

    // sampler-published state
    mutable std::mutex mtx_state;
    std::deque<sample> window;      // last window_ns of samples
    server_obs_clock::snapshot last_clock;
    double energy_board   = 0.0;
    double energy_prefill = 0.0;
    double energy_decode  = 0.0;
    double energy_idle    = 0.0;
    double idle_watts     = -1.0;   // < 0 = not calibrated
    uint64_t energy_samples = 0;    // sampling periods with an energy delta
    server_obs_energy_mark mark;
    bool   energy_available = false;
};

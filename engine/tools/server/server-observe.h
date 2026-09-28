#pragma once

// Observability for llama-server: request log (--request-log-jsonl), the GET /events record stream,
// GET /telemetry, and board energy attribution.
//
// Threading contract:
//  - the server loop thread (start_loop) is the only writer of the phase clock and the token
//    counters, and it only ever performs relaxed atomic stores and enqueues POD records
//  - a 10 Hz sampler thread reads the NVML energy counter and the phase clock, and attributes
//    energy deltas to prefill / decode / idle by the loop's phase time share
//  - a writer thread formats the records, appends them to the JSONL file and fans them out to
//    the /events subscribers
// No NVML call, JSON formatting, or file I/O ever happens on the server loop thread.
//
// Payload schema: docs/observability.md

#include "server-common.h"

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

// counters written by the loop thread (relaxed stores), read by the sampler and /telemetry
struct server_obs_counters {
    std::atomic<uint64_t> prefill_tokens  { 0 }; // computed prompt tokens
    std::atomic<uint64_t> decode_tokens   { 0 }; // committed generated tokens
    std::atomic<uint64_t> drafted         { 0 };
    std::atomic<uint64_t> accepted        { 0 };
    std::atomic<uint64_t> decode_calls    { 0 };
    std::atomic<uint64_t> busy_slot_calls { 0 }; // sum over decode calls of busy slots

    // gauges, snapshot of the loop's own view
    std::atomic<int32_t> slots_busy      { 0 };
    std::atomic<int32_t> slots_prompt    { 0 };
    std::atomic<int32_t> queue_deferred  { 0 };

    // KV budget scheduling (--kv-budget), loop thread
    std::atomic<uint64_t> sched_parks        { 0 }; // idle sessions saved to host memory and evicted from VRAM
    std::atomic<uint64_t> sched_park_drops   { 0 }; // evicted without a new copy (already cached, or no cache room)
    std::atomic<uint64_t> sched_unparks      { 0 }; // sessions restored from host memory
    std::atomic<uint64_t> sched_preempts     { 0 }; // generating slots paused to host memory
    std::atomic<uint64_t> sched_resumes      { 0 };
    std::atomic<uint64_t> sched_grows        { 0 }; // reservations extended in flight
    std::atomic<uint64_t> sched_park_bytes   { 0 }; // device -> host, parks and preempts
    std::atomic<uint64_t> sched_unpark_bytes { 0 }; // host -> device, unparks and resumes
    std::atomic<uint64_t> sched_park_us      { 0 };
    std::atomic<uint64_t> sched_unpark_us    { 0 };
    std::atomic<uint64_t> sched_wait_slot    { 0 }; // requests that waited for a slot
    std::atomic<uint64_t> sched_wait_kv      { 0 }; // requests that waited for KV cells
    std::atomic<uint64_t> sched_rollback_fallbacks { 0 }; // truncations the memory refused
    std::atomic<uint64_t> sched_recompute_tokens   { 0 }; // tokens re-prefilled to recover a sequence
    std::atomic<uint64_t> sched_kv_overflows       { 0 }; // llama_decode() found no cells despite the budget
    std::atomic<int32_t>  kv_cells_total     { 0 };
    std::atomic<int32_t>  kv_cells_used      { 0 }; // cells held by all slots
    std::atomic<int32_t>  kv_cells_reserved  { 0 }; // cells held or promised (max(used, reservation) per slot)
    std::atomic<int32_t>  sched_waiting      { 0 }; // requests in the admission queue
    std::atomic<int32_t>  sched_preempted    { 0 }; // slots paused to host memory
    std::atomic<int32_t>  sessions_parked    { 0 }; // prompt cache entries

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
    SERVER_OBS_REC_REQUEST_REJECTED,
    SERVER_OBS_REC_RAW, // pre-formatted JSON object (server_start, throughput), envelope is added by the writer
};

// POD-like request record, filled on the loop (or HTTP) thread and formatted by the writer thread
// all `const char *` fields must point to string literals
struct server_obs_record {
    server_obs_record_kind kind = SERVER_OBS_REC_RAW;
    int64_t t_unix_ms = 0;

    // request info
    int32_t      id_task    = -1;
    int32_t      id_slot    = -1;
    int32_t      id_parent  = -1;
    const char * task_type  = "completion";
    const char * endpoint   = "native";
    bool         stream     = false;
    int32_t      n_prompt_tokens = 0;
    int32_t      n_predict  = -1;
    int32_t      n_ctx_slot = 0;
    double       queue_wait_ms = -1.0; // < 0 = unknown
    const char * wait_reason   = "none"; // KV budget: "none", "slot" or "kv"
    int32_t      n_unparked_tokens = 0; // tokens restored from host memory at launch

    // done
    const char * finish_reason = "eos";
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
    bool     speculative  = false;
    uint64_t n_drafted     = 0;
    uint64_t n_accepted    = 0;
    uint64_t n_verif_steps = 0;
    std::vector<uint64_t> accepted_per_pos;
    bool     energy_valid   = false;
    double   prefill_joules = 0.0;
    double   decode_joules  = 0.0;

    // rejected / error
    const char * stage      = "http";
    const char * error_type = "server";
    std::string  error_msg;

    // SERVER_OBS_REC_RAW: the body is merged after the envelope
    const char * event  = "";
    json         body;
    bool         retain = false; // keep as the server_start record replayed to every new subscriber
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

struct server_observer {
    explicit server_observer(const server_observer_config & cfg);
    ~server_observer();

    server_observer(const server_observer &) = delete;
    server_observer & operator=(const server_observer &) = delete;

    // starts the sampler and writer threads, emits server_start
    // `server_start` receives the gpu/instance fields from here and must carry the model/config sections
    void start(json server_start);
    void stop();

    // hot path (loop thread)
    server_obs_clock    clock;
    server_obs_counters counters;

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

    // /telemetry sections owned by the observer: gpu, energy, scheduler.loop and loop totals, events
    json gpu_json() const;      // live NVML read (HTTP thread)
    json gpu_static_json() const;
    json energy_json() const;
    json loop_json() const;     // {window_seconds, phase, seconds, fraction}
    json loop_totals_json() const;
    json events_json() const;
    std::string to_prometheus() const;

    const std::string & instance_id() const { return server_instance_id; }
    int64_t t_start_unix_ms() const { return t_start_ms; }

    static int64_t now_unix_ms();

private:
    void sampler_loop();
    void writer_loop();
    std::string format(const server_obs_record & rec, uint64_t seq) const;
    void publish_line(std::string && line, bool retain);

    server_observer_config cfg;
    std::string server_instance_id;
    int64_t     t_start_ms = 0;

    std::unique_ptr<server_nvml> nvml;

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
    server_obs_energy_mark mark;
    bool   energy_available = false;
};

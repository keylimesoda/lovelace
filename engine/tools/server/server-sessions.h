#pragma once

// Per-agent session store (docs/design/session-store.md): every session keeps a write-back mirror of its sequence
// state in a pinned host arena, so that its cells can be dropped from VRAM at no cost and restored at PCIe speed.
//
//  - the mirror of the attention KV is an append-only log of KV deltas (llama_state_seq_get_delta_data_async), one
//    per sync of the target and of the draft context; a rollback truncates it (segments end early, later ones go)
//  - the recurrent state is a snapshot taken with every sync, the context checkpoints move between the slot (while
//    the session is resident) and the mirror (while it is parked)
//  - mirrors of parked sessions are demoted to files in --session-dir when the arena runs short, and promoted back
//    before use; a session is only deleted by its TTL or the disk budget
//
// Threading: everything runs on the server thread, except the file reads and writes, which run on a worker thread
// that owns the buffers of the session being moved until the job is done.

#include "common.h"
#include "llama.h"
#include "server-session-store.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// completion handle of an asynchronous state transfer, waited for on destruction
struct server_xfer {
    server_xfer() = default;
    explicit server_xfer(llama_state_xfer_t h) : h(h) {}
    server_xfer(server_xfer && other) noexcept : t_us(other.t_us), t_end_us(other.t_end_us), h(other.h) { other.h = nullptr; }
    server_xfer & operator=(server_xfer && other) noexcept;
    server_xfer(const server_xfer &) = delete;
    server_xfer & operator=(const server_xfer &) = delete;
    ~server_xfer() { wait(); }

    bool done(); // non-blocking
    void wait();
    void order(); // later graphs of the issuing context run after the copies (see llama_state_xfer_order)

    int64_t t_us     = -1; // device time of the copies, once done
    int64_t t_end_us = -1; // when they were seen complete (ggml_time_us), once done

private:
    llama_state_xfer_t h = nullptr;
};

// bytes appended to chunks of the arena
struct server_session_log {
    std::vector<uint8_t *> chunks;
    size_t size = 0;

    size_t capacity(size_t n_chunk) const { return chunks.size() * n_chunk; }

    // chunks to take for n more bytes
    size_t n_more_chunks(const common_state_arena & arena, size_t n) const;
    bool   grow(common_state_arena & arena, size_t n);
    void   truncate(common_state_arena & arena, size_t n);
    void   release(common_state_arena & arena);

    // the memory of bytes [off, off + n), appended to out
    void views(const common_state_arena & arena, size_t off, size_t n, std::vector<llama_state_chunk> & out) const;
};

// one KV delta in the log
struct server_session_seg {
    size_t    off  = 0;
    size_t    size = 0;
    bool      dft  = false; // of the draft context
    llama_pos pos0 = 0;     // the delta holds the cells at positions >= pos0
    llama_pos end  = std::numeric_limits<llama_pos>::max(); // cells at positions >= end are left out (rollback)

    server_xfer xfer; // the copy that fills it
};

enum server_session_tier {
    SERVER_SESSION_RAM,       // the mirror is in the arena (the session may be resident in VRAM as well)
    SERVER_SESSION_DEMOTING,  // being written to its file, still in the arena
    SERVER_SESSION_DISK,      // only in its file
    SERVER_SESSION_PROMOTING, // being read from its file into the arena
};

const char * server_session_tier_name(server_session_tier tier);

struct server_session {
    uint64_t id = 0;

    llama_tokens tokens;    // the prompt the mirror holds: the KV of tokens[0, n_kv)
    int32_t n_kv     = 0;   // target cells at positions [0, n_kv)
    int32_t n_kv_dft = 0;   // draft cells at positions [0, n_kv_dft)
    int32_t n_rec    = -1;  // tokens the recurrent snapshot is at (usable only at n_kv)

    server_session_log              log;
    std::vector<server_session_seg> segs;
    common_state_buf                rec;
    common_state_buf                rec_dft;

    // the context checkpoints, while parked (the slot owns them while the session is resident)
    std::list<common_prompt_checkpoint> checkpoints;

    int  id_slot = -1;  // the slot the session is bound to (its cells, if any, belong to the session)
    bool stale   = false; // the slot holds cells the mirror has not synced (no arena room at the time)

    server_session_tier tier = SERVER_SESSION_RAM;
    bool     file_current = false; // its file holds the current mirror (demoting it only frees the arena)
    size_t   file_bytes   = 0;     // size of its file, 0 = none

    // the layout of its file; while demoted, the sizes of the freed buffers
    size_t file_head    = 0;
    size_t file_log     = 0;
    size_t file_rec     = 0;
    size_t file_rec_dft = 0;
    std::vector<std::pair<size_t, size_t>> file_ckpt; // target, draft bytes of each checkpoint

    int64_t  t_used_us   = 0; // LRU order
    uint64_t t_used_s    = 0; // wall clock, for the TTL
    int64_t  t_sync_us   = 0; // issue of the last sync
    int64_t  t_synced_us = 0; // completion of the last sync, 0 while in flight

    // a restore in flight into id_slot
    std::vector<server_xfer> restore_xfers;
    int64_t                  t_restore_us     = 0;
    size_t                   restore_bytes    = 0;
    bool                     restore_prefetch = false;
    bool                     restore_ordered  = false; // the compute streams wait for the copies, not the host

    // stats
    uint64_t n_sync           = 0;
    uint64_t n_restore        = 0;
    uint64_t n_prefetch       = 0;
    uint64_t n_prefetch_hit   = 0;
    uint64_t n_truncate       = 0; // rollbacks that truncated the mirror
    uint64_t n_demote         = 0;
    uint64_t n_promote        = 0;
    size_t   last_restore_bytes = 0;
    double   last_restore_ms    = 0.0; // device time of the copies
    double   last_restore_wall_ms = 0.0; // issue to completion
    int64_t  last_restore_end_us  = 0;   // when the copies of the last restore completed

    // bytes of the mirror in the arena (or that it takes when promoted)
    size_t bytes() const;
    size_t ckpt_bytes() const;
};

struct server_session_store {
    struct params_t {
        std::string dir;              // demoted mirrors, "" = no disk tier
        size_t      disk_limit = 0;   // bytes, 0 = unlimited
        uint64_t    ttl_s      = 0;   // 0 = never expire
        size_t      reserve    = 0;   // bytes kept free in the arena for syncs and checkpoints
        size_t      n_ckpt     = 0;   // context checkpoints a parked mirror keeps (the newest), 0 = all
    };

    // wake: called from the worker thread when a file job completes, to run the server loop
    server_session_store(common_state_arena & arena, llama_context * ctx_tgt, llama_context * ctx_dft,
            params_t params, std::function<void()> wake);
    ~server_session_store();

    server_session_store(const server_session_store &) = delete;
    server_session_store & operator=(const server_session_store &) = delete;

    common_state_arena & arena;
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    params_t params;

    server_session_index index;

    // the session a prompt belongs to, a new id if it continues none (see the rule in the implementation)
    // anchor: the end of the prompt's first user message, 0 if unknown
    uint64_t identify(const llama_tokens & tokens, size_t anchor);

    server_session * get(uint64_t id);
    server_session & create(uint64_t id);

    // mirror the slot's sequence after a request: truncate the mirror to its common prefix with `tokens`, append the
    // new cells and snapshot the recurrent state, all asynchronously. Returns the bytes being copied; 0: no arena
    // room (without wait_room) or a failure, the session is then stale and the sync has to be retried.
    size_t sync(server_session & s, llama_seq_id seq, const llama_tokens & tokens, bool wait_room = false);

    // the copies of the last sync are done, i.e. the slot's cells may change
    bool sync_done(server_session & s);
    void sync_wait(server_session & s);

    // the mirror holds all n_tokens of the slot and the recurrent state there
    bool current(const server_session & s, int32_t n_tokens) const;

    // start restoring the mirror into seq, whose memory must be empty; returns the tokens it will hold (<= n_kv:
    // without a recurrent snapshot at n_kv, only up to the newest checkpoint). The checkpoints move to `ckpts`.
    int32_t restore(server_session & s, llama_seq_id seq, std::list<common_prompt_checkpoint> & ckpts, bool prefetch);
    bool restore_done(server_session & s); // non-blocking, false while in flight
    void restore_wait(server_session & s);

    // the cells of a session go: its checkpoints go back to its mirror (those past the mirror are dropped)
    void park(server_session & s, std::list<common_prompt_checkpoint> && ckpts);

    // no transfer of the session is in flight
    bool idle(server_session & s) { return sync_done(s) && restore_done(s); }

    // the first sessions of the queue that making room never demotes (unless the room is a must): the head and the
    // next in line stay in the arena, ready to restore
    static constexpr size_t KEEP_QUEUED = 2;

    // arena room for n chunks: demotes parked sessions (their chunks free once written), except the first `keep` of
    // the queue; wait: until they are written, demoting any parked session if it has to
    bool make_room(size_t n_chunks, bool wait, uint64_t except = 0, size_t keep = KEEP_QUEUED);

    // start reading a demoted session back; true if it is in the arena already
    bool promote(server_session & s);

    // reap file jobs, keep the arena reserve free, promote the wanted sessions (queued requests, in order) and keep
    // them from being demoted; returns true if a session became usable
    bool poll(const std::vector<uint64_t> & wanted);

    // delete a session (TTL), its file as well
    void erase(uint64_t id);
    std::vector<uint64_t> expired(uint64_t now_s) const;

    // read the index of the files in params.dir (at startup), returns the next free session id
    uint64_t load_dir();

    struct stats_t {
        size_t arena_bytes = 0, arena_free = 0, arena_chunk = 0;
        size_t ram_bytes = 0, disk_bytes = 0;
        size_t n_sessions = 0, n_gpu = 0, n_ram = 0, n_disk = 0, n_io = 0;
        uint64_t n_demote = 0, n_promote = 0, n_deleted = 0, n_sync_stale = 0;
        uint64_t n_prefetch = 0, n_prefetch_hit = 0;
        uint64_t io_bytes_written = 0, io_bytes_read = 0;
        double   io_write_s = 0.0, io_read_s = 0.0;
    };
    stats_t stats() const;

    std::unordered_map<uint64_t, std::unique_ptr<server_session>> sessions;

    uint64_t n_deleted            = 0;
    uint64_t n_sync_stale         = 0;
    uint64_t n_demote_total       = 0;
    uint64_t n_promote_total      = 0;
    uint64_t n_prefetch_total     = 0;
    uint64_t n_prefetch_hit_total = 0;

private:
    uint64_t next_id = 1;

    // the sessions of queued requests, in queue order (see poll)
    std::vector<uint64_t> wanted;

    // file jobs, run by the worker thread
    struct job {
        enum kind_t { WRITE, READ } kind = WRITE;
        uint64_t id = 0;
        std::filesystem::path path;
        std::vector<uint8_t> head;                         // WRITE: header and tables
        size_t head_size = 0;                              // READ: bytes to skip
        std::vector<std::pair<uint8_t *, size_t>> spans;   // the bytes after the head, in file order
        std::atomic<bool> cancel { false };
        std::atomic<int>  state  { 0 };                    // 0 pending, 1 ok, 2 failed
        std::string error;
        int64_t t_us = 0;
    };

    std::thread                       worker;
    std::mutex                        mutex;
    std::condition_variable           cv;
    std::deque<std::shared_ptr<job>>  queue;
    bool                              stop = false;
    std::unordered_map<uint64_t, std::shared_ptr<job>> jobs; // by session
    std::function<void()>             wake;

    uint64_t io_bytes_written = 0, io_bytes_read = 0;
    int64_t  io_write_us = 0, io_read_us = 0;
    size_t   disk_bytes = 0;

    void run_worker();
    void run_job(job & j);
    void submit(std::shared_ptr<job> j);

    std::filesystem::path path_of(uint64_t id) const;

    void truncate(server_session & s, int32_t keep, int32_t keep_dft);
    bool demote(server_session & s);
    bool reap_jobs();
    void finish_job(server_session & s, job & j);
    void release_ram(server_session & s);
    void drop(uint64_t id, const char * reason);
    void enforce_disk_limit(uint64_t except);

    // arena chunks the buffers of a session hold
    size_t arena_chunks(const server_session & s) const;
};

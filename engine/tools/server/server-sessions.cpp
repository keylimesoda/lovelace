#include "server-sessions.h"

#include "server-common.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

static constexpr uint32_t SESSION_FILE_MAGIC   = 0x3153564c; // "LVS1"
static constexpr uint32_t SESSION_FILE_VERSION = 1;

static constexpr llama_state_seq_flags SESSION_KV_FLAGS  = LLAMA_STATE_SEQ_FLAGS_KV_ONLY;
static constexpr llama_state_seq_flags SESSION_REC_FLAGS = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

static uint64_t session_now_s() {
    const std::time_t now = std::time(nullptr);
    return now > 0 ? (uint64_t) now : 0;
}

const char * server_session_tier_name(server_session_tier tier) {
    switch (tier) {
        case SERVER_SESSION_RAM:       return "ram";
        case SERVER_SESSION_DEMOTING:  return "demoting";
        case SERVER_SESSION_DISK:      return "disk";
        case SERVER_SESSION_PROMOTING: return "promoting";
    }
    return "unknown";
}

//
// server_xfer
//

server_xfer & server_xfer::operator=(server_xfer && other) noexcept {
    if (this != &other) {
        wait();
        h        = other.h;
        t_us     = other.t_us;
        t_end_us = other.t_end_us;
        other.h  = nullptr;
    }
    return *this;
}

bool server_xfer::done() {
    if (h && llama_state_xfer_done(h)) {
        t_us     = llama_state_xfer_t_us(h);
        t_end_us = llama_state_xfer_t_end_us(h);
        llama_state_xfer_free(h);
        h = nullptr;
    }
    return h == nullptr;
}

void server_xfer::wait() {
    if (h) {
        llama_state_xfer_wait(h);
        t_us     = llama_state_xfer_t_us(h);
        t_end_us = llama_state_xfer_t_end_us(h);
        llama_state_xfer_free(h);
        h = nullptr;
    }
}

void server_xfer::order() {
    if (h) {
        llama_state_xfer_order(h);
    }
}

//
// server_session_log
//

size_t server_session_log::n_more_chunks(const common_state_arena & arena, size_t n) const {
    const size_t need = size + n;
    const size_t cap  = capacity(arena.chunk_size());
    return need <= cap ? 0 : arena.n_chunks(need - cap);
}

bool server_session_log::grow(common_state_arena & arena, size_t n) {
    const size_t k = n_more_chunks(arena, n);
    return k == 0 || arena.take(k, chunks);
}

void server_session_log::truncate(common_state_arena & arena, size_t n) {
    GGML_ASSERT(n <= size);
    size = n;
    const size_t keep = arena.n_chunks(n);
    if (chunks.size() > keep) {
        std::vector<uint8_t *> rest(chunks.begin() + keep, chunks.end());
        chunks.resize(keep);
        arena.give(rest);
    }
}

void server_session_log::release(common_state_arena & arena) {
    arena.give(chunks);
    size = 0;
}

void server_session_log::views(const common_state_arena & arena, size_t off, size_t n, std::vector<llama_state_chunk> & out) const {
    const size_t cs = arena.chunk_size();
    GGML_ASSERT(off + n <= capacity(cs));
    while (n > 0) {
        const size_t i = off / cs;
        const size_t o = off % cs;
        const size_t k = std::min(n, cs - o);
        out.push_back({ chunks[i] + o, k });
        off += k;
        n   -= k;
    }
}

//
// server_session
//

size_t server_session::ckpt_bytes() const {
    size_t res = 0;
    for (const auto & c : checkpoints) {
        res += c.size();
    }
    return res;
}

size_t server_session::bytes() const {
    if (tier == SERVER_SESSION_DISK) {
        size_t res = file_log + file_rec + file_rec_dft;
        for (const auto & c : file_ckpt) {
            res += c.first + c.second;
        }
        return res;
    }
    return log.size + rec.size() + rec_dft.size() + ckpt_bytes();
}

//
// file layout: head (header, tokens, segment and checkpoint tables), then the log, the recurrent snapshots and the
// data of each checkpoint
//

namespace {

struct head_writer {
    std::vector<uint8_t> buf;

    template <typename T>
    void put(const T & v) {
        const auto * p = reinterpret_cast<const uint8_t *>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }

    void put_bytes(const void * data, size_t n) {
        const auto * p = static_cast<const uint8_t *>(data);
        buf.insert(buf.end(), p, p + n);
    }
};

struct head_reader {
    const uint8_t * data;
    size_t n;
    size_t off = 0;

    template <typename T>
    T get() {
        T v;
        get_bytes(&v, sizeof(T));
        return v;
    }

    void get_bytes(void * dst, size_t k) {
        if (off + k > n) {
            throw std::runtime_error("truncated session file head");
        }
        memcpy(dst, data + off, k);
        off += k;
    }
};

// the fixed part of the head
struct head_fixed {
    uint32_t magic;
    uint32_t version;
    uint64_t id;
    uint64_t n_tokens;
    int32_t  n_kv;
    int32_t  n_kv_dft;
    int32_t  n_rec;
    uint32_t n_segs;
    uint32_t n_ckpts;
    uint32_t pad;
    uint64_t log_size;
    uint64_t rec_size;
    uint64_t rec_dft_size;
    uint64_t head_size;
};

bool write_all(int fd, const uint8_t * p, size_t n, const std::atomic<bool> & cancel) {
    constexpr size_t piece = 64u << 20;
    while (n > 0) {
        if (cancel.load(std::memory_order_relaxed)) {
            errno = ECANCELED;
            return false;
        }
        const ssize_t k = ::write(fd, p, std::min(n, piece));
        if (k < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        p += k;
        n -= (size_t) k;
    }
    return true;
}

bool read_all(int fd, uint8_t * p, size_t n) {
    constexpr size_t piece = 64u << 20;
    while (n > 0) {
        const ssize_t k = ::read(fd, p, std::min(n, piece));
        if (k < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (k == 0) {
            errno = EIO; // short file
            return false;
        }
        p += k;
        n -= (size_t) k;
    }
    return true;
}

} // namespace

static std::vector<uint8_t> session_encode_head(const server_session & s) {
    head_writer w;

    head_fixed h {};
    h.magic        = SESSION_FILE_MAGIC;
    h.version      = SESSION_FILE_VERSION;
    h.id           = s.id;
    h.n_tokens     = s.tokens.size();
    h.n_kv         = s.n_kv;
    h.n_kv_dft     = s.n_kv_dft;
    h.n_rec        = s.n_rec;
    h.n_segs       = s.segs.size();
    h.n_ckpts      = s.checkpoints.size();
    h.log_size     = s.log.size;
    h.rec_size     = s.rec.size();
    h.rec_dft_size = s.rec_dft.size();
    w.put(h);

    w.put_bytes(s.tokens.data(), s.tokens.size() * sizeof(llama_token));

    for (const auto & seg : s.segs) {
        w.put<uint64_t>(seg.off);
        w.put<uint64_t>(seg.size);
        w.put<uint32_t>(seg.dft ? 1 : 0);
        w.put<int32_t>(seg.pos0);
        w.put<int32_t>(seg.end);
    }

    for (const auto & c : s.checkpoints) {
        w.put<int64_t>(c.n_tokens);
        w.put<int32_t>(c.pos_min);
        w.put<int32_t>(c.pos_max);
        w.put<int32_t>(c.id_task);
        w.put<uint64_t>(c.data_tgt.size());
        w.put<uint64_t>(c.data_dft.size());
        w.put<uint64_t>(c.data_spec.size());
        w.put_bytes(c.data_spec.data(), c.data_spec.size());
    }

    // the head size goes into the fixed part
    reinterpret_cast<head_fixed *>(w.buf.data())->head_size = w.buf.size();

    return std::move(w.buf);
}

// fills the metadata of a demoted session from its file head
static void session_decode_head(const uint8_t * data, size_t n, server_session & s) {
    head_reader r { data, n };

    const auto h = r.get<head_fixed>();
    if (h.magic != SESSION_FILE_MAGIC || h.version != SESSION_FILE_VERSION) {
        throw std::runtime_error("unsupported session file format");
    }
    if (h.head_size != n) {
        throw std::runtime_error("session file head size mismatch");
    }

    s.id       = h.id;
    s.n_kv     = h.n_kv;
    s.n_kv_dft = h.n_kv_dft;
    s.n_rec    = h.n_rec;

    s.tokens.resize(h.n_tokens);
    r.get_bytes(s.tokens.data(), h.n_tokens * sizeof(llama_token));

    s.segs.clear();
    for (uint32_t i = 0; i < h.n_segs; ++i) {
        server_session_seg seg;
        seg.off  = r.get<uint64_t>();
        seg.size = r.get<uint64_t>();
        seg.dft  = r.get<uint32_t>() != 0;
        seg.pos0 = r.get<int32_t>();
        seg.end  = r.get<int32_t>();
        if (seg.off + seg.size > h.log_size) {
            throw std::runtime_error("session file segment out of range");
        }
        s.segs.push_back(std::move(seg));
    }

    s.checkpoints.clear();
    s.file_ckpt.clear();
    for (uint32_t i = 0; i < h.n_ckpts; ++i) {
        auto & c = s.checkpoints.emplace_back();
        c.n_tokens = r.get<int64_t>();
        c.pos_min  = r.get<int32_t>();
        c.pos_max  = r.get<int32_t>();
        c.id_task  = r.get<int32_t>();
        const uint64_t tgt  = r.get<uint64_t>();
        const uint64_t dft  = r.get<uint64_t>();
        const uint64_t spec = r.get<uint64_t>();
        c.data_spec.resize(spec);
        r.get_bytes(c.data_spec.data(), spec);
        s.file_ckpt.emplace_back(tgt, dft);
    }

    s.file_head    = h.head_size;
    s.file_log     = h.log_size;
    s.file_rec     = h.rec_size;
    s.file_rec_dft = h.rec_dft_size;
}

//
// server_session_store
//

server_session_store::server_session_store(common_state_arena & arena, llama_context * ctx_tgt, llama_context * ctx_dft,
        params_t params, std::function<void()> wake)
    : arena(arena), ctx_tgt(ctx_tgt), ctx_dft(ctx_dft), params(std::move(params)), wake(std::move(wake)) {
    if (!this->params.dir.empty()) {
        std::filesystem::create_directories(this->params.dir);
        worker = std::thread([this]() { run_worker(); });
    }
}

server_session_store::~server_session_store() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
        for (auto & it : jobs) {
            it.second->cancel = true;
        }
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    // the transfers of the mirrors complete before the arena goes
    for (auto & it : sessions) {
        auto & s = *it.second;
        sync_wait(s);
        restore_wait(s);
        release_ram(s);
    }
}

std::filesystem::path server_session_store::path_of(uint64_t id) const {
    return std::filesystem::path(params.dir) / ("session-" + std::to_string(id) + ".bin");
}

uint64_t server_session_store::identify(const llama_tokens & tokens, size_t anchor) {
    // a prompt continues the session it shares the longest prefix with, if that prefix covers the first exchange of
    // the prompt (anchor: the end of its first user message): siblings of an agent share the system prompt and tools,
    // never the first user message. Without message spans, the prompt must hold at least half of the session.
    const auto m = index.find(tokens);
    if (m) {
        const server_session * s = get(m->id);
        const size_t n_s = s ? s->tokens.size() : 0;
        const bool cont = anchor > 0 ? m->n_common >= anchor : (n_s > 0 && 2 * m->n_common >= n_s);
        if (cont) {
            return m->id;
        }
    }
    while (sessions.count(next_id) || next_id == 0) {
        next_id++;
    }
    return next_id++;
}

server_session * server_session_store::get(uint64_t id) {
    const auto it = sessions.find(id);
    return it == sessions.end() ? nullptr : it->second.get();
}

server_session & server_session_store::create(uint64_t id) {
    auto & p = sessions[id];
    GGML_ASSERT(!p);
    p = std::make_unique<server_session>();
    p->id        = id;
    p->t_used_us = ggml_time_us();
    p->t_used_s  = session_now_s();
    next_id = std::max(next_id, id + 1);
    return *p;
}

bool server_session_store::sync_done(server_session & s) {
    bool res = true;
    for (auto & seg : s.segs) {
        res = seg.xfer.done() && res;
    }
    res = s.rec.done() && s.rec_dft.done() && res;
    if (res && s.t_synced_us == 0 && s.t_sync_us != 0) {
        s.t_synced_us = ggml_time_us();
    }
    return res;
}

void server_session_store::sync_wait(server_session & s) {
    for (auto & seg : s.segs) {
        seg.xfer.wait();
    }
    s.rec.wait();
    s.rec_dft.wait();
    sync_done(s);
}

bool server_session_store::current(const server_session & s, int32_t n_tokens) const {
    return !s.stale && s.tier == SERVER_SESSION_RAM && s.n_kv == n_tokens && s.n_rec == s.n_kv;
}

void server_session_store::truncate(server_session & s, int32_t keep, int32_t keep_dft) {
    // segments that start at or past the new end go, the one that straddles it ends there
    // a dropped segment of the target can sit before a kept one of the draft (their positions differ by a token),
    // the log then keeps its bytes until the next rebuild
    for (auto it = s.segs.begin(); it != s.segs.end();) {
        const llama_pos lim = it->dft ? keep_dft : keep;
        if (it->pos0 >= lim) {
            it->xfer.wait();
            it = s.segs.erase(it);
        } else {
            it->end = std::min(it->end, lim);
            ++it;
        }
    }
    size_t used = 0;
    for (const auto & seg : s.segs) {
        used = std::max(used, seg.off + seg.size);
    }
    s.log.truncate(arena, used);

    s.tokens.resize(std::min<size_t>(s.tokens.size(), keep));
    s.n_kv     = keep;
    s.n_kv_dft = keep_dft;
    if (s.n_rec != keep) {
        s.n_rec = -1;
    }
    s.n_truncate++;
    s.file_current = false;
}

size_t server_session_store::sync(server_session & s, llama_seq_id seq, const llama_tokens & tokens, bool wait_room) {
    GGML_ASSERT(s.tier == SERVER_SESSION_RAM);

    // the previous transfers of the mirror complete first (they finished long ago in practice)
    sync_wait(s);
    restore_wait(s);

    const int32_t n_mem     = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq) + 1;
    const int32_t n_now     = std::min<int32_t>((int32_t) tokens.size(), n_mem);
    const int32_t n_dft_now = ctx_dft ? llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq) + 1 : 0;

    // the mirror stays valid up to its common prefix with the slot
    int32_t c = 0;
    {
        const int32_t lim = std::min<int32_t>((int32_t) s.tokens.size(), n_now);
        while (c < lim && s.tokens[c] == tokens[c]) {
            c++;
        }
    }
    const int32_t keep     = std::min(c, s.n_kv);
    const int32_t keep_dft = std::min({ c, s.n_kv_dft, n_dft_now });
    if (keep < s.n_kv || keep_dft < s.n_kv_dft) {
        truncate(s, keep, keep_dft);
    }

    const size_t sz         = llama_state_seq_get_delta_size_ext(ctx_tgt, seq, keep, SESSION_KV_FLAGS);
    const size_t sz_dft     = ctx_dft ? llama_state_seq_get_delta_size_ext(ctx_dft, seq, keep_dft, SESSION_KV_FLAGS) : 0;
    const size_t sz_rec     = llama_state_seq_get_size_ext(ctx_tgt, seq, SESSION_REC_FLAGS);
    const size_t sz_rec_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq, SESSION_REC_FLAGS) : 0;
    if (sz == 0 || (ctx_dft && sz_dft == 0) || sz_rec == 0 || (ctx_dft && sz_rec_dft == 0)) {
        SRV_ERR("session %llu: the state sizes are unavailable\n", (unsigned long long) s.id);
        s.stale = true;
        n_sync_stale++;
        return 0;
    }

    // arena room: the new log bytes, and a recurrent snapshot of another size
    auto buf_chunks = [&](size_t n) { return n >= 4096 ? arena.n_chunks(n) : 0; };
    size_t need = s.log.n_more_chunks(arena, sz + sz_dft);
    if (s.rec.size() != sz_rec) {
        need += buf_chunks(sz_rec);
    }
    if (s.rec_dft.size() != sz_rec_dft) {
        need += buf_chunks(sz_rec_dft);
    }
    if (arena.n_free() < need && !make_room(need, wait_room, s.id)) {
        SRV_WRN("session %llu: no arena room to mirror %.1f MiB yet, the sync is retried\n",
                (unsigned long long) s.id, (sz + sz_dft + sz_rec) / 1048576.0);
        s.stale = true;
        n_sync_stale++;
        return 0;
    }

    bool ok = s.log.grow(arena, sz + sz_dft);

    std::vector<llama_state_chunk> views;
    auto add_seg = [&](llama_context * ctx, size_t size, bool dft, llama_pos pos0) {
        server_session_seg seg;
        seg.off  = s.log.size;
        seg.size = size;
        seg.dft  = dft;
        seg.pos0 = pos0;

        views.clear();
        s.log.views(arena, seg.off, size, views);

        llama_state_xfer_t h = nullptr;
        const size_t n = llama_state_seq_get_delta_data_async(ctx, views.data(), views.size(), seq, pos0, SESSION_KV_FLAGS, &h);
        seg.xfer = server_xfer(h);
        if (n != size) {
            return false;
        }
        s.log.size += size;
        s.segs.push_back(std::move(seg));
        return true;
    };

    ok = ok && add_seg(ctx_tgt, sz, false, keep);
    ok = ok && (!ctx_dft || add_seg(ctx_dft, sz_dft, true, keep_dft));

    // the recurrent snapshot copies in order with the compute stream (recurrent rows move between cells)
    ok = ok && (s.rec.size() == sz_rec || s.rec.alloc(&arena, sz_rec));
    ok = ok && s.rec.get_async(ctx_tgt, seq, SESSION_REC_FLAGS);
    if (ctx_dft) {
        ok = ok && (s.rec_dft.size() == sz_rec_dft || s.rec_dft.alloc(&arena, sz_rec_dft));
        ok = ok && s.rec_dft.get_async(ctx_dft, seq, SESSION_REC_FLAGS);
    }

    if (!ok) {
        SRV_ERR("session %llu: failed to start the mirror copies, the sync is retried\n", (unsigned long long) s.id);
        sync_wait(s);
        truncate(s, keep, keep_dft);
        s.stale = true;
        n_sync_stale++;
        return 0;
    }

    s.tokens.assign(tokens.begin(), tokens.begin() + n_now);
    s.n_kv     = n_now;
    s.n_kv_dft = n_dft_now;
    s.n_rec    = n_now;
    s.stale    = false;
    s.file_current = false;

    s.t_sync_us   = ggml_time_us();
    s.t_synced_us = 0;
    s.t_used_us   = s.t_sync_us;
    s.t_used_s    = session_now_s();
    s.n_sync++;

    index.update(s.id, s.tokens);

    return sz + sz_dft + sz_rec + sz_rec_dft;
}

int32_t server_session_store::restore(server_session & s, llama_seq_id seq, std::list<common_prompt_checkpoint> & ckpts, bool prefetch) {
    GGML_ASSERT(s.tier == SERVER_SESSION_RAM && s.restore_xfers.empty() && (s.id_slot < 0 || s.id_slot == seq));

    sync_wait(s);

    // the recurrent state bounds the usable prefix: the snapshot at n_kv, else the newest checkpoint below
    int32_t n = s.n_kv;
    const common_prompt_checkpoint * ckpt = nullptr;
    if (s.n_rec != s.n_kv || s.rec.empty()) {
        for (auto it = s.checkpoints.rbegin(); it != s.checkpoints.rend(); ++it) {
            if (it->n_tokens > 0 && it->n_tokens <= s.n_kv && !it->empty()) {
                ckpt = &*it;
                break;
            }
        }
        n = ckpt ? (int32_t) ckpt->n_tokens : 0;
        if (n < s.n_kv) {
            SRV_WRN("session %llu: no recurrent snapshot at %d tokens, restoring %d\n", (unsigned long long) s.id, s.n_kv, n);
        }
    }
    const int32_t n_dft = std::min(s.n_kv_dft, n);

    s.restore_bytes    = 0;
    s.t_restore_us     = ggml_time_us();
    s.restore_prefetch = prefetch;

    bool ok = true;

    std::vector<llama_state_chunk> views;
    if (n > 0) {
        for (auto & seg : s.segs) {
            const llama_pos end = std::min<llama_pos>(seg.end, seg.dft ? n_dft : n);
            if (end <= seg.pos0) {
                continue;
            }
            views.clear();
            s.log.views(arena, seg.off, seg.size, views);

            llama_state_xfer_t h = nullptr;
            const size_t k = llama_state_seq_set_delta_data_async(seg.dft ? ctx_dft : ctx_tgt, views.data(), views.size(),
                    seg.size, seq, seg.pos0, end, SESSION_KV_FLAGS, &h);
            s.restore_xfers.emplace_back(h);
            // the graphs that read these cells wait for the copies on the device, the host does not
            s.restore_xfers.back().order();
            if (k != seg.size) {
                ok = false;
                break;
            }
            s.restore_bytes += seg.size;
        }

        if (ok && ckpt) {
            ckpt->load_tgt(ctx_tgt, seq, SESSION_REC_FLAGS);
            ckpt->load_dft(ctx_dft, seq, SESSION_REC_FLAGS);
            s.restore_bytes += ckpt->data_tgt.size() + ckpt->data_dft.size();
        } else if (ok) {
            auto set_rec = [&](llama_context * ctx, const common_state_buf & buf) {
                if (buf.empty()) {
                    return true;
                }
                const auto v = buf.views();
                llama_state_xfer_t h = nullptr;
                const size_t k = llama_state_seq_set_data_async(ctx, v.data(), v.size(), buf.size(), seq,
                        SESSION_REC_FLAGS | LLAMA_STATE_SEQ_FLAGS_ORDERED, &h);
                s.restore_xfers.emplace_back(h);
                s.restore_bytes += buf.size();
                return k == buf.size();
            };
            ok = set_rec(ctx_tgt, s.rec) && (!ctx_dft || set_rec(ctx_dft, s.rec_dft));
        }
    }
    s.restore_ordered = ok;

    if (!ok) {
        SRV_ERR("session %llu: failed to restore the mirror, its prompt is processed from scratch\n", (unsigned long long) s.id);
        restore_wait(s);
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, -1, -1);
        }
        truncate(s, 0, 0);
        n = 0;
    } else if (n < s.n_kv || n_dft < s.n_kv_dft) {
        truncate(s, n, n_dft);
    }

    ckpts.splice(ckpts.end(), s.checkpoints);

    s.id_slot   = seq;
    s.t_used_us = ggml_time_us();
    s.t_used_s  = session_now_s();
    s.n_restore++;
    if (prefetch) {
        s.n_prefetch++;
        n_prefetch_total++;
    }

    return n;
}

bool server_session_store::restore_done(server_session & s) {
    if (s.restore_xfers.empty()) {
        return true;
    }
    for (auto & x : s.restore_xfers) {
        if (!x.done()) {
            return false;
        }
    }
    int64_t t_us   = 0;
    int64_t t_end  = s.t_restore_us;
    for (auto & x : s.restore_xfers) {
        t_us += std::max<int64_t>(0, x.t_us);
        t_end = std::max(t_end, x.t_end_us);
    }
    s.restore_xfers.clear();
    s.restore_ordered      = false;
    s.last_restore_bytes   = s.restore_bytes;
    s.last_restore_ms      = t_us / 1e3;
    s.last_restore_end_us  = t_end;
    s.last_restore_wall_ms = (t_end - s.t_restore_us) / 1e3;
    return true;
}

void server_session_store::restore_wait(server_session & s) {
    for (auto & x : s.restore_xfers) {
        x.wait();
    }
    restore_done(s);
}

void server_session_store::park(server_session & s, std::list<common_prompt_checkpoint> && ckpts) {
    // a checkpoint past the mirror belongs to tokens the mirror does not hold
    ckpts.remove_if([&](const common_prompt_checkpoint & c) { return c.n_tokens > s.n_kv || c.empty(); });
    s.checkpoints.splice(s.checkpoints.end(), ckpts);
    s.id_slot      = -1;
    s.stale        = false; // the cells the mirror lags are gone
    s.file_current = false;
    s.t_used_us    = ggml_time_us();
}

//
// tiering
//

void server_session_store::release_ram(server_session & s) {
    s.file_log     = s.log.size;
    s.file_rec     = s.rec.size();
    s.file_rec_dft = s.rec_dft.size();
    s.file_ckpt.clear();
    for (auto & c : s.checkpoints) {
        s.file_ckpt.emplace_back(c.data_tgt.size(), c.data_dft.size());
        c.data_tgt.clear();
        c.data_dft.clear();
    }
    s.log.release(arena);
    s.rec.clear();
    s.rec_dft.clear();
}

void server_session_store::submit(std::shared_ptr<job> j) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        jobs[j->id] = j;
        queue.push_back(std::move(j));
    }
    cv.notify_one();
}

bool server_session_store::demote(server_session & s) {
    if (params.dir.empty() || s.id_slot >= 0 || s.tier != SERVER_SESSION_RAM) {
        return false;
    }

    sync_wait(s);
    restore_wait(s);

    if (s.file_current) {
        release_ram(s);
        s.tier = SERVER_SESSION_DISK;
        s.n_demote++;
        n_demote_total++;
        return true;
    }

    auto j = std::make_shared<job>();
    j->kind = job::WRITE;
    j->id   = s.id;
    j->path = path_of(s.id);
    j->head = session_encode_head(s);

    auto add = [&](const std::vector<llama_state_chunk> & v) {
        for (const auto & c : v) {
            j->spans.emplace_back(c.data, c.size);
        }
    };
    std::vector<llama_state_chunk> v;
    s.log.views(arena, 0, s.log.size, v);
    add(v);
    add(s.rec.views());
    add(s.rec_dft.views());
    for (const auto & c : s.checkpoints) {
        c.data_tgt.wait();
        c.data_dft.wait();
        add(c.data_tgt.views());
        add(c.data_dft.views());
    }

    s.file_head = j->head.size();
    s.tier = SERVER_SESSION_DEMOTING;
    submit(std::move(j));
    return true;
}

bool server_session_store::promote(server_session & s) {
    switch (s.tier) {
        case SERVER_SESSION_RAM:
            return true;
        case SERVER_SESSION_PROMOTING:
            return false;
        case SERVER_SESSION_DEMOTING:
            {
                // still in the arena: stop writing it
                std::lock_guard<std::mutex> lock(mutex);
                const auto it = jobs.find(s.id);
                if (it != jobs.end()) {
                    it->second->cancel = true;
                }
                return false;
            }
        case SERVER_SESSION_DISK:
            break;
    }

    auto buf_chunks = [&](size_t n) { return n >= 4096 ? arena.n_chunks(n) : 0; };
    size_t need = arena.n_chunks(s.file_log) + buf_chunks(s.file_rec) + buf_chunks(s.file_rec_dft);
    for (const auto & c : s.file_ckpt) {
        need += buf_chunks(c.first) + buf_chunks(c.second);
    }
    if (arena.n_free() < need && !make_room(need, false, s.id)) {
        return false;
    }

    bool ok = s.log.grow(arena, s.file_log);
    ok = ok && s.rec.alloc(&arena, s.file_rec) && s.rec_dft.alloc(&arena, s.file_rec_dft);
    {
        size_t i = 0;
        for (auto & c : s.checkpoints) {
            ok = ok && c.data_tgt.alloc(&arena, s.file_ckpt[i].first) && c.data_dft.alloc(&arena, s.file_ckpt[i].second);
            i++;
        }
    }
    if (!ok) {
        release_ram(s);
        return false;
    }

    auto j = std::make_shared<job>();
    j->kind      = job::READ;
    j->id        = s.id;
    j->path      = path_of(s.id);
    j->head_size = s.file_head;

    auto add = [&](const std::vector<llama_state_chunk> & v) {
        for (const auto & c : v) {
            j->spans.emplace_back(c.data, c.size);
        }
    };
    std::vector<llama_state_chunk> v;
    s.log.views(arena, 0, s.file_log, v);
    add(v);
    add(s.rec.views());
    add(s.rec_dft.views());
    for (const auto & c : s.checkpoints) {
        add(c.data_tgt.views());
        add(c.data_dft.views());
    }

    s.tier = SERVER_SESSION_PROMOTING;
    submit(std::move(j));
    return false;
}

void server_session_store::finish_job(server_session & s, job & j) {
    const bool ok = j.state.load() == 1;

    if (j.kind == job::WRITE) {
        io_write_us += j.t_us;
        if (ok) {
            size_t size = j.head.size();
            for (const auto & sp : j.spans) {
                size += sp.second;
            }
            io_bytes_written += size;
            disk_bytes = disk_bytes - std::min(disk_bytes, s.file_bytes) + size;
            s.file_bytes   = size;
            s.file_current = true;
            release_ram(s);
            s.tier = SERVER_SESSION_DISK;
            s.n_demote++;
            n_demote_total++;
            SRV_INF("session %llu demoted to disk: %.1f MiB in %.2f s\n", (unsigned long long) s.id, size / 1048576.0, j.t_us / 1e6);
            enforce_disk_limit(s.id);
        } else {
            if (!j.cancel.load()) {
                SRV_ERR("session %llu: failed to write %s (%s), it stays in RAM\n", (unsigned long long) s.id, j.path.c_str(), j.error.c_str());
            }
            s.tier = SERVER_SESSION_RAM;
        }
    } else {
        io_read_us += j.t_us;
        if (ok) {
            size_t size = 0;
            for (const auto & sp : j.spans) {
                size += sp.second;
            }
            io_bytes_read += size;
            s.log.size     = s.file_log;
            s.file_current = true;
            s.tier = SERVER_SESSION_RAM;
            s.n_promote++;
            n_promote_total++;
            SRV_INF("session %llu promoted from disk: %.1f MiB in %.2f s\n", (unsigned long long) s.id, size / 1048576.0, j.t_us / 1e6);
        } else {
            SRV_ERR("session %llu: failed to read %s (%s)\n", (unsigned long long) s.id, j.path.c_str(), j.error.c_str());
            release_ram(s);
            s.tier = SERVER_SESSION_DISK;
            drop(s.id, "its file is unreadable");
        }
    }
}

void server_session_store::run_worker() {
    while (true) {
        std::shared_ptr<job> j;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&]() { return stop || !queue.empty(); });
            if (queue.empty()) {
                return;
            }
            j = queue.front();
            queue.pop_front();
        }

        run_job(*j);

        if (wake) {
            wake();
        }
    }
}

void server_session_store::run_job(job & j) {
    const int64_t t0 = ggml_time_us();

    auto fail = [&](const char * what) {
        j.error = std::string(what) + ": " + strerror(errno);
        j.t_us  = ggml_time_us() - t0;
        j.state = 2;
    };

    if (j.kind == job::WRITE) {
        const std::string tmp = j.path.string() + ".tmp";
        const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) {
            return fail("open");
        }
        bool ok = write_all(fd, j.head.data(), j.head.size(), j.cancel);
        for (const auto & sp : j.spans) {
            ok = ok && write_all(fd, sp.first, sp.second, j.cancel);
        }
        ok = ok && ::fdatasync(fd) == 0;
        if (ok) {
            // the data is on disk, the page cache would only compete with the pinned arena
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        }
        const int err = errno;
        ::close(fd);
        errno = err;
        if (!ok || ::rename(tmp.c_str(), j.path.c_str()) != 0) {
            const int err2 = errno;
            ::unlink(tmp.c_str());
            errno = err2;
            return fail("write");
        }
    } else {
        const int fd = ::open(j.path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return fail("open");
        }
        bool ok = ::lseek(fd, (off_t) j.head_size, SEEK_SET) == (off_t) j.head_size;
        for (const auto & sp : j.spans) {
            ok = ok && read_all(fd, sp.first, sp.second);
        }
        if (ok) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        }
        const int err = errno;
        ::close(fd);
        errno = err;
        if (!ok) {
            return fail("read");
        }
    }

    j.t_us  = ggml_time_us() - t0;
    j.state = 1;
}

bool server_session_store::make_room(size_t n_chunks, bool wait, uint64_t except) {
    while (arena.n_free() < n_chunks) {
        size_t incoming = 0;
        for (const auto & it : sessions) {
            if (it.second->tier == SERVER_SESSION_DEMOTING) {
                incoming += arena_chunks(*it.second);
            }
        }

        if (arena.n_free() + incoming < n_chunks) {
            // the least recently used parked mirror goes; the sessions of queued requests last, farthest first
            server_session * victim = nullptr;
            int victim_rank = -1;
            for (auto & it : sessions) {
                auto & s = *it.second;
                if (s.id == except || s.id_slot >= 0 || s.tier != SERVER_SESSION_RAM || !s.restore_xfers.empty()) {
                    continue;
                }
                const auto w = std::find(wanted.begin(), wanted.end(), s.id);
                const int rank = w == wanted.end() ? -1 : (int) (wanted.end() - w); // 1 = last in the queue
                const bool better = victim == nullptr ||
                    (rank < 0 && victim_rank >= 0) ||
                    (rank < 0 && victim_rank < 0 && s.t_used_us < victim->t_used_us) ||
                    (rank >= 0 && victim_rank >= 0 && rank < victim_rank);
                if (better) {
                    victim = &s;
                    victim_rank = rank;
                }
            }
            if (victim != nullptr && demote(*victim)) {
                continue;
            }
            if (incoming == 0) {
                return false;
            }
        }

        if (!wait) {
            return false;
        }

        // wait for a demotion to complete
        std::shared_ptr<job> j;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto & it : jobs) {
                if (it.second->kind == job::WRITE) {
                    j = it.second;
                    break;
                }
            }
        }
        if (!j) {
            return false;
        }
        while (j->state.load() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        reap_jobs();
    }
    return true;
}

bool server_session_store::reap_jobs() {
    std::vector<std::shared_ptr<job>> done;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = jobs.begin(); it != jobs.end();) {
            if (it->second->state.load() != 0) {
                done.push_back(it->second);
                it = jobs.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto & j : done) {
        server_session * s = get(j->id);
        if (s != nullptr) {
            finish_job(*s, *j);
        }
    }
    return !done.empty();
}

size_t server_session_store::arena_chunks(const server_session & s) const {
    auto buf = [&](const common_state_buf & b) { return b.in_arena() ? arena.n_chunks(b.size()) : 0; };
    size_t res = s.log.chunks.size() + buf(s.rec) + buf(s.rec_dft);
    for (const auto & c : s.checkpoints) {
        res += buf(c.data_tgt) + buf(c.data_dft);
    }
    return res;
}

bool server_session_store::poll(const std::vector<uint64_t> & wanted_ids) {
    wanted = wanted_ids;

    const bool changed = reap_jobs();

    for (auto & it : sessions) {
        auto & s = *it.second;
        if (s.t_synced_us == 0 && s.t_sync_us != 0) {
            sync_done(s);
        }
    }

    // the sessions of queued requests come back first, in order, one read at a time
    bool reading = false;
    for (const auto & it : sessions) {
        reading = reading || it.second->tier == SERVER_SESSION_PROMOTING;
    }
    for (uint64_t id : wanted) {
        server_session * s = get(id);
        if (s == nullptr || s->tier == SERVER_SESSION_RAM) {
            continue;
        }
        if (s->tier == SERVER_SESSION_DEMOTING) {
            promote(*s); // cancels the write
            continue;
        }
        if (!reading && s->tier == SERVER_SESSION_DISK) {
            reading = true;
            promote(*s);
        }
        break;
    }

    // room for the next syncs and checkpoints
    const size_t reserve = arena.n_chunks(params.reserve);
    if (arena.n_free() < reserve) {
        make_room(reserve, false);
    }

    return changed;
}

void server_session_store::enforce_disk_limit(uint64_t except) {
    if (params.disk_limit == 0) {
        return;
    }
    while (disk_bytes > params.disk_limit) {
        // stale copies of sessions in RAM go first, then the least recently used demoted sessions
        server_session * victim = nullptr;
        for (auto & it : sessions) {
            auto & s = *it.second;
            if (s.id == except || s.file_bytes == 0 || s.tier == SERVER_SESSION_DEMOTING || s.tier == SERVER_SESSION_PROMOTING) {
                continue;
            }
            const bool on_disk = s.tier == SERVER_SESSION_DISK;
            if (victim == nullptr ||
                (!on_disk && victim->tier == SERVER_SESSION_DISK) ||
                (on_disk == (victim->tier == SERVER_SESSION_DISK) && s.t_used_us < victim->t_used_us)) {
                victim = &s;
            }
        }
        if (victim == nullptr) {
            return;
        }
        if (victim->tier == SERVER_SESSION_DISK) {
            drop(victim->id, "the session disk budget is exceeded");
        } else {
            std::error_code ec;
            std::filesystem::remove(path_of(victim->id), ec);
            disk_bytes -= std::min(disk_bytes, victim->file_bytes);
            victim->file_bytes   = 0;
            victim->file_current = false;
        }
    }
}

void server_session_store::drop(uint64_t id, const char * reason) {
    server_session * s = get(id);
    if (s == nullptr) {
        return;
    }
    GGML_ASSERT(s->id_slot < 0);

    std::shared_ptr<job> j;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = jobs.find(id);
        if (it != jobs.end()) {
            j = it->second;
            j->cancel = true;
        }
    }
    if (j) {
        while (j->state.load() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        std::lock_guard<std::mutex> lock(mutex);
        jobs.erase(id);
    }

    SRV_WRN("session %llu deleted: %s\n", (unsigned long long) id, reason);

    sync_wait(*s);
    restore_wait(*s);
    release_ram(*s);

    if (s->file_bytes > 0 || s->tier != SERVER_SESSION_RAM) {
        std::error_code ec;
        std::filesystem::remove(path_of(id), ec);
        disk_bytes -= std::min(disk_bytes, s->file_bytes);
    }

    index.erase(id);
    sessions.erase(id);
    n_deleted++;
}

void server_session_store::erase(uint64_t id) {
    drop(id, "its inactivity TTL expired");
}

std::vector<uint64_t> server_session_store::expired(uint64_t now_s) const {
    std::vector<uint64_t> res;
    if (params.ttl_s == 0) {
        return res;
    }
    for (const auto & it : sessions) {
        const auto & s = *it.second;
        if (s.t_used_s != 0 && now_s >= s.t_used_s && now_s - s.t_used_s >= params.ttl_s) {
            res.push_back(s.id);
        }
    }
    return res;
}

uint64_t server_session_store::load_dir() {
    if (params.dir.empty()) {
        return next_id;
    }

    const uint64_t now_s = session_now_s();

    for (const auto & entry : std::filesystem::directory_iterator(params.dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            // an interrupted write
            std::error_code ec;
            std::filesystem::remove(entry.path(), ec);
            continue;
        }
        if (name.rfind("session-", 0) != 0 || name.size() < 13 || name.compare(name.size() - 4, 4, ".bin") != 0) {
            continue;
        }

        try {
            const int fd = ::open(entry.path().c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                throw std::runtime_error(strerror(errno));
            }
            head_fixed h {};
            std::vector<uint8_t> head;
            bool ok = read_all(fd, reinterpret_cast<uint8_t *>(&h), sizeof(h));
            ok = ok && h.magic == SESSION_FILE_MAGIC && h.version == SESSION_FILE_VERSION && h.head_size >= sizeof(h) &&
                h.head_size <= (size_t) entry.file_size();
            if (ok) {
                head.resize(h.head_size);
                memcpy(head.data(), &h, sizeof(h));
                ok = read_all(fd, head.data() + sizeof(h), h.head_size - sizeof(h));
            }
            ::close(fd);
            if (!ok) {
                throw std::runtime_error("unreadable or unsupported head");
            }

            auto s = std::make_unique<server_session>();
            session_decode_head(head.data(), head.size(), *s);

            size_t payload = s->file_log + s->file_rec + s->file_rec_dft;
            for (const auto & c : s->file_ckpt) {
                payload += c.first + c.second;
            }
            if (s->file_head + payload != (size_t) entry.file_size() || sessions.count(s->id)) {
                throw std::runtime_error("size or identity mismatch");
            }

            const auto age = std::chrono::duration_cast<std::chrono::seconds>(
                    std::filesystem::file_time_type::clock::now() - entry.last_write_time()).count();

            s->tier         = SERVER_SESSION_DISK;
            s->file_bytes   = entry.file_size();
            s->file_current = true;
            s->t_used_s     = age > 0 && (uint64_t) age < now_s ? now_s - age : now_s;
            s->t_used_us    = ggml_time_us() - age * 1000000;

            disk_bytes += s->file_bytes;
            index.update(s->id, s->tokens);
            next_id = std::max(next_id, s->id + 1);

            SRV_INF("session %llu found on disk: %zu tokens, %.1f MiB\n", (unsigned long long) s->id, s->tokens.size(), s->file_bytes / 1048576.0);

            const uint64_t id = s->id;
            sessions[id] = std::move(s);
        } catch (const std::exception & e) {
            SRV_ERR("ignoring session file %s: %s\n", entry.path().c_str(), e.what());
        }
    }

    enforce_disk_limit(0);

    return next_id;
}

server_session_store::stats_t server_session_store::stats() const {
    stats_t st;
    st.arena_chunk = arena.chunk_size();
    st.arena_bytes = arena.n_total() * arena.chunk_size();
    st.arena_free  = arena.n_free() * arena.chunk_size();
    st.disk_bytes  = disk_bytes;
    st.n_sessions  = sessions.size();
    for (const auto & it : sessions) {
        const auto & s = *it.second;
        if (s.id_slot >= 0) {
            st.n_gpu++;
        }
        switch (s.tier) {
            case SERVER_SESSION_RAM:       st.n_ram++; st.ram_bytes += s.bytes(); break;
            case SERVER_SESSION_DISK:      st.n_disk++; break;
            case SERVER_SESSION_DEMOTING:
            case SERVER_SESSION_PROMOTING: st.n_io++; st.ram_bytes += s.bytes(); break;
        }
    }
    st.n_demote         = n_demote_total;
    st.n_promote        = n_promote_total;
    st.n_deleted        = n_deleted;
    st.n_sync_stale     = n_sync_stale;
    st.n_prefetch       = n_prefetch_total;
    st.n_prefetch_hit   = n_prefetch_hit_total;
    st.io_bytes_written = io_bytes_written;
    st.io_bytes_read    = io_bytes_read;
    st.io_write_s       = io_write_us / 1e6;
    st.io_read_s        = io_read_us / 1e6;
    return st;
}

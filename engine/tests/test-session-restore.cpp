// Byte-exact restore gate for the per-agent session store (tools/server/server-sessions.h), on the real model and GPU.
//
//   test-session-restore MODEL.gguf SESSION_DIR
//
// A session is decoded into sequence 0 and mirrored, grows after that sync (so the mirror holds two KV deltas),
// is mirrored again, parked, and restored into sequence 1 while sequence 0 holds other cells (so the restored cells
// land in a different cell range). Its attention KV (every layer, K and V, matched by position) and its recurrent
// state (with the pending ReplaySSM records) are compared byte for byte against the state before the park. Then the
// mirror is demoted to a file, a fresh store finds it at startup, promotes it and restores it again, and the same
// comparison runs. Pass = 0 differing bytes in both. Run it under deploy/gpu-lock.sh (bench/session_restore_check.sh).

#include "server-sessions.h"

#include "common.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

// the state of one sequence: attention KV rows by position (per layer K, then V), and the recurrent state bytes
struct seq_state {
    std::map<llama_pos, std::vector<std::vector<uint8_t>>> rows;
    std::vector<uint8_t> rec;
};

template <typename T>
static T rd(const uint8_t * & p) {
    T v;
    memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

// parses a KV delta blob (see llama_kv_cache::state_write_cells) for a cell-ext stride, false if it does not fit
static bool parse_kv(const std::vector<uint8_t> & blob, size_t ext, seq_state & st) {
    const uint8_t * p   = blob.data();
    const uint8_t * end = blob.data() + blob.size();
    rd<uint32_t>(p); // magic
    rd<int32_t>(p);  // pos_min
    const uint32_t n_stream = rd<uint32_t>(p);
    st.rows.clear();
    for (uint32_t s = 0; s < n_stream; ++s) {
        if (p + 4 > end) {
            return false;
        }
        const uint32_t n = rd<uint32_t>(p);
        if (n == 0) {
            continue;
        }
        std::vector<llama_pos> pos(n);
        for (uint32_t i = 0; i < n; ++i) {
            if (p + 8 + ext > end) {
                return false;
            }
            pos[i] = rd<int32_t>(p);
            const uint32_t n_seq = rd<uint32_t>(p);
            if (n_seq != 1) {
                return false;
            }
            p += ext + 4;
        }
        if (p + 8 > end) {
            return false;
        }
        const uint32_t v_trans = rd<uint32_t>(p);
        const uint32_t n_layer = rd<uint32_t>(p);
        if (v_trans != 0 || n_layer == 0 || n_layer > 1024) {
            return false;
        }
        for (int kv = 0; kv < 2; ++kv) {
            for (uint32_t l = 0; l < n_layer; ++l) {
                if (p + 12 > end) {
                    return false;
                }
                rd<int32_t>(p); // type
                const uint64_t row = rd<uint64_t>(p);
                if (row == 0 || p + row * n > end) {
                    return false;
                }
                for (uint32_t i = 0; i < n; ++i) {
                    st.rows[pos[i]].emplace_back(p, p + row);
                    p += row;
                }
            }
        }
    }
    return p == end;
}

static seq_state snapshot(llama_context * ctx, llama_seq_id seq) {
    seq_state st;

    const size_t n = llama_state_seq_get_delta_size_ext(ctx, seq, 0, LLAMA_STATE_SEQ_FLAGS_KV_ONLY);
    std::vector<uint8_t> blob(n);
    llama_state_chunk chunk { blob.data(), blob.size() };
    llama_state_xfer_t x = nullptr;
    GGML_ASSERT(llama_state_seq_get_delta_data_async(ctx, &chunk, 1, seq, 0, LLAMA_STATE_SEQ_FLAGS_KV_ONLY, &x) == n);
    llama_state_xfer_free(x);

    bool ok = false;
    for (size_t ext = 0; ext <= 64 && !ok; ext += 4) {
        ok = parse_kv(blob, ext, st);
    }
    GGML_ASSERT(ok && "cannot parse the KV delta");

    // the recurrent state, without its header (magic, sequence id)
    const size_t nr = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<uint8_t> rec(nr);
    GGML_ASSERT(llama_state_seq_get_data_ext(ctx, rec.data(), rec.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == nr);
    st.rec.assign(rec.begin() + 8, rec.end());
    return st;
}

// differing bytes (a missing or extra position counts all of its bytes)
static size_t compare(const seq_state & a, const seq_state & b, const char * what) {
    size_t diff = 0, total = 0;
    for (const auto & [pos, rows] : a.rows) {
        const auto it = b.rows.find(pos);
        for (size_t l = 0; l < rows.size(); ++l) {
            total += rows[l].size();
            if (it == b.rows.end() || l >= it->second.size() || it->second[l].size() != rows[l].size()) {
                diff += rows[l].size();
                continue;
            }
            for (size_t i = 0; i < rows[l].size(); ++i) {
                diff += rows[l][i] != it->second[l][i];
            }
        }
    }
    for (const auto & [pos, rows] : b.rows) {
        if (!a.rows.count(pos)) {
            for (const auto & r : rows) {
                diff += r.size();
            }
        }
    }
    size_t diff_rec = a.rec.size() != b.rec.size() ? std::max(a.rec.size(), b.rec.size()) : 0;
    if (diff_rec == 0) {
        for (size_t i = 0; i < a.rec.size(); ++i) {
            diff_rec += a.rec[i] != b.rec[i];
        }
    }
    printf("%s: %zu cells, KV %.1f MiB: %zu differing bytes; recurrent %.1f MiB: %zu differing bytes\n", what,
            a.rows.size(), total / 1048576.0, diff, a.rec.size() / 1048576.0, diff_rec);
    return diff + diff_rec;
}

static void decode(llama_context * ctx, llama_seq_id seq, const llama_tokens & t, int32_t from, int32_t to) {
    llama_batch batch = llama_batch_init(1024, 0, 1);
    for (int32_t i = from; i < to;) {
        batch.n_tokens = 0;
        for (; i < to && batch.n_tokens < 1024; ++i) {
            const int k = batch.n_tokens++;
            batch.token[k]     = t[i];
            batch.pos[k]       = i;
            batch.n_seq_id[k]  = 1;
            batch.seq_id[k][0] = seq;
            batch.logits[k]    = i == to - 1;
        }
        GGML_ASSERT(llama_decode(ctx, batch) == 0);
    }
    llama_synchronize(ctx);
    llama_batch_free(batch);
}

static bool wait_ram(server_session_store & store, server_session & s) {
    for (int i = 0; i < 12000 && s.tier != SERVER_SESSION_RAM; ++i) {
        store.promote(s);
        store.poll({});
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return s.tier == SERVER_SESSION_RAM;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf SESSION_DIR\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    GGML_ASSERT(model);

    auto cp = llama_context_default_params();
    cp.n_ctx           = 32768;
    cp.n_batch         = 1024;
    cp.n_ubatch        = 1024;
    cp.n_seq_max       = 2;
    cp.kv_unified      = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k          = GGML_TYPE_Q5_0;
    cp.type_v          = GGML_TYPE_Q5_0;
    cp.n_rs_seq        = 3; // ReplaySSM, as with MTP: the recurrent state carries pending records
    llama_context * ctx = llama_init_from_model(model, cp);
    GGML_ASSERT(ctx);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_tokens t(6000), filler(3000);
    for (int32_t i = 0; i < (int32_t) t.size(); ++i) {
        t[i] = 1000 + (int32_t) ((i * 7919ll) % (n_vocab - 2000));
    }
    for (int32_t i = 0; i < (int32_t) filler.size(); ++i) {
        filler[i] = 1000 + (int32_t) ((i * 104729ll + 17) % (n_vocab - 2000));
    }

    common_state_arena arena((size_t) 4 << 30, (size_t) 16 << 20);
    server_session_store::params_t sp;
    sp.dir = argv[2];
    sp.reserve = 0;

    std::filesystem::remove_all(sp.dir);

    llama_memory_t mem = llama_get_memory(ctx);

    size_t diff = 0;
    seq_state golden;
    {
        server_session_store store(arena, ctx, nullptr, sp, {});
        server_session & s = store.create(1);

        decode(ctx, 0, t, 0, 4096);
        GGML_ASSERT(store.sync(s, 0, llama_tokens(t.begin(), t.begin() + 4096)) > 0);
        store.sync_wait(s);

        // the session grows after its last sync: the next sync appends a second delta
        decode(ctx, 0, t, 4096, 5000);
        golden = snapshot(ctx, 0);
        GGML_ASSERT(store.sync(s, 0, llama_tokens(t.begin(), t.begin() + 5000)) > 0);
        store.sync_wait(s);
        printf("mirror: %zu deltas, %.1f MiB\n", s.segs.size(), s.bytes() / 1048576.0);
        GGML_ASSERT(s.segs.size() == 2);

        // park, and let another sequence take the low cells
        store.park(s, {});
        llama_memory_seq_rm(mem, 0, -1, -1);
        decode(ctx, 0, filler, 0, (int32_t) filler.size());

        std::list<common_prompt_checkpoint> ckpts;
        GGML_ASSERT(store.restore(s, 1, ckpts, false) == 5000);
        store.restore_wait(s);
        llama_synchronize(ctx);
        diff += compare(golden, snapshot(ctx, 1), "delta restore into other cells");

        // demote it to its file
        llama_memory_seq_rm(mem, 1, -1, -1);
        store.park(s, {});
        store.make_room(arena.n_total(), /* wait */ true);
        GGML_ASSERT(s.tier == SERVER_SESSION_DISK);
        printf("demoted: %.1f MiB file\n", s.file_bytes / 1048576.0);
    }
    {
        // a fresh store (a restart) finds the file, promotes and restores it
        server_session_store store(arena, ctx, nullptr, sp, {});
        store.load_dir();
        server_session * s = store.get(1);
        GGML_ASSERT(s && s->tier == SERVER_SESSION_DISK);
        GGML_ASSERT(wait_ram(store, *s));

        std::list<common_prompt_checkpoint> ckpts;
        GGML_ASSERT(store.restore(*s, 1, ckpts, false) == 5000);
        store.restore_wait(*s);
        llama_synchronize(ctx);
        diff += compare(golden, snapshot(ctx, 1), "restore after disk promotion (fresh store)");
        llama_memory_seq_rm(mem, 1, -1, -1);
        store.park(*s, {});
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("%s: %zu differing bytes\n", diff == 0 ? "PASS" : "FAIL", diff);
    return diff == 0 ? 0 : 1;
}

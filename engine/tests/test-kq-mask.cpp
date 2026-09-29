// The KQ mask built on a backend by ggml_kq_mask() from llama_kv_mask_set_keys() must be bitwise equal to the host
// mask (llama_kv_mask_set_host()) whenever llama_kv_mask_on_device() accepts the ubatch, and the cases it must refuse
// (M-RoPE 2D rule hiding a cell, more than 8 sequences in a stream) must be refused.
// The cache layouts mirror production decode: a ~286k-cell unified cache holding six ~47.6k sequences in interleaved
// chunks with holes, idle sequences, a prefix shared by two batch sequences, and 6 (1 row per sequence) or 24 (4 rows
// per sequence, MTP verify) row ubatches with M-RoPE positions.

#include "ggml.h"
#include "ggml-backend.h"

#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-kv-cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct ubatch_data {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_data;
    std::vector<llama_seq_id *> seq_id;
    llama_ubatch ub = {};
};

// one row per (seq, pos) in rows, in that order; n_pos = 4 gives M-RoPE sections, filled from xy (text: all == pos)
static void make_ubatch(ubatch_data & u, const std::vector<std::pair<llama_seq_id, llama_pos>> & rows, uint32_t n_pos,
        const std::vector<std::pair<llama_pos, llama_pos>> * xy = nullptr) {
    const uint32_t n = rows.size();
    u.pos.assign(n*n_pos, 0);
    u.n_seq_id.assign(n, 1);
    u.seq_id_data.resize(n);
    u.seq_id.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        u.seq_id_data[i] = rows[i].first;
        u.seq_id[i]      = &u.seq_id_data[i];
        for (uint32_t j = 0; j < n_pos; ++j) {
            u.pos[j*n + i] = rows[i].second;
        }
        if (xy && n_pos >= 3) {
            u.pos[1*n + i] = (*xy)[i].first;  // y
            u.pos[2*n + i] = (*xy)[i].second; // x
        }
    }
    u.ub = {};
    u.ub.n_tokens     = n;
    u.ub.n_seq_tokens = 1;
    u.ub.n_seqs       = n;
    u.ub.n_pos        = n_pos;
    u.ub.pos          = u.pos.data();
    u.ub.n_seq_id     = u.n_seq_id.data();
    u.ub.seq_id       = u.seq_id.data();
}

struct mask_case {
    std::string name;
    llama_kv_cells_vec    cells;
    std::vector<uint32_t> seq_to_stream;
    ubatch_data           u;
    int64_t n_kv;
    int64_t n_stream;
    bool    causal;
    bool    expect_device;
};

static std::mt19937 rng(1234);

// n_seqs sequences of len cells each (plus n_idle idle ones), in chunks of `chunk` cells placed round-robin with ~1% holes
static llama_kv_cells make_unified(uint32_t n_cells, int n_seqs, int n_idle, uint32_t len, uint32_t chunk,
        std::vector<llama_pos> & next_pos) {
    llama_kv_cells cells;
    cells.resize(n_cells);
    next_pos.assign(n_seqs + n_idle, 0);
    std::vector<uint32_t> left(n_seqs + n_idle, len);
    uint32_t i = 0;
    for (bool any = true; any && i < n_cells; ) {
        any = false;
        for (int s = 0; s < n_seqs + n_idle && i < n_cells; ++s) {
            for (uint32_t c = 0; c < chunk && left[s] > 0 && i < n_cells; ++c, ++i) {
                if (rng() % 100 == 0) {
                    continue; // hole
                }
                const llama_pos p = next_pos[s]++;
                cells.pos_set(i, p);
                llama_kv_cell_ext ext;
                ext.x = p;
                ext.y = p;
                cells.ext_set(i, ext);
                cells.seq_add(i, s);
                left[s]--;
                any = true;
            }
        }
    }
    return cells;
}

// the last n_rows positions of each batch sequence (the ubatch's own cells, already stored as after apply())
static std::vector<std::pair<llama_seq_id, llama_pos>> tail_rows(const std::vector<llama_seq_id> & seqs,
        const std::vector<llama_pos> & next_pos, int n_rows) {
    std::vector<std::pair<llama_seq_id, llama_pos>> rows;
    for (llama_seq_id s : seqs) {
        for (int r = 0; r < n_rows; ++r) {
            rows.push_back({ s, next_pos[s] - n_rows + r });
        }
    }
    return rows;
}

static std::vector<mask_case> make_cases() {
    std::vector<mask_case> res;
    const uint32_t n_cells = 286720;

    for (int n_rows : { 1, 4 }) {
        for (uint32_t n_pos : { 1u, 4u }) {
            for (bool shared : { false, true }) {
                mask_case c;
                std::vector<llama_pos> next_pos;
                c.cells.push_back(make_unified(n_cells, 6, 2, 47600, 1024, next_pos));
                if (shared) {
                    // seq 1 also sees the first 300 cells of seq 0 (a copied prefix): cells in two batch sequences
                    int added = 0;
                    for (uint32_t i = 0; i < n_cells && added < 300; ++i) {
                        if (!c.cells[0].is_empty(i) && c.cells[0].seq_has(i, 0)) {
                            c.cells[0].seq_add(i, 1);
                            added++;
                        }
                    }
                }
                c.seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
                make_ubatch(c.u, tail_rows({ 0, 1, 2, 3, 4, 5 }, next_pos, n_rows), n_pos);
                c.n_kv     = n_cells;
                c.n_stream = 1;
                c.causal   = true;
                c.expect_device = true;
                c.name = std::to_string(6*n_rows) + " rows, n_pos " + std::to_string(n_pos) + (shared ? ", shared prefix" : "");
                res.push_back(std::move(c));
            }
        }
    }

    // non-causal, n_kv shorter than the cache and not a multiple of 8
    {
        mask_case c;
        std::vector<llama_pos> next_pos;
        c.cells.push_back(make_unified(8192, 3, 1, 2000, 256, next_pos));
        c.seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
        make_ubatch(c.u, tail_rows({ 0, 2 }, next_pos, 3), 1);
        c.n_kv = 8189; c.n_stream = 1; c.causal = false; c.expect_device = true;
        c.name = "non-causal, 6 rows, n_kv 8189";
        res.push_back(std::move(c));
    }

    // one stream per sequence (non-unified cache)
    {
        mask_case c;
        std::vector<llama_pos> next_pos;
        c.seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
        std::vector<std::pair<llama_seq_id, llama_pos>> rows;
        for (int s = 0; s < 3; ++s) {
            llama_kv_cells cells;
            cells.resize(4096);
            for (uint32_t i = 0; i < 3000; ++i) {
                cells.pos_set(i, i);
                cells.seq_add(i, s);
            }
            c.cells.push_back(std::move(cells));
            c.seq_to_stream[s] = s;
            for (int r = 0; r < 4; ++r) {
                rows.push_back({ s, 2996 + r });
            }
        }
        make_ubatch(c.u, rows, 1);
        c.n_kv = 4096; c.n_stream = 3; c.causal = true; c.expect_device = true;
        c.name = "3 streams x 4 rows";
        res.push_back(std::move(c));
    }

    // fallback: an image block where cells share a position and the 2D rule hides some of them
    {
        mask_case c;
        std::vector<llama_pos> next_pos;
        c.cells.push_back(make_unified(16384, 2, 0, 4000, 512, next_pos));
        const llama_pos t = next_pos[0];
        std::vector<std::pair<llama_seq_id, llama_pos>> rows;
        std::vector<std::pair<llama_pos, llama_pos>> xy;
        uint32_t i = 0;
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                while (!c.cells[0].is_empty(i)) {
                    ++i;
                }
                c.cells[0].pos_set(i, t);
                llama_kv_cell_ext ext;
                ext.x = t + x;
                ext.y = t + y;
                c.cells[0].ext_set(i, ext);
                c.cells[0].seq_add(i, 0);
                rows.push_back({ 0, t });
                xy.push_back({ t + y, t + x });
            }
        }
        c.seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
        make_ubatch(c.u, rows, 4, &xy);
        c.n_kv = 16384; c.n_stream = 1; c.causal = true; c.expect_device = false;
        c.name = "M-RoPE image block (fallback)";
        res.push_back(std::move(c));
    }

    // fallback: 9 sequences in one stream
    {
        mask_case c;
        std::vector<llama_pos> next_pos;
        c.cells.push_back(make_unified(16384, 9, 0, 1500, 256, next_pos));
        c.seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
        make_ubatch(c.u, tail_rows({ 0, 1, 2, 3, 4, 5, 6, 7, 8 }, next_pos, 1), 1);
        c.n_kv = 16384; c.n_stream = 1; c.causal = true; c.expect_device = false;
        c.name = "9 sequences (fallback)";
        res.push_back(std::move(c));
    }

    return res;
}

static std::vector<uint8_t> device_mask(ggml_backend_t backend, const std::vector<int32_t> & keys,
        const std::vector<int32_t> & rows, int64_t n_kv, int64_t n_tps, int64_t n_stream, ggml_type type) {
    ggml_init_params params = { 4*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * k   = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, n_kv, 1, 1, n_stream);
    ggml_tensor * r   = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2, n_tps, 1, n_stream);
    ggml_tensor * out = ggml_kq_mask(ctx, k, r, type);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);
    ggml_backend_tensor_set(k, keys.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(r, rows.data(), 0, ggml_nbytes(r));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> res(ggml_nbytes(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size());

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    ggml_backend_load_all();

    std::vector<ggml_backend_t> backends;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend != nullptr);
        backends.push_back(backend);
    }
    if (backends.empty()) {
        fprintf(stderr, "no backends\n");
        return 1;
    }

    const llama_hparams hparams = {};

    int n_fail = 0;
    llama_kv_mask_scratch scratch;

    for (mask_case & c : make_cases()) {
        const int64_t n_tps = c.u.ub.n_tokens/c.n_stream;
        const llama_kv_mask_args args = {
            /*.hparams          =*/ hparams,
            /*.ubatch           =*/ &c.u.ub,
            /*.v_cells          =*/ c.cells,
            /*.seq_to_stream    =*/ c.seq_to_stream,
            /*.n_swa            =*/ 0,
            /*.swa_type         =*/ LLAMA_SWA_TYPE_NONE,
            /*.n_kv             =*/ c.n_kv,
            /*.n_stream         =*/ c.n_stream,
            /*.n_tps            =*/ n_tps,
        };

        const bool on_device = llama_kv_mask_on_device(args, c.causal, scratch);
        if (on_device != c.expect_device) {
            fprintf(stderr, "FAIL %s: on_device = %d, expected %d\n", c.name.c_str(), on_device, c.expect_device);
            n_fail++;
            continue;
        }
        if (!on_device) {
            fprintf(stderr, "ok   %s: host mask\n", c.name.c_str());
            continue;
        }

        std::vector<int32_t> keys(c.n_kv*c.n_stream);
        std::vector<int32_t> rows(2*c.u.ub.n_tokens);
        llama_kv_mask_set_keys(args, c.causal, keys.data(), rows.data());

        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            ggml_init_params params = { (size_t) (c.n_kv*n_tps*c.n_stream*ggml_type_size(type)) + ggml_tensor_overhead(), nullptr, false };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * host = ggml_new_tensor_4d(ctx, type, c.n_kv, n_tps, 1, c.n_stream);
            llama_kv_mask_set_host(args, host, c.causal, scratch);

            size_t n_keep = 0;
            for (int64_t i = 0; i < ggml_nelements(host); ++i) {
                n_keep += type == GGML_TYPE_F16 ? ((const uint16_t *) host->data)[i] == 0 : ((const uint32_t *) host->data)[i] == 0;
            }

            for (ggml_backend_t backend : backends) {
                const std::vector<uint8_t> dev = device_mask(backend, keys, rows, c.n_kv, n_tps, c.n_stream, type);
                const bool ok = dev.size() == ggml_nbytes(host) && memcmp(dev.data(), host->data, dev.size()) == 0;
                fprintf(stderr, "%s %s, %s, %s: %zu of %lld visible\n", ok ? "ok  " : "FAIL", c.name.c_str(),
                        ggml_type_name(type), ggml_backend_name(backend), n_keep, (long long) ggml_nelements(host));
                n_fail += !ok;
            }
            ggml_free(ctx);
        }
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }

    fprintf(stderr, "%s\n", n_fail == 0 ? "all KQ mask checks passed" : "KQ mask checks FAILED");
    return n_fail == 0 ? 0 : 1;
}

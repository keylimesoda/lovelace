#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_context * init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // Use a full ubatch so buffer discovery preserves prefill allocation sizes.
    const uint32_t n_tokens = llama_n_ubatch(ctx);
    if (!decode_tokens(ctx, std::vector<llama_token>(n_tokens, 0), n_tokens)) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint8_t fill) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return init_ctx(model, cparams, fill);
}

static float logit_diff(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity();
}

static double nmse(const float * a, const float * b, int n) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double diff = (double) a[i] - b[i];
        mse_ab += diff*diff;
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 == 0.0 ? (mse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : mse_ab/mse_a0;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab, uint8_t fill) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return init_ctx(model, cparams, fill);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // removing more than the last ubatch can restore must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // identical ubatch shapes should produce identical states, but the larger
    // stdev makes the model sensitive to backend scheduling/rounding noise
    constexpr float nmse_eps = 1e-5f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    double   nmse_ab   = 0.0;
    double   nmse_a0   = 0.0;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            const float diff = logit_diff(r, f);
            if (diff > 0.0f && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_ab += d*d;
                nmse_a0 += (double) r*r;
            } else {
                nmse_ab = std::numeric_limits<double>::infinity();
                nmse_a0 = 1.0;
            }
        }
    }
    const double nmse_val = nmse_a0 == 0.0 ? (nmse_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_ab/nmse_a0;

    if (nmse_val > nmse_eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, nmse %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, nmse_val, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g, nmse %g)\n", __func__, (double) diff_max, nmse_val);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    double nmse_tail_ab = 0.0;
    double nmse_tail_a0 = 0.0;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            const float r = l_roll[t];
            const float f = l_ref[t];
            diff_tail = std::max(diff_tail, logit_diff(r, f));
            if (std::isfinite(r) && std::isfinite(f)) {
                const double d = (double) r - f;
                nmse_tail_ab += d*d;
                nmse_tail_a0 += (double) r*r;
            } else {
                nmse_tail_ab = std::numeric_limits<double>::infinity();
                nmse_tail_a0 = 1.0;
            }
        }
    }
    const double nmse_tail = nmse_tail_a0 == 0.0 ? (nmse_tail_ab == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : nmse_tail_ab/nmse_tail_a0;

    if (!ok || nmse_tail > nmse_eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g, nmse %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail, nmse_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g, nmse %g)\n", __func__, (double) diff_tail, nmse_tail);
    cleanup();
    return true;
}

//
// [ReplaySSM] replay vs snapshot rollback
//

// creates a context with the rollback mode forced through LLAMA_RS_ROLLBACK (read at context creation)
static llama_context * make_ctx_mode(const common_params & params, llama_model * model, uint8_t fill, const char * mode,
        uint32_t n_seq_max, uint32_t n_rs_seq, bool kv_unified) {
    const char * env = getenv("LLAMA_RS_ROLLBACK");
    const std::string env_old = env ? env : "";

    setenv("LLAMA_RS_ROLLBACK", mode, 1);

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq_max;
    cparams.n_rs_seq   = n_rs_seq;
    cparams.n_ctx      = 1024;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 64;
    cparams.kv_unified = kv_unified;
    llama_context * ctx = init_ctx(model, cparams, fill);

    if (env) {
        setenv("LLAMA_RS_ROLLBACK", env_old.c_str(), 1);
    } else {
        unsetenv("LLAMA_RS_ROLLBACK");
    }
    return ctx;
}

// bitwise comparison of the first n_outputs logits rows
static bool logits_same(llama_context * a, llama_context * b, int n_outputs, int n_vocab, const std::string & what) {
    for (int i = 0; i < n_outputs; ++i) {
        const float * la = llama_get_logits_ith(a, i);
        const float * lb = llama_get_logits_ith(b, i);
        if (la == nullptr || lb == nullptr) {
            fprintf(stderr, "%s : %s: missing logits at output %d\n", __func__, what.c_str(), i);
            return false;
        }
        if (memcmp(la, lb, n_vocab*sizeof(float)) != 0) {
            fprintf(stderr, "%s : %s: logits differ at output %d (nmse %g)\n", __func__, what.c_str(), i, nmse(la, lb, n_vocab));
            return false;
        }
    }
    return true;
}

// Six sequences verify drafts of changing length in one batch, with changing subsets of active sequences (so inactive
// cells get moved around) and a different rollback depth per sequence. Replay and snapshot mode run identical ubatch
// shapes and must produce bitwise-equal logits.
static bool test_ab_multi_seq(const common_params & params, llama_model * model, int n_vocab, uint8_t fill, bool kv_unified) {
    constexpr uint32_t n_seqs   = 6;
    constexpr uint32_t n_rs_seq = 7;
    constexpr uint32_t K        = n_rs_seq + 1;
    const std::string  tag      = std::string("kv_unified=") + (kv_unified ? "1" : "0");

    llama_context * ctx_rep  = make_ctx_mode(params, model, fill, "replay",   n_seqs, n_rs_seq, kv_unified);
    llama_context * ctx_snap = make_ctx_mode(params, model, fill, "snapshot", n_seqs, n_rs_seq, kv_unified);
    if (ctx_rep == nullptr || ctx_snap == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_rep);
        llama_free(ctx_snap);
    };

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((13*(uint32_t) pos + 29*seq + 3) % (uint32_t) n_vocab);
    };

    std::vector<llama_pos> pos(n_seqs, 0);
    llama_batch batch = llama_batch_init(256, 0, 1);

    const auto decode_both = [&](const std::string & what) {
        if (llama_decode(ctx_rep, batch) != 0 || llama_decode(ctx_snap, batch) != 0) {
            fprintf(stderr, "%s : %s %s: decode failed\n", __func__, tag.c_str(), what.c_str());
            return false;
        }
        return logits_same(ctx_rep, ctx_snap, batch.n_tokens, n_vocab, tag + " " + what);
    };

    // prefill with different lengths per seq
    common_batch_clear(batch);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < 5 + 3*s; ++i) {
            common_batch_add(batch, tok(s, pos[s]), pos[s], { (llama_seq_id) s }, true);
            pos[s]++;
        }
    }
    bool ok = decode_both("prefill");

    std::mt19937 rng(7);
    const std::vector<std::vector<uint32_t>> subsets = { { 0, 1, 2, 3, 4, 5 }, { 0, 2, 4 }, { 1, 2, 5 }, { 3 }, { 5, 0 } };

    for (int round = 0; ok && round < 30; ++round) {
        const auto & active = subsets[round % subsets.size()];
        const uint32_t T = 1 + (uint32_t) round % K;

        common_batch_clear(batch);
        for (uint32_t s : active) {
            for (uint32_t i = 0; i < T; ++i) {
                common_batch_add(batch, tok(s, pos[s] + i), pos[s] + i, { (llama_seq_id) s }, true);
            }
        }
        ok = decode_both("round " + std::to_string(round));

        // accept a different number of tokens per seq, roll back the rest
        for (uint32_t s : active) {
            const uint32_t r = (uint32_t) (rng() % T);
            if (ok && r > 0) {
                const llama_pos p0 = pos[s] + (llama_pos) (T - r);
                const bool ok_rep  = llama_memory_seq_rm(llama_get_memory(ctx_rep),  (llama_seq_id) s, p0, -1);
                const bool ok_snap = llama_memory_seq_rm(llama_get_memory(ctx_snap), (llama_seq_id) s, p0, -1);
                if (!ok_rep || !ok_snap) {
                    fprintf(stderr, "%s : %s round %d: rollback of %u failed (replay %d, snapshot %d)\n",
                            __func__, tag.c_str(), round, r, ok_rep, ok_snap);
                    ok = false;
                }
            }
            pos[s] += (llama_pos) (T - r);
        }
    }

    if (ok) {
        common_batch_clear(batch);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            common_batch_add(batch, tok(s, pos[s]), pos[s], { (llama_seq_id) s }, true);
        }
        ok = decode_both("final");
    }

    llama_batch_free(batch);
    cleanup();

    if (ok) {
        fprintf(stderr, "%s : %s replay and snapshot logits bitwise equal\n", __func__, tag.c_str());
    }
    return ok;
}

// replay mode only: stacked rollbacks compose, and removals beyond the pending records are refused
static bool test_replay_rollback_rules(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    constexpr uint32_t n_seqs   = 2;
    constexpr uint32_t n_rs_seq = 7;
    constexpr uint32_t T        = n_rs_seq + 1;

    llama_context * ctx_a = make_ctx_mode(params, model, fill, "replay", n_seqs, n_rs_seq, true);
    llama_context * ctx_b = make_ctx_mode(params, model, fill, "replay", n_seqs, n_rs_seq, true);
    if (ctx_a == nullptr || ctx_b == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return false;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((17*(uint32_t) pos + 5*seq + 11) % (uint32_t) n_vocab);
    };

    llama_batch batch = llama_batch_init(64, 0, 1);

    const auto decode_range = [&](llama_context * ctx, llama_pos p0, llama_pos p1) {
        common_batch_clear(batch);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (llama_pos p = p0; p < p1; ++p) {
                common_batch_add(batch, tok(s, p), p, { (llama_seq_id) s }, true);
            }
        }
        return llama_decode(ctx, batch) == 0;
    };

    // prompt of 12 tokens, then a verify of T tokens for both seqs
    bool ok = true;
    for (llama_context * ctx : { ctx_a, ctx_b }) {
        ok = ok && decode_range(ctx, 0, 12) && decode_range(ctx, 12, 12 + T);
    }

    auto * mem_a = llama_get_memory(ctx_a);
    auto * mem_b = llama_get_memory(ctx_b);
    const llama_pos p_end = 12 + T;

    // seq 0: 2 + 3 stacked in ctx_a vs 5 at once in ctx_b; seq 1: 1 in both
    ok = ok && llama_memory_seq_rm(mem_a, 0, p_end - 2, -1);
    ok = ok && llama_memory_seq_rm(mem_a, 0, p_end - 5, -1);
    ok = ok && llama_memory_seq_rm(mem_b, 0, p_end - 5, -1);
    ok = ok && llama_memory_seq_rm(mem_a, 1, p_end - 1, -1);
    ok = ok && llama_memory_seq_rm(mem_b, 1, p_end - 1, -1);
    if (!ok) {
        fprintf(stderr, "%s : stacked rollback failed\n", __func__);
    }

    // 3 of the T recorded tokens of seq 0 remain; 4 more cannot be removed, and the refusal changes nothing
    if (ok && llama_memory_seq_rm(mem_a, 0, p_end - 9, -1)) {
        fprintf(stderr, "%s : rollback beyond the pending records was accepted\n", __func__);
        ok = false;
    }

    const llama_pos pos[n_seqs] = { p_end - 5, p_end - 1 };
    for (int step = 0; ok && step < 3; ++step) {
        common_batch_clear(batch);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            for (llama_pos p = pos[s] + 2*step; p < pos[s] + 2*step + 2; ++p) {
                common_batch_add(batch, tok(s, p + 1000), p, { (llama_seq_id) s }, true);
            }
        }
        ok = llama_decode(ctx_a, batch) == 0 && llama_decode(ctx_b, batch) == 0;
        ok = ok && logits_same(ctx_a, ctx_b, batch.n_tokens, n_vocab, "stacked vs single rollback step " + std::to_string(step));
    }

    llama_batch_free(batch);
    llama_free(ctx_a);
    llama_free(ctx_b);

    if (ok) {
        fprintf(stderr, "%s : stacked rollback and refused over-rollback OK\n", __func__);
    }
    return ok;
}

// full state save/restore while rollbacks are pending: the restored replay and snapshot contexts must continue with
// bitwise-equal logits. (a full restore compacts the attention KV cells, so restored vs original is not bitwise equal
// in either mode; the per-seq checkpoint restore in test_rollback covers that with pending records.)
static bool test_ab_state_restore(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    constexpr uint32_t n_seqs   = 3;
    constexpr uint32_t n_rs_seq = 7;
    constexpr uint32_t T        = n_rs_seq + 1;

    const char * modes[2] = { "replay", "snapshot" };
    llama_context * orig[2] = {};
    llama_context * rest[2] = {};

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((19*(uint32_t) pos + 7*seq + 2) % (uint32_t) n_vocab);
    };

    llama_batch batch = llama_batch_init(64, 0, 1);
    const llama_pos p_end = 10 + T;
    const llama_pos rollback[n_seqs] = { 5, 1, 7 };

    bool ok = true;
    for (int i = 0; ok && i < 2; ++i) {
        orig[i] = make_ctx_mode(params, model, fill, modes[i], n_seqs, n_rs_seq, true);
        rest[i] = make_ctx_mode(params, model, fill, modes[i], n_seqs, n_rs_seq, true);
        ok = orig[i] != nullptr && rest[i] != nullptr;

        for (llama_pos p0 : { (llama_pos) 0, (llama_pos) 10 }) {
            common_batch_clear(batch);
            for (uint32_t s = 0; s < n_seqs; ++s) {
                for (llama_pos p = p0; p < (p0 == 0 ? 10 : p_end); ++p) {
                    common_batch_add(batch, tok(s, p), p, { (llama_seq_id) s }, true);
                }
            }
            ok = ok && llama_decode(orig[i], batch) == 0;
        }
        for (uint32_t s = 0; ok && s < n_seqs; ++s) {
            ok = llama_memory_seq_rm(llama_get_memory(orig[i]), (llama_seq_id) s, p_end - rollback[s], -1);
        }

        if (ok) {
            std::vector<uint8_t> state(llama_state_get_size(orig[i]));
            ok = llama_state_get_data(orig[i], state.data(), state.size()) == state.size() &&
                 llama_state_set_data(rest[i], state.data(), state.size()) == state.size();
        }
        if (!ok) {
            fprintf(stderr, "%s : %s save/restore failed\n", __func__, modes[i]);
        }
    }

    for (int step = 0; ok && step < 3; ++step) {
        common_batch_clear(batch);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const llama_pos p0 = p_end - rollback[s] + 2*step;
            for (llama_pos p = p0; p < p0 + 2; ++p) {
                common_batch_add(batch, tok(s, p + 1000), p, { (llama_seq_id) s }, true);
            }
        }
        for (int i = 0; ok && i < 2; ++i) {
            ok = llama_decode(orig[i], batch) == 0 && llama_decode(rest[i], batch) == 0;
        }
        const std::string t = " step " + std::to_string(step);
        ok = ok && logits_same(orig[0], orig[1], batch.n_tokens, n_vocab, "original replay vs snapshot" + t);
        ok = ok && logits_same(rest[0], rest[1], batch.n_tokens, n_vocab, "restored replay vs snapshot" + t);
    }

    llama_batch_free(batch);
    for (int i = 0; i < 2; ++i) {
        llama_free(orig[i]);
        llama_free(rest[i]);
    }

    if (ok) {
        fprintf(stderr, "%s : restore with pending rollbacks bitwise equal across modes\n", __func__);
    }
    return ok;
}

static int test_rollback(const common_params & params, llama_model * model, uint8_t fill) {
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // TODO: use smart pointers
    llama_context * ctx_src = make_ctx(params, model, fill);
    llama_context * ctx_dst = make_ctx(params, model, fill);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float nmse_eps = 0.0;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            const double nmse_val = nmse(logits_src, logits_dst, n_vocab);
            int token_first = -1;
            for (int token = 0; token < n_vocab; ++token) {
                if (logit_diff(logits_src[token], logits_dst[token]) > 0.0f && token_first < 0) {
                    token_first = token;
                }
            }
            if (nmse_val > nmse_eps) {
                fprintf(stderr, "%s : %s logits mismatch at position %d, first token %d, nmse %g\n",
                        __func__, mode, pos, token_first, nmse_val);
                return false;
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    // TODO: this test is invalid because RS rollback is only correct once after a ubatch with more than n_rs_seq tokens
    //       this is not the case here. add asserts and guardrails to prevent such attempts
    //if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
    //    !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
    //    fprintf(stderr, "%s : partial rollback failed\n", __func__);
    //    return 1;
    //}

    //constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    //common_prompt_checkpoint ckpt_partial;
    //ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    //ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    //if (!replay_and_compare("partial")) {
    //    return 1;
    //}

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model, fill);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        const double nmse_dirty = nmse(logits_src_replay[i].data(), logits_dirty, n_vocab);
        int token_first = -1;
        for (int token = 0; token < n_vocab; ++token) {
            if (logit_diff(logits_src_replay[i][token], logits_dirty[token]) > 0.0f && token_first < 0) {
                token_first = token;
            }
        }
        if (nmse_dirty > nmse_eps) {
            fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, first token %d, nmse %g\n",
                    __func__, pos, token_first, nmse_dirty);
            return 1;
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    if (!test_multi_seq_split_replay(params, model, n_vocab, fill)) {
        return 1;
    }

    // [ReplaySSM] only the qwen35 family implements the replay mode
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (strcmp(arch, "qwen35") == 0 || strcmp(arch, "qwen35moe") == 0) {
        for (bool kv_unified : { false, true }) {
            if (!test_ab_multi_seq(params, model, n_vocab, fill, kv_unified)) {
                return 1;
            }
        }
        if (!test_replay_rollback_rules(params, model, n_vocab, fill)) {
            return 1;
        }
        if (!test_ab_state_restore(params, model, n_vocab, fill)) {
            return 1;
        }
    }

    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    for (uint8_t fill : { 0, 0x3e }) {
        fprintf(stderr, "%s : testing with cache fill 0x%02x\n", __func__, fill);
        if (test_rollback(params, model, fill) != 0) {
            return 1;
        }
    }

    return 0;
}

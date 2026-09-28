// ReplaySSM: ggml_gated_delta_net_rr / ggml_ssm_conv_fold must reproduce the sequential recurrence bit for bit on the
// same backend. The reference is ggml_gated_delta_net with K snapshot slots (slot i = state i tokens back).

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct gdn_shape {
    int64_t S;   // head size
    int64_t H_k; // q/k heads
    int64_t H_v; // v heads
};

// inputs of n_tokens tokens of n_seqs seqs, laid out as the ops expect
struct gdn_inputs {
    int64_t n_tokens = 0;
    int64_t n_seqs   = 0;
    std::vector<float> q, k, v, g, beta; // [S,H_k,T,B] [S,H_k,T,B] [S,H_v,T,B] [1,H_v,T,B] [1,H_v,T,B]
};

static std::mt19937 rng(42);

static float urand(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
}

// k/q unit-norm (tiny norms included), extreme gates, beta at the edges
static gdn_inputs make_inputs(const gdn_shape & sh, int64_t n_tokens, int64_t n_seqs) {
    gdn_inputs in;
    in.n_tokens = n_tokens;
    in.n_seqs   = n_seqs;

    auto unit = [&](std::vector<float> & dst) {
        dst.resize(sh.S*sh.H_k*n_tokens*n_seqs);
        for (size_t off = 0; off < dst.size(); off += sh.S) {
            const float amp = urand(0.0f, 1.0f) < 0.1f ? 1e-6f : 1.0f;
            double norm = 0.0;
            for (int64_t i = 0; i < sh.S; ++i) {
                dst[off + i] = amp*urand(-1.0f, 1.0f);
                norm += (double) dst[off + i]*dst[off + i];
            }
            const float inv = (float) (1.0/std::sqrt(norm + 1e-12));
            for (int64_t i = 0; i < sh.S; ++i) {
                dst[off + i] *= inv;
            }
        }
    };
    unit(in.q);
    unit(in.k);

    in.v.resize(sh.S*sh.H_v*n_tokens*n_seqs);
    for (auto & x : in.v) {
        x = urand(-0.3f, 5.0f);
    }

    in.g.resize(sh.H_v*n_tokens*n_seqs);
    in.beta.resize(sh.H_v*n_tokens*n_seqs);
    for (size_t i = 0; i < in.g.size(); ++i) {
        const float r = urand(0.0f, 1.0f);
        in.g[i]    = r < 0.1f ? -20.0f : r < 0.2f ? 20.0f*urand(-1.0f, 0.0f) : urand(-3.0f, 0.0f);
        const float rb = urand(0.0f, 1.0f);
        in.beta[i] = rb < 0.1f ? 0.0f : rb < 0.2f ? 1.0f : urand(0.0f, 1.0f);
    }

    return in;
}

// token range [t0, t0 + n) of seq s of `in`, as a single-seq input
static gdn_inputs slice(const gdn_shape & sh, const gdn_inputs & in, int64_t s, int64_t t0, int64_t n) {
    gdn_inputs out;
    out.n_tokens = n;
    out.n_seqs   = 1;
    auto cut = [&](const std::vector<float> & src, std::vector<float> & dst, int64_t per_token) {
        const float * p = src.data() + (s*in.n_tokens + t0)*per_token;
        dst.assign(p, p + n*per_token);
    };
    cut(in.q,    out.q,    sh.S*sh.H_k);
    cut(in.k,    out.k,    sh.S*sh.H_k);
    cut(in.v,    out.v,    sh.S*sh.H_v);
    cut(in.g,    out.g,    sh.H_v);
    cut(in.beta, out.beta, sh.H_v);
    return out;
}

static void append(gdn_inputs & dst, const gdn_inputs & src) {
    GGML_ASSERT(dst.n_seqs <= 1 && src.n_seqs == 1);
    dst.n_seqs    = 1;
    dst.n_tokens += src.n_tokens;
    dst.q   .insert(dst.q   .end(), src.q   .begin(), src.q   .end());
    dst.k   .insert(dst.k   .end(), src.k   .begin(), src.k   .end());
    dst.v   .insert(dst.v   .end(), src.v   .begin(), src.v   .end());
    dst.g   .insert(dst.g   .end(), src.g   .begin(), src.g   .end());
    dst.beta.insert(dst.beta.end(), src.beta.begin(), src.beta.end());
}

//
// graph runner: every call builds, allocates and computes one graph on the backend
//

struct runner {
    ggml_backend_t backend;

    ggml_context          * ctx = nullptr;
    ggml_backend_buffer_t   buf = nullptr;

    std::vector<std::pair<ggml_tensor *, const void *>> uploads;

    void begin() {
        ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(params);
        uploads.clear();
    }

    ggml_tensor * input(ggml_type type, std::initializer_list<int64_t> ne, const void * data) {
        const std::vector<int64_t> v(ne);
        ggml_tensor * t = ggml_new_tensor(ctx, type, (int) v.size(), v.data());
        uploads.emplace_back(t, data);
        return t;
    }

    void compute(ggml_tensor * out) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buf != nullptr);
        for (auto & [t, data] : uploads) {
            ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    }

    std::vector<float> get(ggml_tensor * t) const {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        return v;
    }

    void end() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
    }
};

// ggml_gated_delta_net(K) from s0: returns scores followed by K snapshots
static std::vector<float> run_gdn(runner & r, const gdn_shape & sh, const gdn_inputs & in, const std::vector<float> & s0, int64_t K) {
    const int64_t T = in.n_tokens;
    const int64_t B = in.n_seqs;
    r.begin();
    ggml_tensor * q = r.input(GGML_TYPE_F32, { sh.S, sh.H_k, T, B }, in.q.data());
    ggml_tensor * k = r.input(GGML_TYPE_F32, { sh.S, sh.H_k, T, B }, in.k.data());
    ggml_tensor * v = r.input(GGML_TYPE_F32, { sh.S, sh.H_v, T, B }, in.v.data());
    ggml_tensor * g = r.input(GGML_TYPE_F32, { 1, sh.H_v, T, B }, in.g.data());
    ggml_tensor * b = r.input(GGML_TYPE_F32, { 1, sh.H_v, T, B }, in.beta.data());
    ggml_tensor * s = r.input(GGML_TYPE_F32, { sh.S, sh.S, sh.H_v, B }, s0.data());
    ggml_tensor * out = ggml_gated_delta_net(r.ctx, q, k, v, g, b, s, K);
    r.compute(out);
    std::vector<float> res = r.get(out);
    r.end();
    return res;
}

// ggml_gated_delta_net_rr; `rec` is updated in place with the arena after the op
static std::vector<float> run_rr(runner & r, const gdn_shape & sh, const gdn_inputs & in, const std::vector<float> & state,
        std::vector<float> & rec, int64_t n_rows, const std::vector<int32_t> & rs, int64_t K, int64_t dst_row0) {
    const int64_t T = in.n_tokens;
    const int64_t B = in.n_seqs;
    const int64_t n_embd_rec = sh.S*sh.H_k + sh.S*sh.H_v + 2*sh.H_v;
    r.begin();
    ggml_tensor * q   = r.input(GGML_TYPE_F32, { sh.S, sh.H_k, T, B }, in.q.data());
    ggml_tensor * k   = r.input(GGML_TYPE_F32, { sh.S, sh.H_k, T, B }, in.k.data());
    ggml_tensor * v   = r.input(GGML_TYPE_F32, { sh.S, sh.H_v, T, B }, in.v.data());
    ggml_tensor * g   = r.input(GGML_TYPE_F32, { 1, sh.H_v, T, B }, in.g.data());
    ggml_tensor * b   = r.input(GGML_TYPE_F32, { 1, sh.H_v, T, B }, in.beta.data());
    ggml_tensor * s   = r.input(GGML_TYPE_F32, { sh.S, sh.S, sh.H_v, B }, state.data());
    ggml_tensor * a   = r.input(GGML_TYPE_F32, { n_embd_rec, K, n_rows }, rec.data());
    ggml_tensor * rst = r.input(GGML_TYPE_I32, { 2, B }, rs.data());
    ggml_tensor * out = ggml_gated_delta_net_rr(r.ctx, q, k, v, g, b, s, a, rst, (int32_t) K, (int32_t) dst_row0);
    r.compute(out);
    std::vector<float> res = r.get(out);
    rec = r.get(a);
    r.end();
    return res;
}

// ggml_ssm_conv_fold; `recx` is updated in place
static std::vector<float> run_fold(runner & r, int64_t n_hist, int64_t C, const std::vector<float> & cs, std::vector<float> & recx,
        int64_t n_rows, const std::vector<float> & x, int64_t T, int64_t B, const std::vector<int32_t> & rs, int64_t K, int64_t dst_row0) {
    r.begin();
    ggml_tensor * c   = r.input(GGML_TYPE_F32, { n_hist, C, B }, cs.data());
    ggml_tensor * a   = r.input(GGML_TYPE_F32, { C, K, n_rows }, recx.data());
    ggml_tensor * xt  = r.input(GGML_TYPE_F32, { C, T, B }, x.data());
    ggml_tensor * rst = r.input(GGML_TYPE_I32, { 2, B }, rs.data());
    ggml_tensor * out = ggml_ssm_conv_fold(r.ctx, c, a, xt, rst, (int32_t) K, (int32_t) dst_row0);
    r.compute(out);
    std::vector<float> res = r.get(out);
    recx = r.get(a);
    r.end();
    return res;
}

//
// checks
//

static int n_fail = 0;

static bool same_bits(const float * a, const float * b, size_t n, const std::string & what) {
    if (memcmp(a, b, n*sizeof(float)) == 0) {
        return true;
    }
    size_t i = 0;
    while (i < n && memcmp(a + i, b + i, sizeof(float)) == 0) {
        ++i;
    }
    fprintf(stderr, "  FAIL %s: first mismatch at %zu of %zu (%.9g vs %.9g)\n", what.c_str(), i, n, (double) a[i], (double) b[i]);
    n_fail++;
    return false;
}

// test 1-3 and 5: record T tokens of B seqs, then fold m = 0..T records per seq (different m per seq) under one new token
static void test_record_fold(runner & r, const gdn_shape & sh, int64_t T, int64_t K, int64_t B) {
    const int64_t D          = sh.S*sh.S*sh.H_v;
    const int64_t n_embd_rec = sh.S*sh.H_k + sh.S*sh.H_v + 2*sh.H_v;
    const int64_t n_rows     = B + 2; // rows 0 and B + 1 are never written
    const int64_t dst_row0   = 1;
    const std::string tag    = "S=" + std::to_string(sh.S) + " H=" + std::to_string(sh.H_k) + "/" + std::to_string(sh.H_v) +
                               " T=" + std::to_string(T) + " K=" + std::to_string(K) + " B=" + std::to_string(B);
    GGML_ASSERT(T <= K);

    const gdn_inputs in = make_inputs(sh, T, B);

    std::vector<float> s0(D*B);
    for (auto & x : s0) {
        x = urand(-1.0f, 1.0f);
    }

    // 1. reference: all T snapshots (slot i = state after T - i tokens)
    const std::vector<float> ref = run_gdn(r, sh, in, s0, T);
    const int64_t n_scores = sh.S*sh.H_v*T*B;
    auto ref_state = [&](int64_t s, int64_t n) -> const float * {
        return n == 0 ? s0.data() + s*D : ref.data() + n_scores + (T - n)*D*B + s*D;
    };

    // 2. record: committed == s0 (nothing before the recorded tokens), scores == reference scores
    std::vector<float> rec(n_embd_rec*K*n_rows);
    for (auto & x : rec) {
        x = urand(-1.0f, 1.0f);
    }
    const std::vector<float> rec_before = rec;

    std::vector<int32_t> rs0(2*B);
    for (int64_t s = 0; s < B; ++s) {
        rs0[2*s + 0] = 0;
        rs0[2*s + 1] = (int32_t) (dst_row0 + s);
    }
    const std::vector<float> out_rec = run_rr(r, sh, in, s0, rec, n_rows, rs0, K, dst_row0);
    same_bits(out_rec.data(), ref.data(), n_scores, tag + " record scores");
    same_bits(out_rec.data() + n_scores, s0.data(), D*B, tag + " record committed");

    // 5. isolation: rows outside [dst_row0, dst_row0 + B) and unused slots are untouched
    same_bits(rec.data(), rec_before.data(), n_embd_rec*K, tag + " record row 0 untouched");
    same_bits(rec.data() + (n_rows - 1)*n_embd_rec*K, rec_before.data() + (n_rows - 1)*n_embd_rec*K, n_embd_rec*K, tag + " record row B+1 untouched");
    for (int64_t s = 0; s < B; ++s) {
        const int64_t off = ((dst_row0 + s)*K + T)*n_embd_rec;
        same_bits(rec.data() + off, rec_before.data() + off, (K - T)*n_embd_rec, tag + " record unused slots untouched");
    }

    // 3. fold: m_s = (s + round) % (T + 1) records under one new token; the rejected suffix is poisoned
    const gdn_inputs x1 = make_inputs(sh, 1, B);
    for (int64_t round = 0; round <= T; ++round) {
        std::vector<float> rec_fold = rec;
        std::vector<int32_t> rs(2*B);
        std::vector<float> expect(D*B);
        for (int64_t s = 0; s < B; ++s) {
            const int64_t m = (s + round) % (T + 1);
            rs[2*s + 0] = (int32_t) m;
            rs[2*s + 1] = (int32_t) (dst_row0 + s);
            memcpy(expect.data() + s*D, ref_state(s, m), D*sizeof(float));
            for (int64_t u = m; u < K; ++u) {
                float * p = rec_fold.data() + ((dst_row0 + s)*K + u)*n_embd_rec;
                std::fill(p, p + n_embd_rec, std::numeric_limits<float>::quiet_NaN());
            }
        }

        const std::vector<float> out = run_rr(r, sh, x1, s0, rec_fold, n_rows, rs, K, dst_row0);
        const std::string t = tag + " fold round " + std::to_string(round);
        same_bits(out.data() + sh.S*sh.H_v*B, expect.data(), D*B, t + " committed");

        // the new token's score equals a plain K=1 step from the snapshot
        const std::vector<float> ref1 = run_gdn(r, sh, x1, expect, 1);
        same_bits(out.data(), ref1.data(), sh.S*sh.H_v*B, t + " new token score");
    }

    // swapped sources: seq s folds the records of seq B - 1 - s from their row, while the op records seq s there
    if (B > 1) {
        std::vector<float> rec_swap = rec;
        std::vector<int32_t> rs(2*B);
        std::vector<float> state(D*B), expect(D*B);
        for (int64_t s = 0; s < B; ++s) {
            const int64_t o = B - 1 - s;
            const int64_t m = (s % 2 == 0) ? T : 1;
            rs[2*s + 0] = (int32_t) m;
            rs[2*s + 1] = (int32_t) (dst_row0 + o);
            memcpy(state .data() + s*D, s0.data() + o*D, D*sizeof(float));
            memcpy(expect.data() + s*D, ref_state(o, m), D*sizeof(float));
        }
        const std::vector<float> out = run_rr(r, sh, x1, state, rec_swap, n_rows, rs, K, dst_row0);
        same_bits(out.data() + sh.S*sh.H_v*B, expect.data(), D*B, tag + " swapped sources committed");
    }

    // conv fold: record T columns, then fold m = 0..T columns per seq against the conv_input windows
    {
        const int64_t n_hist = 3;
        const int64_t C      = 97;

        std::vector<float> cs0(n_hist*C*B), x(C*T*B), recx(C*K*n_rows);
        for (auto & v : cs0)  { v = urand(-1.0f, 1.0f); }
        for (auto & v : x)    { v = urand(-1.0f, 1.0f); }
        for (auto & v : recx) { v = urand(-1.0f, 1.0f); }

        const std::vector<float> out0 = run_fold(r, n_hist, C, cs0, recx, n_rows, x, T, B, rs0, K, dst_row0);
        same_bits(out0.data(), cs0.data(), n_hist*C*B, tag + " conv record history");

        const std::vector<float> x1c(C*B, 0.5f);
        for (int64_t round = 0; round <= T; ++round) {
            std::vector<float> recx_fold = recx;
            std::vector<int32_t> rs(2*B);
            std::vector<float> expect(n_hist*C*B);
            for (int64_t s = 0; s < B; ++s) {
                const int64_t m = (s + round) % (T + 1);
                rs[2*s + 0] = (int32_t) m;
                rs[2*s + 1] = (int32_t) (dst_row0 + s);
                // conv_input[s] = cs0[s] || x[s]; the history after m tokens is columns [m, m + n_hist)
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t i = 0; i < n_hist; ++i) {
                        const int64_t col = m + i;
                        expect[(s*C + c)*n_hist + i] = col < n_hist ? cs0[(s*C + c)*n_hist + col] : x[(s*T + col - n_hist)*C + c];
                    }
                }
                for (int64_t u = m; u < K; ++u) {
                    float * p = recx_fold.data() + ((dst_row0 + s)*K + u)*C;
                    std::fill(p, p + C, std::numeric_limits<float>::quiet_NaN());
                }
            }
            const std::vector<float> out = run_fold(r, n_hist, C, cs0, recx_fold, n_rows, x1c, 1, B, rs, K, dst_row0);
            same_bits(out.data(), expect.data(), n_hist*C*B, tag + " conv fold round " + std::to_string(round));
        }
    }
}

// test 4: rounds of verify T <= K tokens with m <= T accepted, against one long run over the accepted inputs only
static void test_chain(runner & r, const gdn_shape & sh, int64_t K, int n_rounds) {
    const int64_t D          = sh.S*sh.S*sh.H_v;
    const int64_t n_embd_rec = sh.S*sh.H_k + sh.S*sh.H_v + 2*sh.H_v;
    const int64_t n_rows     = 2;
    const int64_t row        = 1;
    const std::string tag    = "chain S=" + std::to_string(sh.S) + " K=" + std::to_string(K);

    std::vector<float> s0(D);
    for (auto & x : s0) {
        x = urand(-1.0f, 1.0f);
    }

    std::vector<float> rec(n_embd_rec*K*n_rows, 0.0f);
    std::vector<float> committed = s0;
    int32_t m_pending = 0;

    gdn_inputs accepted;
    std::vector<int64_t> prefix;                 // accepted tokens before each round
    std::vector<gdn_inputs> rounds;              // verify inputs of each round
    std::vector<std::vector<float>> committed_r; // committed state after each round's op
    std::vector<std::vector<float>> scores_r;    // scores of each round

    for (int round = 0; round < n_rounds; ++round) {
        const int64_t T = 1 + (int64_t) (rng() % K);
        const gdn_inputs in = make_inputs(sh, T, 1);

        const std::vector<int32_t> rs = { m_pending, (int32_t) row };
        const std::vector<float> out = run_rr(r, sh, in, committed, rec, n_rows, rs, K, row);

        prefix.push_back(accepted.n_tokens);
        rounds.push_back(in);
        scores_r.emplace_back(out.begin(), out.begin() + sh.S*sh.H_v*T);
        committed.assign(out.begin() + sh.S*sh.H_v*T, out.end());
        committed_r.push_back(committed);

        // accept m <= T tokens: rec_m = T, lowered by the rejected suffix
        const int64_t m = (int64_t) (rng() % (T + 1));
        append(accepted, slice(sh, in, 0, 0, m));
        m_pending = (int32_t) m;
    }

    const int64_t n_acc = accepted.n_tokens;
    GGML_ASSERT(n_acc > 0);

    const std::vector<float> ref = run_gdn(r, sh, accepted, s0, n_acc);
    const int64_t n_scores = sh.S*sh.H_v*n_acc;
    auto ref_state = [&](int64_t n) -> const float * {
        return n == 0 ? s0.data() : ref.data() + n_scores + (n_acc - n)*D;
    };

    for (int round = 0; round < n_rounds; ++round) {
        const std::string t = tag + " round " + std::to_string(round);
        if (!same_bits(committed_r[round].data(), ref_state(prefix[round]), D, t + " committed")) {
            break;
        }
        const std::vector<float> ref_scores = run_gdn(r, sh, rounds[round], committed_r[round], 1);
        if (!same_bits(scores_r[round].data(), ref_scores.data(), sh.S*sh.H_v*rounds[round].n_tokens, t + " scores")) {
            break;
        }
    }

    // the last pending records fold into the long run's final state
    const gdn_inputs x1 = make_inputs(sh, 1, 1);
    const std::vector<int32_t> rs = { m_pending, (int32_t) row };
    const std::vector<float> out = run_rr(r, sh, x1, committed, rec, n_rows, rs, K, row);
    same_bits(out.data() + sh.S*sh.H_v, ref_state(n_acc), D, tag + " final fold");
}

static bool backend_supports(ggml_backend_t backend, const gdn_shape & sh) {
    ggml_init_params params = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * rs = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    ggml_tensor * rr = ggml_gated_delta_net_rr(ctx,
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.S, sh.H_k, 1, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.S, sh.H_k, 1, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.S, sh.H_v, 1, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, sh.H_v, 1, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, sh.H_v, 1, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.S, sh.S, sh.H_v, 1),
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.S*sh.H_k + sh.S*sh.H_v + 2*sh.H_v, 4, 1),
            rs, 4, 0);
    ggml_tensor * fold = ggml_ssm_conv_fold(ctx,
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 8, 1),
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 1),
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 1, 1),
            rs, 4, 0);
    const bool ok = ggml_backend_supports_op(backend, rr) && ggml_backend_supports_op(backend, fold);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    ggml_backend_load_all();

    const gdn_shape shapes[] = {
        { 32,  2,  6 }, // broadcast q/k heads
        { 128, 4,  4 }, // qwen35 head size, H_k == H_v
        { 128, 4, 12 }, // qwen35 head size, broadcast
    };

    int n_backends = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend != nullptr);

        runner r;
        r.backend = backend;

        for (const gdn_shape & sh : shapes) {
            if (!backend_supports(backend, sh)) {
                fprintf(stderr, "%s: S=%lld not supported, skipping\n", ggml_backend_name(backend), (long long) sh.S);
                continue;
            }

            const int n_fail_before = n_fail;

            for (int64_t K : { 4, 8 }) {
                for (int64_t T : { (int64_t) 1, K/2, K }) {
                    for (int64_t B : { 1, 6 }) {
                        test_record_fold(r, sh, T, K, B);
                    }
                }
                test_chain(r, sh, K, 50);
            }

            fprintf(stderr, "%s: S=%lld H=%lld/%lld: %s\n", ggml_backend_name(backend), (long long) sh.S,
                    (long long) sh.H_k, (long long) sh.H_v, n_fail == n_fail_before ? "OK" : "FAIL");
        }

        n_backends++;
        ggml_backend_free(backend);
    }

    if (n_backends == 0) {
        fprintf(stderr, "no backends\n");
        return 1;
    }

    fprintf(stderr, "%s\n", n_fail == 0 ? "all ReplaySSM checks passed" : "ReplaySSM checks FAILED");
    return n_fail == 0 ? 0 : 1;
}

#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

// One transition of the state column `col` owned by a warp; each lane holds rows_per_lane rows (row i = r*warp_size + lane).
// The verify loops of gated_delta_net(_rr) and the ReplaySSM replay loop all run this function. The rounding of every
// operation is pinned with intrinsics so that -use_fast_math cannot contract the loops differently, which keeps a folded
// state bit-identical to the verify trajectory.
template <int warp_size, int rows_per_lane, bool KDA>
static __device__ __forceinline__ void gdn_step(
        float * s_shard, const float * k_reg, const float * g_t, const float v_col, const float beta_val, const int lane) {
    if constexpr (!KDA) {
        const float g_val = expf(*g_t);

        // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard = __fmaf_rn(s_shard[r], k_reg[r], kv_shard);
        }
        const float kv_col = warp_reduce_sum<warp_size>(kv_shard);

        // delta[col] = (v[col] - g * kv[col]) * beta
        const float delta_col = __fmul_rn(__fsub_rn(v_col, __fmul_rn(g_val, kv_col)), beta_val);

        // S[i][col] = g * S[i][col] + k[i] * delta[col]
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = __fmaf_rn(k_reg[r], delta_col, __fmul_rn(g_val, s_shard[r]));
        }
    } else {
        float g_reg[rows_per_lane];

        // kv[col] = sum_i g[i] * S[i][col] * k[i]
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            g_reg[r] = expf(g_t[i]);
            kv_shard = __fmaf_rn(__fmul_rn(g_reg[r], s_shard[r]), k_reg[r], kv_shard);
        }
        const float kv_col = warp_reduce_sum<warp_size>(kv_shard);

        // delta[col] = (v[col] - kv[col]) * beta
        const float delta_col = __fmul_rn(__fsub_rn(v_col, kv_col), beta_val);

        // S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = __fmaf_rn(k_reg[r], delta_col, __fmul_rn(g_reg[r], s_shard[r]));
        }
    }
}

// attn[col] = (S^T @ q)[col] * scale
template <int warp_size, int rows_per_lane>
static __device__ __forceinline__ float gdn_readout(const float * s_shard, const float * q_reg, const float scale) {
    float attn_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        attn_partial = __fmaf_rn(s_shard[r], q_reg[r], attn_partial);
    }
    return __fmul_rn(warp_reduce_sum<warp_size>(attn_partial), scale);
}

// one verify step: gdn_step on token t, then the readout into attn_data[col]
template <int warp_size, int rows_per_lane, bool KDA>
static __device__ __forceinline__ void gdn_token(
        float * s_shard, const float * q_t, const float * k_t, const float * v_t, const float * g_t, const float * beta_t,
        float * attn_data, const int col, const int lane, const float scale) {
    const float beta_val = *beta_t;

    // Cache k and q in registers
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        k_reg[r] = k_t[i];
        q_reg[r] = q_t[i];
    }

    gdn_step<warp_size, rows_per_lane, KDA>(s_shard, k_reg, g_t, v_t[col], beta_val, lane);

    const float attn_col = gdn_readout<warp_size, rows_per_lane>(s_shard, q_reg, scale);
    if (lane == 0) {
        attn_data[col] = attn_col;
    }
}

// Each warp owns one state column of one head of one seq; a block holds NS seqs (threadIdx.z). With state_ids the
// input state of seq s is row state_ids[s] of the cache the kernel writes into: the block reads its columns of every
// seq before it writes any, so a seq may read the row another seq writes (the launcher then puts all seqs in a block).
template <int S_v, bool KDA, bool keep_rs_t, int NS>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4 * NS, NS == 1 ? 2 : 1)
gated_delta_net_cuda(const float *   q,
                     const float *   k,
                     const float *   v,
                     const float *   g,
                     const float *   beta,
                     const float *   curr_state,
                     const int32_t * state_ids,
                     float *         dst,
                     float *         state,
                     int64_t         H,
                     int64_t         n_tokens,
                     int64_t         n_seqs,
                     int64_t         sq1,
                     int64_t         sq2,
                     int64_t         sq3,
                     int64_t         sv1,
                     int64_t         sv2,
                     int64_t         sv3,
                     int64_t         sb1,
                     int64_t         sb2,
                     int64_t         sb3,
                     const uint3     neqk1_magic,
                     const uint3     rq3_magic,
                     float           scale,
                     int64_t         state_slot_stride,
                     int             K) {
    const uint32_t h_idx    = blockIdx.x;
    const int64_t  sequence = (int64_t) blockIdx.y * NS + threadIdx.z;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous
    float         s_shard[rows_per_lane];

    // input state holds s0 only: [S_v, S_v, H, n_seqs] (or cache rows), seq stride is D = H * S_v * S_v
    ggml_cuda_pdl_sync();
    if (sequence < n_seqs) {
        const int64_t src_row = state_ids != nullptr ? state_ids[sequence] : sequence;
        const float * cs      = curr_state + (src_row * H + h_idx) * S_v * S_v + col * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = cs[r * warp_size + lane];
        }
    }
    if constexpr (NS > 1) {
        __syncthreads();
    }
    if (sequence >= n_seqs) {
        return;
    }

    {
        const uint32_t iq3 = fastdiv((uint32_t) sequence, rq3_magic);

        // output state layout (per-slot D * n_seqs)
        float * state_j   = state + (sequence * H + h_idx) * S_v * S_v;
        float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

        for (int t = 0; t < n_tokens; t++) {
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            const float * beta_t = beta + gb_offset;
            const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

            gdn_token<warp_size, rows_per_lane, KDA>(s_shard, q_t, k_t, v_t, g_t, beta_t, attn_data, col, lane, scale);

            attn_data += S_v * H;

            if constexpr (keep_rs_t) {
                // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
                // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * slot_state = state_j + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        slot_state[col * S_v + i] = s_shard[r];
                    }
                }
            }
        }

        if constexpr (!keep_rs_t) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i            = r * warp_size + lane;
                state_j[col * S_v + i] = s_shard[r];
            }
        }
    }
}

// ReplaySSM phase 1: replay the m pending records of the source row, store the committed state before token
// j_commit = n_tokens - min(n_tokens, K), then run the new tokens. The records are only read here.
// Seqs per block and state_ids as in gated_delta_net_cuda.
template <int S_v, bool KDA, int NS>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4 * NS, NS == 1 ? 2 : 1)
gated_delta_net_rr_cuda(const float *   q,
                        const float *   k,
                        const float *   v,
                        const float *   g,
                        const float *   beta,
                        const float *   curr_state,
                        const int32_t * state_ids,
                        const float *   rec,
                        const int32_t * rs,
                        float *         dst,
                        float *         state,
                        int64_t         H,
                        int64_t         n_tokens,
                        int64_t         n_seqs,
                        int64_t         sq1,
                        int64_t         sq2,
                        int64_t         sq3,
                        int64_t         sv1,
                        int64_t         sv2,
                        int64_t         sv3,
                        int64_t         sb1,
                        int64_t         sb2,
                        int64_t         sb3,
                        const uint3     neqk1_magic,
                        const uint3     rq3_magic,
                        float           scale,
                        int             K,
                        int64_t         rec_row_stride,
                        int64_t         rec_slot_stride,
                        int64_t         rec_off_v,
                        int64_t         rec_off_g,
                        int64_t         rec_off_b) {
    const uint32_t h_idx    = blockIdx.x;
    const int64_t  sequence = (int64_t) blockIdx.y * NS + threadIdx.z;
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];

    ggml_cuda_pdl_sync();
    if (sequence < n_seqs) {
        const int64_t src_row = state_ids != nullptr ? state_ids[sequence] : sequence;
        const float * cs      = curr_state + (src_row * H + h_idx) * S_v * S_v + col * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = cs[r * warp_size + lane];
        }
    }
    if constexpr (NS > 1) {
        __syncthreads();
    }
    if (sequence >= n_seqs) {
        return;
    }

    const int j_commit = (int) n_tokens - min((int) n_tokens, K);

    {
        const uint32_t iq3 = fastdiv((uint32_t) sequence, rq3_magic);

        float * state_j   = state + (sequence * H + h_idx) * S_v * S_v;
        float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

        const int m       = rs[2 * sequence + 0];
        const int src_row = rs[2 * sequence + 1];

        const float * rec_src = rec + (int64_t) src_row * rec_row_stride;
        for (int u = 0; u < m; u++) {
            const float * r_t = rec_src + u * rec_slot_stride;

            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = r_t[iq1 * S_v + i];
            }

            gdn_step<warp_size, rows_per_lane, KDA>(s_shard, k_reg, r_t + rec_off_g + h_idx * (KDA ? S_v : 1),
                    r_t[rec_off_v + h_idx * S_v + col], r_t[rec_off_b + h_idx], lane);
        }

        for (int t = 0; t < n_tokens; t++) {
            if (t == j_commit) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i            = r * warp_size + lane;
                    state_j[col * S_v + i] = s_shard[r];
                }
            }

            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            const float * beta_t = beta + gb_offset;
            const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

            gdn_token<warp_size, rows_per_lane, KDA>(s_shard, q_t, k_t, v_t, g_t, beta_t, attn_data, col, lane, scale);

            attn_data += S_v * H;
        }
    }
}

// ReplaySSM phase 2: copy the inputs of the last n_rec tokens of each seq into rec row dst_row0 + s.
// Launched after phase 1 on the same stream, so every read of the source records has completed.
static __global__ void gated_delta_net_rr_record_cuda(const float * k,
                                                      const float * v,
                                                      const float * g,
                                                      const float * beta,
                                                      float *       rec,
                                                      int64_t       n_embd_rec,
                                                      int64_t       rec_off_v,
                                                      int64_t       rec_off_g,
                                                      int64_t       rec_off_b,
                                                      int64_t       S_v,
                                                      int64_t       G,
                                                      int64_t       j_commit,
                                                      int64_t       K,
                                                      int64_t       dst_row0,
                                                      int64_t       sk1,
                                                      int64_t       sk2,
                                                      int64_t       sk3,
                                                      int64_t       rk3,
                                                      int64_t       sv1,
                                                      int64_t       sv2,
                                                      int64_t       sv3,
                                                      int64_t       sb1,
                                                      int64_t       sb2,
                                                      int64_t       sb3) {
    const int64_t e = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t u = blockIdx.y;
    const int64_t s = blockIdx.z;
    const int64_t t = j_commit + u;

    ggml_cuda_pdl_sync();

    if (e >= n_embd_rec) {
        return;
    }

    float val;
    if (e < rec_off_v) {
        const int64_t hk = e / S_v;
        val = k[(s / rk3) * sk3 + t * sk2 + hk * sk1 + (e - hk * S_v)];
    } else if (e < rec_off_g) {
        const int64_t i = e - rec_off_v;
        const int64_t h = i / S_v;
        val = v[s * sv3 + t * sv2 + h * sv1 + (i - h * S_v)];
    } else if (e < rec_off_b) {
        const int64_t i = e - rec_off_g;
        const int64_t h = i / G;
        val = g[(s * sb3 + t * sb2 + h * sb1) * G + (i - h * G)];
    } else {
        const int64_t h = e - rec_off_b;
        val = beta[s * sb3 + t * sb2 + h * sb1];
    }

    rec[((dst_row0 + s) * K + u) * n_embd_rec + e] = val;
}

// seqs per block: all of them when the state is gathered from the cache it is written to (see gated_delta_net_cuda)
static int gdn_seqs_per_warp(const int32_t * state_ids, int64_t n_seqs) {
    if (state_ids == nullptr) {
        return 1;
    }
    GGML_ASSERT(n_seqs <= GGML_CUDA_GDN_STATE_IDS_MAX_SEQS);
    return n_seqs <= 1 ? 1 : n_seqs <= 2 ? 2 : n_seqs <= 4 ? 4 : 8;
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d, const int32_t * state_ids,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int ns = gdn_seqs_per_warp(state_ids, n_seqs);
    GGML_ASSERT(ns == 1 || (S_v == 128 && !KDA));

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, (n_seqs + ns - 1) / ns, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, ns > 1 ? n_seqs : 1); // grid.y == 1 when ns > 1

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

#define GDN_LAUNCH(S, NS_)                                                                                     \
    ggml_cuda_kernel_launch(gated_delta_net_cuda<S, KDA, keep_rs_t, NS_>, launch_params,                      \
        q_d, k_d, v_d, g_d, b_d, s_d, state_ids, dst_d, state_d, H,                                            \
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                                        \
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K)

    switch (S_v) {
        case 16:  GDN_LAUNCH(16, 1); break;
        case 32:  GDN_LAUNCH(32, 1); break;
        case 64:  GDN_LAUNCH(64, 1); break;
        case 128:
            if constexpr (!KDA) {
                switch (ns) {
                    case 2:  GDN_LAUNCH(128, 2); return;
                    case 4:  GDN_LAUNCH(128, 4); return;
                    case 8:  GDN_LAUNCH(128, 8); return;
                    default: break;
                }
            }
            GDN_LAUNCH(128, 1);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
#undef GDN_LAUNCH
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    // the input state, or the cache rows it was gathered from when the gather is fused
    const float *   s_d       = (const float *) src_state->data;
    const int32_t * state_ids = nullptr;
    if (cache != nullptr && cache->state_ids != nullptr) {
        s_d       = cache->state_src;
        state_ids = cache->state_ids;
    }
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, state_ids, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, state_ids, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, state_ids, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, state_ids, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

template <int S_v, bool KDA, int NS>
static void launch_gated_delta_net_rr(
        const ggml_cuda_kernel_launch_params & launch_params,
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        const float * s_d, const int32_t * state_ids, const float * rec_d, const int32_t * rs_d, float * dst_d,
        float * state_d, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        uint3 neqk1_magic, uint3 rq3_magic, float scale, int K,
        int64_t rec_row_stride, int64_t rec_slot_stride, int64_t rec_off_v, int64_t rec_off_g, int64_t rec_off_b) {
    ggml_cuda_kernel_launch(gated_delta_net_rr_cuda<S_v, KDA, NS>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, state_ids, rec_d, rs_d, dst_d, state_d, H, n_tokens, n_seqs,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K,
        rec_row_stride, rec_slot_stride, rec_off_v, rec_off_g, rec_off_b);
}

static void ggml_cuda_op_gated_delta_net_rr_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];
    ggml_tensor * src_rec   = dst->src[6];
    ggml_tensor * src_rs    = dst->src[7];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;
    const int64_t H_k      = nek1;
    const int64_t G        = src_g->ne[0];
    const bool    kda      = (G == S_v);

    GGML_ASSERT(neq1 == nek1);
    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(ggml_is_contiguous(src_rec));

    // the records are written in place: rec must be the arena itself, not a scheduler copy
    GGML_ASSERT(src_rec->buffer != nullptr && ggml_backend_buffer_get_usage(src_rec->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    const int     K        = ggml_get_op_params_i32(dst, 0);
    const int64_t dst_row0 = ggml_get_op_params_i32(dst, 1);

    const int64_t n_embd_rec = src_rec->ne[0];
    const int64_t rec_off_v  = S_v * H_k;
    const int64_t rec_off_g  = rec_off_v + S_v * H;
    const int64_t rec_off_b  = rec_off_g + G * H;

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    const float *   q_d   = (const float *) src_q->data;
    const float *   k_d   = (const float *) src_k->data;
    const float *   v_d   = (const float *) src_v->data;
    const float *   g_d   = (const float *) src_g->data;
    const float *   b_d   = (const float *) src_beta->data;
    const float *   s_d   = cache != nullptr && cache->state_ids != nullptr ? cache->state_src : (const float *) src_state->data;
    const int32_t * ids_d = cache != nullptr ? cache->state_ids : nullptr;
    float *         rec_d = (float *) src_rec->data;
    const int32_t * rs_d  = (const int32_t *) src_rs->data;
    float *         dst_d = (float *) dst->data;

    // committed state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d = cache != nullptr ? cache->data : dst_d + S_v * H * n_tokens * n_seqs;

    cudaStream_t stream = ctx.stream();

    // phase 1: replay + committed store + verify
    {
        const int  ns        = gdn_seqs_per_warp(ids_d, n_seqs);
        GGML_ASSERT(ns == 1 || (S_v == 128 && !kda));
        const int  warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
        const int  num_warps = 4;
        const dim3 grid_dims(H, (n_seqs + ns - 1) / ns, (S_v + num_warps - 1) / num_warps);
        const dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, ns > 1 ? n_seqs : 1); // grid.y == 1 when ns > 1
        const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);

        const uint3 neqk1_magic = init_fastdiv_values(neq1);
        const uint3 rq3_magic   = init_fastdiv_values(nev3 / neq3);

#define GDN_RR_LAUNCH(S, KDA_, NS_)                                                                                  \
        launch_gated_delta_net_rr<S, KDA_, NS_>(launch_params, q_d, k_d, v_d, g_d, b_d, s_d, ids_d, rec_d, rs_d,   \
            dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,                         \
            neqk1_magic, rq3_magic, scale, K, K * n_embd_rec, n_embd_rec, rec_off_v, rec_off_g, rec_off_b)

        switch (S_v) {
            case 16:  if (kda) { GDN_RR_LAUNCH(16, true, 1); } else { GDN_RR_LAUNCH(16, false, 1); } break;
            case 32:  if (kda) { GDN_RR_LAUNCH(32, true, 1); } else { GDN_RR_LAUNCH(32, false, 1); } break;
            case 64:  if (kda) { GDN_RR_LAUNCH(64, true, 1); } else { GDN_RR_LAUNCH(64, false, 1); } break;
            case 128:
                if (kda) {
                    GDN_RR_LAUNCH(128, true, 1);
                } else {
                    switch (ns) {
                        case 2:  GDN_RR_LAUNCH(128, false, 2); break;
                        case 4:  GDN_RR_LAUNCH(128, false, 4); break;
                        case 8:  GDN_RR_LAUNCH(128, false, 8); break;
                        default: GDN_RR_LAUNCH(128, false, 1); break;
                    }
                }
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
#undef GDN_RR_LAUNCH
    }

    // phase 2: record the inputs of the last min(n_tokens, K) tokens
    {
        const int64_t n_rec = std::min<int64_t>(n_tokens, K);
        const int     block = 256;
        const dim3    grid_dims((n_embd_rec + block - 1) / block, n_rec, n_seqs);
        const ggml_cuda_kernel_launch_params launch_params(grid_dims, dim3(block, 1, 1), 0, stream);

        ggml_cuda_kernel_launch(gated_delta_net_rr_record_cuda, launch_params,
            k_d, v_d, g_d, b_d, rec_d, n_embd_rec, rec_off_v, rec_off_g, rec_off_b, S_v, G,
            n_tokens - n_rec, (int64_t) K, dst_row0,
            (int64_t) (nbk1 / sizeof(float)), (int64_t) (nbk2 / sizeof(float)), (int64_t) (nbk3 / sizeof(float)), nev3 / nek3,
            sv1, sv2, sv3, sb1, sb2, sb3);
    }
}

void ggml_cuda_op_gated_delta_net_rr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_rr_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_rr_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_rr_impl(ctx, dst, &cache);
}

// ReplaySSM conv history fold

// phase 1: out[:, c, s] = tail_{n_hist}(conv_state[:, c, s] || recx[src][0:m][c]), a pure gather
static __global__ void ssm_conv_fold_cuda(const float *   conv_state,
                                          const float *   recx,
                                          const int32_t * rs,
                                          float *         dst,
                                          int64_t         n_hist,
                                          int64_t         C,
                                          int64_t         K,
                                          int64_t         cs1,
                                          int64_t         cs2) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t s = blockIdx.y;

    ggml_cuda_pdl_sync();

    if (c >= C) {
        return;
    }

    const int     m       = rs[2 * s + 0];
    const int64_t src_row = rs[2 * s + 1];

    const float * cs  = conv_state + s * cs2 + c * cs1;
    float *       out = dst + (s * C + c) * n_hist;

    for (int64_t i = 0; i < n_hist; ++i) {
        const int64_t idx = i + m;
        out[i] = idx < n_hist ? cs[idx] : recx[(src_row * K + idx - n_hist) * C + c];
    }
}

// phase 2: recx[dst_row0 + s][u] = x[:, j_commit + u, s]
static __global__ void ssm_conv_fold_record_cuda(const float * x,
                                                 float *       recx,
                                                 int64_t       C,
                                                 int64_t       K,
                                                 int64_t       j_commit,
                                                 int64_t       dst_row0,
                                                 int64_t       sx1,
                                                 int64_t       sx2) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t u = blockIdx.y;
    const int64_t s = blockIdx.z;

    ggml_cuda_pdl_sync();

    if (c >= C) {
        return;
    }

    recx[((dst_row0 + s) * K + u) * C + c] = x[s * sx2 + (j_commit + u) * sx1 + c];
}

void ggml_cuda_op_ssm_conv_fold(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_cs   = dst->src[0];
    ggml_tensor       * src_recx = dst->src[1];
    const ggml_tensor * src_x    = dst->src[2];
    const ggml_tensor * src_rs   = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous_rows(src_cs));
    GGML_ASSERT(ggml_is_contiguous_rows(src_x));
    GGML_ASSERT(ggml_is_contiguous(src_recx));
    GGML_ASSERT(ggml_is_contiguous(dst));

    // the records are written in place: recx must be the arena itself, not a scheduler copy
    GGML_ASSERT(src_recx->buffer != nullptr && ggml_backend_buffer_get_usage(src_recx->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    const int64_t n_hist   = src_cs->ne[0];
    const int64_t C        = src_cs->ne[1];
    const int64_t n_seqs   = src_cs->ne[2];
    const int64_t n_tokens = src_x->ne[1];

    const int64_t K        = ggml_get_op_params_i32(dst, 0);
    const int64_t dst_row0 = ggml_get_op_params_i32(dst, 1);
    const int64_t n_rec    = std::min<int64_t>(n_tokens, K);

    cudaStream_t stream = ctx.stream();
    const int    block  = 256;

    {
        const ggml_cuda_kernel_launch_params launch_params(dim3((C + block - 1) / block, n_seqs, 1), dim3(block, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(ssm_conv_fold_cuda, launch_params,
            (const float *) src_cs->data, (const float *) src_recx->data, (const int32_t *) src_rs->data, (float *) dst->data,
            n_hist, C, K, (int64_t) (src_cs->nb[1] / sizeof(float)), (int64_t) (src_cs->nb[2] / sizeof(float)));
    }
    {
        const ggml_cuda_kernel_launch_params launch_params(dim3((C + block - 1) / block, n_rec, n_seqs), dim3(block, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(ssm_conv_fold_record_cuda, launch_params,
            (const float *) src_x->data, (float *) src_recx->data, C, K, n_tokens - n_rec, dst_row0,
            (int64_t) (src_x->nb[1] / sizeof(float)), (int64_t) (src_x->nb[2] / sizeof(float)));
    }
}

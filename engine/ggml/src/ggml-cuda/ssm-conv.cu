#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// l2 norm of a 128-channel head held one channel per thread (threadIdx.x) of the seq threadIdx.y; reduced in the order
// of rms_norm_f32<256> + scale, so the fused chain is bit-identical to the unfused graph
static __device__ __forceinline__ float ssm_conv_chain_l2(const float y, const float eps, const float scale_out, float (*s_sum)[4]) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    float tmp = warp_reduce_sum(y * y);
    if (lane == 0) {
        s_sum[threadIdx.y][warp] = tmp;
    }
    __syncthreads();
    tmp = lane < 4 ? s_sum[threadIdx.y][lane] : 0.0f;
    tmp = warp_reduce_sum(tmp);
    __syncthreads(); // s_sum is rewritten by the next token

    const int   ncols = 128;
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    return scale_out * (scale * y);
}

// thread (c, s): channel c of seq s, all seqs of the ubatch in one block; the last block computes the gates
template <int NS>
static __global__ void __launch_bounds__(128 * NS) ssm_conv_chain_f32(const ggml_cuda_ssm_conv_chain p) {
    constexpr int d_conv = 4;
    constexpr int n_hist = d_conv - 1;
    constexpr int block  = 128;

    __shared__ float s_sum[NS][block / WARP_SIZE];

    const int tid = threadIdx.x;
    const int s   = threadIdx.y;

    ggml_cuda_pdl_sync();

    if (blockIdx.x == p.C / block) {
        const int64_t n = p.H_v * p.T * p.n_seqs;
        for (int64_t e = s * block + tid; e < n; e += block * blockDim.y) {
            const int64_t h = e % p.H_v;
            p.gate[e] = ggml_cuda_op_softplus_single(p.alpha[e] + p.dt[h]) * p.A[h];
            p.beta[e] = ggml_cuda_op_sigmoid_single(p.b[e]);
        }
        return;
    }

    const int64_t c = (int64_t) blockIdx.x * block + tid;

    float w[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        w[j] = p.w[c * d_conv + j];
    }

    // gather the history of every seq before the first write: the conv cache and the records are updated in place,
    // and the source row of one seq can be the destination row of another
    float hist[n_hist];
    {
        const float * cs = p.cache + p.ids[s] * p.cache_row + c * n_hist;
        if (p.recx != nullptr) {
            const int     m       = p.rs[2 * s + 0];
            const int64_t src_row = p.rs[2 * s + 1];
#pragma unroll
            for (int i = 0; i < n_hist; ++i) {
                const int idx = i + m;
                hist[i] = idx < n_hist ? cs[idx] : p.recx[(src_row * p.rec_K + idx - n_hist) * p.C + c];
            }
        } else {
#pragma unroll
            for (int i = 0; i < n_hist; ++i) {
                hist[i] = cs[i];
            }
        }
    }
    __syncthreads();

    const float * x = p.x + s * p.sx2 + c;

    // window[k]: history for k < n_hist, then the inputs
    const auto window = [&](const int64_t k) {
        return k < n_hist ? (k == 0 ? hist[0] : k == 1 ? hist[1] : hist[2]) : x[(k - n_hist) * p.sx1];
    };

    if (p.recx != nullptr) {
        const int64_t n_rec    = min(p.T, p.rec_K);
        const int64_t j_commit = p.T - n_rec;
        for (int64_t u = 0; u < n_rec; ++u) {
            p.recx[((p.rec_dst_row0 + s) * p.rec_K + u) * p.C + c] = x[(j_commit + u) * p.sx1];
        }
    }

    for (int k = 0; k < p.n_cpy; ++k) {
        float * dst = p.cpy_dst[k] + s * p.cpy_row[k] + c * n_hist;
#pragma unroll
        for (int i = 0; i < n_hist; ++i) {
            dst[i] = window(p.cpy_off[k] + i);
        }
    }

    const bool l2   = p.qn != nullptr && blockIdx.x < 2 * p.H_k;
    const bool is_q = blockIdx.x < p.H_k;

    float xw[d_conv] = { hist[0], hist[1], hist[2], 0.0f };
    for (int64_t t = 0; t < p.T; ++t) {
        xw[d_conv - 1] = x[t * p.sx1];

        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += xw[j] * w[j];
        }
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            xw[j] = xw[j + 1];
        }

        const float y = ggml_cuda_op_silu_single(sumf);
        p.y[(s * p.T + t) * p.C + c] = y;

        if (l2) {
            const int64_t hk = is_q ? blockIdx.x : blockIdx.x - p.H_k;
            const float   o  = ssm_conv_chain_l2(y, is_q ? p.q_eps : p.k_eps, is_q ? p.q_scale : p.k_scale, s_sum);
            (is_q ? p.qn : p.kn)[((s * p.T + t) * p.H_k + hk) * block + tid] = o;
        }
    }
}

void ggml_cuda_op_ssm_conv_chain(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_chain & p) {
    GGML_ASSERT(p.C % 128 == 0 && p.T <= GGML_CUDA_SSM_CONV_CHAIN_MAX_TOKENS);
    GGML_ASSERT(p.n_seqs <= GGML_CUDA_SSM_CONV_CHAIN_MAX_SEQS && p.n_cpy <= GGML_CUDA_SSM_CONV_CHAIN_MAX_CPY);

    const dim3 grid(p.C / 128 + (p.gate != nullptr ? 1 : 0));
    const ggml_cuda_kernel_launch_params launch_params(grid, dim3(128, p.n_seqs), 0, ctx.stream());

    if (p.n_seqs == 1) {
        ggml_cuda_kernel_launch(ssm_conv_chain_f32<1>, launch_params, p);
    } else if (p.n_seqs == 2) {
        ggml_cuda_kernel_launch(ssm_conv_chain_f32<2>, launch_params, p);
    } else if (p.n_seqs <= 4) {
        ggml_cuda_kernel_launch(ssm_conv_chain_f32<4>, launch_params, p);
    } else {
        ggml_cuda_kernel_launch(ssm_conv_chain_f32<8>, launch_params, p);
    }
}

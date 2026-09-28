#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// the gated delta net input chain of one layer as a single kernel: gather (and ReplaySSM-fold) the conv history of each
// seq from the conv cache, write the new history back, conv + silu, optionally l2-normalize the q/k heads and compute
// the gates (see ggml_cuda_try_ssm_conv_chain_fusion)
#define GGML_CUDA_SSM_CONV_CHAIN_MAX_SEQS 8
#define GGML_CUDA_SSM_CONV_CHAIN_MAX_TOKENS 32
#define GGML_CUDA_SSM_CONV_CHAIN_MAX_CPY 4

struct ggml_cuda_ssm_conv_chain {
    int64_t C;      // channels
    int64_t T;      // tokens per seq
    int64_t n_seqs;

    // conv history source: row ids[s] of the conv cache, [d_conv - 1] floats per channel
    const float   * cache;
    int64_t         cache_row; // floats
    const int32_t * ids;

    // ReplaySSM fold (recx == nullptr: none): rs[2*s] pending records m, rs[2*s + 1] source row
    float         * recx;
    const int32_t * rs;
    int64_t         rec_K;
    int64_t         rec_dst_row0;

    // input x[c + t*sx1 + s*sx2] and the conv weights [d_conv, C]
    const float * x;
    int64_t       sx1;
    int64_t       sx2;
    const float * w;

    // history write-back: dst[s*row + c*(d_conv - 1) + i] = window[off + i], window = history || x
    int     n_cpy;
    float * cpy_dst[GGML_CUDA_SSM_CONV_CHAIN_MAX_CPY];
    int64_t cpy_row[GGML_CUDA_SSM_CONV_CHAIN_MAX_CPY];
    int64_t cpy_off[GGML_CUDA_SSM_CONV_CHAIN_MAX_CPY];

    float * y; // silu(conv) [C, T, n_seqs]

    // l2 norm of the first 2*H_k heads of size 128 (q then k), qn/kn [128, H_k, T, n_seqs] (qn == nullptr: none)
    float * qn;
    float * kn;
    int64_t H_k;
    float   q_eps;
    float   q_scale;
    float   k_eps;
    float   k_scale;

    // gates (gate == nullptr: none), [H_v, T, n_seqs]: gate = softplus(alpha + dt) * A, beta = sigmoid(b)
    const float * alpha;
    const float * dt;
    const float * A;
    const float * b;
    float       * gate;
    float       * beta;
    int64_t       H_v;
};

void ggml_cuda_op_ssm_conv_chain(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_chain & p);

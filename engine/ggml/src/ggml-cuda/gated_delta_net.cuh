#include "common.cuh"
#include "ggml.h"

// max seqs of a ubatch for which the input state can be gathered in-kernel from the cache it is written to
#define GGML_CUDA_GDN_STATE_IDS_MAX_SEQS 8

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)

    // input state of seq s = row state_ids[s] (stride D) of state_src, replacing the get_rows that feeds the op
    // (state_ids == nullptr: read the state operand)
    const float *   state_src = nullptr;
    const int32_t * state_ids = nullptr;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// ReplaySSM: gated_delta_net_rr (replay + committed state + record) and ssm_conv_fold
void ggml_cuda_op_gated_delta_net_rr(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the committed state into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_rr_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                                 ggml_cuda_gated_delta_net_fused_cache cache);

void ggml_cuda_op_ssm_conv_fold(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

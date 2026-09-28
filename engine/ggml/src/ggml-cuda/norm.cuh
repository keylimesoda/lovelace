#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// rms_norm -> mul -> (silu(g) -> mul): the gated output norm of the gated delta net
void ggml_cuda_op_rms_norm_fused_silu_gate(ggml_backend_cuda_context & ctx,
                                           ggml_tensor *               dst,
                                           ggml_tensor *               mul_tensor,
                                           ggml_tensor *               silu_tensor,
                                           ggml_tensor *               gate_tensor);

// add -> rms_norm -> mul: the residual add and the norm of the next block; writes both the sum and the norm
void ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               add_tensor,
                                     ggml_tensor *               rms_tensor,
                                     ggml_tensor *               mul_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

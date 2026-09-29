#include "kq-mask.cuh"

// Every thread writes 8 consecutive cells of one row: 16 B (f16) or 32 B (f32) of mask from 32 B of keys, which stay in
// L2 across the rows. The mask values are the bit patterns of 0 and -INFINITY, so the result is bitwise what the host
// fill (llama_cast<T>(0.0f) / llama_cast<T>(-INFINITY)) produces.

#define CUDA_KQ_MASK_BLOCK_SIZE 128

template <typename U, U drop>
static __device__ __forceinline__ U kq_mask_val(const int32_t key, const int2 r) {
    return (key & r.x) && (key & 0xFFFFFF) <= r.y ? U(0) : drop;
}

template <typename U, U drop>
static __global__ void kq_mask_kernel(
        const int32_t * __restrict__ keys, const int2 * __restrict__ rows, U * __restrict__ dst,
        const int64_t n_kv, const int64_t n_rows, const int64_t n_all) {
    const int64_t j0 = 8*((int64_t) blockIdx.x*blockDim.x + threadIdx.x);
    if (j0 >= n_kv) {
        return;
    }

    for (int64_t ir = blockIdx.y; ir < n_all; ir += gridDim.y) {
        const int2      r   = rows[ir];
        const int32_t * key = keys + (ir/n_rows)*n_kv;
        U             * row = dst + ir*n_kv;

        if (n_kv % 8 == 0) {
            // n_kv % 8 == 0 keeps every stream's keys and every row 8-element aligned
            const int4 k0 = *(const int4 *) (key + j0);
            const int4 k1 = *(const int4 *) (key + j0 + 4);
            alignas(16) U v[8] = {
                kq_mask_val<U, drop>(k0.x, r), kq_mask_val<U, drop>(k0.y, r), kq_mask_val<U, drop>(k0.z, r), kq_mask_val<U, drop>(k0.w, r),
                kq_mask_val<U, drop>(k1.x, r), kq_mask_val<U, drop>(k1.y, r), kq_mask_val<U, drop>(k1.z, r), kq_mask_val<U, drop>(k1.w, r),
            };
            int4 * out = (int4 *) (row + j0);
#pragma unroll
            for (int q = 0; q < (int) (8*sizeof(U)/sizeof(int4)); ++q) {
                out[q] = ((const int4 *) v)[q];
            }
        } else {
            const int64_t j1 = min(j0 + 8, n_kv);
            for (int64_t j = j0; j < j1; ++j) {
                row[j] = kq_mask_val<U, drop>(key[j], r);
            }
        }
    }
}

void ggml_cuda_op_kq_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * keys = dst->src[0];
    const ggml_tensor * rows = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(keys) && ggml_is_contiguous(rows));

    const int64_t n_kv   = dst->ne[0];
    const int64_t n_rows = dst->ne[1];
    const int64_t n_all  = n_rows*dst->ne[3];

    if (n_kv == 0 || n_all == 0) {
        return;
    }

    const dim3 block_dims(CUDA_KQ_MASK_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((n_kv + 8*CUDA_KQ_MASK_BLOCK_SIZE - 1)/(8*CUDA_KQ_MASK_BLOCK_SIZE), std::min<int64_t>(n_all, 65535), 1);

    cudaStream_t stream = ctx.stream();

    const int32_t * keys_d = (const int32_t *) keys->data;
    const int2    * rows_d = (const int2    *) rows->data;

    switch (dst->type) {
        case GGML_TYPE_F16:
            kq_mask_kernel<uint16_t, 0xFC00u><<<block_nums, block_dims, 0, stream>>>(
                keys_d, rows_d, (uint16_t *) dst->data, n_kv, n_rows, n_all);
            break;
        case GGML_TYPE_F32:
            kq_mask_kernel<uint32_t, 0xFF800000u><<<block_nums, block_dims, 0, stream>>>(
                keys_d, rows_d, (uint32_t *) dst->data, n_kv, n_rows, n_all);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}

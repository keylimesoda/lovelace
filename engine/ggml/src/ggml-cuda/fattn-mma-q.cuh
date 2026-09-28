#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// FlashAttention for quantized K/V and few queries per K/V head: decode and speculative verify batches.
//
// Each CUDA block handles one tile of up to FATTN_MMA_Q_MAX_TOKENS query tokens times all Q heads of one K/V head,
// flattened into ncols_tiles*16 rows of (token, head) pairs, over one chunk of the K/V cells (split-K).
// The quantized K/V rows are copied into shared memory as they are (cp.async, multi-stage) and dequantized from there
// directly into mma fragments, so the K/V cache is read from VRAM exactly once per step and no f16 copy is made.
// KQ is computed with int8 mma from Q quantized to 8 bit per 32 values (like q8_1) and the raw K values,
// VKQ with f16 mma, accumulated per K/V tile in f16 and across tiles in f32.
// Before touching K/V a block scans its mask chunk and only visits K/V tiles that some query of the tile can see:
// with a unified KV cache shared by several sequences each tile then reads only the cells of its own sequences.
// The partial results of the chunks are merged by flash_attn_ext_mma_q_combine with a fixed number of chunks per row.

#define FATTN_MMA_Q_NWARPS          8
#define FATTN_MMA_Q_NTHREADS        (FATTN_MMA_Q_NWARPS*WARP_SIZE)
#define FATTN_MMA_Q_NBATCH          64  // K/V cells per tile, 8 per warp for KQ
#define FATTN_MMA_Q_MAX_TOKENS      8   // query tokens per tile
#define FATTN_MMA_Q_MAX_CHUNK_TILES 128 // K/V tiles per chunk
#define FATTN_MMA_Q_SMEM_MAX        (99*1024)

static constexpr __host__ __device__ int fattn_mma_q_block_bytes(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 ? sizeof(block_q4_0) :
           type == GGML_TYPE_Q4_1 ? sizeof(block_q4_1) :
           type == GGML_TYPE_Q5_0 ? sizeof(block_q5_0) :
           type == GGML_TYPE_Q5_1 ? sizeof(block_q5_1) :
           type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) : 0;
}

static constexpr __host__ __device__ bool fattn_mma_q_type_supported(const ggml_type type) {
    return fattn_mma_q_block_bytes(type) != 0;
}

// Bytes of one K/V row (one cell of one K/V head) in shared memory.
static constexpr __host__ __device__ int fattn_mma_q_row_bytes(const int D, const ggml_type type) {
    return (D/QK8_0) * fattn_mma_q_block_bytes(type);
}

// Granularity of the asynchronous K/V copies, the K/V strides and data pointers must be aligned to it.
static constexpr __host__ __device__ int fattn_mma_q_cpy_bytes(const int D, const ggml_type type) {
    return fattn_mma_q_row_bytes(D, type) % 16 == 0 ? 16 : (fattn_mma_q_row_bytes(D, type) % 8 == 0 ? 8 : 4);
}

// Shared memory layout, in bytes. Q and P rows are padded by 16 bytes to avoid bank conflicts for ldmatrix.
static constexpr __host__ __device__ int fattn_mma_q_stride_Q(const int D) {
    return D + 16; // int8
}
static constexpr __host__ __device__ int fattn_mma_q_stride_P() {
    return FATTN_MMA_Q_NBATCH + 8; // halves
}
static constexpr __host__ __device__ int fattn_mma_q_stride_mask() {
    return FATTN_MMA_Q_NBATCH + 8; // halves
}
static constexpr __host__ __device__ int fattn_mma_q_smem_fixed(const int D, const int ncols_tiles) {
    return 16*ncols_tiles*fattn_mma_q_stride_Q(D)        // Q, int8
         + 16*ncols_tiles*(D/QK8_1)*8                    // Q scales and sums
         + 16*ncols_tiles*fattn_mma_q_stride_P()*2       // P
         + FATTN_MMA_Q_NWARPS*16*ncols_tiles*4           // cross-warp row max/sum
         + FATTN_MMA_Q_MAX_CHUNK_TILES*4 + 16;           // visible tile list + count
}
static constexpr __host__ __device__ int fattn_mma_q_smem_stage(const int D, const ggml_type type) {
    return 2*FATTN_MMA_Q_NBATCH*fattn_mma_q_row_bytes(D, type)            // raw K and V
         + FATTN_MMA_Q_MAX_TOKENS*fattn_mma_q_stride_mask()*2;           // mask
}
// Number of K/V tiles in flight: 2 if it fits into shared memory, else 1.
static constexpr __host__ __device__ int fattn_mma_q_nstages(const int D, const ggml_type type, const int ncols_tiles) {
    return fattn_mma_q_smem_fixed(D, ncols_tiles) + 2*fattn_mma_q_smem_stage(D, type) <= FATTN_MMA_Q_SMEM_MAX ? 2 : 1;
}
static constexpr __host__ __device__ int fattn_mma_q_smem_total(const int D, const ggml_type type, const int ncols_tiles) {
    return fattn_mma_q_smem_fixed(D, ncols_tiles) + fattn_mma_q_nstages(D, type, ncols_tiles)*fattn_mma_q_smem_stage(D, type);
}

// ------------------------------------------------------------------------------------------------------------------
// Dequantization from shared memory into mma operands.

// Two nibbles/bytes in the low byte of each 16 bit half of q -> half2 of (1024 + value), exact.
static __device__ __forceinline__ half2 fattn_mma_q_u8x2_to_h2(const uint32_t q) {
    const uint32_t h = q | 0x64006400;
    return *(const half2 *) &h;
}

// Dimension j of two K/V rows.
template <ggml_type type>
static __device__ __forceinline__ half2 fattn_mma_q_dq1x2(const uint8_t * row0, const uint8_t * row1, const int j) {
    const int ib = j / QK8_0;
    const int jj = j % QK8_0;
    const uint8_t * b0 = row0 + ib*fattn_mma_q_block_bytes(type);
    const uint8_t * b1 = row1 + ib*fattn_mma_q_block_bytes(type);

    if constexpr (type == GGML_TYPE_Q4_0) {
        const int shift = 4*(jj / 16);
        const uint32_t q = ((b0[2 + jj % 16] >> shift) & 0x0F) | (((b1[2 + jj % 16] >> shift) & 0x0F) << 16);
        const half2 d = __halves2half2(*(const half *) b0, *(const half *) b1);
        return __hmul2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1032.0f, 1032.0f)), d);
    } else if constexpr (type == GGML_TYPE_Q4_1) {
        const int shift = 4*(jj / 16);
        const uint32_t q = ((b0[4 + jj % 16] >> shift) & 0x0F) | (((b1[4 + jj % 16] >> shift) & 0x0F) << 16);
        const half2 dm0 = *(const half2 *) b0;
        const half2 dm1 = *(const half2 *) b1;
        return __hfma2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1024.0f, 1024.0f)), __lows2half2(dm0, dm1), __highs2half2(dm0, dm1));
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        const int shift = 4*(jj / 16);
        const int ih = 2 + 2*(jj / 16);
        const uint32_t qh0 = *(const uint16_t *) (b0 + ih) >> (jj % 16);
        const uint32_t qh1 = *(const uint16_t *) (b1 + ih) >> (jj % 16);
        const uint32_t q = ((b0[6 + jj % 16] >> shift) & 0x0F) | ((qh0 & 1) << 4) |
                          (((b1[6 + jj % 16] >> shift) & 0x0F) << 16) | ((qh1 & 1) << 20);
        const half2 d = __halves2half2(*(const half *) b0, *(const half *) b1);
        return __hmul2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1040.0f, 1040.0f)), d);
    } else if constexpr (type == GGML_TYPE_Q5_1) {
        const int shift = 4*(jj / 16);
        const uint32_t qh0 = *(const uint32_t *) (b0 + 4) >> jj;
        const uint32_t qh1 = *(const uint32_t *) (b1 + 4) >> jj;
        const uint32_t q = ((b0[8 + jj % 16] >> shift) & 0x0F) | ((qh0 & 1) << 4) |
                          (((b1[8 + jj % 16] >> shift) & 0x0F) << 16) | ((qh1 & 1) << 20);
        const half2 dm0 = *(const half2 *) b0;
        const half2 dm1 = *(const half2 *) b1;
        return __hfma2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1024.0f, 1024.0f)), __lows2half2(dm0, dm1), __highs2half2(dm0, dm1));
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "bad type");
        const uint32_t q = (b0[2 + jj] ^ 0x80) | ((b1[2 + jj] ^ 0x80) << 16);
        const half2 d = __halves2half2(*(const half *) b0, *(const half *) b1);
        return __hmul2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1152.0f, 1152.0f)), d);
    }
}

// int8 values of dimensions 4*lq...4*lq+3 (b0) and 16+4*lq...16+4*lq+3 (b1) of a block, the B operand of an m16n8k32 mma.
template <ggml_type type>
static __device__ __forceinline__ void fattn_mma_q_K_i8(const uint8_t * b, const int lq, uint32_t & b0, uint32_t & b1) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        const uint16_t * qs = (const uint16_t *) (b + 2);
        b0 = qs[2*lq + 0] | (uint32_t(qs[2*lq + 1]) << 16);
        b1 = qs[2*lq + 8] | (uint32_t(qs[2*lq + 9]) << 16);
    } else {
        constexpr int iqs = type == GGML_TYPE_Q4_0 ? 2 : type == GGML_TYPE_Q4_1 ? 4 : type == GGML_TYPE_Q5_0 ? 6 : 8;
        const uint16_t * qs = (const uint16_t *) (b + iqs);
        const uint32_t x = qs[2*lq + 0] | (uint32_t(qs[2*lq + 1]) << 16);
        b0 = (x >> 0) & 0x0F0F0F0F;
        b1 = (x >> 4) & 0x0F0F0F0F;
        if constexpr (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            const uint16_t * qh16 = (const uint16_t *) (b + (type == GGML_TYPE_Q5_0 ? 2 : 4));
            const uint32_t qh = qh16[0] | (uint32_t(qh16[1]) << 16);
            const uint32_t h0 = qh >> (4*lq);
            const uint32_t h1 = qh >> (4*lq + 16);
            b0 |= ((h0 << 4) & 0x00000010) | ((h0 << 11) & 0x00001000) | ((h0 << 18) & 0x00100000) | ((h0 << 25) & 0x10000000);
            b1 |= ((h1 << 4) & 0x00000010) | ((h1 << 11) & 0x00001000) | ((h1 << 18) & 0x00100000) | ((h1 << 25) & 0x10000000);
        }
        if constexpr (type == GGML_TYPE_Q4_0) {
            b0 = __vsubss4(b0, 0x08080808);
            b1 = __vsubss4(b1, 0x08080808);
        } else if constexpr (type == GGML_TYPE_Q5_0) {
            b0 = __vsubss4(b0, 0x10101010);
            b1 = __vsubss4(b1, 0x10101010);
        }
    }
}

// Scale d and offset m of a block: value = d*q + m, with q as returned by fattn_mma_q_K_i8.
template <ggml_type type>
static __device__ __forceinline__ float2 fattn_mma_q_K_dm(const uint8_t * b) {
    if constexpr (type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1) {
        return __half22float2(*(const half2 *) b);
    } else {
        return make_float2(__half2float(*(const half *) b), 0.0f);
    }
}

static constexpr __host__ __device__ bool fattn_mma_q_has_m(const ggml_type type) {
    return type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1;
}

// int32 -> float for |x| < 2^22, cheaper than I2F.
static __device__ __forceinline__ float fattn_mma_q_i2f(const int x) {
    return __int_as_float(x + 0x4B400000) - 12582912.0f;
}

// ------------------------------------------------------------------------------------------------------------------
// PTX wrappers.

static __device__ __forceinline__ uint32_t fattn_mma_q_h2_as_u32(const half2 x) {
    return *(const uint32_t *) &x;
}

// C (16x8, f32) += A (16x16, f16, row-major) * B (16x8, f16, column-major)
static __device__ __forceinline__ void fattn_mma_q_mma_f32(float * c, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// C (16x8, int32) += A (16x32, int8, row-major) * B (32x8, int8, column-major)
static __device__ __forceinline__ void fattn_mma_q_mma_i8(int * c, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// C (16x8, f16) += A (16x16, f16, row-major) * B (16x8, f16, column-major)
static __device__ __forceinline__ void fattn_mma_q_mma_f16(uint32_t * c, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// A fragment of a 16x16 f16 tile at tile (row-major, stride in halves).
static __device__ __forceinline__ void fattn_mma_q_ldmatrix_A(uint32_t * a, const half * tile, const int stride) {
#ifdef AMPERE_MMA_AVAILABLE
    const half * p = tile + (threadIdx.x % 16)*stride + (threadIdx.x / 16)*8;
    const uint32_t addr = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addr));
#else
    GGML_UNUSED_VARS(a, tile, stride);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// Asynchronous copy of nbytes (4, 8, or 16), zero-filled if !valid.
template <int nbytes>
static __device__ __forceinline__ void fattn_mma_q_cp_async(const uint32_t dst, const void * src, const bool valid) {
#ifdef CP_ASYNC_AVAILABLE
    const int src_size = valid ? nbytes : 0;
    if constexpr (nbytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" : : "r"(dst), "l"(src), "r"(src_size));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" : : "r"(dst), "l"(src), "n"(nbytes), "r"(src_size));
    }
#else
    GGML_UNUSED_VARS(dst, src, valid);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ void fattn_mma_q_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

template <int n>
static __device__ __forceinline__ void fattn_mma_q_cp_async_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

// Copies the raw rows of K/V tile k0 into shared memory.
template <int D, ggml_type type>
static __device__ __forceinline__ void fattn_mma_q_load_KV(
        uint8_t * tile, const char * KV, const int64_t stride, const int k0, const int ne11) {
    constexpr int row_bytes = fattn_mma_q_row_bytes(D, type);
    constexpr int cpy       = fattn_mma_q_cpy_bytes(D, type);
    constexpr int per_row   = row_bytes / cpy;
    const uint32_t tile_s = __cvta_generic_to_shared(tile);
#pragma unroll
    for (int i0 = 0; i0 < FATTN_MMA_Q_NBATCH*per_row; i0 += FATTN_MMA_Q_NTHREADS) {
        const int i = i0 + threadIdx.y*WARP_SIZE + threadIdx.x;
        if (i0 + FATTN_MMA_Q_NTHREADS > FATTN_MMA_Q_NBATCH*per_row && i >= FATTN_MMA_Q_NBATCH*per_row) {
            break;
        }
        const int row = i / per_row;
        const int c   = i % per_row;
        const bool valid = k0 + row < ne11;
        fattn_mma_q_cp_async<cpy>(tile_s + row*row_bytes + c*cpy, KV + (valid ? (k0 + row)*stride : 0) + c*cpy, valid);
    }
}

// Copies the mask values of tile k0 for the query tokens of the tile into shared memory.
static __device__ __forceinline__ void fattn_mma_q_load_mask(
        half * tile_mask, const half * mask, const int stride_mask, const int ntok, const int k0, const int ne11, const bool mask_vec) {
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    if (mask_vec) {
        // rows are 16 byte aligned and ne11 % 8 == 0
        const uint32_t tile_s = __cvta_generic_to_shared(tile_mask);
        if (tid < ntok*(FATTN_MMA_Q_NBATCH/8)) {
            const int t = tid / (FATTN_MMA_Q_NBATCH/8);
            const int c = tid % (FATTN_MMA_Q_NBATCH/8);
            const bool valid = k0 + 8*c < ne11;
            fattn_mma_q_cp_async<16>(tile_s + (t*fattn_mma_q_stride_mask() + 8*c)*sizeof(half),
                mask + int64_t(t)*stride_mask + (valid ? k0 + 8*c : 0), valid);
        }
    } else {
        for (int i = tid; i < ntok*FATTN_MMA_Q_NBATCH; i += FATTN_MMA_Q_NTHREADS) {
            const int t = i / FATTN_MMA_Q_NBATCH;
            const int c = i % FATTN_MMA_Q_NBATCH;
            tile_mask[t*fattn_mma_q_stride_mask() + c] = k0 + c < ne11 ? mask[int64_t(t)*stride_mask + k0 + c] : __float2half(0.0f);
        }
    }
}

// For each tile of ntok_tile query tokens, marks the K/V tiles that at least one of the tokens can see.
// One CUDA block per 32 K/V tiles, 8 threads per K/V tile.
__launch_bounds__(256, 1)
static __global__ void flash_attn_ext_mma_q_visible(
        const half * GGML_CUDA_RESTRICT mask, uint8_t * GGML_CUDA_RESTRICT vis,
        const int ne01, const int ne11, const int64_t stride_mask1, const int64_t stride_mask3,
        const int ntok_tile, const int mask_vec) {
    ggml_cuda_pdl_sync();
    const int ntiles_kv = (ne11 + FATTN_MMA_Q_NBATCH - 1) / FATTN_MMA_Q_NBATCH;
    const int tile = blockIdx.x*(256/8) + threadIdx.x/8;
    const int k    = tile*FATTN_MMA_Q_NBATCH + 8*(threadIdx.x % 8);
    const int tok0 = blockIdx.y*ntok_tile;
    const int ntok = min(ntok_tile, ne01 - tok0);

    const half * mask_t = mask + blockIdx.z*stride_mask3 + tok0*stride_mask1;

    bool visible = false;
    if (mask_vec) {
        // rows are 16 byte aligned and ne11 % 8 == 0
        if (k < ne11) {
            for (int t = 0; t < ntok; ++t) {
                uint4 v;
                ggml_cuda_memcpy_1<16>(&v, mask_t + t*stride_mask1 + k);
                const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    visible |= (w[l] & 0xFFFF) != 0xFC00 || (w[l] >> 16) != 0xFC00;
                }
            }
        }
    } else {
        for (int t = 0; t < ntok; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                if (k + l < ne11) {
                    visible |= __half2float(mask_t[t*stride_mask1 + k + l]) != -INFINITY;
                }
            }
        }
    }
    visible |= __shfl_xor_sync(0xFFFFFFFF, visible, 1);
    visible |= __shfl_xor_sync(0xFFFFFFFF, visible, 2);
    visible |= __shfl_xor_sync(0xFFFFFFFF, visible, 4);
    if (threadIdx.x % 8 == 0 && tile < ntiles_kv) {
        vis[(blockIdx.z*gridDim.y + blockIdx.y)*int64_t(ntiles_kv) + tile] = visible;
    }
    ggml_cuda_pdl_lc();
}

template <int D, ggml_type type, int ncols_tiles>
__launch_bounds__(FATTN_MMA_Q_NTHREADS, 1)
static __global__ void flash_attn_ext_mma_q(
        const char  * GGML_CUDA_RESTRICT Q,
        const char  * GGML_CUDA_RESTRICT K,
        const char  * GGML_CUDA_RESTRICT V,
        const char  * GGML_CUDA_RESTRICT mask,
        const uint8_t * GGML_CUDA_RESTRICT vis,
        const float * GGML_CUDA_RESTRICT sinks,
        float       * GGML_CUDA_RESTRICT dst,
        half2       * GGML_CUDA_RESTRICT dst_parts,
        float2      * GGML_CUDA_RESTRICT dst_meta,
        const float scale,
        const int ne01, const int ne02, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int ne11, const int ne12, const int64_t nb11, const int64_t nb12, const int64_t nb13,
                                        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int ne33, const int64_t nb31, const int64_t nb33,
        const int gqa_ratio, const int ntok_tile, const int chunk_tiles, const int nchunks, const int mask_vec) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    constexpr int nstages   = fattn_mma_q_nstages(D, type, ncols_tiles);
    constexpr int row_bytes = fattn_mma_q_row_bytes(D, type);
    constexpr int nrows     = 16*ncols_tiles;     // (token, head) rows of the tile
    constexpr int DW        = D/FATTN_MMA_Q_NWARPS; // VKQ columns per warp
    constexpr int NT        = DW/8;                 // VKQ n tiles per warp
    constexpr int stride_Q  = fattn_mma_q_stride_Q(D);
    constexpr int stride_P  = fattn_mma_q_stride_P();
    constexpr int stride_M  = fattn_mma_q_stride_mask();
    static_assert(D % 64 == 0, "bad D");

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp*WARP_SIZE + lane;
    const int lq   = lane % 4; // thread in quad
    const int lr   = lane / 4; // quad in warp

    // the K/V heads of a chunk are adjacent so that they share the L2 sectors of K/V rows and the mask
    const int chunk    = blockIdx.x / ne12;
    const int head_kv  = blockIdx.x % ne12;
    const int tok0     = blockIdx.y*ntok_tile;
    const int sequence = blockIdx.z;

    const int ntok       = min(ntok_tile, ne01 - tok0);
    const int rows_valid = ntok*gqa_ratio;

    const int ntiles_kv = (ne11 + FATTN_MMA_Q_NBATCH - 1) / FATTN_MMA_Q_NBATCH;
    const int kt_begin  = chunk*chunk_tiles;
    const int kt_num    = min(chunk_tiles, ntiles_kv - kt_begin);

    const char * K_h = K + sequence*nb13 + head_kv*nb12;
    const char * V_h = V + sequence*nb23 + head_kv*nb22;
    const half * mask_h = mask ? (const half *) (mask + (sequence % ne33)*nb33 + tok0*nb31) : nullptr;
    const int stride_mask = nb31 / sizeof(half);

    extern __shared__ __align__(16) uint8_t fattn_mma_q_smem[];
    int8_t  * tile_Q    = (int8_t *) fattn_mma_q_smem;
    float2  * tile_Qds  = (float2 *) (tile_Q + nrows*stride_Q);
    half    * tile_P    = (half *) (tile_Qds + nrows*(D/QK8_1));
    float   * red       = (float *) (tile_P + nrows*stride_P);
    int     * tile_list = (int *) (red + FATTN_MMA_Q_NWARPS*nrows);
    int     * tile_num  = tile_list + FATTN_MMA_Q_MAX_CHUNK_TILES;
    uint8_t * stages    = fattn_mma_q_smem + fattn_mma_q_smem_fixed(D, ncols_tiles);
    // stage s: raw K, raw V, mask
    auto stage_K    = [&](const int s) { return stages + s*fattn_mma_q_smem_stage(D, type); };
    auto stage_V    = [&](const int s) { return stage_K(s) + FATTN_MMA_Q_NBATCH*row_bytes; };
    auto stage_mask = [&](const int s) { return (half *) (stage_V(s) + FATTN_MMA_Q_NBATCH*row_bytes); };

    ggml_cuda_pdl_sync();

    // List the K/V tiles of the chunk that are visible to at least one query of the tile:
    const uint8_t * vis_c = vis ? vis + ((sequence % ne33)*gridDim.y + blockIdx.y)*int64_t(ntiles_kv) + kt_begin : nullptr;
    if (warp == 0) {
        int n = 0;
        for (int it0 = 0; it0 < kt_num; it0 += WARP_SIZE) {
            const int it = it0 + lane;
            const bool visible = it < kt_num && (!vis_c || vis_c[it]);
            const uint32_t ballot = __ballot_sync(0xFFFFFFFF, visible);
            if (visible) {
                tile_list[n + __popc(ballot & ((1u << lane) - 1))] = kt_begin + it;
            }
            n += __popc(ballot);
        }
        if (lane == 0) {
            *tile_num = n;
        }
    }
    __syncthreads();
    const int nvis = *tile_num;

    if (nvis == 0 && nchunks > 1) {
        // nothing visible, the merge skips this chunk
        for (int r = tid; r < rows_valid; r += FATTN_MMA_Q_NTHREADS) {
            const int t    = tok0 + r / gqa_ratio;
            const int head = head_kv*gqa_ratio + r % gqa_ratio;
            dst_meta[((int64_t(sequence)*ne01 + t)*ne02 + head)*nchunks + chunk] = make_float2(-INFINITY, 0.0f);
        }
        return;
    }

    // Start loading the first K/V tiles, the load of Q overlaps with them:
#pragma unroll
    for (int s = 0; s < nstages; ++s) {
        if (s < nvis) {
            const int k0 = tile_list[s]*FATTN_MMA_Q_NBATCH;
            fattn_mma_q_load_KV<D, type>(stage_K(s), K_h, nb11, k0, ne11);
            if (mask_h) {
                fattn_mma_q_load_mask(stage_mask(s), mask_h, stride_mask, ntok, k0, ne11, mask_vec);
            }
        }
        fattn_mma_q_cp_async_commit();
        if (s < nvis) {
            fattn_mma_q_load_KV<D, type>(stage_V(s), V_h, nb21, tile_list[s]*FATTN_MMA_Q_NBATCH, ne11);
        }
        fattn_mma_q_cp_async_commit();
    }

    // Q is quantized to int8 per 32 values, each thread does 8 values, 4 threads a block:
    static_assert(D/8 <= WARP_SIZE, "bad D");
    for (int r = warp; r < nrows; r += FATTN_MMA_Q_NWARPS) {
        if (lane >= D/8) {
            continue;
        }
        float q[8] = {0.0f};
        if (r < rows_valid) {
            const int t = tok0 + r / gqa_ratio;
            const int h = head_kv*gqa_ratio + r % gqa_ratio;
            const float4 * Q_r = (const float4 *) (Q + sequence*nb03 + t*nb01 + h*nb02) + 2*lane;
            const float4 q0 = Q_r[0];
            const float4 q1 = Q_r[1];
            q[0] = q0.x; q[1] = q0.y; q[2] = q0.z; q[3] = q0.w;
            q[4] = q1.x; q[5] = q1.y; q[6] = q1.z; q[7] = q1.w;
        }
        float amax = 0.0f;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            q[l] *= scale;
            amax = fmaxf(amax, fabsf(q[l]));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF >> (WARP_SIZE - D/8), amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF >> (WARP_SIZE - D/8), amax, 2));
        const float d  = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        int sum = 0;
        uint32_t qi[2];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 4) {
            const int x0 = __float2int_rn(q[l0 + 0]*id);
            const int x1 = __float2int_rn(q[l0 + 1]*id);
            const int x2 = __float2int_rn(q[l0 + 2]*id);
            const int x3 = __float2int_rn(q[l0 + 3]*id);
            sum += x0 + x1 + x2 + x3;
            qi[l0/4] = (x0 & 0xFF) | ((x1 & 0xFF) << 8) | ((x2 & 0xFF) << 16) | (uint32_t(x3 & 0xFF) << 24);
        }
        sum += __shfl_xor_sync(0xFFFFFFFF >> (WARP_SIZE - D/8), sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFF >> (WARP_SIZE - D/8), sum, 2);
        *(uint2 *) (tile_Q + r*stride_Q + 8*lane) = make_uint2(qi[0], qi[1]);
        if (lane % 4 == 0) {
            tile_Qds[r*(D/QK8_1) + lane/4] = make_float2(d, d*sum);
        }
    }

    // The token of each row of this thread, used for the mask:
    int row_tok[ncols_tiles][2];
#pragma unroll
    for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            row_tok[m][h] = min((16*m + 8*h + lr) / gqa_ratio, ntok - 1);
        }
    }

    float VKQ[ncols_tiles][NT][4];
    float KQ_max[ncols_tiles][2];
    float KQ_sum[ncols_tiles][2];
#pragma unroll
    for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                VKQ[m][j][l] = 0.0f;
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            KQ_max[m][h] = -INFINITY;
            KQ_sum[m][h] = 0.0f;
        }
    }

    for (int it = 0; it < nvis; ++it) {
        const int s  = nstages == 1 ? 0 : it % nstages;
        const int k0 = tile_list[it]*FATTN_MMA_Q_NBATCH;

        fattn_mma_q_cp_async_wait<2*nstages - 1>();
        __syncthreads(); // K and mask of this tile are in shared memory

        // KQ, each warp does 8 K/V cells:
        float KQ[ncols_tiles][4];
#pragma unroll
        for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                KQ[m][l] = 0.0f;
            }
        }
        {
            constexpr int bs = fattn_mma_q_block_bytes(type);
            const uint8_t * K_row = stage_K(s) + (8*warp + lr    )*row_bytes; // B operand
            const uint8_t * K_c0  = stage_K(s) + (8*warp + 2*lq  )*row_bytes; // scales of the columns of C
            const uint8_t * K_c1  = K_c0 + row_bytes;
#pragma unroll
            for (int ib = 0; ib < D/QK8_0; ++ib) {
                uint32_t b0;
                uint32_t b1;
                fattn_mma_q_K_i8<type>(K_row + ib*bs, lq, b0, b1);
                const float2 dm0 = fattn_mma_q_K_dm<type>(K_c0 + ib*bs);
                const float2 dm1 = fattn_mma_q_K_dm<type>(K_c1 + ib*bs);
#pragma unroll
                for (int m = 0; m < ncols_tiles; ++m) {
                    uint32_t a[4];
                    fattn_mma_q_ldmatrix_A(a, (const half *) (tile_Q + 16*m*stride_Q + ib*QK8_0), stride_Q/2);
                    int acc[4] = {0, 0, 0, 0};
                    fattn_mma_q_mma_i8(acc, a, b0, b1);
                    const float2 ds0 = tile_Qds[(16*m + lr    )*(D/QK8_1) + ib];
                    const float2 ds1 = tile_Qds[(16*m + lr + 8)*(D/QK8_1) + ib];
                    KQ[m][0] += dm0.x*ds0.x * fattn_mma_q_i2f(acc[0]);
                    KQ[m][1] += dm1.x*ds0.x * fattn_mma_q_i2f(acc[1]);
                    KQ[m][2] += dm0.x*ds1.x * fattn_mma_q_i2f(acc[2]);
                    KQ[m][3] += dm1.x*ds1.x * fattn_mma_q_i2f(acc[3]);
                    if constexpr (fattn_mma_q_has_m(type)) {
                        KQ[m][0] += dm0.y*ds0.y;
                        KQ[m][1] += dm1.y*ds0.y;
                        KQ[m][2] += dm0.y*ds1.y;
                        KQ[m][3] += dm1.y*ds1.y;
                    }
                }
            }
        }

        // Mask, out-of-bounds cells, and the tile maximum of each row:
        const int  c0     = 8*warp + 2*lq;
        const half * mask_s = stage_mask(s);
#pragma unroll
        for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float & x = KQ[m][2*h + e];
                    if (mask_h) {
                        x += __half2float(mask_s[row_tok[m][h]*stride_M + c0 + e]);
                    }
                    if (k0 + c0 + e >= ne11) {
                        x = -INFINITY;
                    }
                }
                float mx = fmaxf(KQ[m][2*h + 0], KQ[m][2*h + 1]);
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 2));
                if (lq == 0) {
                    red[warp*nrows + 16*m + 8*h + lr] = mx;
                }
            }
        }

        __syncthreads(); // tile maxima are in shared memory, K and mask of this stage are no longer needed

        if (it + nstages < nvis) {
            const int k0_next = tile_list[it + nstages]*FATTN_MMA_Q_NBATCH;
            fattn_mma_q_load_KV<D, type>(stage_K(s), K_h, nb11, k0_next, ne11);
            if (mask_h) {
                fattn_mma_q_load_mask(stage_mask(s), mask_h, stride_mask, ntok, k0_next, ne11, mask_vec);
            }
        }
        fattn_mma_q_cp_async_commit();

        // Softmax, P is written to shared memory as f16:
#pragma unroll
        for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = 16*m + 8*h + lr;
                float mx = KQ_max[m][h];
#pragma unroll
                for (int w = 0; w < FATTN_MMA_Q_NWARPS; ++w) {
                    mx = fmaxf(mx, red[w*nrows + r]);
                }
                const float mx_use = mx == -INFINITY ? 0.0f : mx;
                const float scale_old = expf(KQ_max[m][h] - mx_use);
                KQ_max[m][h] = mx;

                const float p0 = expf(KQ[m][2*h + 0] - mx_use);
                const float p1 = expf(KQ[m][2*h + 1] - mx_use);
                KQ_sum[m][h] = KQ_sum[m][h]*scale_old + (p0 + p1);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    VKQ[m][j][2*h + 0] *= scale_old;
                    VKQ[m][j][2*h + 1] *= scale_old;
                }
                *(half2 *) (tile_P + r*stride_P + c0) = make_half2(p0, p1);
            }
        }

        fattn_mma_q_cp_async_wait<2*nstages - 1>();
        __syncthreads(); // P and V of this tile are in shared memory

        // VKQ, each warp does DW columns:
        {
            uint32_t VKQ_h[ncols_tiles][NT][2];
#pragma unroll
            for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    VKQ_h[m][j][0] = 0;
                    VKQ_h[m][j][1] = 0;
                }
            }
            const uint8_t * V_s = stage_V(s);
#pragma unroll
            for (int k = 0; k < FATTN_MMA_Q_NBATCH; k += 16) {
                const uint8_t * V_r0 = V_s + (k + 2*lq    )*row_bytes;
                const uint8_t * V_r1 = V_s + (k + 2*lq + 1)*row_bytes;
                const uint8_t * V_r8 = V_s + (k + 2*lq + 8)*row_bytes;
                const uint8_t * V_r9 = V_s + (k + 2*lq + 9)*row_bytes;
                uint32_t b0[NT];
                uint32_t b1[NT];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int dim = warp*DW + 8*j + lr;
                    b0[j] = fattn_mma_q_h2_as_u32(fattn_mma_q_dq1x2<type>(V_r0, V_r1, dim));
                    b1[j] = fattn_mma_q_h2_as_u32(fattn_mma_q_dq1x2<type>(V_r8, V_r9, dim));
                }
#pragma unroll
                for (int m = 0; m < ncols_tiles; ++m) {
                    uint32_t a[4];
                    fattn_mma_q_ldmatrix_A(a, tile_P + 16*m*stride_P + k, stride_P);
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        fattn_mma_q_mma_f16(VKQ_h[m][j], a, b0[j], b1[j]);
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const float2 lo = __half22float2(*(const half2 *) &VKQ_h[m][j][0]);
                    const float2 hi = __half22float2(*(const half2 *) &VKQ_h[m][j][1]);
                    VKQ[m][j][0] += lo.x;
                    VKQ[m][j][1] += lo.y;
                    VKQ[m][j][2] += hi.x;
                    VKQ[m][j][3] += hi.y;
                }
            }
        }

        __syncthreads(); // V and P of this stage are no longer needed

        if (it + nstages < nvis) {
            fattn_mma_q_load_KV<D, type>(stage_V(s), V_h, nb21, tile_list[it + nstages]*FATTN_MMA_Q_NBATCH, ne11);
        }
        fattn_mma_q_cp_async_commit();
    }
    fattn_mma_q_cp_async_wait<0>();

    // Sum up the partial KQ row sums of the quads and warps:
#pragma unroll
    for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float sum = KQ_sum[m][h];
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
            if (lq == 0) {
                red[warp*nrows + 16*m + 8*h + lr] = sum;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = 16*m + 8*h + lr;
            if (r >= rows_valid) {
                continue;
            }
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < FATTN_MMA_Q_NWARPS; ++w) {
                sum += red[w*nrows + r];
            }
            float mx = KQ_max[m][h];

            const int t    = tok0 + r / gqa_ratio;
            const int head = head_kv*gqa_ratio + r % gqa_ratio;
            const int64_t row_dst = (int64_t(sequence)*ne01 + t)*ne02 + head;

            if (nchunks == 1) {
                float scale_VKQ = 1.0f;
                if (sinks) {
                    const float sink   = sinks[head];
                    const float mx_new = fmaxf(mx, sink);
                    scale_VKQ = mx == -INFINITY ? 0.0f : expf(mx - mx_new);
                    sum = sum*scale_VKQ + expf(sink - mx_new);
                }
                const float inv = sum > 0.0f ? scale_VKQ/sum : 0.0f;
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int dim = warp*DW + 8*j + 2*lq;
                    *(float2 *) (dst + row_dst*D + dim) = make_float2(inv*VKQ[m][j][2*h + 0], inv*VKQ[m][j][2*h + 1]);
                }
            } else {
                const int64_t row_part = row_dst*nchunks + chunk;
                const float inv = sum > 0.0f ? 1.0f/sum : 0.0f;
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int dim = warp*DW + 8*j + 2*lq;
                    dst_parts[row_part*(D/2) + dim/2] = make_half2(inv*VKQ[m][j][2*h + 0], inv*VKQ[m][j][2*h + 1]);
                }
                if (warp == 0 && lq == 0) {
                    dst_meta[row_part] = make_float2(mx, sum);
                }
            }
        }
    }
    ggml_cuda_pdl_lc();
#else
    GGML_UNUSED_VARS(Q, K, V, mask, vis, sinks, dst, dst_parts, dst_meta, scale,
        ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb13, nb21, nb22, nb23, ne33, nb31, nb33,
        gqa_ratio, ntok_tile, chunk_tiles, nchunks, mask_vec);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// Merges the per-chunk results of flash_attn_ext_mma_q, one CUDA block per dst row (token, head).
template <int D>
__launch_bounds__(D/2, 1)
static __global__ void flash_attn_ext_mma_q_combine(
        const half2  * GGML_CUDA_RESTRICT parts,
        const float2 * GGML_CUDA_RESTRICT meta,
        const float  * GGML_CUDA_RESTRICT sinks,
        float        * GGML_CUDA_RESTRICT dst,
        const int nchunks, const int ne02) {
    ggml_cuda_pdl_sync();
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    parts += row*nchunks*(D/2);
    meta  += row*nchunks;

    float mx = sinks ? sinks[row % ne02] : -INFINITY;
    for (int c = 0; c < nchunks; ++c) {
        const float2 mc = meta[c];
        if (mc.y > 0.0f) {
            mx = fmaxf(mx, mc.x);
        }
    }

    float2 num = make_float2(0.0f, 0.0f);
    float  den = sinks ? expf(sinks[row % ne02] - mx) : 0.0f;
    for (int c = 0; c < nchunks; ++c) {
        const float2 mc = meta[c];
        if (mc.y > 0.0f) {
            const float w = expf(mc.x - mx)*mc.y;
            const float2 p = __half22float2(parts[c*(D/2) + tid]);
            num.x += w*p.x;
            num.y += w*p.y;
            den   += w;
        }
    }
    const float inv = den > 0.0f ? 1.0f/den : 0.0f;
    ((float2 *) (dst + row*D))[tid] = make_float2(inv*num.x, inv*num.y);
    ggml_cuda_pdl_lc();
}

template <int D, ggml_type type, int ncols_tiles>
static void ggml_cuda_flash_attn_ext_mma_q_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int ntok_tile) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    cudaStream_t stream = ctx.stream();

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    const int ne11      = K->ne[1];

    constexpr size_t nbytes_shared = fattn_mma_q_smem_total(D, type, ncols_tiles);
    const auto kernel = flash_attn_ext_mma_q<D, type, ncols_tiles>;
    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shared_memory_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared));
        shared_memory_limit_raised[id] = true;
    }

    const bool mask_vec = mask && mask->nb[1] % 16 == 0 && mask->nb[3] % 16 == 0 && ((uintptr_t) mask->data) % 16 == 0 && ne11 % 8 == 0;

    const int ntiles_tok = (Q->ne[1] + ntok_tile - 1) / ntok_tile;
    const int ntiles_kv  = (ne11 + FATTN_MMA_Q_NBATCH - 1) / FATTN_MMA_Q_NBATCH;

    ggml_cuda_pool_alloc<uint8_t> vis(ctx.pool());
    if (mask) {
        vis.alloc(size_t(mask->ne[3])*ntiles_tok*ntiles_kv);
        ggml_cuda_kernel_launch(flash_attn_ext_mma_q_visible,
            ggml_cuda_kernel_launch_params(dim3((ntiles_kv + 31)/32, ntiles_tok, mask->ne[3]), dim3(256, 1, 1), 0, stream),
            (const half *) mask->data, vis.ptr, int(Q->ne[1]), ne11, int64_t(mask->nb[1]/sizeof(half)), int64_t(mask->nb[3]/sizeof(half)),
            ntok_tile, int(mask_vec));
    }

    // Split the K/V cells into chunks: enough CUDA blocks to fill the GPU, but not so many that writing and merging the
    // per-chunk results costs more than a few percent of the K/V data. With several token tiles the tokens usually
    // belong to different sequences of a unified KV cache and each tile sees only a fraction of the cells.
    const int   nunits   = ntiles_tok * K->ne[2] * Q->ne[3];
    const float frac_vis = std::min(1.0f, float(ntok_tile) / Q->ne[1]);
    const int nblocks_target = 2*nsm;
    int chunk_tiles = std::ceil(frac_vis*ntiles_kv*nunits / nblocks_target);
    chunk_tiles = std::max(chunk_tiles, 16);
    chunk_tiles = std::min(chunk_tiles, FATTN_MMA_Q_MAX_CHUNK_TILES);
    chunk_tiles = std::min(chunk_tiles, ntiles_kv);
    const int nchunks = (ntiles_kv + chunk_tiles - 1) / chunk_tiles;

    const int64_t nrows_dst = Q->ne[1]*Q->ne[2]*Q->ne[3];
    ggml_cuda_pool_alloc<half2>  parts(ctx.pool());
    ggml_cuda_pool_alloc<float2> meta(ctx.pool());
    if (nchunks > 1) {
        parts.alloc(nrows_dst*nchunks*(D/2));
        meta.alloc(nrows_dst*nchunks);
    }

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const dim3 block_dims(WARP_SIZE, FATTN_MMA_Q_NWARPS, 1);
    const dim3 block_nums(nchunks*K->ne[2], ntiles_tok, Q->ne[3]);
    ggml_cuda_kernel_launch(kernel, ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream),
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        mask ? (const char *) mask->data : nullptr, vis.ptr, sinks ? (const float *) sinks->data : nullptr,
        (float *) dst->data, parts.ptr, meta.ptr, scale,
        int(Q->ne[1]), int(Q->ne[2]), int64_t(Q->nb[1]), int64_t(Q->nb[2]), int64_t(Q->nb[3]),
        ne11, int(K->ne[2]), int64_t(K->nb[1]), int64_t(K->nb[2]), int64_t(K->nb[3]),
                             int64_t(V->nb[1]), int64_t(V->nb[2]), int64_t(V->nb[3]),
        mask ? int(mask->ne[3]) : 1, mask ? int64_t(mask->nb[1]) : 0, mask ? int64_t(mask->nb[3]) : 0,
        gqa_ratio, ntok_tile, chunk_tiles, nchunks, int(mask_vec));
    CUDA_CHECK(cudaGetLastError());

    if (nchunks > 1) {
        ggml_cuda_kernel_launch(flash_attn_ext_mma_q_combine<D>,
            ggml_cuda_kernel_launch_params(dim3(nrows_dst, 1, 1), dim3(D/2, 1, 1), 0, stream),
            (const half2 *) parts.ptr, (const float2 *) meta.ptr, sinks ? (const float *) sinks->data : nullptr,
            (float *) dst->data, nchunks, int(Q->ne[2]));
        CUDA_CHECK(cudaGetLastError());
    }
}

template <int D, ggml_type type>
void ggml_cuda_flash_attn_ext_mma_q_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    const int ntok_tile = std::min<int64_t>({Q->ne[1], FATTN_MMA_Q_MAX_TOKENS, 64/gqa_ratio});
    const int nrows     = ntok_tile*gqa_ratio;

    if (nrows <= 16) {
        ggml_cuda_flash_attn_ext_mma_q_launch<D, type, 1>(ctx, dst, ntok_tile);
    } else if (nrows <= 32) {
        ggml_cuda_flash_attn_ext_mma_q_launch<D, type, 2>(ctx, dst, ntok_tile);
    } else if (nrows <= 48) {
        ggml_cuda_flash_attn_ext_mma_q_launch<D, type, 3>(ctx, dst, ntok_tile);
    } else {
        ggml_cuda_flash_attn_ext_mma_q_launch<D, type, 4>(ctx, dst, ntok_tile);
    }
}

#define DECL_FATTN_MMA_Q_CASE(D, type)                                         \
    template void ggml_cuda_flash_attn_ext_mma_q_case                          \
    <D, type>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)              \

#define EXTERN_DECL_FATTN_MMA_Q_CASES(type)        \
    extern DECL_FATTN_MMA_Q_CASE( 64, type);       \
    extern DECL_FATTN_MMA_Q_CASE(128, type);       \
    extern DECL_FATTN_MMA_Q_CASE(256, type);       \

EXTERN_DECL_FATTN_MMA_Q_CASES(GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_MMA_Q_CASES(GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_MMA_Q_CASES(GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_MMA_Q_CASES(GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_MMA_Q_CASES(GGML_TYPE_Q8_0)

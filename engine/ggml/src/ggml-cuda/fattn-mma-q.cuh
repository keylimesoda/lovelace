#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// FlashAttention for quantized K/V and few queries per K/V head: decode and speculative verify batches.
//
// Each CUDA block handles one tile of up to FATTN_MMA_Q_MAX_TOKENS query tokens times all Q heads of one K/V head,
// flattened into ncols_tiles*16 rows of (token, head) pairs, over one chunk of the K/V cells (split-K).
// The quantized K/V rows are copied into shared memory as they are (cp.async, multi-stage) and dequantized from there
// directly into mma fragments, so the K/V cache is read from VRAM exactly once per step and no f16 copy is made.
// KQ is accumulated in f32, VKQ per K/V tile in f16 and across tiles in f32.
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
    return D + 8; // halves
}
static constexpr __host__ __device__ int fattn_mma_q_stride_P() {
    return FATTN_MMA_Q_NBATCH + 8; // halves
}
static constexpr __host__ __device__ int fattn_mma_q_stride_mask() {
    return FATTN_MMA_Q_NBATCH + 8; // halves
}
static constexpr __host__ __device__ int fattn_mma_q_smem_fixed(const int D, const int ncols_tiles) {
    return 16*ncols_tiles*fattn_mma_q_stride_Q(D)*2      // Q
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
// Dequantization from shared memory into f16 mma operands.

// Two nibbles/bytes in the low byte of each 16 bit half of q -> half2 of (1024 + value), exact.
static __device__ __forceinline__ half2 fattn_mma_q_u8x2_to_h2(const uint32_t q) {
    const uint32_t h = q | 0x64006400;
    return *(const half2 *) &h;
}

// Dimensions j and j+1 (j even) of a K/V row.
template <ggml_type type>
static __device__ __forceinline__ half2 fattn_mma_q_dq2(const uint8_t * row, const int j) {
    const int ib = j / QK8_0;
    const int jj = j % QK8_0;
    const uint8_t * b = row + ib*fattn_mma_q_block_bytes(type);

    if constexpr (type == GGML_TYPE_Q4_0) {
        uint32_t q = *(const uint16_t *) (b + 2 + jj % 16);
        q = (q >> (4*(jj / 16))) & 0x0F0F;
        q = (q & 0x0F) | ((q & 0x0F00) << 8);
        const half2 d = __half2half2(*(const half *) b);
        return __hmul2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1032.0f, 1032.0f)), d);
    } else if constexpr (type == GGML_TYPE_Q4_1) {
        uint32_t q = *(const uint16_t *) (b + 4 + jj % 16);
        q = (q >> (4*(jj / 16))) & 0x0F0F;
        q = (q & 0x0F) | ((q & 0x0F00) << 8);
        const half2 dm = *(const half2 *) b;
        return __hfma2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1024.0f, 1024.0f)), __low2half2(dm), __high2half2(dm));
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        const uint32_t qh = *(const uint16_t *) (b + 2) | (uint32_t(*(const uint16_t *) (b + 4)) << 16);
        uint32_t q = *(const uint16_t *) (b + 6 + jj % 16);
        q = (q >> (4*(jj / 16))) & 0x0F0F;
        q = (q & 0x0F) | ((q & 0x0F00) << 8);
        q |= ((qh >> jj) & 1) << 4 | ((qh >> (jj + 1)) & 1) << 20;
        const half2 d = __half2half2(*(const half *) b);
        return __hmul2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1040.0f, 1040.0f)), d);
    } else if constexpr (type == GGML_TYPE_Q5_1) {
        const uint32_t qh = *(const uint32_t *) (b + 4);
        uint32_t q = *(const uint16_t *) (b + 8 + jj % 16);
        q = (q >> (4*(jj / 16))) & 0x0F0F;
        q = (q & 0x0F) | ((q & 0x0F00) << 8);
        q |= ((qh >> jj) & 1) << 4 | ((qh >> (jj + 1)) & 1) << 20;
        const half2 dm = *(const half2 *) b;
        return __hfma2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1024.0f, 1024.0f)), __low2half2(dm), __high2half2(dm));
    } else {
        static_assert(type == GGML_TYPE_Q8_0, "bad type");
        uint32_t q = *(const uint16_t *) (b + 2 + jj) ^ 0x8080;
        q = (q & 0xFF) | ((q & 0xFF00) << 8);
        const half2 d = __half2half2(*(const half *) b);
        return __hmul2(__hsub2(fattn_mma_q_u8x2_to_h2(q), make_half2(1152.0f, 1152.0f)), d);
    }
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

template <int D, ggml_type type, int ncols_tiles>
__launch_bounds__(FATTN_MMA_Q_NTHREADS, 1)
static __global__ void flash_attn_ext_mma_q(
        const char  * GGML_CUDA_RESTRICT Q,
        const char  * GGML_CUDA_RESTRICT K,
        const char  * GGML_CUDA_RESTRICT V,
        const char  * GGML_CUDA_RESTRICT mask,
        const float * GGML_CUDA_RESTRICT sinks,
        float       * GGML_CUDA_RESTRICT dst,
        half2       * GGML_CUDA_RESTRICT dst_parts,
        float2      * GGML_CUDA_RESTRICT dst_meta,
        const float scale,
        const int ne01, const int ne02, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int ne11, const int ne12, const int64_t nb11, const int64_t nb12, const int64_t nb13,
                                        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int ne33, const int64_t nb31, const int64_t nb33,
        const int gqa_ratio, const int ntok_tile, const int chunk_tiles, const int mask_vec) {
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

    const int chunk    = blockIdx.x;
    const int tok0     = blockIdx.y*ntok_tile;
    const int head_kv  = blockIdx.z % ne12;
    const int sequence = blockIdx.z / ne12;

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
    half    * tile_Q    = (half *) fattn_mma_q_smem;
    half    * tile_P    = tile_Q + nrows*stride_Q;
    float   * red       = (float *) (tile_P + nrows*stride_P);
    int     * tile_list = (int *) (red + FATTN_MMA_Q_NWARPS*nrows);
    int     * tile_num  = tile_list + FATTN_MMA_Q_MAX_CHUNK_TILES;
    uint8_t * stages    = fattn_mma_q_smem + fattn_mma_q_smem_fixed(D, ncols_tiles);
    // stage s: raw K, raw V, mask
    auto stage_K    = [&](const int s) { return stages + s*fattn_mma_q_smem_stage(D, type); };
    auto stage_V    = [&](const int s) { return stage_K(s) + FATTN_MMA_Q_NBATCH*row_bytes; };
    auto stage_mask = [&](const int s) { return (half *) (stage_V(s) + FATTN_MMA_Q_NBATCH*row_bytes); };

    ggml_cuda_pdl_sync();

    // Find the K/V tiles of the chunk that are visible to at least one query of the tile:
    if (mask_h) {
        for (int it = warp; it < kt_num; it += FATTN_MMA_Q_NWARPS) {
            const int k0 = (kt_begin + it)*FATTN_MMA_Q_NBATCH;
            bool visible = false;
            for (int i = lane; i < ntok*FATTN_MMA_Q_NBATCH; i += WARP_SIZE) {
                const int t = i / FATTN_MMA_Q_NBATCH;
                const int c = i % FATTN_MMA_Q_NBATCH;
                if (k0 + c < ne11) {
                    visible |= __half2float(mask_h[int64_t(t)*stride_mask + k0 + c]) != -INFINITY;
                }
            }
            visible = __any_sync(0xFFFFFFFF, visible);
            if (lane == 0) {
                tile_list[it] = visible;
            }
        }
        __syncthreads();
        if (warp == 0) {
            int n = 0;
            for (int it0 = 0; it0 < kt_num; it0 += WARP_SIZE) {
                const int it = it0 + lane;
                const bool visible = it < kt_num && tile_list[it];
                const uint32_t ballot = __ballot_sync(0xFFFFFFFF, visible);
                __syncwarp();
                if (visible) {
                    tile_list[n + __popc(ballot & ((1u << lane) - 1))] = kt_begin + it;
                }
                n += __popc(ballot);
                __syncwarp();
            }
            if (lane == 0) {
                *tile_num = n;
            }
        }
    } else {
        for (int it = tid; it < kt_num; it += FATTN_MMA_Q_NTHREADS) {
            tile_list[it] = kt_begin + it;
        }
        if (tid == 0) {
            *tile_num = kt_num;
        }
    }
    __syncthreads();
    const int nvis = *tile_num;

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

    for (int i = tid; i < nrows*(D/2); i += FATTN_MMA_Q_NTHREADS) {
        const int r  = i / (D/2);
        const int k2 = i % (D/2);
        half2 q = make_half2(0.0f, 0.0f);
        if (r < rows_valid) {
            const int t = tok0 + r / gqa_ratio;
            const int h = head_kv*gqa_ratio + r % gqa_ratio;
            const float2 qf = ((const float2 *) (Q + sequence*nb03 + t*nb01 + h*nb02))[k2];
            q = make_half2(scale*qf.x, scale*qf.y);
        }
        ((half2 *) (tile_Q + r*stride_Q))[k2] = q;
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
            const uint8_t * K_row = stage_K(s) + (8*warp + lr)*row_bytes;
#pragma unroll
            for (int k = 0; k < D; k += 16) {
                const uint32_t b0 = fattn_mma_q_h2_as_u32(fattn_mma_q_dq2<type>(K_row, k + 2*lq));
                const uint32_t b1 = fattn_mma_q_h2_as_u32(fattn_mma_q_dq2<type>(K_row, k + 2*lq + 8));
#pragma unroll
                for (int m = 0; m < ncols_tiles; ++m) {
                    uint32_t a[4];
                    fattn_mma_q_ldmatrix_A(a, tile_Q + 16*m*stride_Q + k, stride_Q);
                    fattn_mma_q_mma_f32(KQ[m], a, b0, b1);
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

            if (gridDim.x == 1) {
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
                const int64_t row_part = row_dst*gridDim.x + chunk;
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
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, dst, dst_parts, dst_meta, scale,
        ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb13, nb21, nb22, nb23, ne33, nb31, nb33,
        gqa_ratio, ntok_tile, chunk_tiles, mask_vec);
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

    // Split the K/V cells into chunks: enough CUDA blocks to fill the GPU, but not so many that writing and merging the
    // per-chunk results costs more than a few percent of the K/V data.
    const int ntiles_tok = (Q->ne[1] + ntok_tile - 1) / ntok_tile;
    const int ntiles_kv  = (ne11 + FATTN_MMA_Q_NBATCH - 1) / FATTN_MMA_Q_NBATCH;
    const int nunits     = ntiles_tok * K->ne[2] * Q->ne[3];
    const int nblocks_target = 2*nsm;
    int chunk_tiles = (int64_t(ntiles_kv)*nunits + nblocks_target - 1) / nblocks_target;
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

    const bool mask_vec = mask && mask->nb[1] % 16 == 0 && mask->nb[3] % 16 == 0 && ((uintptr_t) mask->data) % 16 == 0 && ne11 % 8 == 0;

    const dim3 block_dims(WARP_SIZE, FATTN_MMA_Q_NWARPS, 1);
    const dim3 block_nums(nchunks, ntiles_tok, K->ne[2]*Q->ne[3]);
    ggml_cuda_kernel_launch(kernel, ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream),
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        mask ? (const char *) mask->data : nullptr, sinks ? (const float *) sinks->data : nullptr,
        (float *) dst->data, parts.ptr, meta.ptr, scale,
        int(Q->ne[1]), int(Q->ne[2]), int64_t(Q->nb[1]), int64_t(Q->nb[2]), int64_t(Q->nb[3]),
        ne11, int(K->ne[2]), int64_t(K->nb[1]), int64_t(K->nb[2]), int64_t(K->nb[3]),
                             int64_t(V->nb[1]), int64_t(V->nb[2]), int64_t(V->nb[3]),
        mask ? int(mask->ne[3]) : 1, mask ? int64_t(mask->nb[1]) : 0, mask ? int64_t(mask->nb[3]) : 0,
        gqa_ratio, ntok_tile, chunk_tiles, int(mask_vec));
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

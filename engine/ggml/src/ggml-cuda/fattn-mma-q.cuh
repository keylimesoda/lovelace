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
// V is dequantized into an f16 tile for VKQ with f16 mma, accumulated per K/V tile in f16 and across tiles in f32.
// Blocks only visit K/V cells that some query of the tile can see: with one token tile the K/V tiles of a chunk
// that are fully masked are skipped; with several token tiles (usually the sequences of a unified KV cache) the
// visible cells of each chunk are gathered into dense tiles, so each tile reads only the cells of its own sequences.
// The partial results of the chunks are merged by flash_attn_ext_mma_q_combine with a fixed number of chunks per row.

#define FATTN_MMA_Q_NWARPS          8
#define FATTN_MMA_Q_NTHREADS        (FATTN_MMA_Q_NWARPS*WARP_SIZE)
#define FATTN_MMA_Q_NBATCH          64  // K/V cells per tile, 8 per warp for KQ
#define FATTN_MMA_Q_MAX_TOKENS      8   // query tokens per tile
#define FATTN_MMA_Q_MAX_CHUNK_TILES 128 // K/V tiles per list of visible tiles, longer chunks are processed in segments
#define FATTN_MMA_Q_SMEM_MAX        (99*1024)
#define FATTN_MMA_Q_GATHER_CELLS    2048 // K/V cells per chunk when gathering visible cells, 8 per thread
#define FATTN_MMA_Q_MAX_GATHER_BATCH 64  // largest number of query tokens for which visible cells are gathered

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
static constexpr __host__ __device__ int fattn_mma_q_stride_Qds(const int D) {
    return D/QK8_1 + 1; // float2, padded to avoid bank conflicts
}
static constexpr __host__ __device__ int fattn_mma_q_stride_V(const int D) {
    return D + 8; // halves
}
static constexpr __host__ __device__ int fattn_mma_q_stride_P() {
    return FATTN_MMA_Q_NBATCH + 8; // halves
}
static constexpr __host__ __device__ int fattn_mma_q_stride_mask() {
    return FATTN_MMA_Q_NBATCH + 8; // halves
}
static constexpr __host__ __device__ int fattn_mma_q_smem_fixed(const int D, const int ncols_tiles) {
    return 16*ncols_tiles*fattn_mma_q_stride_Q(D)        // Q, int8
         + 16*ncols_tiles*fattn_mma_q_stride_Qds(D)*8     // Q scales and sums
         + FATTN_MMA_Q_NBATCH*fattn_mma_q_stride_V(D)*2  // V, f16
         + 16*ncols_tiles*fattn_mma_q_stride_P()*2       // P
         + FATTN_MMA_Q_NWARPS*16*ncols_tiles*4           // cross-warp row max/sum
         + FATTN_MMA_Q_MAX_CHUNK_TILES*4 + 16;           // visible tile list + count
}
static constexpr __host__ __device__ int fattn_mma_q_smem_stage(const int D, const ggml_type type_K, const ggml_type type_V) {
    return FATTN_MMA_Q_NBATCH*(fattn_mma_q_row_bytes(D, type_K) + fattn_mma_q_row_bytes(D, type_V)) // raw K and V
         + FATTN_MMA_Q_MAX_TOKENS*fattn_mma_q_stride_mask()*2;                                    // mask
}
// Number of K/V tiles in flight: 2 if it fits into shared memory, else 1.
static constexpr __host__ __device__ int fattn_mma_q_nstages(const int D, const ggml_type type_K, const ggml_type type_V, const int ncols_tiles) {
    return fattn_mma_q_smem_fixed(D, ncols_tiles) + 2*fattn_mma_q_smem_stage(D, type_K, type_V) <= FATTN_MMA_Q_SMEM_MAX ? 2 : 1;
}
static constexpr __host__ __device__ int fattn_mma_q_smem_total(const int D, const ggml_type type_K, const ggml_type type_V, const int ncols_tiles) {
    return fattn_mma_q_smem_fixed(D, ncols_tiles) +
        fattn_mma_q_nstages(D, type_K, type_V, ncols_tiles)*fattn_mma_q_smem_stage(D, type_K, type_V);
}
// Largest number of 16 row Q tiles for which the kernel fits into shared memory, 0 if none.
static constexpr __host__ __device__ int fattn_mma_q_max_ncols_tiles(const int D, const ggml_type type_K, const ggml_type type_V) {
    for (int ncols_tiles = 4; ncols_tiles >= 1; --ncols_tiles) {
        if (fattn_mma_q_smem_total(D, type_K, type_V, ncols_tiles) <= FATTN_MMA_Q_SMEM_MAX) {
            return ncols_tiles;
        }
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------
// Dequantization from shared memory into mma operands.

// Two nibbles/bytes in the low byte of each 16 bit half of q -> half2 of (1024 + value), exact.
static __device__ __forceinline__ half2 fattn_mma_q_u8x2_to_h2(const uint32_t q) {
    const uint32_t h = q | 0x64006400;
    return *(const half2 *) &h;
}

static constexpr __host__ __device__ bool fattn_mma_q_has_m(const ggml_type type) {
    return type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1;
}

// Four nibbles/bytes of q (values < 128) -> two half2 of (1024 + value), exact.
static __device__ __forceinline__ void fattn_mma_q_u8x4_to_h2(const uint32_t q, half2 * dst) {
    const uint32_t lo = __byte_perm(q, 0x64646464, 0x5140);
    const uint32_t hi = __byte_perm(q, 0x64646464, 0x5342);
    dst[0] = *(const half2 *) &lo;
    dst[1] = *(const half2 *) &hi;
}

// Bits 0..3 of h -> bit 4 of bytes 0..3.
static __device__ __forceinline__ uint32_t fattn_mma_q_spread_qh(const uint32_t h) {
    return ((h << 4) & 0x00000010) | ((h << 11) & 0x00001000) | ((h << 18) & 0x00100000) | ((h << 25) & 0x10000000);
}

// 16 values of a block as f16: dst[0..3] are the dimensions i0...i0+7, dst[4..7] the dimensions i1...i1+7,
// i0 = 8*h and i1 = 16 + 8*h for the 4/5 bit types (nibbles of the same bytes), i0 = 16*h and i1 = 16*h + 8 for q8_0.
template <ggml_type type>
static __device__ __forceinline__ void fattn_mma_q_dq16(const uint8_t * b, const int h, half2 * dst) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        const half2 d = __half2half2(*(const half *) b);
        const uint16_t * qs = (const uint16_t *) (b + 2) + 8*h;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q = (qs[2*l] | (uint32_t(qs[2*l + 1]) << 16)) ^ 0x80808080;
            half2 tmp[2];
            fattn_mma_q_u8x4_to_h2(q, tmp);
            dst[2*l + 0] = __hmul2(__hsub2(tmp[0], make_half2(1152.0f, 1152.0f)), d);
            dst[2*l + 1] = __hmul2(__hsub2(tmp[1], make_half2(1152.0f, 1152.0f)), d);
        }
    } else {
        constexpr int iqs = type == GGML_TYPE_Q4_0 ? 2 : type == GGML_TYPE_Q4_1 ? 4 : type == GGML_TYPE_Q5_0 ? 6 : 8;
        const uint16_t * qs = (const uint16_t *) (b + iqs) + 4*h;
        const uint32_t x0 = qs[0] | (uint32_t(qs[1]) << 16);
        const uint32_t x1 = qs[2] | (uint32_t(qs[3]) << 16);
        uint32_t q[4] = {x0 & 0x0F0F0F0F, x1 & 0x0F0F0F0F, (x0 >> 4) & 0x0F0F0F0F, (x1 >> 4) & 0x0F0F0F0F};
        if constexpr (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            const uint16_t * qh16 = (const uint16_t *) (b + (type == GGML_TYPE_Q5_0 ? 2 : 4));
            const uint32_t qh = qh16[0] | (uint32_t(qh16[1]) << 16);
            q[0] |= fattn_mma_q_spread_qh(qh >> (8*h +  0));
            q[1] |= fattn_mma_q_spread_qh(qh >> (8*h +  4));
            q[2] |= fattn_mma_q_spread_qh(qh >> (8*h + 16));
            q[3] |= fattn_mma_q_spread_qh(qh >> (8*h + 20));
        }
        half2 d;
        half2 m;
        half2 off;
        if constexpr (fattn_mma_q_has_m(type)) {
            const half2 dm = *(const half2 *) b;
            d   = __low2half2(dm);
            m   = __high2half2(dm);
            off = make_half2(1024.0f, 1024.0f);
        } else {
            d   = __half2half2(*(const half *) b);
            m   = make_half2(0.0f, 0.0f);
            off = type == GGML_TYPE_Q4_0 ? make_half2(1032.0f, 1032.0f) : make_half2(1040.0f, 1040.0f);
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            half2 tmp[2];
            fattn_mma_q_u8x4_to_h2(q[l], tmp);
            if constexpr (fattn_mma_q_has_m(type)) {
                dst[2*l + 0] = __hfma2(__hsub2(tmp[0], off), d, m);
                dst[2*l + 1] = __hfma2(__hsub2(tmp[1], off), d, m);
            } else {
                dst[2*l + 0] = __hmul2(__hsub2(tmp[0], off), d);
                dst[2*l + 1] = __hmul2(__hsub2(tmp[1], off), d);
            }
        }
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
        // per byte subtraction without borrows between bytes, the values are < 128:
        if constexpr (type == GGML_TYPE_Q4_0) {
            b0 = ((b0 | 0x80808080) - 0x08080808) ^ 0x80808080;
            b1 = ((b1 | 0x80808080) - 0x08080808) ^ 0x80808080;
        } else if constexpr (type == GGML_TYPE_Q5_0) {
            b0 = ((b0 | 0x80808080) - 0x10101010) ^ 0x80808080;
            b1 = ((b1 | 0x80808080) - 0x10101010) ^ 0x80808080;
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

// B fragments of two 16x8 f16 tiles at tile (row-major [k][n], stride in halves): b[0..1] for n 0..7, b[2..3] for n 8..15.
static __device__ __forceinline__ void fattn_mma_q_ldmatrix_B_trans_x4(uint32_t * b, const half * tile, const int stride) {
#ifdef AMPERE_MMA_AVAILABLE
    const half * p = tile + (threadIdx.x % 16)*stride + (threadIdx.x / 16)*8;
    const uint32_t addr = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
        : "r"(addr));
#else
    GGML_UNUSED_VARS(b, tile, stride);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// B fragment of one 16x8 f16 tile at tile (row-major [k][n], stride in halves).
static __device__ __forceinline__ void fattn_mma_q_ldmatrix_B_trans_x2(uint32_t * b, const half * tile, const int stride) {
#ifdef AMPERE_MMA_AVAILABLE
    const half * p = tile + (threadIdx.x % 16)*stride;
    const uint32_t addr = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
#else
    GGML_UNUSED_VARS(b, tile, stride);
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

// Gathered tiles are copied with 4 threads per row, each thread needs the cell index of one row.
static_assert(FATTN_MMA_Q_NBATCH*4 == FATTN_MMA_Q_NTHREADS, "bad gather layout");

// Loads the cell index of the row that this thread copies for the gathered tile at list position pos0.
static __device__ __forceinline__ int fattn_mma_q_load_idx(const int * GGML_CUDA_RESTRICT idx, const int pos0, const int n) {
    const int pos = pos0 + (threadIdx.y*WARP_SIZE + threadIdx.x) / 4;
    return pos < n ? idx[pos] : -1;
}

// Copies the raw rows of a gathered tile into shared memory, cell is the result of fattn_mma_q_load_idx.
template <int D, ggml_type type>
static __device__ __forceinline__ void fattn_mma_q_load_KV_gather(uint8_t * tile, const char * KV, const int64_t stride, const int cell) {
    constexpr int row_bytes = fattn_mma_q_row_bytes(D, type);
    constexpr int cpy       = fattn_mma_q_cpy_bytes(D, type);
    constexpr int per_row   = row_bytes / cpy;
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int row = tid / 4;
    const uint32_t tile_s = __cvta_generic_to_shared(tile) + row*row_bytes;
    const char * src = KV + (cell >= 0 ? cell*stride : 0);
#pragma unroll
    for (int c = tid % 4; c < per_row; c += 4) {
        fattn_mma_q_cp_async<cpy>(tile_s + c*cpy, src + c*cpy, cell >= 0);
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

// For each tile of ntok_tile query tokens, marks the K/V tiles that at least one of the tokens can see: 0 = no cell
// visible, 1 = visible cells with mask values to apply, 2 = the mask is 0 for all tokens and cells (no mask needed).
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

    bool visible  = false;
    bool all_zero = true;
    if (mask_vec) {
        // rows are 16 byte aligned and ne11 % 8 == 0
        if (k < ne11) {
            for (int t = 0; t < ntok; ++t) {
                uint4 v;
                ggml_cuda_memcpy_1<16>(&v, mask_t + t*stride_mask1 + k);
                const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    visible  |= (w[l] & 0xFFFF) != 0xFC00 || (w[l] >> 16) != 0xFC00;
                    all_zero &= (w[l] & 0x7FFF7FFF) == 0; // +0 or -0
                }
            }
        }
    } else {
        for (int t = 0; t < ntok; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                if (k + l < ne11) {
                    const float x = __half2float(mask_t[t*stride_mask1 + k + l]);
                    visible  |= x != -INFINITY;
                    all_zero &= x == 0.0f;
                }
            }
        }
    }
    // reduce over the 8 threads of the K/V tile:
#pragma unroll
    for (int offset = 1; offset < 8; offset *= 2) {
        visible  |= __shfl_xor_sync(0xFFFFFFFF, visible,  offset);
        all_zero &= __shfl_xor_sync(0xFFFFFFFF, all_zero, offset);
    }
    if (threadIdx.x % 8 == 0 && tile < ntiles_kv) {
        vis[(blockIdx.z*gridDim.y + blockIdx.y)*int64_t(ntiles_kv) + tile] = !visible ? 0 : all_zero ? 2 : 1;
    }
    ggml_cuda_pdl_lc();
}

// For each chunk of FATTN_MMA_Q_GATHER_CELLS K/V cells and each tile of ntok_tile query tokens, lists the cells that
// at least one of the tokens can see together with the mask values of the tokens for them.
// One CUDA block per chunk and token tile, 8 cells per thread.
__launch_bounds__(256, 1)
static __global__ void flash_attn_ext_mma_q_gather(
        const half * GGML_CUDA_RESTRICT mask, int * GGML_CUDA_RESTRICT idx, half * GGML_CUDA_RESTRICT mask_g, int * GGML_CUDA_RESTRICT count,
        const int ne01, const int ne11, const int64_t stride_mask1, const int64_t stride_mask3,
        const int ntok_tile, const int mask_vec) {
    static_assert(FATTN_MMA_Q_GATHER_CELLS == 256*8, "bad FATTN_MMA_Q_GATHER_CELLS");
    ggml_cuda_pdl_sync();
    const int tid  = threadIdx.x;
    const int k    = blockIdx.x*FATTN_MMA_Q_GATHER_CELLS + 8*tid;
    const int tok0 = blockIdx.y*ntok_tile;
    const int ntok = min(ntok_tile, ne01 - tok0);
    const int64_t gc = (int64_t(blockIdx.z)*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x;

    const half * mask_t = mask + blockIdx.z*stride_mask3 + tok0*stride_mask1;

    half vals[FATTN_MMA_Q_MAX_TOKENS][8];
    uint32_t visible = 0;
#pragma unroll
    for (int t = 0; t < FATTN_MMA_Q_MAX_TOKENS; ++t) {
        if (t >= ntok) {
            break;
        }
        if (mask_vec && k < ne11) {
            // rows are 16 byte aligned and ne11 % 8 == 0
            ggml_cuda_memcpy_1<16>(vals[t], mask_t + t*stride_mask1 + k);
        } else {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                vals[t][l] = k + l < ne11 ? mask_t[t*stride_mask1 + k + l] : __float2half(-INFINITY);
            }
        }
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            if (__half2float(vals[t][l]) != -INFINITY) {
                visible |= 1u << l;
            }
        }
    }

    // exclusive prefix sum of the visible cells over the block:
    __shared__ int warp_sums[256/WARP_SIZE];
    const int n_thread = __popc(visible);
    int n_incl = n_thread;
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        const int tmp = __shfl_up_sync(0xFFFFFFFF, n_incl, offset);
        if (threadIdx.x % WARP_SIZE >= offset) {
            n_incl += tmp;
        }
    }
    if (threadIdx.x % WARP_SIZE == WARP_SIZE - 1) {
        warp_sums[tid / WARP_SIZE] = n_incl;
    }
    __syncthreads();
    int pos = n_incl - n_thread;
    int n_total = 0;
#pragma unroll
    for (int w = 0; w < 256/WARP_SIZE; ++w) {
        pos     += w < tid / WARP_SIZE ? warp_sums[w] : 0;
        n_total += warp_sums[w];
    }

    idx    += gc*FATTN_MMA_Q_GATHER_CELLS;
    mask_g += gc*ntok_tile*FATTN_MMA_Q_GATHER_CELLS;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        if (visible & (1u << l)) {
            idx[pos] = k + l;
#pragma unroll
            for (int t = 0; t < FATTN_MMA_Q_MAX_TOKENS; ++t) {
                if (t >= ntok) {
                    break;
                }
                mask_g[t*FATTN_MMA_Q_GATHER_CELLS + pos] = vals[t][l];
            }
            ++pos;
        }
    }
    if (tid == 0) {
        count[gc] = n_total;
    }
    ggml_cuda_pdl_lc();
}

template <int D, ggml_type type_K, ggml_type type_V, int ncols_tiles, bool gather>
__launch_bounds__(FATTN_MMA_Q_NTHREADS, 1)
static __global__ void flash_attn_ext_mma_q(
        const char  * GGML_CUDA_RESTRICT Q,
        const char  * GGML_CUDA_RESTRICT K,
        const char  * GGML_CUDA_RESTRICT V,
        const char  * GGML_CUDA_RESTRICT mask,
        const uint8_t * GGML_CUDA_RESTRICT vis,
        const int   * GGML_CUDA_RESTRICT gather_idx,
        const half  * GGML_CUDA_RESTRICT gather_mask,
        const int   * GGML_CUDA_RESTRICT gather_count,
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
    constexpr int nstages     = fattn_mma_q_nstages(D, type_K, type_V, ncols_tiles);
    constexpr int row_bytes_K = fattn_mma_q_row_bytes(D, type_K);
    constexpr int row_bytes_V = fattn_mma_q_row_bytes(D, type_V);
    constexpr int nrows     = 16*ncols_tiles;     // (token, head) rows of the tile
    constexpr int DW        = D/FATTN_MMA_Q_NWARPS; // VKQ columns per warp
    constexpr int NT        = DW/8;                 // VKQ n tiles per warp
    constexpr int stride_Q  = fattn_mma_q_stride_Q(D);
    constexpr int stride_P  = fattn_mma_q_stride_P();
    constexpr int stride_Qds = fattn_mma_q_stride_Qds(D);
    constexpr int stride_V  = fattn_mma_q_stride_V(D);
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
    half    * tile_V    = (half *) (tile_Qds + nrows*stride_Qds);
    half    * tile_P    = tile_V + FATTN_MMA_Q_NBATCH*stride_V;
    float   * red       = (float *) (tile_P + nrows*stride_P);
    int     * tile_list = (int *) (red + FATTN_MMA_Q_NWARPS*nrows);
    int     * tile_num  = tile_list + FATTN_MMA_Q_MAX_CHUNK_TILES;
    uint8_t * stages    = fattn_mma_q_smem + fattn_mma_q_smem_fixed(D, ncols_tiles);
    // stage s: raw K, raw V, mask
    auto stage_K    = [&](const int s) { return stages + s*fattn_mma_q_smem_stage(D, type_K, type_V); };
    auto stage_V    = [&](const int s) { return stage_K(s) + FATTN_MMA_Q_NBATCH*row_bytes_K; };
    auto stage_mask = [&](const int s) { return (half *) (stage_V(s) + FATTN_MMA_Q_NBATCH*row_bytes_V); };

    ggml_cuda_pdl_sync();

    // gather: the visible cells of the chunk and their mask values, as dense tiles
    const int64_t gc = gather ? ((int64_t(sequence % ne33)*gridDim.y + blockIdx.y)*nchunks + chunk) : 0;
    const int    * idx_g  = gather ? gather_idx + gc*FATTN_MMA_Q_GATHER_CELLS : nullptr;
    const half   * mask_g = gather ? gather_mask + gc*ntok_tile*FATTN_MMA_Q_GATHER_CELLS : nullptr;
    const int      n_g    = gather ? gather_count[gc] : 0;
    // Issues the loads of K (and the mask) or V for tile it into stage s, gather needs the cell indices idx_regs:
    int cell_cur = -1; // gather: cell of the row that this thread copies
    auto load_K = [&](const int s, const int it) {
        if constexpr (gather) {
            fattn_mma_q_load_KV_gather<D, type_K>(stage_K(s), K_h, nb11, cell_cur);
            fattn_mma_q_load_mask(stage_mask(s), mask_g, FATTN_MMA_Q_GATHER_CELLS, ntok, it*FATTN_MMA_Q_NBATCH, FATTN_MMA_Q_GATHER_CELLS, true);
        } else {
            const int k0 = tile_list[it]*FATTN_MMA_Q_NBATCH;
            fattn_mma_q_load_KV<D, type_K>(stage_K(s), K_h, nb11, k0, ne11);
            if (mask_h) {
                fattn_mma_q_load_mask(stage_mask(s), mask_h, stride_mask, ntok, k0, ne11, mask_vec);
            }
        }
    };
    auto load_V = [&](const int s, const int it) {
        if constexpr (gather) {
            fattn_mma_q_load_KV_gather<D, type_V>(stage_V(s), V_h, nb21, cell_cur);
        } else {
            fattn_mma_q_load_KV<D, type_V>(stage_V(s), V_h, nb21, tile_list[it]*FATTN_MMA_Q_NBATCH, ne11);
        }
    };

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
            tile_Qds[r*stride_Qds + lane/4] = make_float2(d, d*sum);
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

    // A chunk longer than the tile list is processed in segments of FATTN_MMA_Q_MAX_CHUNK_TILES tiles:
    for (int seg0 = 0; seg0 < (gather ? 1 : kt_num); seg0 += FATTN_MMA_Q_MAX_CHUNK_TILES) {
    const int seg_begin = kt_begin + seg0;
    const int seg_num   = min(FATTN_MMA_Q_MAX_CHUNK_TILES, kt_num - seg0);
    if (!gather) {
    // List the K/V tiles of the segment that are visible to at least one query of the tile:
    const uint8_t * vis_c = vis ? vis + ((sequence % ne33)*gridDim.y + blockIdx.y)*int64_t(ntiles_kv) + seg_begin : nullptr;
    if (warp == 0) {
        int n = 0;
        for (int it0 = 0; it0 < seg_num; it0 += WARP_SIZE) {
            const int it = it0 + lane;
            const bool visible = it < seg_num && (!vis_c || vis_c[it]);
            const uint32_t ballot = __ballot_sync(0xFFFFFFFF, visible);
            if (visible) {
                tile_list[n + __popc(ballot & ((1u << lane) - 1))] = seg_begin + it;
            }
            n += __popc(ballot);
        }
        if (lane == 0) {
            *tile_num = n;
        }
    }
    __syncthreads();
    }
    const int nvis = gather ? (n_g + FATTN_MMA_Q_NBATCH - 1) / FATTN_MMA_Q_NBATCH : *tile_num;
    __syncthreads(); // tile_num is rewritten by the next segment

    if (nvis == 0 && nchunks > 1 && kt_num <= FATTN_MMA_Q_MAX_CHUNK_TILES) {
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
            if constexpr (gather) {
                cell_cur = fattn_mma_q_load_idx(idx_g, s*FATTN_MMA_Q_NBATCH, n_g);
            }
            load_K(s, s);
        }
        fattn_mma_q_cp_async_commit();
        if (s < nvis) {
            load_V(s, s);
        }
        fattn_mma_q_cp_async_commit();
    }
    // the cell indices of the next tile to load are fetched one iteration ahead:
    int cell_next = -1;
    if constexpr (gather) {
        cell_next = fattn_mma_q_load_idx(idx_g, nstages*FATTN_MMA_Q_NBATCH, n_g);
    }

    for (int it = 0; it < nvis; ++it) {
        const int s  = nstages == 1 ? 0 : it % nstages;
        // cells c of the tile with c >= k_sup are out of bounds:
        const int k_sup = gather ? n_g - it*FATTN_MMA_Q_NBATCH : ne11 - tile_list[it]*FATTN_MMA_Q_NBATCH;

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
            constexpr int bs = fattn_mma_q_block_bytes(type_K);
            const uint8_t * K_row = stage_K(s) + (8*warp + lr    )*row_bytes_K; // B operand
            const uint8_t * K_c0  = stage_K(s) + (8*warp + 2*lq  )*row_bytes_K; // scales of the columns of C
            const uint8_t * K_c1  = K_c0 + row_bytes_K;
#pragma unroll
            for (int ib = 0; ib < D/QK8_0; ++ib) {
                uint32_t b0;
                uint32_t b1;
                fattn_mma_q_K_i8<type_K>(K_row + ib*bs, lq, b0, b1);
                const float2 dm0 = fattn_mma_q_K_dm<type_K>(K_c0 + ib*bs);
                const float2 dm1 = fattn_mma_q_K_dm<type_K>(K_c1 + ib*bs);
#pragma unroll
                for (int m = 0; m < ncols_tiles; ++m) {
                    uint32_t a[4];
                    fattn_mma_q_ldmatrix_A(a, (const half *) (tile_Q + 16*m*stride_Q + ib*QK8_0), stride_Q/2);
                    int acc[4] = {0, 0, 0, 0};
                    fattn_mma_q_mma_i8(acc, a, b0, b1);
                    const float2 ds0 = tile_Qds[(16*m + lr    )*stride_Qds + ib];
                    const float2 ds1 = tile_Qds[(16*m + lr + 8)*stride_Qds + ib];
                    KQ[m][0] += dm0.x*ds0.x * fattn_mma_q_i2f(acc[0]);
                    KQ[m][1] += dm1.x*ds0.x * fattn_mma_q_i2f(acc[1]);
                    KQ[m][2] += dm0.x*ds1.x * fattn_mma_q_i2f(acc[2]);
                    KQ[m][3] += dm1.x*ds1.x * fattn_mma_q_i2f(acc[3]);
                    if constexpr (fattn_mma_q_has_m(type_K)) {
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
                    if (c0 + e >= k_sup) {
                        x = -INFINITY;
                    }
                }
                float mx = fmaxf(KQ[m][2*h + 0], KQ[m][2*h + 1]);
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 2));
                if (lq == 0) {
                    red[(16*m + 8*h + lr)*FATTN_MMA_Q_NWARPS + warp] = mx;
                }
            }
        }

        fattn_mma_q_cp_async_wait<2*nstages - 2>();
        __syncthreads(); // tile maxima and V are in shared memory, K and mask of this stage are no longer needed

        if (it + nstages < nvis) {
            if constexpr (gather) {
                cell_cur  = cell_next;
                cell_next = fattn_mma_q_load_idx(idx_g, (it + nstages + 1)*FATTN_MMA_Q_NBATCH, n_g);
            }
            load_K(s, it + nstages);
        }
        fattn_mma_q_cp_async_commit();

        // Dequantize V:
        {
            constexpr int bs = fattn_mma_q_block_bytes(type_V);
            const uint8_t * V_s = stage_V(s);
#pragma unroll
            for (int i0 = 0; i0 < FATTN_MMA_Q_NBATCH*(D/16); i0 += FATTN_MMA_Q_NTHREADS) {
                const int i    = i0 + tid;
                const int cell = i / (D/16);
                const int ib   = (i % (D/16)) / 2;
                const int h    = i % 2;
                half2 v[8];
                fattn_mma_q_dq16<type_V>(V_s + cell*row_bytes_V + ib*bs, h, v);
                const int i0v = type_V == GGML_TYPE_Q8_0 ? 16*h : 8*h;
                const int i1v = type_V == GGML_TYPE_Q8_0 ? 16*h + 8 : 16 + 8*h;
                ggml_cuda_memcpy_1<16>(tile_V + cell*stride_V + ib*QK8_0 + i0v, v + 0);
                ggml_cuda_memcpy_1<16>(tile_V + cell*stride_V + ib*QK8_0 + i1v, v + 4);
            }
        }

        // Softmax, P is written to shared memory as f16:
#pragma unroll
        for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = 16*m + 8*h + lr;
                float mx = KQ_max[m][h];
#pragma unroll
                for (int w = 0; w < FATTN_MMA_Q_NWARPS; w += 4) {
                    const float4 tmp = *(const float4 *) (red + r*FATTN_MMA_Q_NWARPS + w);
                    mx = fmaxf(mx, fmaxf(fmaxf(tmp.x, tmp.y), fmaxf(tmp.z, tmp.w)));
                }
                const float mx_use = mx == -INFINITY ? 0.0f : mx;
                const float scale_old = expf(KQ_max[m][h] - mx_use);
                KQ_max[m][h] = mx;

                const float p0 = expf(KQ[m][2*h + 0] - mx_use);
                const float p1 = expf(KQ[m][2*h + 1] - mx_use);
                KQ_sum[m][h] = KQ_sum[m][h]*scale_old + (p0 + p1);
                // the maximum rarely changes after the first tiles:
                if (__any_sync(0xFFFFFFFF, scale_old != 1.0f)) {
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        VKQ[m][j][2*h + 0] *= scale_old;
                        VKQ[m][j][2*h + 1] *= scale_old;
                    }
                }
                *(half2 *) (tile_P + r*stride_P + c0) = make_half2(p0, p1);
            }
        }

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
#pragma unroll
            for (int k = 0; k < FATTN_MMA_Q_NBATCH; k += 16) {
                uint32_t b0[NT];
                uint32_t b1[NT];
#pragma unroll
                for (int j = 0; j < NT; j += 2) {
                    uint32_t b[4];
                    if (j + 1 < NT) {
                        fattn_mma_q_ldmatrix_B_trans_x4(b, tile_V + k*stride_V + warp*DW + 8*j, stride_V);
                        b0[j + 1] = b[2];
                        b1[j + 1] = b[3];
                    } else {
                        fattn_mma_q_ldmatrix_B_trans_x2(b, tile_V + k*stride_V + warp*DW + 8*j, stride_V);
                    }
                    b0[j] = b[0];
                    b1[j] = b[1];
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

        __syncthreads(); // V and P are no longer needed

        if (it + nstages < nvis) {
            load_V(s, it + nstages);
        }
        fattn_mma_q_cp_async_commit();
    }
    fattn_mma_q_cp_async_wait<0>();
    }

    // Sum up the partial KQ row sums of the quads and warps:
#pragma unroll
    for (int m = 0; m < ncols_tiles; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float sum = KQ_sum[m][h];
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
            if (lq == 0) {
                red[(16*m + 8*h + lr)*FATTN_MMA_Q_NWARPS + warp] = sum;
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
            for (int w = 0; w < FATTN_MMA_Q_NWARPS; w += 4) {
                const float4 tmp = *(const float4 *) (red + r*FATTN_MMA_Q_NWARPS + w);
                sum += (tmp.x + tmp.y) + (tmp.z + tmp.w);
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
    GGML_UNUSED_VARS(Q, K, V, mask, vis, gather_idx, gather_mask, gather_count, sinks, dst, dst_parts, dst_meta, scale,
        ne01, ne02, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb13, nb21, nb22, nb23, ne33, nb31, nb33,
        gqa_ratio, ntok_tile, chunk_tiles, nchunks, mask_vec);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// ------------------------------------------------------------------------------------------------------------------
// Large batches (prefill). A block handles FATTN_MMA_Q_PF_ROWS (token, head) rows of one K/V head over a chunk of the
// K/V cells. Two warps share 16 rows: each computes KQ for half of the cells of a tile, they exchange the row maxima and
// P through shared memory (named barrier per pair), and each computes VKQ for half of the dimensions. That keeps the
// f32 accumulators at D/4 registers per thread, so 16 warps fit on an SM and hide the latencies of the mma and of the
// scaling math. The raw K/V rows of a tile of FATTN_MMA_Q_PF_NBATCH cells are copied with cp.async one tile ahead and
// unpacked once per block: K into an int8 tile plus its scales (B operand of the int8 KQ mma, Q quantized to int8 per
// 32 values), V into an f16 tile (B operand of the VKQ mma, f16 accumulation within a tile, f32 across tiles). KQ is in
// log2 units (the Q scale includes log2(e)). K/V tiles that no token of the block can see are skipped, tiles with a
// zero mask skip the mask.

#define FATTN_MMA_Q_PF_NWARPS   16
#define FATTN_MMA_Q_PF_NTHREADS (FATTN_MMA_Q_PF_NWARPS*WARP_SIZE)
#define FATTN_MMA_Q_PF_ROWS     (8*FATTN_MMA_Q_PF_NWARPS) // (token, head) rows per block, 16 per warp pair
#define FATTN_MMA_Q_PF_NBATCH   32                          // K/V cells per tile, 2 tiles per visibility tile

static_assert(FATTN_MMA_Q_NBATCH == 2*FATTN_MMA_Q_PF_NBATCH, "bad FATTN_MMA_Q_PF_NBATCH");

static constexpr __host__ __device__ int fattn_mma_q_pf_stride_K(const int D) {
    return D + 16; // int8
}
static constexpr __host__ __device__ int fattn_mma_q_pf_stride_P() {
    return FATTN_MMA_Q_PF_NBATCH + 8; // halves
}
// Bytes of the scales of one 32 value block of K for all cells of a tile: d (and m) as float.
static constexpr __host__ __device__ int fattn_mma_q_pf_Kdm_bytes(const ggml_type type_K) {
    return FATTN_MMA_Q_PF_NBATCH*(fattn_mma_q_has_m(type_K) ? 8 : 4);
}
static constexpr __host__ __device__ int fattn_mma_q_pf_smem(const int D, const ggml_type type_K, const ggml_type type_V) {
    return FATTN_MMA_Q_PF_ROWS*fattn_mma_q_stride_Q(D)                                                   // Q, int8
         + FATTN_MMA_Q_PF_ROWS*fattn_mma_q_stride_Qds(D)*8                                               // Q scales and sums
         + FATTN_MMA_Q_PF_NBATCH*fattn_mma_q_pf_stride_K(D)                                              // K, int8
         + (D/QK8_0)*fattn_mma_q_pf_Kdm_bytes(type_K)                                                    // K scales
         + FATTN_MMA_Q_PF_NBATCH*fattn_mma_q_stride_V(D)*2                                               // V, f16
         + FATTN_MMA_Q_PF_NBATCH*(fattn_mma_q_row_bytes(D, type_K) + fattn_mma_q_row_bytes(D, type_V))  // raw K and V
         + FATTN_MMA_Q_PF_ROWS*fattn_mma_q_pf_stride_P()*2                                               // P, f16
         + FATTN_MMA_Q_PF_NWARPS*16*4                                                                    // row max/sum exchange
         + FATTN_MMA_Q_MAX_CHUNK_TILES*4 + 16;                                                           // tile list + count
}

// B fragments of two 32x8 int8 tiles (n tiles j and j+1) of the K tile at K_s (rows = cells, stride in bytes):
// b[0..1] for cells 0..7, b[2..3] for cells 8..15, 32 values from byte 0.
static __device__ __forceinline__ void fattn_mma_q_pf_ldmatrix_K(uint32_t * b, const int8_t * K_s, const int stride) {
#ifdef AMPERE_MMA_AVAILABLE
    const int8_t * p = K_s + (threadIdx.x % 8 + 8*(threadIdx.x / 16))*stride + 16*((threadIdx.x / 8) % 2);
    const uint32_t addr = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
        : "r"(addr));
#else
    GGML_UNUSED_VARS(b, K_s, stride);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// Barrier for the two warps of a pair (ids 1...8, 0 is __syncthreads).
static __device__ __forceinline__ void fattn_mma_q_pf_pair_sync(const int pair) {
    asm volatile("bar.sync %0, 64;" : : "r"(pair + 1) : "memory");
}

template <int D, ggml_type type_K, ggml_type type_V>
__launch_bounds__(FATTN_MMA_Q_PF_NTHREADS, 1)
static __global__ void flash_attn_ext_mma_q_pf(
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
    constexpr int NB          = FATTN_MMA_Q_PF_NBATCH;
    constexpr int nrows       = FATTN_MMA_Q_PF_ROWS;
    constexpr int nthreads    = FATTN_MMA_Q_PF_NTHREADS;
    constexpr int row_bytes_K = fattn_mma_q_row_bytes(D, type_K);
    constexpr int row_bytes_V = fattn_mma_q_row_bytes(D, type_V);
    constexpr int cpy_K       = fattn_mma_q_cpy_bytes(D, type_K);
    constexpr int cpy_V       = fattn_mma_q_cpy_bytes(D, type_V);
    constexpr int stride_Q    = fattn_mma_q_stride_Q(D);
    constexpr int stride_Qds  = fattn_mma_q_stride_Qds(D);
    constexpr int stride_K    = fattn_mma_q_pf_stride_K(D);
    constexpr int stride_V    = fattn_mma_q_stride_V(D);
    constexpr int stride_P    = fattn_mma_q_pf_stride_P();
    constexpr bool has_m_K    = fattn_mma_q_has_m(type_K);
    constexpr int DH          = D/2;     // VKQ dimensions per warp
    constexpr int NT          = DH/8;    // VKQ n tiles per warp
    constexpr int NBH         = NB/2;    // KQ cells per warp
    constexpr int NTK         = NBH/8;   // KQ n tiles per warp
    constexpr float log2e     = 1.4426950408889634f;
    static_assert(D % 64 == 0 && D/8 <= WARP_SIZE, "bad D");
    static_assert(NTK == 2 && NT % 2 == 0, "bad tiling");

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp*WARP_SIZE + lane;
    const int lq   = lane % 4; // thread in quad
    const int lr   = lane / 4; // quad in warp
    const int pair = warp / 2; // the 16 rows of the pair
    const int hw   = warp % 2; // KQ cells hw*NBH..., VKQ dimensions hw*DH...

    // the token tiles of a K/V head and chunk are adjacent so that they share the L2 sectors of the K/V rows
    const int tile_tok = blockIdx.x;
    const int chunk    = blockIdx.y / ne12;
    const int head_kv  = blockIdx.y % ne12;
    const int sequence = blockIdx.z;
    const int tok0     = tile_tok*ntok_tile;

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
    int8_t  * tile_K    = (int8_t *) (tile_Qds + nrows*stride_Qds);
    float   * tile_Kdm  = (float *) (tile_K + NB*stride_K);
    half    * tile_V    = (half *) ((uint8_t *) tile_Kdm + (D/QK8_0)*fattn_mma_q_pf_Kdm_bytes(type_K));
    uint8_t * raw_K     = (uint8_t *) (tile_V + NB*stride_V);
    uint8_t * raw_V     = raw_K + NB*row_bytes_K;
    half    * tile_P    = (half *) (raw_V + NB*row_bytes_V);
    float   * red       = (float *) (tile_P + nrows*stride_P);
    int     * tile_list = (int *) (red + FATTN_MMA_Q_PF_NWARPS*16);
    int     * tile_num  = tile_list + FATTN_MMA_Q_MAX_CHUNK_TILES;

    ggml_cuda_pdl_sync();

    // Issues the copies of the raw K/V rows of the cells k0...k0+NB-1:
    auto load_raw = [&](const int k0) {
        const uint32_t raw_K_s = __cvta_generic_to_shared(raw_K);
        const uint32_t raw_V_s = __cvta_generic_to_shared(raw_V);
#pragma unroll
        for (int i0 = 0; i0 < NB*(row_bytes_K/cpy_K); i0 += nthreads) {
            const int i = i0 + tid;
            if (i0 + nthreads > NB*(row_bytes_K/cpy_K) && i >= NB*(row_bytes_K/cpy_K)) {
                break;
            }
            const int row = i / (row_bytes_K/cpy_K);
            const int c   = i % (row_bytes_K/cpy_K);
            const bool valid = k0 + row < ne11;
            fattn_mma_q_cp_async<cpy_K>(raw_K_s + row*row_bytes_K + c*cpy_K, K_h + (valid ? (k0 + row)*nb11 : 0) + c*cpy_K, valid);
        }
#pragma unroll
        for (int i0 = 0; i0 < NB*(row_bytes_V/cpy_V); i0 += nthreads) {
            const int i = i0 + tid;
            if (i0 + nthreads > NB*(row_bytes_V/cpy_V) && i >= NB*(row_bytes_V/cpy_V)) {
                break;
            }
            const int row = i / (row_bytes_V/cpy_V);
            const int c   = i % (row_bytes_V/cpy_V);
            const bool valid = k0 + row < ne11;
            fattn_mma_q_cp_async<cpy_V>(raw_V_s + row*row_bytes_V + c*cpy_V, V_h + (valid ? (k0 + row)*nb21 : 0) + c*cpy_V, valid);
        }
        fattn_mma_q_cp_async_commit();
    };

    // Q is quantized to int8 per 32 values, each thread does 8 values, 4 threads a block:
    for (int r = warp; r < nrows; r += FATTN_MMA_Q_PF_NWARPS) {
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
            q[l] *= scale*log2e;
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
            tile_Qds[r*stride_Qds + lane/4] = make_float2(d, d*sum);
        }
    }

    // The rows of this thread and their tokens (for the mask):
    const int r0 = 16*pair + lr;
    const int row_tok[2] = {min(r0 / gqa_ratio, ntok - 1), min((r0 + 8) / gqa_ratio, ntok - 1)};
    float * red_own   = red + 16*warp;       // this warp's row maxima/sums
    float * red_other = red + 16*(warp ^ 1); // the other warp of the pair

    float VKQ[NT][4];
    float KQ_max[2] = {-INFINITY, -INFINITY};
    float KQ_sum[2] = {0.0f, 0.0f}; // over the cells of this warp
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            VKQ[j][l] = 0.0f;
        }
    }

    // A chunk longer than the tile list is processed in segments of FATTN_MMA_Q_MAX_CHUNK_TILES visibility tiles:
    for (int seg0 = 0; seg0 < kt_num; seg0 += FATTN_MMA_Q_MAX_CHUNK_TILES) {
        const int seg_begin = kt_begin + seg0;
        const int seg_num   = min(FATTN_MMA_Q_MAX_CHUNK_TILES, kt_num - seg0);
        // List the visibility tiles of the segment that some token of the block can see, bit 0 = mask needed:
        const uint8_t * vis_c = vis ? vis + ((sequence % ne33)*gridDim.x + tile_tok)*int64_t(ntiles_kv) + seg_begin : nullptr;
        if (warp == 0) {
            int n = 0;
            for (int it0 = 0; it0 < seg_num; it0 += WARP_SIZE) {
                const int it = it0 + lane;
                const int state = it < seg_num ? (vis_c ? vis_c[it] : (mask_h ? 1 : 2)) : 0;
                const uint32_t ballot = __ballot_sync(0xFFFFFFFF, state != 0);
                if (state != 0) {
                    tile_list[n + __popc(ballot & ((1u << lane) - 1))] = 2*(seg_begin + it) + (state == 1);
                }
                n += __popc(ballot);
            }
            if (lane == 0) {
                *tile_num = n;
            }
        }
        __syncthreads();
        const int nsub = 2*(*tile_num);
        __syncthreads(); // tile_num is rewritten by the next segment

        if (nsub == 0 && nchunks > 1 && kt_num <= FATTN_MMA_Q_MAX_CHUNK_TILES) {
            // nothing visible, the merge skips this chunk
            for (int r = tid; r < rows_valid; r += nthreads) {
                const int t    = tok0 + r / gqa_ratio;
                const int head = head_kv*gqa_ratio + r % gqa_ratio;
                dst_meta[((int64_t(sequence)*ne01 + t)*ne02 + head)*nchunks + chunk] = make_float2(-INFINITY, 0.0f);
            }
            return;
        }

        if (nsub > 0) {
            load_raw((tile_list[0] >> 1)*FATTN_MMA_Q_NBATCH);
        }

        for (int i = 0; i < nsub; ++i) {
            const int  e      = tile_list[i/2];
            const bool masked = e & 1;
            const int  k0     = (e >> 1)*FATTN_MMA_Q_NBATCH + (i % 2)*NB;

            fattn_mma_q_cp_async_wait<0>();
            __syncthreads(); // the raw rows of this tile are in shared memory, all warps are done with the previous tile

            // Unpack K into int8 values and scales, V into f16:
            {
                constexpr int bs = fattn_mma_q_block_bytes(type_K);
#pragma unroll
                for (int u0 = 0; u0 < NB*(D/QK8_0); u0 += nthreads) {
                    const int u = u0 + tid;
                    if (u0 + nthreads > NB*(D/QK8_0) && u >= NB*(D/QK8_0)) {
                        break;
                    }
                    const int cell = u / (D/QK8_0);
                    const int ib   = u % (D/QK8_0);
                    const uint8_t * b = raw_K + cell*row_bytes_K + ib*bs;
                    uint32_t w[8];
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        fattn_mma_q_K_i8<type_K>(b, l, w[l], w[4 + l]);
                    }
                    *(uint4 *) (tile_K + cell*stride_K + ib*QK8_0 +  0) = make_uint4(w[0], w[1], w[2], w[3]);
                    *(uint4 *) (tile_K + cell*stride_K + ib*QK8_0 + 16) = make_uint4(w[4], w[5], w[6], w[7]);
                    const float2 dm = fattn_mma_q_K_dm<type_K>(b);
                    if constexpr (has_m_K) {
                        ((float2 *) tile_Kdm)[ib*NB + cell] = dm;
                    } else {
                        tile_Kdm[ib*NB + cell] = dm.x;
                    }
                }
            }
            {
                constexpr int bs = fattn_mma_q_block_bytes(type_V);
#pragma unroll
                for (int u0 = 0; u0 < NB*(D/16); u0 += nthreads) {
                    const int u = u0 + tid;
                    if (u0 + nthreads > NB*(D/16) && u >= NB*(D/16)) {
                        break;
                    }
                    const int cell = u / (D/16);
                    const int ib   = (u % (D/16)) / 2;
                    const int h    = u % 2;
                    half2 v[8];
                    fattn_mma_q_dq16<type_V>(raw_V + cell*row_bytes_V + ib*bs, h, v);
                    const int i0v = type_V == GGML_TYPE_Q8_0 ? 16*h : 8*h;
                    const int i1v = type_V == GGML_TYPE_Q8_0 ? 16*h + 8 : 16 + 8*h;
                    ggml_cuda_memcpy_1<16>(tile_V + cell*stride_V + ib*QK8_0 + i0v, v + 0);
                    ggml_cuda_memcpy_1<16>(tile_V + cell*stride_V + ib*QK8_0 + i1v, v + 4);
                }
            }

            __syncthreads(); // K and V of this tile are unpacked, the raw rows are no longer needed

            if (i + 1 < nsub) {
                load_raw((tile_list[(i + 1)/2] >> 1)*FATTN_MMA_Q_NBATCH + ((i + 1) % 2)*NB);
            }

            // KQ for the 16 rows of the pair and the NBH cells of this warp:
            float KQ[NTK][4];
#pragma unroll
            for (int j = 0; j < NTK; ++j) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    KQ[j][l] = 0.0f;
                }
            }
#pragma unroll
            for (int ib = 0; ib < D/QK8_0; ++ib) {
                uint32_t a[4];
                fattn_mma_q_ldmatrix_A(a, (const half *) (tile_Q + 16*pair*stride_Q + ib*QK8_0), stride_Q/2);
                const float2 ds0 = tile_Qds[(r0    )*stride_Qds + ib];
                const float2 ds1 = tile_Qds[(r0 + 8)*stride_Qds + ib];
                uint32_t b[4];
                fattn_mma_q_pf_ldmatrix_K(b, tile_K + hw*NBH*stride_K + ib*QK8_0, stride_K);
#pragma unroll
                for (int j = 0; j < NTK; ++j) {
                    int acc[4] = {0, 0, 0, 0};
                    fattn_mma_q_mma_i8(acc, a, b[2*j + 0], b[2*j + 1]);
                    const int c = hw*NBH + 8*j + 2*lq;
                    if constexpr (has_m_K) {
                        const float4 dm = *(const float4 *) ((const float2 *) tile_Kdm + ib*NB + c);
                        KQ[j][0] += dm.x*ds0.x * fattn_mma_q_i2f(acc[0]) + dm.y*ds0.y;
                        KQ[j][1] += dm.z*ds0.x * fattn_mma_q_i2f(acc[1]) + dm.w*ds0.y;
                        KQ[j][2] += dm.x*ds1.x * fattn_mma_q_i2f(acc[2]) + dm.y*ds1.y;
                        KQ[j][3] += dm.z*ds1.x * fattn_mma_q_i2f(acc[3]) + dm.w*ds1.y;
                    } else {
                        const float2 dk = *(const float2 *) (tile_Kdm + ib*NB + c);
                        KQ[j][0] += dk.x*ds0.x * fattn_mma_q_i2f(acc[0]);
                        KQ[j][1] += dk.y*ds0.x * fattn_mma_q_i2f(acc[1]);
                        KQ[j][2] += dk.x*ds1.x * fattn_mma_q_i2f(acc[2]);
                        KQ[j][3] += dk.y*ds1.x * fattn_mma_q_i2f(acc[3]);
                    }
                }
            }

            // Mask and out-of-bounds cells:
            const int kw = k0 + hw*NBH; // first cell of this warp
            if (masked) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const half * mask_r = mask_h + int64_t(row_tok[h])*stride_mask + kw;
#pragma unroll
                    for (int j = 0; j < NTK; ++j) {
                        const int c = 8*j + 2*lq;
                        float2 m = make_float2(0.0f, 0.0f);
                        if (mask_vec) {
                            // ne11 % 8 == 0, so the pair is either in or out of bounds
                            if (kw + c < ne11) {
                                m = __half22float2(*(const half2 *) (mask_r + c));
                            }
                        } else {
                            m.x = kw + c + 0 < ne11 ? __half2float(mask_r[c + 0]) : 0.0f;
                            m.y = kw + c + 1 < ne11 ? __half2float(mask_r[c + 1]) : 0.0f;
                        }
                        KQ[j][2*h + 0] += log2e*m.x;
                        KQ[j][2*h + 1] += log2e*m.y;
                    }
                }
            }
            if (kw + NBH > ne11) {
#pragma unroll
                for (int j = 0; j < NTK; ++j) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        if (kw + 8*j + 2*lq + l % 2 >= ne11) {
                            KQ[j][l] = -INFINITY;
                        }
                    }
                }
            }

            // Row maxima of this warp's cells, exchanged with the other warp of the pair:
            float mx[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                mx[h] = fmaxf(fmaxf(KQ[0][2*h + 0], KQ[0][2*h + 1]), fmaxf(KQ[1][2*h + 0], KQ[1][2*h + 1]));
                mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xFFFFFFFF, mx[h], 1));
                mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xFFFFFFFF, mx[h], 2));
            }
            if (lq == 0) {
                red_own[lr + 0] = mx[0];
                red_own[lr + 8] = mx[1];
            }
            fattn_mma_q_pf_pair_sync(pair);

            // Online softmax, both warps of the pair compute the same row maxima:
            float scale_old[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float m_new = fmaxf(KQ_max[h], fmaxf(mx[h], red_other[lr + 8*h]));
                const float m_use = m_new == -INFINITY ? 0.0f : m_new;
                scale_old[h] = exp2f(KQ_max[h] - m_use);
                KQ_max[h] = m_new;
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < NTK; ++j) {
                    KQ[j][2*h + 0] = exp2f(KQ[j][2*h + 0] - m_use);
                    KQ[j][2*h + 1] = exp2f(KQ[j][2*h + 1] - m_use);
                    sum += KQ[j][2*h + 0] + KQ[j][2*h + 1];
                }
                KQ_sum[h] = KQ_sum[h]*scale_old[h] + sum;
            }
            // the maximum rarely changes after the first tiles:
            if (__any_sync(0xFFFFFFFF, scale_old[0] != 1.0f || scale_old[1] != 1.0f)) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    VKQ[j][0] *= scale_old[0];
                    VKQ[j][1] *= scale_old[0];
                    VKQ[j][2] *= scale_old[1];
                    VKQ[j][3] *= scale_old[1];
                }
            }

            // P of this warp's cells to shared memory, then the P of all NB cells as the A operand of VKQ:
            half * P_pair = tile_P + 16*pair*stride_P;
#pragma unroll
            for (int j = 0; j < NTK; ++j) {
                const int c = hw*NBH + 8*j + 2*lq;
                *(half2 *) (P_pair + (lr + 0)*stride_P + c) = make_half2(KQ[j][0], KQ[j][1]);
                *(half2 *) (P_pair + (lr + 8)*stride_P + c) = make_half2(KQ[j][2], KQ[j][3]);
            }
            fattn_mma_q_pf_pair_sync(pair);
            uint32_t P[NB/16][4];
#pragma unroll
            for (int kk = 0; kk < NB/16; ++kk) {
                fattn_mma_q_ldmatrix_A(P[kk], P_pair + 16*kk, stride_P);
            }

            // VKQ for this warp's dimensions, f16 accumulation within the tile:
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                uint32_t acc[2][2] = {{0, 0}, {0, 0}};
#pragma unroll
                for (int kk = 0; kk < NB/16; ++kk) {
                    uint32_t b[4];
                    fattn_mma_q_ldmatrix_B_trans_x4(b, tile_V + 16*kk*stride_V + hw*DH + 8*j, stride_V);
                    fattn_mma_q_mma_f16(acc[0], P[kk], b[0], b[1]);
                    fattn_mma_q_mma_f16(acc[1], P[kk], b[2], b[3]);
                }
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    const float2 lo = __half22float2(*(const half2 *) &acc[jj][0]);
                    const float2 hi = __half22float2(*(const half2 *) &acc[jj][1]);
                    VKQ[j + jj][0] += lo.x;
                    VKQ[j + jj][1] += lo.y;
                    VKQ[j + jj][2] += hi.x;
                    VKQ[j + jj][3] += hi.y;
                }
            }
        }
        fattn_mma_q_cp_async_wait<0>();
    }

    // Row sums: over the quad, then over the pair (the shared memory of the last tile's maxima is free again after
    // the pair barrier that follows its P exchange):
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        KQ_sum[h] += __shfl_xor_sync(0xFFFFFFFF, KQ_sum[h], 1);
        KQ_sum[h] += __shfl_xor_sync(0xFFFFFFFF, KQ_sum[h], 2);
    }
    __syncthreads();
    if (lq == 0) {
        red_own[lr + 0] = KQ_sum[0];
        red_own[lr + 8] = KQ_sum[1];
    }
    fattn_mma_q_pf_pair_sync(pair);

    // Write this warp's dimensions of the rows of the pair:
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        float sum = KQ_sum[h] + red_other[lr + 8*h];

        const int r = r0 + 8*h;
        if (r >= rows_valid) {
            continue;
        }
        const float mx = KQ_max[h] / log2e; // natural units, like the other kernels

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
                *(float2 *) (dst + row_dst*D + hw*DH + 8*j + 2*lq) = make_float2(inv*VKQ[j][2*h + 0], inv*VKQ[j][2*h + 1]);
            }
        } else {
            const int64_t row_part = row_dst*nchunks + chunk;
            const float inv = sum > 0.0f ? 1.0f/sum : 0.0f;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                dst_parts[row_part*(D/2) + (hw*DH + 8*j)/2 + lq] = make_half2(inv*VKQ[j][2*h + 0], inv*VKQ[j][2*h + 1]);
            }
            if (hw == 0 && lq == 0) {
                dst_meta[row_part] = make_float2(mx, sum);
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

template <int D, ggml_type type_K, ggml_type type_V, int ncols_tiles, bool gather>
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

    constexpr size_t nbytes_shared = fattn_mma_q_smem_total(D, type_K, type_V, ncols_tiles);
    const auto kernel = flash_attn_ext_mma_q<D, type_K, type_V, ncols_tiles, gather>;
    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shared_memory_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared));
        shared_memory_limit_raised[id] = true;
    }

    const bool mask_vec = mask && mask->nb[1] % 16 == 0 && mask->nb[3] % 16 == 0 && ((uintptr_t) mask->data) % 16 == 0 && ne11 % 8 == 0;

    const int ntiles_tok = (Q->ne[1] + ntok_tile - 1) / ntok_tile;
    const int ntiles_kv  = (ne11 + FATTN_MMA_Q_NBATCH - 1) / FATTN_MMA_Q_NBATCH;

    ggml_cuda_pool_alloc<uint8_t> vis(ctx.pool());
    ggml_cuda_pool_alloc<int>     gather_idx(ctx.pool());
    ggml_cuda_pool_alloc<half>    gather_mask(ctx.pool());
    ggml_cuda_pool_alloc<int>     gather_count(ctx.pool());

    int chunk_tiles;
    int nchunks;
    if constexpr (gather) {
        GGML_ASSERT(mask);
        chunk_tiles = FATTN_MMA_Q_GATHER_CELLS/FATTN_MMA_Q_NBATCH;
        nchunks     = (ne11 + FATTN_MMA_Q_GATHER_CELLS - 1) / FATTN_MMA_Q_GATHER_CELLS;
        const size_t ngroups = size_t(mask->ne[3])*ntiles_tok*nchunks;
        gather_idx.alloc(ngroups*FATTN_MMA_Q_GATHER_CELLS);
        gather_mask.alloc(ngroups*ntok_tile*FATTN_MMA_Q_GATHER_CELLS);
        gather_count.alloc(ngroups);
        ggml_cuda_kernel_launch(flash_attn_ext_mma_q_gather,
            ggml_cuda_kernel_launch_params(dim3(nchunks, ntiles_tok, mask->ne[3]), dim3(256, 1, 1), 0, stream),
            (const half *) mask->data, gather_idx.ptr, gather_mask.ptr, gather_count.ptr,
            int(Q->ne[1]), ne11, int64_t(mask->nb[1]/sizeof(half)), int64_t(mask->nb[3]/sizeof(half)), ntok_tile, int(mask_vec));
    } else {
        if (mask) {
            vis.alloc(size_t(mask->ne[3])*ntiles_tok*ntiles_kv);
            ggml_cuda_kernel_launch(flash_attn_ext_mma_q_visible,
                ggml_cuda_kernel_launch_params(dim3((ntiles_kv + 31)/32, ntiles_tok, mask->ne[3]), dim3(256, 1, 1), 0, stream),
                (const half *) mask->data, vis.ptr, int(Q->ne[1]), ne11, int64_t(mask->nb[1]/sizeof(half)), int64_t(mask->nb[3]/sizeof(half)),
                ntok_tile, int(mask_vec));
        }

        // Split the K/V cells into chunks: enough CUDA blocks to fill the GPU, but not so many that writing and merging
        // the per-chunk results costs more than a few percent of the K/V data.
        const int nunits = ntiles_tok * K->ne[2] * Q->ne[3];
        const int nblocks_target = 2*nsm;
        chunk_tiles = (int64_t(ntiles_kv)*nunits + nblocks_target - 1) / nblocks_target;
        chunk_tiles = std::max(chunk_tiles, 16);
        chunk_tiles = std::min(chunk_tiles, ntiles_kv);
        nchunks = (ntiles_kv + chunk_tiles - 1) / chunk_tiles;
    }

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
        mask ? (const char *) mask->data : nullptr, vis.ptr, gather_idx.ptr, gather_mask.ptr, gather_count.ptr,
        sinks ? (const float *) sinks->data : nullptr,
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

template <int D, ggml_type type_K, ggml_type type_V, int ncols_tiles_max, bool gather>
static void ggml_cuda_flash_attn_ext_mma_q_switch_ncols_tiles(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int ntok_tile, const int nrows) {
    if (nrows <= 16) {
        ggml_cuda_flash_attn_ext_mma_q_launch<D, type_K, type_V, 1, gather>(ctx, dst, ntok_tile);
    } else if (nrows <= 32) {
        if constexpr (ncols_tiles_max >= 2) {
            ggml_cuda_flash_attn_ext_mma_q_launch<D, type_K, type_V, 2, gather>(ctx, dst, ntok_tile);
        }
    } else if (nrows <= 48) {
        if constexpr (ncols_tiles_max >= 3) {
            ggml_cuda_flash_attn_ext_mma_q_launch<D, type_K, type_V, 3, gather>(ctx, dst, ntok_tile);
        }
    } else {
        if constexpr (ncols_tiles_max >= 4) {
            ggml_cuda_flash_attn_ext_mma_q_launch<D, type_K, type_V, 4, gather>(ctx, dst, ntok_tile);
        }
    }
}

template <int D, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_q_pf_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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
    GGML_ASSERT(gqa_ratio <= FATTN_MMA_Q_PF_ROWS);
    const int ntok_tile = FATTN_MMA_Q_PF_ROWS / gqa_ratio;

    constexpr size_t nbytes_shared = fattn_mma_q_pf_smem(D, type_K, type_V);
    static_assert(nbytes_shared <= FATTN_MMA_Q_SMEM_MAX, "shared memory");
    const auto kernel = flash_attn_ext_mma_q_pf<D, type_K, type_V>;
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

    // Split the K/V cells into chunks (at least 2048 cells each) so that the waves of blocks (one block per SM) are
    // full: minimize waves * (tiles per chunk + a per-block overhead of ~16 tiles for Q and the partial results).
    const int nunits = ntiles_tok * K->ne[2] * Q->ne[3];
    const int nchunks_max = std::max(1, std::min(16, ntiles_kv/32));
    int nchunks = 1;
    int64_t best_cost = INT64_MAX;
    for (int n = 1; n <= nchunks_max; ++n) {
        const int64_t waves = (int64_t(nunits)*n + nsm - 1) / nsm;
        const int64_t cost  = waves*((ntiles_kv + n - 1)/n + 16);
        if (cost < best_cost) {
            best_cost = cost;
            nchunks   = n;
        }
    }
    const int chunk_tiles = (ntiles_kv + nchunks - 1) / nchunks;
    nchunks = (ntiles_kv + chunk_tiles - 1) / chunk_tiles;

    const int64_t nrows_dst = Q->ne[1]*Q->ne[2]*Q->ne[3];
    ggml_cuda_pool_alloc<half2>  parts(ctx.pool());
    ggml_cuda_pool_alloc<float2> meta(ctx.pool());
    if (nchunks > 1) {
        parts.alloc(nrows_dst*nchunks*(D/2));
        meta.alloc(nrows_dst*nchunks);
    }

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const dim3 block_dims(WARP_SIZE, FATTN_MMA_Q_PF_NWARPS, 1);
    const dim3 block_nums(ntiles_tok, nchunks*K->ne[2], Q->ne[3]);
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

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_q_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    constexpr int ncols_tiles_max = fattn_mma_q_max_ncols_tiles(D, type_K, type_V);
    static_assert(ncols_tiles_max >= 1, "shared memory");

    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // Large batches (prefill): 128 rows per block, K/V unpacked once per block and tile.
    if (Q->ne[1] > FATTN_MMA_Q_MAX_GATHER_BATCH && gqa_ratio <= FATTN_MMA_Q_PF_ROWS) {
        ggml_cuda_flash_attn_ext_mma_q_pf_launch<D, type_K, type_V>(ctx, dst);
        return;
    }

    GGML_ASSERT(gqa_ratio <= 16*ncols_tiles_max);
    const int ntok_tile = std::min<int64_t>({Q->ne[1], FATTN_MMA_Q_MAX_TOKENS, 16*ncols_tiles_max/gqa_ratio});
    const int nrows     = ntok_tile*gqa_ratio;

    // Several token tiles of a verify batch usually belong to different sequences of a unified KV cache, gather their
    // visible cells. Large batches (prefill) skip masked K/V tiles instead: the gather lists would scale with the batch.
    const bool gather = dst->src[3] && Q->ne[1] > ntok_tile && Q->ne[1] <= FATTN_MMA_Q_MAX_GATHER_BATCH;
    if (gather) {
        ggml_cuda_flash_attn_ext_mma_q_switch_ncols_tiles<D, type_K, type_V, ncols_tiles_max, true>(ctx, dst, ntok_tile, nrows);
    } else {
        ggml_cuda_flash_attn_ext_mma_q_switch_ncols_tiles<D, type_K, type_V, ncols_tiles_max, false>(ctx, dst, ntok_tile, nrows);
    }
}

#define DECL_FATTN_MMA_Q_CASE(D, type_K, type_V)                                       \
    template void ggml_cuda_flash_attn_ext_mma_q_case                                   \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)             \

#define EXTERN_DECL_FATTN_MMA_Q_CASES(type_K, type_V)        \
    extern DECL_FATTN_MMA_Q_CASE( 64, type_K, type_V);       \
    extern DECL_FATTN_MMA_Q_CASE(128, type_K, type_V);       \
    extern DECL_FATTN_MMA_Q_CASE(256, type_K, type_V);       \

#define EXTERN_DECL_FATTN_MMA_Q_CASES_V(type_K)               \
    EXTERN_DECL_FATTN_MMA_Q_CASES(type_K, GGML_TYPE_Q4_0)     \
    EXTERN_DECL_FATTN_MMA_Q_CASES(type_K, GGML_TYPE_Q4_1)     \
    EXTERN_DECL_FATTN_MMA_Q_CASES(type_K, GGML_TYPE_Q5_0)     \
    EXTERN_DECL_FATTN_MMA_Q_CASES(type_K, GGML_TYPE_Q5_1)     \
    EXTERN_DECL_FATTN_MMA_Q_CASES(type_K, GGML_TYPE_Q8_0)     \

EXTERN_DECL_FATTN_MMA_Q_CASES_V(GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_MMA_Q_CASES_V(GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_MMA_Q_CASES_V(GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_MMA_Q_CASES_V(GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_MMA_Q_CASES_V(GGML_TYPE_Q8_0)

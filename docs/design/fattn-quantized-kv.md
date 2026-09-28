# FlashAttention for quantized KV at decode and verify batch sizes

`engine/ggml/src/ggml-cuda/fattn-mma-q.cuh` is a CUDA FlashAttention kernel for quantized K/V (q4_0, q4_1, q5_0,
q5_1, q8_0, K and V types independent) and up to 64 query tokens: decode, speculative verify (MTP n=3 gives
4 tokens per sequence), and verify batches of several sequences in a unified KV cache.
It needs Ampere or newer (cp.async, int8 mma). Head sizes 64, 128, 256 with equal K and V head size.

## Why

Before, quantized K/V at these batch sizes took one of two paths (`fattn.cu`):

- 1-2 tokens: the vector kernel. It runs one block per Q head, so every K/V head is read once per Q head
  (6 times for Qwen3.8-27B), and dequantizes K in a per-thread dot product. q4_0 at 196k depth ran at ~250 GB/s.
- 3+ tokens: the f16 MMA kernel, after converting all of K and V to f16 in a temporary buffer each step. Writing and
  reading the f16 copy made q4_0 slower than f16 KV (pp4 at 64k: 118 vs 154 tok/s).

In a unified KV cache every query also went over all cells of all sequences.

## Design

- **One block** handles up to 8 query tokens times all Q heads of one KV head, flattened to 16-64 rows of
  (token, head) pairs (GQA 6: 1 token = 6 rows, 4 tokens = 24 rows, 8 tokens = 48 rows), and one chunk of K/V cells.
- **K/V are read once**: the raw quantized rows of a 64-cell tile are copied with cp.async (two stages when shared
  memory allows) and used from shared memory:
  - KQ with int8 `mma.m16n8k32`: Q is quantized per 32 values like q8_1, K nibbles/bytes are unpacked straight
    into the B fragments, each 32-value block is scaled by d_K*d_Q (plus m_K*s_Q for q4_1/q5_1).
  - V is dequantized once per tile into an f16 tile, VKQ with f16 `mma.m16n8k16` via `ldmatrix.trans`,
    accumulated in f16 within a tile and in f32 across tiles. Softmax state in f32.
- **Split-K**: the cells are split into fixed-size chunks; each chunk writes an f16 partial result normalized by
  its own row sum plus (max, sum); `flash_attn_ext_mma_q_combine` merges them (and applies attention sinks).
- **Masked cells are skipped**:
  - one token tile (single sequence): a pre-pass marks the 64-cell tiles with any visible cell,
  - several token tiles (unified KV of several sequences): a pre-pass lists, per 2048-cell chunk and token tile,
    the visible cells with their mask values; the kernel gathers them into dense tiles. Each tile then reads only
    the cells of its own sequences, also where the sequences' cells are interleaved at single-cell granularity.
- Tokens are tiled 8 at a time, so with n_q in {1, 2, 4, 8} tokens per sequence no sequence spans two tiles
  (its cells are read once). Two sequences in one tile cost compute on masked rows, not bandwidth.

Dispatch (`ggml_cuda_fattn_mma_q_applies`): NVIDIA, cc >= 8.0, quantized K and V with a compiled instance
(`GGML_CUDA_FA_QUANTS`), no ALiBi, no logit softcap, GQA ratio up to what fits into shared memory,
K/V/Q strides aligned, and at most 64 query tokens. Otherwise the previous kernels are used unchanged.

## Numerics

Q is quantized to 8 bit per 32 values for KQ, the same as the vector kernel does for quantized K. P and V are f16.
`test-backend-ops -o FLASH_ATTN_EXT` passes (all 4027 cases, plus the new cases for GQA 6/8/16, mixed K/V types,
sinks, and unified KV masks).

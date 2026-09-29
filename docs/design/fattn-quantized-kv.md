# FlashAttention for quantized KV (decode, verify and prefill)

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

## Prefill (more than 64 query tokens)

`flash_attn_ext_mma_q_pf` takes the large batches, which before went to the f16 MMA kernel after converting all of
K/V to f16 (a pool buffer of n_kv x 4 KB at 4 KV heads x D 256, and every cell of a unified cache was attended,
including the cells of other sequences):

- **128 (token, head) rows per block** (21 tokens x 6 heads for GQA 6), 16 warps. Two warps share 16 rows: each
  computes int8 KQ for half of the 32 cells of a tile, they exchange row maxima and P (f16) through shared memory
  with a named barrier per pair, and each computes VKQ for half of the dimensions. The f32 accumulators are D/4
  registers per thread, so the kernel runs at 128 registers with 16 warps per SM.
- **K/V are unpacked once per block and tile**: the raw rows of a 32-cell tile are copied with cp.async one tile
  ahead, then K becomes an int8 tile plus float scales (B operand of `mma.m16n8k32` via `ldmatrix`), V an f16 tile
  (B operand of `mma.m16n8k16` via `ldmatrix.trans`, f16 accumulation within the 32-cell tile, f32 across tiles).
- KQ is in log2 units (log2(e) folded into the Q scale before quantizing), so the softmax uses `ex2` directly.
- **Masked tiles**: the visibility pre-pass marks each 64-cell tile per token tile as invisible (skipped),
  masked (mask values applied) or zero-mask (no mask read). In a causal prefill only the diagonal tiles read the
  mask; in a unified cache the other sequences' cells are skipped.
- **Chunks**: the cells are split so that the waves of blocks (one per SM) are full (at most 16 chunks of at least
  2048 cells); the partial results are merged by `flash_attn_ext_mma_q_combine`.

Measured: bench/results/prefill-depth/REPORT.md.

Dispatch (`ggml_cuda_fattn_mma_q_applies`): NVIDIA, cc >= 8.0, quantized K and V with a compiled instance
(`GGML_CUDA_FA_QUANTS`), no ALiBi, no logit softcap, GQA ratio up to what fits into shared memory,
K/V/Q strides aligned; any number of query tokens (more than 64: the prefill kernel).

## Numerics

Q is quantized to 8 bit per 32 values for KQ, the same as the vector kernel does for quantized K. P and V are f16
(VKQ accumulated in f16 within a K/V tile, f32 across tiles). `test-backend-ops -o FLASH_ATTN_EXT` passes (all cases,
plus the cases for GQA 6/8/16, mixed K/V types, sinks, unified KV masks, and prefill batches with causal masks).
The KL divergence against the Q8_0 reference (eval/kld) is in the prefill report.

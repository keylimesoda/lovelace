# Prefill at depth: fattn-mma-q for prefill (feat/prefill-depth)

Engine builds: **base** = main e6171e25b (f16 MMA FA after converting q5_0 K/V to f16), **8ea46e9d0** = fattn-mma-q
(int8 KQ) for every batch size, **final** = d4972020b (new prefill kernel, below). Model GPU-5 (IQ4_XS-3.84bpw),
q5_0 KV, `-ub 1024 -b 2048`, RTX 4090. All numbers from committed scripts; raw outputs in this directory.

## Phase 1: gates for 8ea46e9d0

| gate | result |
|---|---|
| test-backend-ops FLASH_ATTN_EXT, all types (`bench/op_gates.sh`) | 4052/4052 pass, test-gdn-replay pass (op-gates-8ea46e9d0.txt) |
| largest `-c` that starts, production profile balanced.args (`bench/max_ctx.sh`, step 8192) | base 344064, 8ea46e9d0 **393216** (+49152 cells) |
| CUDA0 compute buffer at that `-c` (target ctx; MTP ctx 32 MiB) | base 2177.9 MiB, 8ea46e9d0 **1038.3 MiB** (-1140 MiB) |
| greedy e2e (`bench/greedy_gate.sh`, NO_SPEC=1, 6 x 20k) | seq 1/6, par 0/6 identical; every divergence at a baseline top-1/top-2 margin <= 0.20 nats (numerical flips, no bug signature) (greedy-8ea46e9d0/) |
| llama-bench pp2048 d0/65536/131072/196608 (pp2048-phase1.txt) | base 2897/1810/1350/1053, 8ea46e9d0 2927/1780/1271/1003 |
| depth sweep 126k, 1 lane (`BIN= bench/h2h.sh lovelace 1 sweep`) | base 1725, 8ea46e9d0 1692 tok/s |
| depth sweep 126k, 6 lanes (charter B1 config, earlier sessions resident) | base 1126, 8ea46e9d0 **1678** tok/s |

Note: the charter's 1132 tok/s was measured with 6 lanes (acceptance.sh runs `h2h.sh lovelace 6 sweep`). There the
126k prompt lands behind ~95k cells of earlier sessions in the unified cache, and the base path converts and attends
over all of them. With 1 lane the base engine already reaches 1725. 8ea46e9d0 skips invisible 64-cell tiles, which
fixes the 6-lane case (+49%), but its kernel alone is slower than base at depth (-6% at 131072, -5% at 196608).

Recommendation for 8ea46e9d0: **keep**, superseded by the final kernel (same dispatch, faster).

## Phase 2: kernel changes (engine/ggml/src/ggml-cuda/fattn-mma-q.cuh, docs/design/fattn-quantized-kv.md)

New `flash_attn_ext_mma_q_pf` for batches > 64 tokens:
- 128 (token, head) rows per block (21 tokens x GQA 6), 16 warps at 128 registers; two warps share 16 rows: KQ split
  by cells, row maxima and P exchanged through shared memory (named barrier per pair), VKQ split by dimensions.
- K/V raw rows copied with cp.async one 32-cell tile ahead, unpacked once per block: K -> int8 tile + scales (int8
  `mma.m16n8k32`, Q int8 per 32 values as before), V -> f16 tile (f16 `mma.m16n8k16`, f16 accumulation within a
  32-cell tile, f32 across tiles). KQ in log2 units (ex2). I2FP for the int32->float scaling, q5_0 K unpacked 32
  values at once, qh spread with one multiply.
- Visibility pre-pass now 3-state (invisible / masked / zero mask): causal prefill reads the mask only on the
  diagonal tiles; other sessions' cells are skipped.
- Chunks picked for full waves (1 block/SM), partial results capped at 64 MiB of pool.

Profiling (`bench/fa_ncu.sh`, ncu/): v1 (8 warps x 16 rows, 255 regs) ran at 46% tensor-pipe / 42% issue
utilization, latency-bound with 2 warps per scheduler; the warp-pair split (16 warps) and cheaper KQ scaling/unpack
got the rest. SASS phase split of the 16-warp version: KQ+scaling 44% of stall samples, VKQ 30%, K/V unpack 17%,
softmax 7%. int8 PV was rejected on paper: V scales are per (cell, 32 dims), so P would need a per-block requant
that costs more issue slots than it saves in tensor time.

FA op, one 1024-token causal ubatch, q5_0 (test-backend-ops perf, fa-perf-v1.txt, dev/perf-*.txt), ms:

| KV depth | base | 8ea46e9d0 | v1 (8 warps) | final |
|---:|---:|---:|---:|---:|
| 4096 | 0.80 | 0.76 | 0.64 | 0.59 |
| 32768 | 6.49 | 6.58 | 4.55 | 4.36 |
| 131072 | 25.2 | 26.5 | 18.4 | **16.7** (198 TFLOPS equiv.) |

llama-bench pp2048, tok/s (d0 / 65536 / 131072 / 196608):

| build | d0 | d65536 | d131072 | d196608 |
|---|---:|---:|---:|---:|
| base | 2897 | 1810 | 1350 | 1053 |
| 8ea46e9d0 | 2927 | 1780 | 1271 | 1003 |
| final kernel (v6 = d4972020b minus the pool cap, pp2048-pf-v6.txt) | 3065 | 2162 | 1660 | 1349 |

(pp2048-final.txt: base, 8ea46e9d0 and d4972020b in one hold; queued at the time of writing.)

Depth sweep, cold prefill tok/s (sweep/*.sweep.out; final = d4972020b binaries in bins/pf-v6):

| prompt | base 1 lane | final 1 lane | base 6 lanes | 8ea46e9d0 6 lanes | final 6 lanes |
|---:|---:|---:|---:|---:|---:|
| 5k | 2500 | 2541 | 2610 | 2623 | 2544 |
| 30k | 2363 | 2460 | 2348 | 2446 | 2377 |
| 62k | 2113 | 2248 | 1750 | 2132 | 2132 |
| **126k** | 1725 | **1922** | 1126 | 1678 | **1730** |
| 198k | 1092 | 1291 | 821 | 1057 | 1199 |

Target (>= 1620 at 126k): met in both configurations. Shallow depth: no loss beyond run-to-run noise (5k is 1-2
ubatches and varies +-3%). Compute buffer final: 1038.3 MiB (= 8ea46e9d0), max `-c` 393216 (max-ctx-bufs/).

## Accuracy gate (final build)

- eval/kld engine swap (`LLAMA_LIB=... KLD_EXE=build/engine/lovelace-kld TAG=pf-v6 ONLY=wq-iq4xs-q5_0 ./run_kld.sh`),
  same 24 windows 32k-170k, same Q8_0 base file (kld/): **KL mean 0.01721** (limit 0.01757; baseline 0.01707),
  **top-1 95.65%** (limit 95.6%). PASS. Per bucket KL 0.01898 / 0.01458 / 0.02263 / 0.01209.
- test-backend-ops FLASH_ATTN_EXT, all types: 4093/4093 pass (new prefill, causal and mixed-type cases), test-gdn-replay
  pass (op-gates-pf-final.txt).
- greedy gate vs base (greedy-pf-v6/): seq 1/6, par 1/6 identical, all divergences at baseline margins <= 0.10 nats.

# Batched-verify step profile and fixes (branch feat/batched-verify)

Question: where does the production decode step go for 4-6 concurrent sequences at depth, and what makes it cheaper.
Scripts: `bench/step_profile.sh` (server under nsys, `bench/step_drive.py` drives N concurrent decodes from the replay
corpus, `bench/step_table.py` makes the per-step kernel-class table), `bench/greedy_gate.sh` (greedy identity),
`bench/op_gates.sh`, `bench/op_perf.sh`, `bench/h2h.sh` (now takes `BIN=`).

## Step profile, baseline (production flags, 6 sequences x ~47.5k = 286k unified cells, MTP n=3)

20 steady steps, `step-profile/base/six.table.txt`. Step wall 60.8 ms (4 x ~64k: 53.8 ms, 1 x 61.5k: 29.9 ms).

| class | ms/step | share |
|---|---:|---:|
| target matmul (24 rows, MMQ x24) | 22.2 | 36.6% |
| host gap (GPU idle) | 16.1 | 26.4% |
| target FA (16 layers, gather, 556 us/layer) | 9.8 | 16.1% |
| MTP: 3 draft steps (2.17 each) + catch-up (0.96) | 7.4 | 12.3% |
| target DeltaNet | 2.8 | 4.6% |
| logits D2H (24 x 248k f32) | 1.0 | 1.6% |
| norms + misc | 1.3 | 2.1% |

Findings:
- Matmul at 24 rows streams ~620 GB/s over the 11.2 GB of target weights; the 48x5120 Gated-DeltaNet alpha/beta
  products spent 15 us in stream-k fixups each (96 per step, 1.7 ms).
- FA reads each visible cell once (~720 GB/s): not the problem.
- Each draft step is dominated by the full Q6_K LM head (1.11 ms of 2.17; 995 MB) and CUB top-k per row (0.25 ms).
- Host gaps (CPU sampling with DWARF backtraces, `step-profile/base-cpu`): KQ mask fill 45% of in-gap samples
  (a pass over all 290k cells per sequence, testing 256-bit sequence sets), CPU sampling of the verify rows 13%,
  `seq_rm` 10% (scanned every cell), target graph launch ~1.2 ms. All of it serialized between GPU decodes.
- Draft length at batch 6: n=2 shortens the step to 53.9 ms but acceptance-weighted output drops:
  sum of per-request decode 279 tok/s (n=2) vs 292 (n=3, same build). Keep n=3.

## Changes (commits on feat/batched-verify)

| commit | change | measured |
|---|---|---|
| 5217085a3 | bench tooling (above) | |
| 0020326d9 | single-pass KQ mask, `seq_rm` over the sequence's cells (`seq_pos`) | |
| 9af6a7638 | MMQ stream-k grid capped at 2 K iterations/block; perf cases for the model's verify shapes | 48x5120 @ 24 cols 19.7 -> 9.6 us, @16 18.5 -> 8.1 us; large shapes unchanged (`op-perf/streamk.txt`) |
| f1c2a6d34 | server: verify sampling of all slots in parallel (persistent pool) | gap after catch-up 5.6 -> 3.5 ms |
| 4e31dc347 | CUDA top-k for k <= 32 (register lists + warp tournament) | 151936 x 1, k=20: 85 -> 36 us; 6 rows now 2 launches instead of ~24 |
| 2483f2646 | mask precompute from a compact 64-bit copy of each cell's sequence set + PEXT, vectorized rows | CPU microbench, 286k cells, 6 seqs: 6 rows 1.1-1.4 -> 0.22 ms; 24 rows 1.8-3.6 -> 1.3-1.9 ms (store-bound) |

**Final build (2483f2646), `step-profile/v6`:** six-seq step 60.8 -> **54.0 ms** (-11%): host gap 16.1 -> 9.5 ms,
MTP draft steps 2.17 -> 2.00 ms, target matmul/FA unchanged; single stream at 64k 29.9 -> 29.0 ms (gap 4.3 -> 3.3).
Sum of per-request decode in the six-seq drive: 325 tok/s vs 278 base (+17%).

Step profiles after the intermediate builds (`step-profile/v2..v4`): v4 (all but the last commit) six-seq step
61.4 ms, sum of per-request decode 292 vs 278 tok/s base (+5%). The first mask version was no faster in practice:
Qwen3.5 batches carry 4 M-RoPE positions (the 2D path), and the per-cell 256-bit sets made the pass memory-bound;
2483f2646 fixes both.

## 6-agent replay (`bench/h2h.sh lovelace 6 replay:6:60000:1200`, same script, base vs final build)

| run | turns | errors | agg tok/s full | **agg tok/s steady** | decode tok/s per request | TTFT p50/p90 s | accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| base (9311fa379) | 103 | 0 | 64.94 | **75.87** | 32.7 | 18.5 / 59.3 | 0.54 |
| final (2483f2646) | 114 | 0 | 65.99 | **77.33** (+1.9%) | 40.0 (+22%) | 19.3 / 52.3 | 0.55 |

Per-request decode rate rose 22% and the replay finished 11% more turns, but steady aggregate moved only +1.9%:
in the replay the aggregate is bounded by admission/queueing and prefill (TTFT p50 ~19 s, 6 x ~68k sessions over a
327k-cell pool), not by the decode step alone.

## Correctness

- test-backend-ops MUL_MAT 1297/1297, MUL_MAT_ID 929/929, TOP_K 525/525; test-gdn-replay all passed (`gates/v5.txt`).
- KQ mask: dense path bitwise equal to the generic path (6 and 24 rows, M-RoPE positions, 286k cells) in a
  standalone harness.
- Greedy e2e (`greedy/v6/COMPARE.txt`): not token-identical to base. The base build is not self-consistent either:
  base sequential vs base concurrent diverges in 4/6 requests at the same near-tie tokens (' plugin'/' package',
  ' code'/' plugin', ' today'/' from'), so batch-shape numerics already flip these. Top-2 margins could not be
  reported: with MTP the server returns top_logprobs only for the first token of each verify step
  (drafted tokens come back with logprob 0 and no alternatives). The stream-k cap changes the summation order of small
  matmuls, so small-margin flips are expected from it.

## What's left (by size)

1. Host gaps are still the largest lever: 24-row masks are 14 MB of stores per decode (target + catch-up), plus the
   target graph launch (~1.2 ms). Building the mask on the GPU from compact per-cell data (1.4 MB), or launching the
   target graph before its inputs are known, removes most of that.
2. Draft LM head: 3 x 1.1 ms/step. Agent outputs use 19.5k distinct tokens; the 16k most frequent cover 99.7%
   (the id prefix does not: 32k ids cover 93%). A gathered 16k-row draft head (69 MB VRAM) would save ~3 ms/step.
3. MMQ at 24 rows streams ~620 GB/s vs ~720 GB/s for MMVQ at 4 rows; about 3 ms/step is available in the kernel.
4. For the replay aggregate itself: the admission/prefill side (TTFT), since a faster step barely moves it.

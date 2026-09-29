# Prefill cost with other sessions resident (branch feat/resident-cost)

Problem (sched-v2 report): a cold ~148k prefill ran at 1570-1690 tok/s alone but 880-1000 tok/s while another ~150k
session was resident in the unified KV cache, so prefill cost followed every resident cell, not the session's own.

## 1. Reproduction and attribution

Script: `bench/resident_profile.sh OUT alone=128000 resident=150000,128000` (production server flags: q5_0 KV, MTP,
-ub 1024, 327680 unified cells). One GPU-lock hold per spec. `alone` prefills one 125,640-token prompt (OMP corpus) on
a fresh server; `resident` first prefills a 147,684-token prompt from another session (it stays resident, idle), then
the same 125,640-token prompt. Only the measured prefill is captured by nsys; `bench/resident_table.py` splits the
capture into 1024-token ubatches (target pass + its MTP pass) and sums GPU time per kernel class, H2D uploads and the
host gap (wall time with the GPU idle: graph build, host mask build, sync).

Build: main 76f8ee574. Server-reported prefill of the measured prompt:

| run | prompt tokens | prefill tok/s | ms per 1024-token ubatch |
|---|---|---|---|
| alone | 125,640 | 1743 | 557.2 |
| resident (147,684 other cells) | 125,640 | 960 | 1012.0 |

Mean per ubatch over all 129 ubatches (`before/*.table.txt`, bands by depth there too):

| class | alone ms | resident ms | delta ms | share of delta |
|---|---|---|---|---|
| FA kernel (f16 MMA) | 178.7 | 584.7 | +406.0 | 89% |
| H2D upload (KQ mask) | 12.5 (299 MiB) | 33.8 (843 MiB) | +21.3 | 5% |
| K/V q5_0 -> f16 conversion | 6.6 | 21.4 | +14.9 | 3% |
| host gap (GPU idle; host mask build, graph build) | 20.7 | 32.0 | +11.3 | 2% |
| MTP FA | 15.5 | 16.2 | +0.8 | 0% |
| matmul, DeltaNet, norms, misc, MTP rest | 323.3 | 323.8 | +0.5 | 0% |
| **wall** | **557.2** | **1012.0** | **+454.8** | |

Why every cost follows the resident cells:

- `get_n_kv()` returns the padded highest used cell of the whole cache. The K/V views, the KQ mask
  (1024 rows x n_kv f16) and the f16 conversion all span `[0, n_kv)`.
- The resident session occupies cells `[0, 147.7k)`, the new prompt is placed after it. On main the q5_0 prefill runs
  the f16 MMA kernel; its mask pre-pass (`flash_attn_mask_to_KV_max`) only trims masked tiles at the END of the range,
  so the kernel runs KQ/softmax/PV over all of the other session's cells (fully masked, zero contribution).
- The sparse index-list path (`ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse`) needs the `n_kv_max` op param,
  which only DeepSeek-V4 sets; it is never taken for Qwen3.8. The quantized MMA kernel (fattn-mma-q) is used only for
  <= 64 query rows on main.
- The MTP pass does not grow (its FA runs 15.5 vs 16.2 ms).

So the FA kernel is the dominant cause (89%); mask upload, conversion and host mask build add the rest (11%).

## 2. Fix: attend only the cell window of the ubatch's sequences

`llama_kv_cache::get_kv_window()` (engine/src/llama-kv-cache.cpp) replaces `[0, n_kv)` per ubatch with
`[kv_lo, kv_lo + n_kv)`: the first to one past the last cell that holds any sequence of the ubatch (a word scan of
the per-cell sequence bits from both ends), padded to 256 cells as before. The K/V views start at `kv_lo`
(`get_k/get_v` take it), and the KQ mask (host dense and generic builders, pos buckets) covers the same cells. So
every term of the table above follows the ubatch's own cells, for every FA kernel: the FA kernel range, the f16
conversion, the mask size (host build and upload) and the mask pre-pass.

- Graph reuse: the window start is recorded in the (unused) `op_params[0]` of the mask's input tensor (the host mask,
  or the keys of the GPU-built mask) and is part of `can_reuse_kq_mask()`, so a reused graph never keeps stale view
  offsets.
- Scope: on for LLM_ARCH_QWEN35/QWEN35MOE caches without SWA/ALiBi (the MTP draft cache shares the cells and gets its
  own window). Families that index cells directly (indexer caches, relative position buckets, DeepSeek-V4) keep
  `kv_lo = 0`. Sequence ids >= 64 fall back to the full range. `LLAMA_KV_WINDOW=0` turns it off (A/B).
- It narrows a ubatch whose sequences sit in a compact cell range: a prefill and any session whose cells are
  contiguous. A ubatch holding several sessions gets the union of their ranges (6 decoding sessions: the whole cache);
  there the per-tile skipping of the FA kernels (fattn-mma-q decode gather, `flash_attn_ext_mma_q_pf` for prefill)
  carries the FA cost. Placement is unchanged (first fit from the head): a session whose later turns land above
  another session's cells spans them.
- Rebased onto origin/main 3ce297f40 (feat/prefill-depth and feat/step-costs merged): `llama_kv_mask_args` carries
  `kv_lo`, and the GPU-built mask's keys (`llama_kv_mask_set_keys`) and its M-RoPE fallback check read cells
  `kv_lo + j`. test-kq-mask gains a windowed case: a 64-row prefill of the second of two sessions masked over
  `[39936, 50176)` must equal those columns of the full-range mask, and the device mask must equal the host mask.

Gates on the rebased build (origin/main 3ce297f40 + this branch; `rebased/`):

- test-backend-ops FLASH_ATTN_EXT 4093/4093 and KQ_MASK 22/22 passed, test-gdn-replay `all ReplaySSM checks passed`
  (`rebased/gates-ops.txt`); test-kq-mask `all KQ mask checks passed`, including the windowed case
  (`rebased/gate-kq-mask.txt`).
- greedy gate vs origin/main: **sequential 6/6 identical with MTP on and off**. Concurrent: 0/6 (MTP on) and 3/6 (off);
  the off-run divergences are near-ties (margins 0.005-0.13) and the MTP-on ones hit the same tokens as main-vs-main
  runs (' package'/' plugin' at 12, ' a'/' the' at 3); concurrent batching is not a gate (not deterministic on main).

Gates on the pre-rebase build (896206c36 vs main 76f8ee574):

- `bench/op_gates.sh gates-ops.txt <bin> FLASH_ATTN_EXT`: test-backend-ops FLASH_ATTN_EXT 4052/4052 passed (no ggml op
  is touched), test-gdn-replay `all ReplaySSM checks passed`.
- `bench/greedy_gate.sh` (six sessions of ~20k in one unified cache, so each later session sits above the earlier
  ones and gets a window with kv_lo > 0): **sequential 6/6 identical with MTP on (greedy-mtp) and off
  (greedy-nospec)**. Concurrent mode: 2/6 identical in both; every divergence is at a near-tie (baseline top-1/top-2
  margin 0.016-0.15 where reported) and concurrent batching is not deterministic between runs on main either (the
  earlier main gates in bench/results/greedy report 0-1/6, the same ' a'/' the' tie at token 3).

## 3. After

Same script and prompts (`after/*.drive.out`, `after/*.table.txt`):

| run | main tok/s | main ms/ubatch | window tok/s | window ms/ubatch |
|---|---|---|---|---|
| alone | 1743 | 557.2 | 1743 | 557.2 |
| resident (147,684 other cells) | 960 | 1012.0 | **1741** | **558.2** |

With the other session resident the prefill is now within 0.2% of alone (target: within 10%). Per ubatch the resident
run matches alone class by class: FA kernel 179.4 vs 178.7 ms, H2D 300 vs 299 MiB (12.5 ms), conversion 6.6 ms,
host gap 20.7 ms.

Mixed: the first session generates (streaming) while the second prompt is prefilled (`mixed=150000,128000+decode`).
The capture has 267 segments for 129 prefill ubatches, so the server ran the other session's decode as separate
decode calls between prompt chunks [INFERENCE from the segment count], and each call gets its own window:

| run | main | window |
|---|---|---|
| mixed: prefill tok/s | 934 | **1637** |
| mixed: concurrent decode, stream events/s | 3.22 | **5.07** |

No regression alone, `bench/pp_depth.sh` (llama-bench, one sequence, so the window equals `[0, n_kv)`), tok/s:

| build | pp2048 | pp2048 @ d65536 | pp2048 @ d131072 |
|---|---|---|---|
| main 76f8ee574 | 3006 ± 52 | 1872 ± 24 | 1356 ± 9 |
| window | 2998 ± 58 | 1860 ± 3 | 1350 ± 7 |

(within run-to-run noise: the code path is the same for a single sequence).

### On the rebased build (origin/main with the prefill kernel and the GPU mask)

feat/prefill-depth's prefill kernel already skips 64-cell tiles that no row of a block sees, so origin/main no longer
runs PV over the other session; what is left is the tile-visibility pre-pass and the per-tile work over the full
range. Same script (`rebased/main`, `rebased/window`):

| run | origin/main tok/s | ms/ubatch | window tok/s | ms/ubatch |
|---|---|---|---|---|
| alone | - | - | 2001 | 488.3 |
| resident (147,684 other cells) | 1901 | 511.2 | **2009** | **486.4** |
| mixed: prefill tok/s | 1764 | 267.2 | **1862** | 253.2 |
| mixed: concurrent decode events/s | 5.53 | | **5.84** | |

(alone is the same code path on both builds: one sequence, window = `[0, n_kv)`.) Resident per ubatch, origin/main vs
window: FA kernel 146.3 vs 123.4 ms, FA mask pre-pass 7.5 vs 2.6 ms, H2D 90 vs 73 MiB; the window run equals alone
class by class. So with the window the resident prefill is at alone speed (2009 vs 2001 tok/s), +5.7% over
origin/main; the mixed prefill gains 5.6% and the concurrent decode 5.6%.

pp2048 at depth (`rebased/pp-depth.txt`), tok/s: origin/main 3083 ± 21 / 2198 ± 11 / 1698 ± 4 vs window
3084 ± 19 / 2199 ± 7 / 1699 ± 6 at d0 / d65536 / d131072 (no change alone).

## Commands

```sh
bench/resident_profile.sh bench/results/resident-cost/<dir> alone=128000 resident=150000,128000 mixed=150000,128000+decode   # BIN=<bin dir>
bench/op_gates.sh OUT <bin> FLASH_ATTN_EXT KQ_MASK
deploy/gpu-lock.sh docker run --rm --gpus all -v <bin>:/e lovelace-engine-runtime:cuda13.1 /e/test-kq-mask
bench/greedy_gate.sh OUT <main bin> <branch bin>; NO_SPEC=1 bench/greedy_gate.sh OUT2 <main bin> <branch bin>
DEPTHS=0,65536,131072 bench/pp_depth.sh OUT <main bin> <branch bin>
```

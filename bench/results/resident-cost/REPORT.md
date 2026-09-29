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

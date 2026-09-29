# Bake-off session-store (steady state = requests finishing after t=360s)

| run | fit | turns | errors | agg_tps_full | agg_tps_steady | ttft_p50 | ttft_p90 | decode_tps_mean | cache_hit | recompute_p50 | recompute_p90 | depth_p50 | accept | peak_mib | mean_w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sess-6a-150k | ? | 29 | 0 | 21.25 | 25.00 | 82.11 | 731.92 | 52.83 | 0.73 | 3250 | 149229 | 150292 | 0.51 | nan | nan |
| sess-6a-60k | ? | 111 | 0 | 65.73 | 76.77 | 18.38 | 41.39 | 35.27 | 0.93 | 1420 | 4558 | 68211 | 0.55 | nan | nan |

## sess-6a-150k server log errors (2 lines, first 5)

```
0.00.621.301 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
0.05.388.131 I srv    load_model: KV budget: 327680 cells shared by 6 slots, reserving up to 8192 generated tokens per request, headroom 256 cells
```

## sess-6a-60k server log errors (2 lines, first 5)

```
0.00.752.963 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
0.05.771.825 I srv    load_model: KV budget: 327680 cells shared by 6 slots, reserving up to 8192 generated tokens per request, headroom 256 cells
```

## Session store (branch feat/session-store, commits 6629d8a39 + 6fc82a517)

Flags: `--session-ram 30720 --session-dir /sessions` (container-local dir), otherwise the run_replay.sh profile.
Baseline: bench/results/main-327k (replay-6a, MIN_DEPTH 60000): steady 78.4 tok/s, TTFT p50/p90 19.4/48.4 s,
cache hit 0.93, recompute p50/p90 1367/4558.

| metric | sess-6a-60k | sess-6a-150k | smoke 2a-150k |
|---|---|---|---|
| errors / sessions deleted | 0 / 0 | 0 / 0 | 0 / 0 |
| parks (free, host ms p50/p90) | 101 (12.5/15.9) | 23 (24.2/35.3) | 12 (20.3/23.2) |
| restores | 97 | 19 | 11 |
| restore MiB p50/p90 | 1840/2293 | 3636/3800 | 4244/4332 |
| restore device ms p50/p90 | 71/103 | 67/297 | 417/424 (sum of per-delta spans) |
| restore wall ms p50/p90 (issue to done) | 1140/1764 | 1627/1847 | 183/189 |
| restore issue (host) ms p50/p90 | 152/282 | 238/299 | 43/58 |
| prefetches / hits | 1 / 1 | 0 / 0 | - |
| demotions / promotions (disk) | 0 / 0 | 35 / 34 (~5 GB in 0.8 s avg) | 0 / 0 |
| recompute p50/p90, first contact excluded | 1366/4173 | 720/4091 | 1251/3297 |

The wall time of a restore under load is dominated by queueing behind the compute stream (the copies start after the
graphs already queued, and the recurrent snapshot is copied in order with them), not by PCIe (device time 70-100 ms for
~2 GB). The 150k run is dominated by first-contact prefills (6 x 150k tokens at ~530 tok/s); in 20 minutes sessions reached
158k, not 200k.

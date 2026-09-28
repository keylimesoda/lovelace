# Bake-off main-327k (steady state = requests finishing after t=360s)

| run | fit | turns | errors | agg_tps_full | agg_tps_steady | ttft_p50 | ttft_p90 | decode_tps_mean | cache_hit | recompute_p50 | recompute_p90 | depth_p50 | accept | peak_mib | mean_w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| replay-6a | ? | 104 | 0 | 67.00 | 78.38 | 19.40 | 48.35 | 32.77 | 0.93 | 1367 | 4558 | 67595 | 0.55 | nan | nan |

## replay-6a server log errors (2 lines, first 5)

```
0.00.682.399 W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
0.07.942.166 I srv    load_model: KV budget: 327680 cells shared by 6 slots, reserving up to 8192 generated tokens per request, headroom 256 cells
```

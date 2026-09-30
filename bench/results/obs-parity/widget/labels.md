# omarchy-ninfer widget against lovelace

Model.js: `/home/ric/Work/omarchy-ninfer/Model.js`; feed: `/home/ric/Work/lovelace-wt/obs/bench/results/obs-parity/widget/info-2.txt`

| label | value |
|---|---|
| modelName | Qwen 3.8 27B |
| contextLabel | 262,144 tok |
| kvLabel | RK4V4 E8 |
| specLabel | DFLASH2 · 7 draft |
| gpuLabel | RTX 4090 · 23.6/24 GB |
| prefixHitLabel | 83% |
| specAcceptLabel | 45% accepted |
| requestsLabel | Idle |
| generatedLabel | 1.5k tokens |
| ctxDeviceLabel | 1.7k / 376.8k tok |
| ctxDeviceFraction | 0.004529870074728261 |
| ctxHostLabel | 0.6 / 24 GiB |
| ctxStateLabel | dev 4 · host 4 |
| ctxPressureLabel | Stable |
| tps (Generation) | 129 tok/s |
| tps (Prefill) | 0.0 tok/s |
| watts (power_watts) | 357 W |
| uptime (epoch) | 44s |

| feed key | value |
|---|---|
| state | ready |
| model_id | qwen3.8-27b |
| model_name | Qwen 3.8 27B |
| context_window | 262144 |
| kv_dtype | rk4v4-e8 |
| spec | dflash2 |
| draft_tokens | 7 |
| vision | off |
| endpoint | 127.0.0.1:8080 |
| web | http://127.0.0.1:8080/ |
| image | ninfer-4090:hybrid |
| gpu_name | RTX 4090 |
| vram_used | 25368199168 |
| vram_total | 25757220864 |
| power_watts | 356.528 |
| epoch | 1790746290 |
| prompt_tokens | 237 |
| prompt_seconds | 0.716967 |
| predicted_tokens | 1476 |
| predicted_seconds | 14.7316 |
| requests_processing | 0 |
| requests_deferred | 0 |
| requests_total | 6 |
| prefix_hit_tokens | 1143 |
| draft_total | 1876 |
| draft_accepted | 838 |
| ctx_device_tokens | 1707 |
| ctx_device_capacity | 376832 |
| ctx_host_bytes | 682753568 |
| ctx_host_capacity | 25769803776 |
| ctx_state_device | 4 |
| ctx_state_host | 4 |
| ctx_state_host_capacity | 0 |
| ctx_dropped | 0 |
| ctx_degraded | 0 |
| ctx_evicted | 0 |
| ctx_spill | 0 |

## Checks: PASS

none failed

## omarchy-ninfer checkout

    HEAD e72d47c6da79189d16f259601235766e0a59445e
    working tree NOT clean (uncommitted changes of the checkout itself; what ran is hashed below):
       M BarWidget.qml
       M Model.js
       M ninfer-qwen
    203cc318aec9086a6c0b8e4d606a2c8951a1c6a972a52fe5468f6fdd0128159b  Model.js
    cab79bb792e339e46c9fafe2172ec875d5c0538417745032dc2a234ea53f0c04  ninfer-qwen
    unchanged during the check

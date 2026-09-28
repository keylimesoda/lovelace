# Main gate report

## Build

Build step skipped as directed by Main: the stale CMake configure was fixed and all requested targets were rebuilt at `engine/build-sm89/bin`. This gate used those binaries. `build.log` is the earlier pre-fix build attempt and is retained only as historical raw output.

## Correctness

All backend-op commands targeted CUDA0; CPU backend was skipped by the test runner.

| Suite | Result |
|---|---:|
| `FLASH_ATTN_EXT` | 4052/4052 passed |
| `MUL_MAT` | 1297/1297 passed |
| `GATED_DELTA_NET` | 36/36 passed |
| `GATED_DELTA_NET_RR` | 118/118 passed |
| `SSM_CONV` | 45/45 passed |
| `SSM_CONV_FOLD` | 56/56 passed |
| `RMS_NORM` | 51/51 passed |
| `test-gdn-replay` | All ReplaySSM checks passed |
| `test-llama-archs -a qwen35` | 3/3 passed |
| `test-recurrent-state-rollback` | Passed with generated tiny `qwen35-dense.gguf` fixture (`-ngl 0`; CPU) |

## Performance

Model: `Qwen3.8-27B-IQ4_XS-3.84bpw.gguf`; `-ngl 99 -fa 1 -ub 1024 -p 1,4,24 -n 32 -d 0,65536,196608 -r 3`. Speeds are tokens/s (mean ± standard deviation).

### KV q5_0/q5_0

| Test | t/s |
|---|---:|
| pp1 | 55.80 ± 7.23 |
| pp4 | 210.44 ± 20.23 |
| pp24 | 886.90 ± 96.74 |
| tg32 | 60.20 ± 0.04 |
| pp1 @ d65536 | 48.55 ± 5.05 |
| pp4 @ d65536 | 185.80 ± 14.48 |
| pp24 @ d65536 | 698.24 ± 43.60 |
| tg32 @ d65536 | 54.32 ± 0.67 |
| pp1 @ d196608 | 41.98 ± 2.26 |
| pp4 @ d196608 | 159.31 ± 12.21 |
| pp24 @ d196608 | 502.27 ± 34.20 |
| tg32 @ d196608 | 45.89 ± 0.25 |

### KV q4_0/q4_0

| Test | t/s |
|---|---:|
| pp1 | 50.55 ± 6.73 |
| pp4 | 210.02 ± 16.06 |
| pp24 | 876.04 ± 105.54 |
| tg32 | 60.26 ± 0.24 |
| pp1 @ d65536 | 50.33 ± 3.25 |
| pp4 @ d65536 | 184.88 ± 20.50 |
| pp24 @ d65536 | 724.70 ± 47.16 |
| tg32 @ d65536 | 55.38 ± 0.68 |
| pp1 @ d196608 | 44.02 ± 2.96 |
| pp4 @ d196608 | 163.26 ± 16.44 |
| pp24 @ d196608 | 526.60 ± 10.65 |
| tg32 @ d196608 | 47.77 ± 0.25 |

## Fit

`/health` returned HTTP 200 first at **C = 262144**. The preceding capacities 393216, 360448, 327680, and 294912 did not reach healthy startup. `nvidia-smi` reported **23349 MiB memory used** after the winner started. The server used the requested q5_0 KV and draft-KV configuration, six unified slots, and 32 GiB prompt cache. The server was stopped after replay.

## Smoke and telemetry

Six concurrent chat completions used the README material repeated into a 2,643-token prompt (reported by the server), `max_tokens=256`, and `temperature=0`. The initial reasoning-enabled trial spent the output limit on reasoning without user-visible content; the successful six-request smoke used `chat_template_kwargs: {"enable_thinking": false}` to get concise summaries within the same 256-token limit. All six returned coherent summary text. Their first 200 characters were:

```text
1. **System Overview**\nLovelace is a llama.cpp-based serving engine for the Qwen3.8-27B model. It features a replay capture proxy for client traffic, a KV-budget scheduler that parks sessions in pinned h
2. **System Overview**\nLovelace is a llama.cpp-based serving engine for the Qwen3.8-27B model. It features a replay capture proxy for client traffic, a KV-budget scheduler that parks sessions in pinned h
3. **System Overview**\nLovelace is a llama.cpp-based serving engine for the Qwen3.8-27B model. It features a replay capture proxy for client traffic, a KV-budget scheduler that parks sessions in pinned h
4. **System Overview**\nLovelace is a llama.cpp-based serving engine for the Qwen3.8-27B model. It features a replay capture proxy for client traffic, a KV-budget scheduler that parks sessions in pinned h
5. **System Overview**\nLovelace is a llama.cpp-based serving engine for the Qwen3.8-27B model. It features a replay capture proxy for client traffic, a KV-budget scheduler that parks sessions in pinned h
6. **System Overview**\nLovelace is a llama.cpp-based serving engine for the Qwen3.8-27B model. It features a replay capture proxy for client traffic, a KV-budget scheduler that parks sessions in pinned h
```

The final server log has no error-level (`E`) records. It does contain warnings for the explicit `-ngl 99` memory-fit attempt and host prompt-cache eviction while parking sessions.

`GET /telemetry` snapshot (captured at 61.402 s uptime after the initial reasoning-only diagnostics; scheduler was idle):

```json
{
  "scheduler": {
    "slots_total": 6,
    "slots_busy": 0,
    "slots_idle": 6,
    "slots_processing_prompt": 0,
    "slots_generating": 0,
    "queue_deferred": 0,
    "kv_budget": {
      "enabled": true,
      "queue_waiting": 0,
      "slots_preempted": 0,
      "gen_reserve": 8192,
      "headroom": 256,
      "prefill_chunk_gen": 1024,
      "timeslice": 0.0,
      "totals": {
        "parks": 0,
        "park_drops": 0,
        "unparks": 0,
        "preempts": 0,
        "resumes": 0,
        "grows": 0,
        "park_bytes": 0,
        "unpark_bytes": 0,
        "park_seconds": 0.0,
        "unpark_seconds": 0.0,
        "park_host_seconds": 0.0,
        "unpark_host_seconds": 0.0,
        "park_gbps": 0.0,
        "unpark_gbps": 0.0,
        "waited_slot": 0,
        "waited_kv": 0,
        "rollback_fallbacks": 0,
        "recompute_tokens": 0,
        "kv_overflows": 0
      }
    },
    "totals": {
      "prompt_tokens": 16122,
      "prompt_tokens_cached": 16074,
      "predicted_tokens": 3072,
      "prompt_seconds": 6.992918,
      "predicted_seconds": 54.284159,
      "n_decode": 184,
      "n_busy_slots_per_decode": 5.635869565217392,
      "drafted": 2880,
      "accepted": 2087,
      "requests_started": 12,
      "requests_done": 12,
      "requests_rejected": 0
    },
    "loop": {
      "window_seconds": 10.049176,
      "phase": "idle",
      "seconds": {"prefill": 0.03965, "decode": 3.418279, "host": 0.547923, "idle": 6.043324},
      "fraction": {"prefill": 0.003946, "decode": 0.340155, "host": 0.054524, "idle": 0.601375}
    },
    "loop_totals_seconds": {"prefill": 7.005802, "decode": 6.957629, "host": 1.075689, "idle": 46.370272}
  },
  "memory": {
    "kv": {
      "cells_total": 262144,
      "cells_used": 17628,
      "cells_reserved": 17628,
      "unified": true,
      "n_ctx_slot": 262144,
      "cache_type_k": "q5_0",
      "cache_type_v": "q5_0"
    },
    "prompt_cache": {
      "enabled": true,
      "bytes_used": 0,
      "bytes_limit": 34359738368,
      "tokens": 0,
      "tokens_limit": 262144,
      "entries": 0
    },
    "vram": {
      "devices": [{
        "name": "CUDA0",
        "description": "NVIDIA GeForce RTX 4090",
        "total_bytes": 25271402496,
        "free_bytes": 788725760,
        "model_bytes": 12396633088,
        "context_bytes": 6932316160,
        "compute_bytes": 1780073216,
        "unaccounted_bytes": 3373654272
      }],
      "host": {"model_bytes": 675430400, "context_bytes": 0, "compute_bytes": 580260160}
    }
  }
}
```

## Replay: `replay-6a`

The requested 1,200-second, six-agent run completed with 103 successful requests, six cancellations at the duration boundary (`--no-drain`), and zero errors/timeouts. Full per-request output is in `replay-6a/requests.jsonl`.

Summary block:

```text
label                        replay-6a
turns (ok/total)             103/109  status={'ok': 103, 'cancelled': 6}
errors+timeouts              0
active span                  1200.0 s (wall 1200.0 s)
aggregate output tok/s       68.63
mean decode tok/s per agent  40.73
decode tok/s p10/p50/p90     19.12 / 41.22 / 64.70
TTFT s p50/p90/p99           25.49 / 77.96 / 247.59
e2e s p50/p90                42.03 / 132.14
prompt depth p50/p90/max     67690 / 83915 / 99969
cache hit ratio              0.919 (103 reqs reported)
prefill tok/s p50            509
draft acceptance             0.539
------------------------------------------------------------------------
agent  turns   ok  out tok  decode tok/s  ttft p50  sessions
0         23   22    10965         44.29     25.54  2026-09-26T09-54-32_01a0dd23
1         20   19    16247         43.07     21.85  2026-09-26T09-53-08_01a0dd22
2         11   10    18161         31.38     23.34  2026-09-25T05-26-18_01a0d707
3         21   20     8048         41.20     29.36  2026-09-27T08-42-53_01a0e208
4         16   15    15881         44.41     25.25  2026-09-25T07-09-44_01a0d766,2026-09-26T07-38-39_01a0dca6
5         18   17    13053         40.02     27.65  2026-09-27T08-18-54_01a0e1f2
========================================================================
```

`analyze.py` table:
Analyzer heading: `# Bake-off main-gate (steady state = requests finishing after t=360s)`; the requested `replay-6a/warmup_s` override is 180 seconds and is used for this row's summary.

| run | fit | turns | errors | agg_tps_full | agg_tps_steady | ttft_p50 | ttft_p90 | decode_tps_mean | cache_hit | recompute_p50 | recompute_p90 | depth_p50 | accept | peak_mib | mean_w |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| replay-6a | ? | 109 | 0 | 68.63 | 80.53 | 25.49 | 77.96 | 42.04 | 0.93 | 1420 | 4558 | 68058 | 0.54 | nan | nan |

`peak_mib` and `mean_w` are NaN because this replay did not produce a `gpu.csv` sampler file. The server was stopped after the run.

## Raw logs and artifacts

All files are under `~/Work/lovelace/bench/results/main-gate/`: correctness logs (`test-*.log`), `bench-q5_0.log`, `bench-q4_0.log`, `server-fit.log` and per-capacity logs, `server.log`, smoke outputs, `telemetry.json`, `replay-run.log`, `replay-6a/requests.jsonl`, `replay-6a/summary.json`, and `analyze.log`.
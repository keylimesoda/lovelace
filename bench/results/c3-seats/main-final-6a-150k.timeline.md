# Seat timeline: 6a-150k

window 180-1200 s (1020 s), 100 turns ending in it, 67655 tokens (66.3 t/s), median depth 161.5k, 376832 cells -> 2 seats; client/server clock offset 532.03 s from 118 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 1166 | 57.2% |
| prefill_cold | 324 | 15.9% |
| prefill_delta | 190 | 9.3% |
| restore | 47 | 2.3% |
| idle_q_disk | 36 | 1.8% |
| idle_q_other | 276 | 13.5% |
| idle_no_demand | 0 | 0.0% |
| (extra tasks beyond 2 seats) | 0 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 324 | 31.8% | 1246 | 3.8 |
| delta_prefill | 72 | 7.1% | 2342 | 32.5 |
| decode_only | 533 | 52.3% | 63452 | 118.9 |
| restore_only | 8 | 0.8% | 0 | 0.0 |
| idle_queue | 82 | 8.1% | 7 | 0.1 |
| idle | 0 | 0.0% | 0 | 0.0 |

decode-only steps: 1.73 decoding tasks on average

cold prefills launched in the window: 4 (575259 tokens, 325 s)
disk promotions: 556 (2863 GiB, 465 s of reads); demotions (written): 97; parks that synced first: 1 (0.7 s)
queue wait (arrival -> admission): p50 31.6 s, p90 106.9 s, max 182.5 s
restore after admission: 95, mean 494 ms, total 47.0 s

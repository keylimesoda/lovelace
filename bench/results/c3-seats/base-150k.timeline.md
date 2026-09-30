# Seat timeline: base-150k

window 180-1200 s (1020 s), 99 turns ending in it, 66160 tokens (64.9 t/s), median depth 161.5k, 376832 cells -> 2 seats; client/server clock offset 828.32 s from 116 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 986 | 48.3% |
| prefill_cold | 68 | 3.3% |
| prefill_delta | 117 | 5.7% |
| restore | 46 | 2.3% |
| idle_q_disk | 48 | 2.4% |
| idle_q_other | 776 | 38.0% |
| idle_no_demand | 0 | 0.0% |
| (extra tasks beyond 2 seats) | 0 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 68 | 6.6% | 163 | 2.4 |
| delta_prefill | 78 | 7.6% | 2685 | 34.5 |
| decode_only | 550 | 53.9% | 63016 | 114.6 |
| restore_only | 10 | 1.0% | 0 | 0.0 |
| idle_queue | 315 | 30.9% | 15 | 0.0 |
| idle | 0 | 0.0% | 0 | 0.0 |

decode-only steps: 1.62 decoding tasks on average

cold prefills launched in the window: 1 (129471 tokens, 68 s)
disk promotions: 498 (2582 GiB, 472 s of reads); demotions (written): 97; parks that synced first: 0 (0.0 s)
queue wait (arrival -> admission): p50 32.1 s, p90 93.3 s, max 274.7 s
restore after admission: 97, mean 476 ms, total 46.2 s

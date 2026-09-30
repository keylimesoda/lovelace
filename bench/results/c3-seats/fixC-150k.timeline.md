# Seat timeline: fixC-150k

window 180-1200 s (1020 s), 120 turns ending in it, 80737 tokens (79.2 t/s), median depth 162.1k, 376832 cells -> 2 seats; client/server clock offset 888.49 s from 138 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 1588 | 77.8% |
| prefill_cold | 95 | 4.7% |
| prefill_delta | 150 | 7.3% |
| restore | 65 | 3.2% |
| idle_q_disk | 14 | 0.7% |
| idle_q_other | 129 | 6.3% |
| idle_no_demand | 0 | 0.0% |
| (extra tasks beyond 2 seats) | 0 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 95 | 9.3% | 533 | 5.6 |
| delta_prefill | 142 | 14.0% | 3735 | 26.2 |
| decode_only | 748 | 73.3% | 76121 | 101.8 |
| restore_only | 2 | 0.2% | 0 | 0.0 |
| idle_queue | 33 | 3.2% | 2 | 0.1 |
| idle | 0 | 0.0% | 170 | 3400.0 |

decode-only steps: 1.82 decoding tasks on average

cold prefills launched in the window: 1 (129471 tokens, 96 s)
disk promotions: 299 (1271 GiB, 290 s of reads); demotions (written): 108; parks that synced first: 1 (0.3 s)
queue wait (arrival -> admission): p50 28.4 s, p90 59.4 s, max 131.4 s
restore after admission: 120, mean 542 ms, total 65.1 s

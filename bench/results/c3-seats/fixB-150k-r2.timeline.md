# Seat timeline: fixB-150k-r2

window 180-1200 s (1020 s), 138 turns ending in it, 93435 tokens (91.6 t/s), median depth 162.3k, 376832 cells -> 2 seats; client/server clock offset 900.80 s from 150 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 1594 | 78.1% |
| prefill_cold | 73 | 3.6% |
| prefill_delta | 181 | 8.9% |
| restore | 75 | 3.7% |
| idle_q_disk | 10 | 0.5% |
| idle_q_other | 107 | 5.2% |
| idle_no_demand | 0 | 0.0% |
| (extra tasks beyond 2 seats) | 0 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 73 | 7.2% | 624 | 8.6 |
| delta_prefill | 158 | 15.5% | 4612 | 29.2 |
| decode_only | 771 | 75.6% | 84240 | 109.2 |
| restore_only | 7 | 0.7% | 0 | 0.0 |
| idle_queue | 11 | 1.1% | 1 | 0.1 |
| idle | 0 | 0.0% | 0 | 0.0 |

decode-only steps: 1.81 decoding tasks on average

cold prefills launched in the window: 1 (129471 tokens, 73 s)
disk promotions: 278 (1199 GiB, 240 s of reads); demotions (written): 121; parks that synced first: 8 (12.1 s)
queue wait (arrival -> admission): p50 24.7 s, p90 42.6 s, max 122.9 s
restore after admission: 138, mean 546 ms, total 75.3 s

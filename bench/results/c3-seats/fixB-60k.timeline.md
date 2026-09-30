# Seat timeline: fixB-60k

window 180-1200 s (1020 s), 181 turns ending in it, 139801 tokens (137.1 t/s), median depth 75.1k, 376832 cells -> 4 seats; client/server clock offset 6.10 s from 185 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 3513 | 86.1% |
| prefill_cold | 133 | 3.3% |
| prefill_delta | 212 | 5.2% |
| restore | 59 | 1.5% |
| idle_q_disk | 0 | 0.0% |
| idle_q_other | 156 | 3.8% |
| idle_no_demand | 7 | 0.2% |
| (extra tasks beyond 4 seats) | 279 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 141 | 13.8% | 2612 | 18.5 |
| delta_prefill | 219 | 21.4% | 17445 | 79.8 |
| decode_only | 660 | 64.7% | 118569 | 179.5 |
| restore_only | 0 | 0.0% | 0 | 0.0 |
| idle_queue | 0 | 0.0% | 0 | 0.0 |
| idle | 0 | 0.0% | 0 | 0.0 |

decode-only steps: 3.96 decoding tasks on average

cold prefills launched in the window: 4 (247516 tokens, 139 s)
disk promotions: 0 (0 GiB, 0 s of reads); demotions (written): 3; parks that synced first: 0 (0.0 s)
queue wait (arrival -> admission): p50 7.9 s, p90 18.5 s, max 49.7 s
restore after admission: 135, mean 496 ms, total 66.9 s

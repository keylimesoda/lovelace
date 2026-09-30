# Seat timeline: fixB-150k

window 180-1200 s (1020 s), 137 turns ending in it, 94993 tokens (93.1 t/s), median depth 162.9k, 376832 cells -> 2 seats; client/server clock offset 852.09 s from 152 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 1638 | 80.3% |
| prefill_cold | 75 | 3.7% |
| prefill_delta | 154 | 7.5% |
| restore | 64 | 3.2% |
| idle_q_disk | 4 | 0.2% |
| idle_q_other | 104 | 5.1% |
| idle_no_demand | 0 | 0.0% |
| (extra tasks beyond 2 seats) | 0 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 75 | 7.4% | 315 | 4.2 |
| delta_prefill | 144 | 14.1% | 5138 | 35.7 |
| decode_only | 800 | 78.4% | 88735 | 110.9 |
| restore_only | 1 | 0.1% | 0 | 0.2 |
| idle_queue | 0 | 0.0% | 0 | 0.0 |
| idle | 0 | 0.0% | 0 | 0.0 |

decode-only steps: 1.80 decoding tasks on average

cold prefills launched in the window: 1 (129471 tokens, 75 s)
disk promotions: 304 (1329 GiB, 243 s of reads); demotions (written): 118; parks that synced first: 11 (48.3 s)
queue wait (arrival -> admission): p50 24.0 s, p90 48.3 s, max 103.0 s
restore after admission: 138, mean 469 ms, total 64.8 s

# Seat timeline: fixD-150k

window 180-1200 s (1020 s), 128 turns ending in it, 87522 tokens (85.8 t/s), median depth 162.8k, 376832 cells -> 2 seats; client/server clock offset 947.56 s from 147 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 1679 | 82.3% |
| prefill_cold | 85 | 4.2% |
| prefill_delta | 169 | 8.3% |
| restore | 64 | 3.1% |
| idle_q_disk | 0 | 0.0% |
| idle_q_other | 43 | 2.1% |
| idle_no_demand | 0 | 0.0% |
| (extra tasks beyond 2 seats) | 0 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 85 | 8.3% | 370 | 4.4 |
| delta_prefill | 166 | 16.3% | 5639 | 34.0 |
| decode_only | 769 | 75.4% | 80594 | 104.8 |
| restore_only | 0 | 0.0% | 0 | 0.0 |
| idle_queue | 0 | 0.0% | 0 | 0.0 |
| idle | 0 | 0.0% | 0 | 0.0 |

decode-only steps: 1.86 decoding tasks on average

cold prefills launched in the window: 1 (129471 tokens, 85 s)
disk promotions: 117 (517 GiB, 134 s of reads); demotions (written): 103; parks that synced first: 0 (0.0 s)
queue wait (arrival -> admission): p50 27.1 s, p90 46.0 s, max 137.7 s
restore after admission: 127, mean 506 ms, total 64.3 s

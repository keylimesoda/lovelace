# Seat timeline: fixD-150k-r2

window 180-1200 s (1020 s), 134 turns ending in it, 90161 tokens (88.4 t/s), median depth 163.3k, 376832 cells -> 2 seats; client/server clock offset 845.23 s from 157 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 1624 | 79.6% |
| prefill_cold | 78 | 3.8% |
| prefill_delta | 220 | 10.8% |
| restore | 70 | 3.4% |
| idle_q_disk | 5 | 0.3% |
| idle_q_other | 42 | 2.1% |
| idle_no_demand | 0 | 0.0% |
| (extra tasks beyond 2 seats) | 0 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 78 | 7.7% | 381 | 4.9 |
| delta_prefill | 190 | 18.6% | 6164 | 32.5 |
| decode_only | 749 | 73.5% | 82477 | 110.1 |
| restore_only | 2 | 0.2% | 0 | 0.0 |
| idle_queue | 1 | 0.1% | 0 | 0.0 |
| idle | 0 | 0.0% | 0 | 0.0 |

decode-only steps: 1.86 decoding tasks on average

cold prefills launched in the window: 1 (129471 tokens, 79 s)
disk promotions: 126 (564 GiB, 135 s of reads); demotions (written): 107; parks that synced first: 0 (0.0 s)
queue wait (arrival -> admission): p50 23.7 s, p90 49.4 s, max 116.9 s
restore after admission: 134, mean 526 ms, total 70.5 s

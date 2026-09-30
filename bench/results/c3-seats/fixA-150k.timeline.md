# Seat timeline: fixA-150k

window 180-1200 s (1020 s), 139 turns ending in it, 87269 tokens (85.6 t/s), median depth 163.1k, 376832 cells -> 2 seats; client/server clock offset 892.50 s from 155 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 1497 | 73.4% |
| prefill_cold | 75 | 3.7% |
| prefill_delta | 176 | 8.6% |
| restore | 72 | 3.5% |
| idle_q_disk | 48 | 2.4% |
| idle_q_other | 172 | 8.4% |
| idle_no_demand | 0 | 0.0% |
| (extra tasks beyond 2 seats) | 0 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 75 | 7.3% | 396 | 5.3 |
| delta_prefill | 164 | 16.1% | 5480 | 33.5 |
| decode_only | 734 | 72.0% | 80477 | 109.6 |
| restore_only | 9 | 0.9% | 0 | 0.0 |
| idle_queue | 38 | 3.7% | 6 | 0.2 |
| idle | 0 | 0.0% | 0 | 0.0 |

decode-only steps: 1.76 decoding tasks on average

cold prefills launched in the window: 1 (129471 tokens, 75 s)
disk promotions: 141 (753 GiB, 152 s of reads); demotions (written): 138; parks that synced first: 4 (0.6 s)
queue wait (arrival -> admission): p50 28.1 s, p90 48.7 s, max 97.2 s
restore after admission: 139, mean 519 ms, total 72.1 s

# Seat timeline: base-60k

window 180-1200 s (1020 s), 187 turns ending in it, 136159 tokens (133.5 t/s), median depth 75.5k, 376832 cells -> 4 seats; client/server clock offset 6.09 s from 191 matched requests

| seat state | seat-s | share |
|---|---:|---:|
| decode | 3479 | 85.3% |
| prefill_cold | 100 | 2.5% |
| prefill_delta | 249 | 6.1% |
| restore | 62 | 1.5% |
| idle_q_disk | 4 | 0.1% |
| idle_q_other | 182 | 4.5% |
| idle_no_demand | 4 | 0.1% |
| (extra tasks beyond 4 seats) | 255 | |

| GPU phase | s | share | output tokens | t/s in phase |
|---|---:|---:|---:|---:|
| cold_prefill | 106 | 10.4% | 1846 | 17.4 |
| delta_prefill | 243 | 23.8% | 17580 | 72.4 |
| decode_only | 671 | 65.8% | 115731 | 172.5 |
| restore_only | 0 | 0.0% | 0 | 0.0 |
| idle_queue | 0 | 0.0% | 0 | 0.0 |
| idle | 0 | 0.0% | 0 | 0.0 |

decode-only steps: 3.94 decoding tasks on average

cold prefills launched in the window: 3 (183288 tokens, 100 s)
disk promotions: 124 (434 GiB, 72 s of reads); demotions (written): 15; parks that synced first: 0 (0.0 s)
queue wait (arrival -> admission): p50 7.2 s, p90 17.7 s, max 47.9 s
restore after admission: 147, mean 486 ms, total 71.4 s

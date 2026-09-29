# Final acceptance (main 812043c2e, CTX 376832, session arena 24 GiB, engine capped at 34 GiB)

Commands: `BIN=engine/build-main/bin CTX=376832 bench/final_acceptance.sh bench/results/final` (sweep and 6x60k ran
on fca2e7862 with a 30 GiB arena; the scheduler wake-up fix and arena change don't touch those paths).

## B1: single stream vs ninfer (bench/depth_sweep.py)

| prompt | cold prefill tok/s | ninfer | warm decode tok/s | ninfer |
|---:|---:|---:|---:|---:|
| 5,442 | 2,603 | 3,122 | 132.6 | 108.7 |
| 29,541 | 2,554 | 1,899 | 139.3 | 110.8 |
| 61,500 | 2,356 | 2,018 | 116.9 | 89.6 |
| 125,640 | 2,025 | 1,619 | 97.6 | 78.9 |
| 197,622 | 1,394 | 851 | 113.3 | 80.6 |

## Multi-agent replays (20 min timed window, 0 errors in every run)

| run | turns | steady agg tok/s | TTFT p50 / p90 s | median depth | busy slots per decode | demotions |
|---|---:|---:|---:|---:|---:|---:|
| 6 x 60k (B2/C2) | 230 | 153.6 | 8.7 / 16.7 | 78.7k | 3.87 | 3 |
| 6 x 150k, warm-up (C3) | 124 | 66.3 | 33.3 / 108.3 | 161.6k | 1.80 | 698 |
| 3 x 200k, warm-up (C1) | 250 | 113.9 | 1.0 / 7.5 | 74.9k | 1.98 | 4 |

Previous: ninfer 3 lanes 41.0 t/s (TTFT 62/159 s) at 6 x 60k; lovelace 6 x 150k was 25.0 (2026-09-28) and 41.2
(after restore-latency).

C1: 71 turns at >= 200k tokens across 4 sessions (max 220,486), median 208,650 tokens served from cache, while
3 agents shared a 377k pool (parks/restores/demotions, 0 errors).

C3: 66.3 / 153.6 = 43% of the 60k aggregate (target 70%): no collapse (25.0 -> 66.3), but short of the target. At a
161k median depth only ~2 sessions fit the pool (1.8 busy slots per decode vs 3.9 at 60k), and with 6 x ~5.5 GB mirrors
the 24 GiB RAM tier cycles through the NVMe (698 demotions, prefetch 0/113).

RAM (bench/results/final/mem.log): host available >= 25 GiB throughout, guard never fired.

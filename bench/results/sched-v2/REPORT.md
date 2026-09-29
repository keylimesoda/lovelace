# Scheduling v2 (charter C3), branch feat/sched-v2

Status: **C3 still FAIL**. Scheduling v2 is implemented and passes its gates, but it does not move the 6 x 150k
steady aggregate (41.2 vs 42.1 t/s). The measurements below point to the cost that remains: attention and prefill
at depth scale with every cell resident in the unified cache, not only the sequence's own cells.

## What changed

- `engine/tools/server/server-context.cpp` (with `server-sched.h`) changes the queue order: cheapest uncached work first,
  with aging, `uncached x (1 - wait/bound)`. A request that has waited `--sched-age-bound` (default 120 s) is overdue:
  overdue requests go first, and nothing is admitted past an overdue request that waits for cells. A request that
  waits twice the bound pauses the slot that has generated longest (the starvation bound). Otherwise turns run to
  completion.
- Prefill lane: a request with more than `--sched-lane-tokens` uncached tokens (default 16384) is admitted only while no
  other deep prompt is being processed. It waits without holding cells.
- Prompt batch filling gives the step's prompt budget to the fewest remaining tokens first. Every step still runs all
  generating slots.
- Victims are idle slots, longest-idle first. Sessions that a queued request continues are parked last. Prefetch
  targets the next request in line (admission order), not the FIFO head.
- Telemetry: `/telemetry kv_budget.scheduler` shows the queue in admission order, max/mean/oldest wait, aged admits,
  starvation preempts, and lane holder, busy seconds and occupancy. Each request record carries
  `sched {queue_wait_ms, wait_reason, n_uncached, lane, aged}`. The schema is in docs/observability.md.
- Harness: `replay.py --warmup`, or `WARMUP=1` for run_replay.sh and h2h.sh. The default is unchanged.
  `bench/run_replay.sh` also writes the server request log. `bench/sched_report.py` builds the table below.

## Results (steady = requests ending after 180 s of the timed window, bakeoff/analyze.py)

base = sched-base build of 72a1f2890; after = 29ecc4916. Both 150k runs used the first version of `--warmup`, which ran
only the 6 initial sessions' first contacts before the window. Agents that ran out of turns started new sessions
inside the window: 1 cold 148k prefill (before) and 2 (after), about 220 s each, with output near 10 t/s meanwhile.
The harness now warms up every session the agents may use (cba972cfb, reworked in the last commit), so a rerun
measures a window with no cold starts.

| run | turns | errors | agg_steady t/s | TTFT p50/p90 s | depth p50 | busy slots/decode | max wait s | aged admits | lane occupancy | preempts | demotions | prefetch hits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before-150k-warm | 92 | 0 | 42.11 | 48.5 / 88.2 | 160560 | 1.53 | - | - | - | 0 | 206 | 16/84 |
| after-150k-warm | 85 | 0 | 41.19 | 37.4 / 252.6 | 159705 | 2.04 | 891 (warm-up) | 14 | 0.68 | 3 | 123 | 7/72 |
| after smoke 3 x 60k, 240 s | 34 | 0 | 80.61 | 4.2 / 7.9 | 72180 | 2.48 | 69 | 0 | 0.33 | 0 | 0 | 0/0 |

Reference 6 x 60k on base: 82.8 t/s (bench/results/restore-latency). 70% of that is 58 t/s. The after 6 x 60k runs, the
before 6 x 60k rerun and the greedy identity check were queued, but the shared GPU queue did not reach them. They
were cancelled and have not run.

Gates on 29ecc4916: test-gdn-replay `all ReplaySSM checks passed` (gate-gdn-replay.txt). Byte-exact session restore
`PASS: 0 differing bytes` (gate-session-restore.txt). test-server-sched passes, with new cases for queue order and
queued-session victims.

## Findings (from the logs of these runs)

1. **Attention and prefill cost follow the whole used cell range.** `get_n_kv` returns the highest used cell. The FA
   paths process `[0, n_kv)`: decode does not skip masked cells, and the q5_0 prefill path makes an f16 copy of the
   range. Measured prefill of a cold 148k prompt: 1570-1690 tok/s alone vs 880-1000 tok/s at 70-88k own depth while
   another 148k session was resident. Decode ran at 63 t/s per agent at 160k, against 90+ single-stream. Two deep
   sessions batched gain little: busy slots per decode rose from 1.53 to 2.04 with no throughput change. This is
   kernel or KV-placement work, not the scheduler (FA sparse or masked-tile skip for decode and prefill, and
   compaction or lowest-cell placement).
2. **The pool fits two 150k sessions only below about 163k each.** Median depth is 160k, so the scheduler runs one or
   two seats at a time.
3. **A cold 150k prefill with decoders resident stalls them.** Steps take about 1.15-1.3 s per 1024 prompt tokens and
   the decoders get about 2.8 t/s. Serializing cold prefills (the lane) made warm-up slower (1155 vs 912 s): each lane
   prefill ran next to the previous agent's decode.
4. **Mirrors (5.2-6 GB each with checkpoints) exceed the 30 GB arena**, so round-robin access thrashes the disk tier
   (206 demotions, 1.1 TB read). Promotions overlap the queue wait.

In windows without a cold start, after produced 70.8 t/s (t = 519-898 s) and before 50.6 t/s (t = 400-1200 s). This
suggests that the order and backfill help steady state, but the after window is short and follows a backlog. The
rerun with every session warmed up measures it properly.

## Commands

```sh
# base build: git worktree add ../sched-base 72a1f2890; ENGINE_DIR=../sched-base/engine deploy/build-engine.sh llama-server test-gdn-replay
WARMUP=1 bench/run_replay.sh ../sched-base/engine/build-sm89/bin bench/results/sched-v2 before-150k-warm 327680 6 1200 150000 --session-ram 30720 --session-dir /sessions
WARMUP=1 bench/run_replay.sh <after bin>                         bench/results/sched-v2 after-150k-warm  327680 6 1200 150000 --session-ram 30720 --session-dir /sessions
WARMUP=1 bench/run_replay.sh <after bin> bench/results/sched-v2 smoke-3a-60k 327680 3 240 60000 --session-ram 30720 --session-dir /sessions
deploy/gpu-lock.sh docker run --rm --gpus all -v <after bin>:/e lovelace-engine-runtime:cuda13.1 /e/test-gdn-replay
bench/session_restore_check.sh <after bin>
python3 bench/sched_report.py bench/results/sched-v2/{before-150k-warm,after-150k-warm,smoke-3a-60k}
# not yet run: 6 x 60k before/after (x2), 150k before/after with the all-session warm-up, greedy identity
bench/run_replay.sh <bin> bench/results/sched-v2 <label> 327680 6 1200 60000 --session-ram 30720 --session-dir /sessions
bench/restore_latency.sh <base bin> OUT greedy-base; bench/restore_latency.sh <after bin> OUT greedy-after
(cd bench/replay && uv run python restore_latency.py --compare OUT/greedy-base OUT/greedy-after)
```

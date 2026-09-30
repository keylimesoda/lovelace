# C3 seats: keep the two seats busy at 6 x 150k (branch feat/c3-seats)

Build kept: **fix B** = main 812043c2e + (A) queued sessions stay in the arena + (B) `--session-ckpts 2`
(commits dfe3a0744, 700be8271; engine source at HEAD == the measured `c3-fixB-bin`). Harness: replay `--warmup`
now warms every session the agents may use (cherry-pick of 75bba37f2 from feat/sched-v2), so the timed window has no
cold first contact (the charter metric: "steady after first contacts"). All numbers: `bench/c3_seats.sh`
(= `bench/h2h.sh` lovelace 6, CTX 376832, `--spec-draft-vocab`, WARMUP=1 for 150k, 34 GiB cap, sessions on the NVMe),
steady window 180-1200 s, `bench/sched_report.py` + `bench/seat_timeline.py`.

## 1. Attribution (bench/seat_timeline.py)

The script rebuilds each request on the server from the server log (launch, restore, prefill, decode, release),
joins it with the client records (arrival), and samples the window every 50 ms. Seats = floor(cells / (median depth
+ 8.4k)) = 2 at 161k. Per-run tables: `*.timeline.md` in this directory.

| seat state (share of 2 x 1020 seat-s) | main final (old harness) | base (new harness) | A | B | B r2 | C | D | D r2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode | 57.2% | 48.3% | 73.4% | 80.3% | 78.1% | 77.8% | 82.3% | 79.6% |
| prefill cold | 15.9% | 3.3% | 3.7% | 3.7% | 3.6% | 4.7% | 4.2% | 3.8% |
| prefill delta | 9.3% | 5.7% | 8.6% | 7.5% | 8.9% | 7.3% | 8.3% | 10.8% |
| restore (RAM -> VRAM after admission) | 2.3% | 2.3% | 3.5% | 3.2% | 3.7% | 3.2% | 3.1% | 3.4% |
| idle, head waits on a disk promotion | 1.8% | 2.4% | 2.4% | 0.2% | 0.5% | 0.7% | 0.0% | 0.3% |
| idle, head waits (other) | 13.5% | **38.0%** | 8.4% | 5.1% | 5.2% | 6.3% | 2.1% | 2.1% |
| GPU fully idle with requests queued | 8.1% | **30.9%** | 3.7% | 0.0% | 1.1% | 3.2% | 0.0% | 0.1% |
| decode-only step rate, t/s (tasks) | 118.9 (1.73) | 114.6 (1.62) | 109.5 (1.76) | 110.8 (1.80) | 109.2 (1.81) | 101.7 (1.82) | 104.8 (1.86) | 110.1 (1.86) |

Findings:
- **Old harness (main final, 66.3 t/s):** 4 cold 130-150k prefills inside the window (3 agents switched to new
  sessions, 1 compaction) took 31.8% of the GPU at 3.8 t/s of output. With every session warmed, 1 remains (the
  compaction, 129k tokens, 75 s).
- **Dominant loss: a promotion livelock.** With room for one parked mirror, `poll()` promoted the head, then the next
  in line, whose `make_room()` demoted the head (instant: its file was current), and so on: 691 promotions (3.8 TB
  read) in 35 min, and the head was never in the arena when admission ran. GPU idle with requests waiting: 31% of the
  window (gaps up to 23 s); demotion writes slowed to 10-25 s by the read storm.
- After A, the remaining losses are delta prefill (~8% of seat time, the other seat decodes at ~34 t/s meanwhile),
  restores after admission (~0.5 s x each turn, 3%), and the one compaction prefill; VRAM prefetch never applies
  (0 hits): two 165k sessions leave no room for a third.

## 2. Fixes, one at a time (6 x 150k, same harness)

| run | agg t/s steady | TTFT p50 / p90 s | max queue wait in window s | busy slots / decode | demotions / promotions | prefetch hits | host RAM min GiB | host load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main final, old harness (reference) | 66.3 | 33.3 / 108.3 | 182.5 | 1.80 | 698 / 691 | 0/113 | >= 25 | - |
| base (main bin, new harness) | 64.9 | 32.1 / 60.3 | 274.7 | 1.74 | 701 / 694 | 0/115 | 16 | edge ~10 cores |
| A: head + next stay in the arena | 85.6 | 30.8 / 52.4 | 97.2 | 1.85 | 217 / 211 | 0/157 | 14 | edge ~10 cores |
| **B: A + `--session-ckpts 2`** | **93.1** | 26.3 / 61.3 | 103.0 | 1.96 | 320 / 314 | 0/153 | 18 | edge ~10 cores |
| B, rerun | **91.6** | 28.1 / 54.3 | 122.9 | 1.92 | 294 / 289 | 0/152 | 21 | load 15-21 |
| C: B + separate read/write workers + write-ahead | 79.2 | 30.9 / 63.7 | 131.4 | 1.94 | 368 / 363 | 0/140 | 17 | edge ~10 cores |
| D: C + only head/next promoted, reserve for the head only | 85.8 | 28.2 / 47.3 | 137.7 | 1.98 | 132 / 127 | 0/147 | 17 | edge ~10 cores |
| D, rerun | 88.4 | 25.4 / 59.7 | 116.9 | 1.98 | 142 / 137 | 0/158 | 18 | load 2.5-14 |

0 errors in every run (warm-up included). "host load": the user's browser (msedge, software rendering) used ~10 cores
from 19:28 through all runs; load average was logged from 22:33 (`loadavg.log`, and in `*.mem.log` of later runs).

- **A** (dfe3a0744): `make_room()` never demotes the first 2 sessions of the queue (unless the room is a must), and a
  promotion never pushes out a session ahead of it. 64.9 -> 85.6 t/s; promotions 694 -> 211.
  Regression test `test-server-session-tiers` (fails on main: 787 promotions, head on disk; passes: 1).
- **B** (700be8271): a parked mirror keeps only its newest 2 context checkpoints (restores start at the recurrent
  snapshot; the next turn rolls back to the newest one). ~1 GB less per 160k mirror. 85.6 -> 93.1 / 91.6 t/s.
- **C, D: dropped.** D removed the last seat idling (idle 2.1%, 0 GPU idle, promotions 127) but both D runs were
  slower per step (delta prefill 1.56-1.61 s/turn vs 1.44-1.52 for B, same delta tokens): 87.1 vs 92.4 t/s over two
  runs each. The engine is back to B (commit "keep fix B"); the design doc records the attempt.
- sched-v2 (run-to-completion, prefill lane) was not ported: no preemption happens at 150k (preempts 0), and with
  every session warmed there is at most one cold prefill in the window, so there are no concurrent cold prefills to
  serialize.

## 3. Gates (on c3-fixB-bin)

| gate | result |
|---|---|
| test-server-sched, test-server-wakeup | pass (gate-cpu-tests.txt) |
| test-server-session-tiers (new) | PASS on B (main: FAIL, 787 promotions, head on disk) |
| test-gdn-replay | all ReplaySSM checks passed (gate-gdn-replay.txt) |
| session_restore_check.sh (byte-exact, incl. disk demotion/promotion) | PASS: 0 differing bytes (gate-session-restore.txt) |
| greedy_gate.sh vs main, both with `--session-ram 24576` | sequential **6/6 identical**; parallel 3/6 (divergences at batch-composition-dependent tokens, greedy/COMPARE.txt) |

The same GPU gates also passed on D before it was dropped (gates-D/).

## 4. 6 x 60k (no regression check)

Back to back, same host conditions (the user's browser on ~10 cores; load 2.5-16):

| run | agg t/s steady | TTFT p50 / p90 s | max wait s | busy slots / decode | demotions | prefetch hits | host RAM min GiB | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B (this branch) | **137.1** | 9.8 / 24.6 | 49.7 | 4.11 | 3 | 3/138 | 18 | 2.5-9.7 |
| main 812043c2e | 133.5 | 8.6 / 23.3 | 47.8 | 4.08 | 127 | 5/151 | 17 | 2.8-16.5 |

No regression (+2.7%). Both are below the 153.6 t/s of bench/results/final (measured 04:56 on an idle desktop), which
is the host load, not the build.

## C3 status

6 x 150k steady: **92.4 t/s (two-run mean) vs the 107 t/s target** (60% of 153.6; was 66.3 = 43%). The seats now
decode ~79% of the time; what is left is not idling but per-step cost: two batched 160k decoders run at ~55-60 t/s
each (decode-only 109-111 t/s), each turn prefills ~1.4k delta tokens at ~950 tok/s during which the other seat
decodes at ~34 t/s, plus 0.5 s restores and the in-window compaction prefill. Even with both seats always decoding,
2 x ~60 t/s x (1 - ~12% delta prefill/restore) is ~105 t/s, so 107 needs faster decode/prefill at depth (kernels,
out of scope here) or a third seat (pool).

## Commands

```sh
# bins: c3-base-bin = main 812043c2e (engine/build-main/bin copy); c3-fix{A,B,C,D}-bin = builds of each commit
BIN=<bin> bench/c3_seats.sh bench/results/c3-seats <label> replay:6:150000:1200 1
BIN=<bin> bench/c3_seats.sh bench/results/c3-seats <label> replay:6:60000:1200 0
python3 bench/seat_timeline.py bench/results/c3-seats/<label>        # attribution table
(cd bench/replay && uv run python ../sched_report.py ../results/c3-seats/<label>)
deploy/gpu-lock.sh docker run --rm --gpus all -v <bin>:/e lovelace-engine-runtime:cuda13.1 /e/test-gdn-replay
bench/session_restore_check.sh <bin>
SERVER_ARGS="--spec-draft-vocab /draft-vocab/qwen3.8-27b.txt --session-ram 24576" bench/greedy_gate.sh bench/results/c3-seats/greedy <main bin> <bin>
docker run --rm -e LD_LIBRARY_PATH=/e:/opt/cuda-stubs -v <bin>:/e lovelace-engine-devel:cuda13.1 /e/test-server-session-tiers   # also test-server-sched, test-server-wakeup
```

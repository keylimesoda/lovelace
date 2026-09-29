# Session-switch latency (feat/restore-latency, commit 49ab13628)

Scripts: `bench/restore_latency.sh` (N agents taking greedy turns one request at a time on a pool that cannot hold them;
`BUSY=K` adds agents prefilling fresh ~16k-token prompts), `bench/run_replay.sh` (6-agent replay).
"before" = main build (e88e7e403 + docs), "after" = this branch. Phases are ms from admission (request_done.session).

## Switch phases, 3 agents at 60-70k, pool 131072

| phase (switch requests, p50/p90) | idle before | idle after | loaded before (BUSY=1) | loaded after (BUSY=1) |
|---|---|---|---|---|
| restored GiB | 1.53 | 1.50 | 1.51 | 1.51 |
| park of the victim (host) | - | 8.8 / 13.1 | - | 13.7 / 19.0 |
| restore issue (host) | - | 14.2 / 16.2 | - | 17.8 / 19.2 |
| restore copies done (from admit) | - | 79 / 82 | - | 86 / 94 |
| restore device time | 157 / 171 | 109 / 115 | 159 / 168 | 111 / 117 |
| prompt batched (from admit) | - | 24 / 28 | - | 5736 / 8138 |
| first token after admit | 1732 / 2833 | 2007 / 2382 | 9845 / 13276 | 8925 / 13334 |

Restore wall on the critical path is now ~80-95 ms for 1.5 GiB (park + issue + copies; the first prefill graph waits
for the copies on the device, the host does not). Under load, the remaining delay (prompt batched 5.7 s after admit) is
not the restore: the busy agent's 16k prompt takes every batch until it is done (batch filling is first come, first
served). That is Scheduling v2 (stall-free prefill sharing).

## 6-agent replay, 60k, 20 min (replay-6a-after)

0 errors (98 ok, 5 cancelled at the deadline), steady 82.8 t/s (bakeoff/analyze.py, warm-up 360 s), TTFT p50/p90
17.2/64.7 s (p90 swings 40-60 s run to run on the unchanged build), cache hit 0.92, 0 preemptions, 0 demotions.
Switches: 75 restores on the critical path + 15 prefetches, all 15 used: hit rate 15/90 = 17%. Most requests are
admitted as they arrive (the pool has room once idle sessions are parked), so there is no wait to prefetch into.
Restore issue host time p50/p90 121/169 ms in the replay, vs 14-18 ms in the scripted runs: not explained yet
(suspect: the host blocks on a full launch queue of the compute stream while a long graph is in flight).

## Greedy identity (restored vs resident)

Pool 131072 (parks) vs 327680 (all resident), 3 agents x 4 turns: 6 of 12 outputs differ. One of the differing turns
is a first contact (no restore involved), and every divergence is a coherent alternative continuation after 15-217
identical characters. The pool size changes cell placement and FA accumulation order, so this check cannot isolate
restores. An exact check needs identical cell layouts, or a byte comparison of restored cells against the mirror.

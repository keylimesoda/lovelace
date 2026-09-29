# fix/sched-wakeup: lost wake-up of KV budget admission

Branch `fix/sched-wakeup` (fix 4cb7be843) against main (engine fca2e7862, bins `lovelace-wt/wakeup-main-bin`), 2026-09-29.

## Root cause

A finished request leaves its slot idle with its session mirror sync in flight (`slot.release()`, XFER_SYNC).

1. The next `update_slots()` pass tries the queued first contact while the sync runs. `sched_sync()` counts the syncing slot as active, so `plan_commit()` finds no victim: `SCHED_WAIT_KV`. `sched_admit()` clears `sched_dirty`.
2. No slot runs, so `update_slots()` waits for the sync (`sched_poll_xfers(true)`, server-context.cpp:4389, the log line "done 202.9 ms after issue"). Then it calls `sched_admit()` again, which returns at once. `sched_finish_xfer()` set `sched_dirty` only for a completed park (server-context.cpp:2362-2366), never for a sync or a restore, and `sched_admit()` returns early when the flag is clear (server-context.cpp:3128).
3. `update_slots()` returns with every slot idle. `server_queue::start_loop()` then waits for a new task only (server-queue.cpp:356). Nothing retries the admission until an unrelated request arrives.

## Fix

- `sched_finish_xfer()` marks the admission gate when any transfer completes: a park, a sync or a restore.
- `sched_dirty` becomes `server_sched_gate` (server-sched.h). While work waits, it also retries at least once per second as a backstop. `server_queue::update_slots_after()` makes the loop run `update_slots()` again at that deadline without a new task, and keeps it out of the sleeping state meanwhile.

## Regression test

`test-server-wakeup` drives the real `server_queue` loop, the gate and the KV budget through the hang scenario. The victim is mid-sync, the sync completion goes unmarked, and no task ever arrives.

| build | result |
|---|---|
| fix | `admitted 100.5 ms after the sync completed, 2 passes, 0 tasks`, pass |
| main's `start_loop()` (server-queue.cpp from main, everything else the same) | `FAIL: the task waiting for KV was not admitted within 5 s of the sync (1 passes, 0 tasks)` |

`test-server-sched` also covers the gate's backstop and wait deadline. All checks pass.

## End-to-end repro: `bench/wakeup_repro.sh BIN OUT LABEL`

Setup:
- the balanced profile with `-np 3 -c 376832`, q5_0 KV, `--kv-budget`, `--session-ram 24576`, MTP n=3 and the draft vocab
- `--memory 34g --memory-swap 34g`, sessions in `~/.cache/lovelace-bench/wakeup-sessions`
- `replay.py --warmup --agents 3 --start-turn-min-depth 200000`

A 1 s telemetry sampler records stalls: no slot busy while requests wait. A stall of 90 s gets one tiny probe request, so that a hanging build still finishes.

| build | first contacts | stalls | longest | probes | result |
|---|---|---|---|---|---|
| main | 3/3 (199942, 199106, 198496 tokens) | 2 | 92 s | 2 | FAIL |
| fix  | 3/3 (199942, 199106, 198496 tokens) | 0 | 0 s | 0 | PASS |

Timeline from server.log:
- **main:** the sync is done at 2:32.97. Slot 0 is parked and the next first contact launched at 4:03.81, right after the probe; the same happens again at 6:01.74 → 7:32.29.
- **fix:** the sync is done at 2:32.428, slot 0 is parked at 2:32.460 and the waiting first contact is launched at 2:32.465, 36 ms after the sync. That comes from the completion event, not the 1 s backstop. The other hand-offs land 38-89 ms after their syncs.

Artifacts: `main/`, `fix/` (RESULT.md, samples.jsonl, server.log).

## Gates

- `test-gdn-replay` (fix build): all ReplaySSM checks passed (test-gdn-replay.txt).
- `bench/greedy_gate.sh`, main vs fix: sequential 6/6 identical, concurrent 6/6 identical (greedy/COMPARE.txt).

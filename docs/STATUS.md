# Live status (update at every milestone; read with docs/CHARTER.md at the start of a session)

Updated: 2026-09-28 ~20:50 local.

## Production
- Stopped during measurement. Restart with `deploy/serve.sh balanced` (lovelace engine :8090, capture proxy :8080).
  The old ninfer container `ninfer-qwen` is stopped and kept only for comparison.

## Acceptance (docs/CHARTER.md rows)
- A1 quality: **PASS**. lovelace KL 0.0171 vs ninfer 0.0282 against the same Q8_0 reference (docs/quality-kld.md).
- B1 single stream (bench/results/h2h/*-sweep.jsonl): **decode PASS** (equal or faster at 5k-198k); **prefill FAIL at depth**
  (126k: 1132 vs 1619 tok/s; 62k: 1739 vs 2018). A fix is in progress (feat/prefill-depth).
- B2 multi-agent: ninfer 3 lanes = 41.0 t/s steady, TTFT p50/p90 62/159 s; ninfer 6 lanes can't start at >= 163k.
  lovelace same-day run `bench/results/h2h/lovelace-6a` was finishing at 20:50; earlier lovelace runs got 76.8 t/s,
  TTFT 18/41 s. `bench/acceptance.sh` (background) then tries ninfer 6 lanes at lower context and writes
  bench/results/acceptance/REPORT.md. Regenerate the report any time: `bench/acceptance.sh report`.

## In-flight engineering (worktrees in ~/Work/lovelace-wt/, branches not yet merged)
| branch | agent (model) | goal | state at 20:50 |
|---|---|---|---|
| feat/batched-verify | BatchedVerify (Opus) | raise 6-agent aggregate tok/s above ~77 (profile, then cut the top step costs) | 5 commits |
| feat/restore-latency | SessionCore (Opus) | agent-switch restore wall 1.1-1.8 s -> <= 0.4 s; working prefetch | 1 commit |
| feat/prefill-depth | FattnQuantKV (Opus) | prefill at 126k >= 1620 tok/s (ninfer parity), no compute-buffer growth; 4 h timebox | started |

Merge rule for each branch: test-backend-ops for the touched ops, test-gdn-replay, the greedy e2e identity check,
and before/after numbers from the branch's own script; then `bench/acceptance.sh` (only missing steps run) and a
report regeneration.

## Decisions made (with evidence)
- Weights GPU-5 (ShapeLearn IQ4_XS-3.84bpw); KV q5_0; MTP n=3; E8 KV not ported (quality); DFlash2 rejected;
  shared-prefix pinning dropped (risk/benefit); GPU-4, ISTA IQ3_S and ninfer artifact all measured worse.

## Next after the in-flight work
1. Merge the three branches as they pass; rerun the acceptance script; redeploy (`deploy/serve.sh balanced`).
2. Update CHARTER status rows B1/B2 and README results.

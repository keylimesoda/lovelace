# Live status (update at every milestone; read with docs/CHARTER.md at the start of a session)

Updated: 2026-09-28 ~23:00 local.

## Production
- Stopped during measurement. Restart with `deploy/serve.sh balanced` (lovelace engine :8090, capture proxy :8080).
  The old ninfer container `ninfer-qwen` is stopped and kept only for comparison.

## Acceptance (docs/CHARTER.md rows)
- A1 quality: **PASS**. lovelace KL 0.0171 vs ninfer 0.0282 against the same Q8_0 reference (docs/quality-kld.md).
- B1 single stream (bench/results/h2h/*-sweep.jsonl): **decode PASS** (equal or faster at 5k-198k); **prefill FAIL at depth**
  (126k: 1132 vs 1619 tok/s; 62k: 1739 vs 2018). Cause (bench/results/prefill-profile on feat/prefill-depth): at 128k,
  FA is 53% of a prefill ubatch and the f16 kernel already runs near the f16/f32-acc peak (~140 of 165 TFLOPS), so
  the fix needs cheaper math (int8 PV, blockwise f16 accumulation), gated by KLD.
- B2 multi-agent: **PASS** (bench/results/acceptance/REPORT.md): lovelace 76.3 t/s steady, TTFT 19/40 s vs ninfer
  3 lanes 41.0 t/s, 62/159 s; ninfer 6 lanes can't start at >= 163k.
- C3 no collapse at 6 x 150k: **FAIL** (25.0 vs 76.8 t/s); Scheduling v2 in progress.
- TTFT p90 swings ~40-60 s run to run on an unchanged build; judge TTFT with 2 runs per side or by p50.

## In-flight engineering (worktrees in ~/Work/lovelace-wt/, branches not yet merged)
| branch | agent | goal | state at 23:40 |
|---|---|---|---|
| feat/batched-verify | BatchedVerify | 6-seq step 60.8 -> 58.6 ms (+3.4% summed decode) on the merged build; the 54.0 ms / +17% was a pre-rebase build with the stream-k cap | **merged** 4b6187d5d (stream-k cap dropped: broke greedy identity). Open: +0.5 ms graph-launch gap after rebase (StepCosts item 0) |
| feat/restore-latency | SessionCore | switch park+issue+copies 80-95 ms (was 1.1-1.8 s wall); replay 82.8 t/s | **merged** 7ea3ef641 (byte-exact restore gate, gdn-replay pass) |
| feat/prefill-depth | PrefillFA | phase 1: gates + compute buffer for 8ea46e9d0 (no f16 K/V copy; +2.5-4% to 64k, -4% at 192k); phase 2: cheaper FA math, 126k >= 1620 tok/s, KLD <= 0.01707+0.0005 | started 23:00 |
| feat/sched-v2 (stacked on restore-latency) | SchedV2-2 | charter C3 (Scheduling v2 in docs/design/session-store.md) | started 23:00 |
| feat/step-costs | StepCosts | from main's 58.6 ms step (host gap 12.7): 0) +0.5 ms graph-launch gap, 1) GPU KQ mask, 2) gathered draft LM head (~3 ms), 3) MMQ 24 rows | started 23:40 |

Merge rule for each branch: test-backend-ops for the touched ops, test-gdn-replay, the greedy e2e identity check,
and before/after numbers from the branch's own script; then `bench/acceptance.sh` (only missing steps run) and a
report regeneration.

## Decisions made (with evidence)
- Model family (2026-09-28): Qwen3.8-27B dense stays. Qwen3.8-Flash-Next is 180B total params (~100 GB at 4 bpw),
  which doesn't fit 24 GB VRAM + 60 GB RAM; Gemma 4 31B has more KV per token (27.5 vs 22 KB at q5_0) and ranks lower
  on agentic coding. Per-agent context is 262k via swapping (host RAM/disk parking); 6 x 170k resident isn't required.
- Weights GPU-5 (ShapeLearn IQ4_XS-3.84bpw); KV q5_0; MTP n=3; E8 KV not ported (quality); DFlash2 rejected;
  shared-prefix pinning dropped (risk/benefit); GPU-4, ISTA IQ3_S and ninfer artifact all measured worse.

## Next after the in-flight work
1. Merge the three branches as they pass; rerun the acceptance script; redeploy (`deploy/serve.sh balanced`).
2. Update CHARTER status rows B1/B2 and README results.

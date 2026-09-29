# Live status (update at every milestone; read with docs/CHARTER.md at the start of a session)

Updated: 2026-09-29 ~03:30 local.

## Production
- Stopped during measurement. Restart with `deploy/serve.sh balanced` (lovelace engine :8090, capture proxy :8080).
  The old ninfer container `ninfer-qwen` is stopped and kept only for comparison.

## Acceptance (docs/CHARTER.md rows)
- A1 quality: **PASS**. lovelace KL 0.0171 vs ninfer 0.0282 against the same Q8_0 reference (docs/quality-kld.md).
- B1 single stream: **PASS** on branch results (decode equal or faster at every depth; cold prefill at 126k 1922/1730
  tok/s at 1/6 lanes vs ninfer 1619). The final depth sweep on merged main is still to run.
- B2 multi-agent: **PASS** (bench/results/acceptance/REPORT.md): lovelace 76.3 t/s steady, TTFT 19/40 s vs ninfer
  3 lanes 41.0 t/s, 62/159 s; ninfer 6 lanes can't start at >= 163k. restore-latency build: 82.8 t/s.
- C3 no collapse at 6 x 150k: **FAIL** so far (41.2 t/s). Cause fixed on feat/resident-cost (prefill attended every
  resident cell); re-measure after merge.
- TTFT p90 swings ~40-60 s run to run on an unchanged build; judge TTFT with 2 runs per side or by p50.

## In-flight engineering (worktrees in ~/Work/lovelace-wt/, branches not yet merged)
| branch | agent | result | state at 03:30 |
|---|---|---|---|
| feat/batched-verify | BatchedVerify | 6-seq step 60.8 -> 58.6 ms (+3.4% summed decode) | **merged** (stream-k cap dropped: broke greedy identity) |
| feat/restore-latency | SessionCore | switch park+issue+copies 80-95 ms (was 1.1-1.8 s wall) | **merged** (byte-exact restore gate) |
| feat/prefill-depth | PrefillFA | prefill kernel flash_attn_ext_mma_q_pf: 126k cold prefill 1126/1725 -> 1730/1922 tok/s (6/1 lanes); compute buffer 2178 -> 1038 MiB; max ctx 344k -> 393k; KLD 0.01721 | **merged** c21a03251's parent |
| feat/step-costs | StepCosts, StepCosts2 | GPU KQ mask (step 56.4 -> 52.1 ms, greedy 6/6); 32k MTP draft vocab (held-out acceptance +0.11 pt, ~3-4% decode; 16k rejected, -1.21 pt); graph-launch "gap" was driver noise | **merged** c21a03251 |
| feat/resident-cost | ResidentCost | attention covers only the ubatch's sequences' cell window (get_kv_window): prefill with a 148k session resident 1741 vs alone 1743 tok/s (was ~half) | rebasing onto main (kv_lo into GPU-mask keys + graph reuse), then gates |
| feat/sched-v2 (pushed, NOT merged) | SchedV2-2 (done) | 6 x 150k: 41.2 vs 42.1 t/s base: no gain while resident cost dominated. Open: lane starvation (max wait 891 s in warm-up vs 120 s bound), mirrors 5-6 GB/session outgrow the 30 GB RAM tier, 6 x 60k no-regression and greedy gate not run | re-evaluate on top of resident-cost |

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

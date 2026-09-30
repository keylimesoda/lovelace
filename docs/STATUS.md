# Live status (update at every milestone; read with docs/CHARTER.md at the start of a session)

Updated: 2026-09-29 ~12:00 local.

## Incidents 2026-09-29
- **Benches OOM-killed the desktop** (05:09 and 09:42, both in the 6 x 150k replay): the session disk tier of the bench
  scripts was /tmp (tmpfs = RAM). Fixed on main: session dirs on the NVMe (~/.cache/lovelace-bench), hard
  `--memory 34g` cap on every engine container (bench and production), arena 30 -> 24 GiB, RAM sampler + guard in
  bench/final_acceptance.sh. Verified: container 26.1 of 34 GiB, host >= 27 GiB available during a 200k warm-up.
- **Scheduler lost wake-up** (10:06-10:17, 3 agents x 200k): a finished session sync didn't re-run admission, so the
  server idled with 2 queued requests until an unrelated request arrived. Fixed and merged (fix/sched-wakeup,
  812043c2e; test-server-wakeup; bench/wakeup_repro.sh).

## Production
- **Stopped 2026-09-29 ~12:10 for the C3 push** (user's choice). Last good build: main 812043c2e, image
  lovelace-serve:latest; restart with `deploy/serve.sh balanced` (CTX 376832, 24 GiB arena, 34 GiB cap).
  C3 work: feat/c3-seats (agent C3Seats): per-seat timeline attribution of the 6 x 150k run, then fixes (disk->RAM
  promotion ahead of turn, smaller mirrors, sched-v2 run-to-completion), target >= ~107 t/s.
  The old ninfer container `ninfer-qwen` is stopped and kept only for comparison.

## Acceptance (docs/CHARTER.md rows; evidence bench/results/final/REPORT.md)
- A1 quality **PASS**; B1 single stream **PASS** (except cold prefill of a ~5k prompt, 2603 vs 3122 tok/s);
  B2 6 x 60k **PASS** 153.6 t/s vs ninfer 41.0; C1 >= 200k per agent **PASS**; C2 no stalls/errors **PASS**.
- C3 no collapse at depth **PARTIAL**: 6 x 150k 66.3 t/s (was 25.0) = 43% of the 60k aggregate vs a 70% target.
  Limits: ~2 sessions fit the pool at 161k median depth; the 24 GiB RAM tier cycles via NVMe (698 demotions,
  prefetch 0/113). Candidates: feat/sched-v2 run-to-completion + prefetch on top of main, smaller mirrors
  (checkpoints), or more pool.
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

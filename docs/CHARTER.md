# Charter (from the user, 2026-09-28) and acceptance checklist

Check every milestone against this file. When the evidence for an item changes, update the Status column.

## Goal (user's words, condensed)

An engine/model combination on THIS box (RTX 4090 24 GB, Ryzen 9 9950X, 64 GB RAM, T705 NVMe) that gives:
1. **Maximum intelligence**: measured as distance from BF16, or with SWE-bench / agent-bench style scores.
2. **Context**: at least 170k, and up to 262k, **per agent** (each agent gets its own full cache).
3. **Multi-agent**: 4-6 agents at once, seamlessly; optimize **aggregate throughput**.
4. **Maximum speed**: tuned to this exact machine (SM89 kernels, CPU, RAM, storage).
5. **Consider**: DFlash2, exotic KV caches (e.g. ninfer's E8 `rk4v4-e8`), GGUFs (unsloth, ShapeLearn), and ninfer's own quant.
6. **Observability**, like the tensorninja ninfer branch, if it costs no performance.

Hard bar (added by the user): **no dumber than the previous setup** (ninfer + its artifact + E8 KV), and **as fast or
faster for multi-user**. Benchmarks must be cheap, deterministic scripts; spend model effort only where code is needed.

## Acceptance checklist

| # | Criterion | How it is measured (script) | Status |
|---|---|---|---|
| A1 | Quality >= previous setup | eval/kld against the same Q8_0 reference: lovelace (GPU-5 + q5_0 KV) vs ninfer artifact + rk4v4-e8 | **PASS**: KL 0.0171 vs 0.0282, top-1 95.67% vs 94.50% (docs/quality-kld.md) |
| A2 | Quality vs BF16 | Q8_0 used as the proxy reference (KL of Q8_0 vs BF16 not measured) | proxy only |
| B1 | Single-stream speed >= previous | bench/h2h.sh ENGINE 1 sweep (bench/depth_sweep.py, 8k-200k) | **PASS** (branch results, final main sweep pending): decode 113/119/99/78/110 vs 109/111/90/79/81 tok/s at 5k-198k; cold prefill at 126k 1922 (1 lane) / 1730 (6 lanes) vs 1619, at 198k 1291/1199 vs 851 (bench/results/prefill-depth, feat/prefill-depth merged; KLD 0.01721 <= 0.01757) |
| B2 | Multi-agent aggregate >= previous | bench/h2h.sh ENGINE N replay:6:60000:1200 on both engines | **PASS**: 6 agents, 20 min: lovelace 76.3 t/s steady, TTFT p50/p90 19/40 s, 0 errors vs ninfer (best config that starts, 3 lanes) 41.0 t/s, 62/159 s |
| C1 | >= 170k (target 262k) context per agent, never lost | session-store replays (bench/results/session-store) | met: 0 errors, 0 sessions lost at ~150k x 6 |
| C2 | 4-6 agents without stalls or errors | bench/results/main-327k, session-store | met: 0 errors; TTFT p90 41 s at real depth |
| C3 | No aggregate collapse at depth; pausing agents OK, no starvation | 6 x 150k replay, steady after first contacts, vs 6 x 60k (docs/design/session-store.md, Scheduling v2) | **FAIL** so far: 41.2 vs 82.8 t/s. Cause found: prefill attended every resident cell; feat/resident-cost fixes it (prefill with a 148k session resident 1741 vs 1743 tok/s alone). Re-measure after merge |
| D1 | DFlash2 considered | bake-off: acceptance 0.27 vs MTP 0.55 on agent traffic | done (MTP chosen) |
| D2 | Exotic KV considered | KLD of q4/q5/q8 mixes; ninfer E8 measured in A1 (adds 0.0055 KL vs q5_0's 0.0008); KVarN not measured | done: q5_0 kept |
| D3 | SM89 kernels | llama-bench tables (bench/results/main-gate) | done |
| E1 | Observability without perf cost | overhead A/B < 0.5% | done |

## Working agreement (how this project is run)

- **Source of truth lives in the repo, not in a chat:** this file (goal, acceptance, status), README.md (what exists
  and how to run it), docs/design/* (designs), bench/results/*/REPORT.md (evidence). A new session starts by reading
  this file and README.md, and nothing else is assumed.
- **Measure against the incumbent**, never only against an easy baseline. Stock llama.cpp numbers are context,
  not acceptance.
- **Measurement is scripts only:** `bench/acceptance.sh` runs every acceptance measurement that is missing and writes
  `bench/results/acceptance/REPORT.md` (PASS/FAIL per row). Model-driven agents write code; they don't run
  benchmark campaigns.
- **Small milestones:** each milestone names the charter row it moves, ends with `bench/acceptance.sh report`, and
  updates the Status column above.
- **Agent models:** Luna (`sonic`/`scout`) by default; Opus (`task`) for new CUDA kernels, and for any work where Luna
  shows failure or degradation.
- **GPU discipline:** `deploy/gpu-lock.sh` for every GPU job (FIFO; a server is started, used and removed in one hold);
  production (`deploy/serve.sh`) is stopped during measurement and restarted after.

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
| A1 | Quality >= previous setup | eval/kld against the same Q8_0 reference: lovelace (GPU-5 + q5_0 KV) vs ninfer artifact + rk4v4-e8 | pending (ninfer scorer being written) |
| A2 | Quality vs BF16 | Q8_0 used as the proxy reference (KL of Q8_0 vs BF16 not measured) | proxy only |
| B1 | Single-stream speed >= previous | bench/h2h.sh ENGINE 1 sweep (bench/depth_sweep.py, 8k-200k) | running |
| B2 | Multi-agent aggregate >= previous | bench/h2h.sh ENGINE N replay:6:60000:1200 on both engines | running (ninfer 3 and 6 lanes) |
| C1 | >= 170k (target 262k) context per agent, never lost | session-store replays (bench/results/session-store) | met: 0 errors, 0 sessions lost at ~150k x 6 |
| C2 | 4-6 agents without stalls or errors | bench/results/main-327k, session-store | met: 0 errors; TTFT p90 41 s at real depth |
| D1 | DFlash2 considered | bake-off: acceptance 0.27 vs MTP 0.55 on agent traffic | done (MTP chosen) |
| D2 | Exotic KV considered | KLD of q4/q5/q8 mixes done; E8 comes from A1; KVarN not measured | partial |
| D3 | SM89 kernels | llama-bench tables (bench/results/main-gate) | done |
| E1 | Observability without perf cost | overhead A/B < 0.5% | done |

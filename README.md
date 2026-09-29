# lovelace

A llama.cpp-based serving engine for **Qwen3.8-27B** on one NVIDIA RTX 4090 (24 GB, sm_89) with a Ryzen 9 9950X
host. It is built for 4–6 concurrent coding agents, each with its own context of up to 262,144 tokens.

## Configuration (decided by measurement)

| Choice | Value | Evidence |
|---|---|---|
| Weights | ShapeLearn `Qwen3.8-27B-IQ4_XS-3.84bpw.gguf` (embedded MTP head) | lowest KL vs Q8_0 of the candidates that fit (docs/quality-kld.md) |
| KV cache | q5_0 / q5_0 | KL 0.00155 vs f16 KV through 250k, ~10% of the weight-quant error |
| Speculation | MTP, 3 draft tokens | acceptance 0.54–0.57 on replayed agent traffic (DFlash2: 0.27) |
| GPU KV pool | 327,680 cells shared by 6 slots | largest fit, with `-ubd 32` draft ubatch |
| Sessions | every agent keeps its full context; host mirror in pinned RAM (30 GiB), NVMe overflow | docs/design/session-store.md |

## Architecture

`engine/` is a llama.cpp subtree (upstream b11223) with:

- **ReplaySSM rollback** for the Gated-DeltaNet layers: speculative verify records inputs and folds only the
  accepted tokens, instead of snapshotting the 151 MB recurrent state per draft token (7.2 GB → 1.06 GB at 6 slots).
- **KV-budget scheduler**: admits requests against real KV capacity (no "context size exceeded"), interleaves
  prefill chunks with decode, preempts at token boundaries, and places checkpoints at message boundaries.
- **Per-agent session store**: sessions are identified by prompt prefix. Each one has an append-only mirror of its
  KV cells plus its recurrent state in pinned host memory, synced after every request (parking is free),
  restored at ~25 GB/s, and demoted to NVMe instead of ever being dropped.
- **sm_89 kernels**: FlashAttention that reads quantized KV directly for decode and verify batches (int8 tensor-core
  QK), IQ-quant matmul tuning for 1–24 rows, and fused DeltaNet input/state/norm chains.
- **Observability**: `/telemetry`, an `/events` SSE stream, a JSONL request log, NVML board-energy attribution,
  per-session tier and restore stats, and the dashboard (`apps/web`) served on the API port.

`bench/replay/capture_proxy.py` fronts the engine on :8080 and records request bodies for future replay and eval.

## Measured results (RTX 4090)

llama-bench, q5_0 KV (bench/results/main-gate/REPORT.md), stock llama.cpp b11223 → lovelace:

| test | stock | lovelace |
|---|---:|---:|
| decode, depth 0 | 54.2 t/s | 60.2 t/s |
| decode @ 64k | 39.7 t/s | 54.3 t/s |
| 4-token verify @ 64k | 116 t/s | 185.8 t/s |
| 4-token verify @ 196k | 70 t/s (q4_0) | 159.3 t/s |

Replayed OMP agent traffic, 6 agents, 20 min (bench/results/main-327k, bench/results/session-store):

| workload | errors | steady aggregate | TTFT p50 / p90 | cache hit |
|---|---:|---:|---:|---:|
| stock llama.cpp, 6 agents ≥60k | most requests failed (context exceeded) | – | – | – |
| lovelace, real depth (median ~68k) | 0 | 76.8 t/s | 18.4 / 41.4 s | 0.93 |
| lovelace, 6 sessions of ~150k each (≈3x the GPU pool) | 0, no session lost | 25.0 t/s | queue-bound | restores from mirrors |

## Operations

```sh
deploy/build-engine.sh llama-server                       # CUDA 13.1 devel image, sm_89
(cd apps/web && bun install && bun run build)             # dashboard
docker build -f deploy/serve.Dockerfile -t lovelace-serve:latest .
deploy/serve.sh balanced                                  # engine :8090, capture proxy :8080
deploy/serve.sh status | logs [engine|proxy] | stop
```

Profiles: `balanced` (q5_0 KV, 327,680 cells), `quality` (q8_0/q5_0 KV, 262,144), `vision` (balanced plus
`--mmproj`, 294,912). `CTX=` overrides the pool. State lives in `~/.local/state/lovelace` (`log/` request log,
`lib/captures/` proxy captures, `lib/sessions/` spilled sessions). The dashboard is at http://127.0.0.1:8080/.

GPU work while developing: every GPU job goes through `deploy/gpu-lock.sh` (FIFO queue; it holds until GPU memory
is idle again). `deploy/gpu-watchdog.sh` flags a queue stalled on an idle GPU. Stop production first
(`deploy/serve.sh stop`).

## Benchmarks and evaluation

- `bench/replay/`: multi-agent replay of recorded OMP sessions (README there). `bench/run_replay.sh` runs one replay in
  a single GPU-lock hold; `bench/bakeoff/analyze.py` summarizes the steady state.
- `eval/kld/`: sparse top-k KL divergence at 32k–250k depth on recorded agent replies (weights and KV types).

## Known follow-ups

- Session restore copies take 70–300 ms but 1.1–1.8 s wall, and prefetch rarely fires (docs/design/session-store.md).
- Aggregate throughput saturates at ~80 t/s with 5–6 agents at ~68k depth: next target is batched-verify efficiency.
- Parked capacity option: large-batch quantized FA without the f16 copy (branch `feat/fattn-prefill`, +~10% pool,
  −11–15% prefill at depth).

## Upstream

```sh
git subtree pull --prefix engine llama-upstream <tag> --squash
```

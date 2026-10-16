# lovelace: developer and operations notes

The front page is [README.md](../README.md). This page has the configuration, the operational detail and the
development workflow. Current state and open work: [STATUS.md](STATUS.md); goals and acceptance rows:
[CHARTER.md](CHARTER.md).

## Configuration (decided by measurement)

| Choice | Value | Evidence |
|---|---|---|
| Weights | ShapeLearn `Qwen3.8-27B-IQ4_XS-3.84bpw.gguf` ("GPU-5", embedded MTP head) | lowest KL vs Q8_0 of the candidates that fit ([quality-kld.md](quality-kld.md)) |
| KV cache | q5_0 / q5_0 | KL 0.00155 vs f16 KV through 250k, ~10% of the weight-quant error ([quality-kld.md](quality-kld.md)) |
| Speculation | MTP, 3 draft tokens, 32k-token draft vocabulary | DFlash2 acceptance 0.27 vs MTP 0.55 ([CHARTER.md](CHARTER.md) D1); draft vocab +0.11 pt held-out acceptance, +3.9% summed decode ([step-costs](../bench/results/step-costs/REPORT.md)) |
| GPU KV pool | 262,144 unified cells shared by 6 slots (`balanced`) | 376,832 starts (largest is 393,216, [prefill-depth](../bench/results/prefill-depth/REPORT.md)) but leaves ~0.5 GB free; long prefills then abort with CUDA OOM in `cudaGraphInstantiate` |
| Sessions | every agent keeps its full context; 24 GiB pinned-RAM mirror arena, NVMe overflow, engine container capped at 34 GiB | [session-store.md](design/session-store.md), [STATUS.md](STATUS.md) incidents |

## Engine changes (`engine/`)

`engine/` is a llama.cpp subtree (upstream b11223, commit `8eeca010d`) with:

- **Quantized-KV FlashAttention for sm_89** (`ggml-cuda/fattn-mma-q.cuh`): reads q4/q5/q8 K/V rows once into shared
  memory and dequantizes straight into mma fragments; KQ runs as int8 mma against Q quantized per 32 values; no f16
  copy of K/V. One kernel serves decode and speculative verify, a second path serves prefill. Design:
  [fattn-quantized-kv.md](design/fattn-quantized-kv.md); results: [prefill-depth](../bench/results/prefill-depth/REPORT.md).
- **ReplaySSM rollback** for the Gated-DeltaNet layers: verify records the per-token inputs and a fold applies only
  the accepted ones, bit-exact, instead of snapshotting the recurrent state per draft token. Port of NInfer's design:
  [replayssm-llamacpp.md](design/replayssm-llamacpp.md).
- **KV-budget scheduler** (`--kv-budget`): admits requests against real KV capacity (no "context size exceeded"),
  interleaves prefill chunks with decode, preempts at token boundaries, checkpoints at message boundaries.
- **Per-agent session store** (`--session-ram`, `--session-dir`): sessions are identified by prompt prefix; each
  keeps an append-only mirror of its KV cells plus its recurrent state in pinned host memory, synced after every
  request, restored asynchronously, demoted to NVMe instead of dropped. Design: [session-store.md](design/session-store.md).
- **Own-cells attention window**: K/V views, the KQ mask and FA cover only the cell window of the ubatch's
  sequences, so a resident session no longer slows another session's prefill ([resident-cost](../bench/results/resident-cost/REPORT.md)).
- **GPU-built KQ mask** (`GGML_OP_KQ_MASK`) and a **reduced-vocab MTP draft head** (`--spec-draft-vocab`,
  `deploy/draft-vocab/`) ([step-costs](../bench/results/step-costs/REPORT.md)).
- **Observability** (`--telemetry`, `--request-log-jsonl`, `--web-dir`): `/telemetry`, `/events` SSE, JSONL
  request log, NVML board-energy attribution, per-session tier and restore stats. Schema:
  [observability.md](observability.md); dashboard: [dashboard.md](dashboard.md), source `apps/web`.

Every flag is documented in [engine/tools/server/README.md](../engine/tools/server/README.md).

## Operations

```sh
# one-time images (CUDA 13.1, Ubuntu 24.04)
docker build -f deploy/devel.Dockerfile   -t lovelace-engine-devel:cuda13.1   deploy
docker build -f deploy/engines.Dockerfile -t lovelace-engine-runtime:cuda13.1 deploy

deploy/build-engine.sh llama-server                       # sm_89 build in the devel image -> engine/build-sm89/bin
(cd apps/web && bun install && bun run build)             # dashboard -> apps/web/dist
docker build -f deploy/serve.Dockerfile -t lovelace-serve:latest .
deploy/serve.sh balanced                                  # engine :8090, capture proxy :8080
deploy/serve.sh status | logs [engine|proxy] | stop
```

Profiles (`deploy/profiles/*.args`, default pool from `deploy/serve.sh`):

| profile | KV K / V | pool (cells) | extra |
|---|---|---:|---|
| `balanced` | q5_0 / q5_0 | 262,144 | |
| `quality` | q8_0 / q5_0 | 262,144 | |
| `vision` | q5_0 / q5_0 | 294,912 | `--mmproj` (ggml-org Q8_0 projector) |

`CTX=` overrides the pool; `ENGINE_MEM=` the container memory cap (default `34g`). Models are mounted read-only
from `~/models`. State lives in `${LOVELACE_STATE:-~/.local/state/lovelace}`: `log/requests.jsonl` (request log),
`lib/captures/` (every request body the capture proxy saw), `lib/sessions/` (spilled sessions).

**Memory safety.** The session arena is pinned host memory, which the kernel cannot swap or reclaim. Two bench runs
OOM-killed the desktop on 2026-09-29 (session tier on tmpfs); since then every engine container runs under a hard
`--memory 34g` cap and the arena is 24 GiB, so an overrun kills and restarts the engine (sessions survive on disk),
never the desktop. `bench/final_acceptance.sh` also samples host RAM and stops the engine below 4 GiB available
([STATUS.md](STATUS.md); [final/mem.log](../bench/results/final/mem.log)).

**GPU discipline while developing.** Every GPU job goes through `deploy/gpu-lock.sh` (FIFO queue; it holds until GPU
memory is idle again). `deploy/gpu-watchdog.sh` flags a queue stalled on an idle GPU. Stop production first
(`deploy/serve.sh stop`) and restart it after.

## Measurement

| what | script | output |
|---|---|---|
| charter acceptance, only missing steps | `bench/acceptance.sh` (`report` to regenerate) | `bench/results/acceptance/REPORT.md` |
| final acceptance on one build | `BIN=... CTX=... bench/final_acceptance.sh OUT` | `bench/results/final/` |
| lovelace vs ninfer, same workload | `bench/h2h.sh ENGINE LANES sweep` or `replay:AGENTS:MIN_DEPTH:SECONDS` | `bench/results/h2h/` |
| single-stream depth sweep | `bench/depth_sweep.py` | `*.sweep.out` |
| multi-agent replay | `bench/run_replay.sh`, `bench/replay/` ([README](../bench/replay/README.md)) | `*.replay.out` |
| quality (KL vs Q8_0) | `eval/kld/run_kld.sh` ([README](../eval/kld/README.md)) | `eval/kld/results/` (untracked) |
| greedy identity gate | `bench/greedy_gate.sh RESULT_DIR BIN_A BIN_B` (`NO_SPEC=1` for no speculation) | `COMPARE.txt` |
| op gates | `bench/op_gates.sh` (test-backend-ops, test-gdn-replay) | `*.txt` |

Merge rule for an engine branch: test-backend-ops for the touched ops, test-gdn-replay, the greedy identity gate,
before/after numbers from the branch's own script, and the KLD gate when numerics can change; then
`bench/acceptance.sh` and a report regeneration.

Private data: the replay corpus (`bench/replay/corpus/sessions/`), the KLD windows and results, and proxy captures
are raw agent transcripts and are gitignored. Keep it that way.

## Upstream sync

```sh
git subtree pull --prefix engine llama-upstream <tag> --squash
```

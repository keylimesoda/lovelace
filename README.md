# lovelace

Lovelace is a llama.cpp-based serving engine for Qwen3.8-27B, targeted at a single NVIDIA RTX 4090 (sm_89) with an AMD Ryzen 9 9950X host.

## Architecture

The `engine/` directory is a llama.cpp subtree with ReplaySSM rollback for recurrent state, a KV-budget scheduler that parks sessions in pinned host memory, sm_89 CUDA kernels, and telemetry, request logging, and a web dashboard. `bench/replay/capture_proxy.py` is the front door for client traffic and stores captures separately from the engine's request log.

## Measured quality

The current `docs/quality-kld.md` results use sparse top-64 KL over recorded agent replies. On the weight study, IQ4_XS-3.84bpw weights against the Q8_0 reference had mean KL 0.0163, median KL 0.0011, 95.64% top-1 agreement, and ΔNLL +0.0096. On the KV study, q5_0/q5_0 used 22 KB/token and measured mean KL 0.00155, 98.63% top-1 agreement, ΔNLL +0.00101, and KL 0.00170 at 250k tokens. The quality profile uses q8_0/q5_0 KV; that row measured mean KL 0.00117 and 98.89% top-1 agreement.

## Quick start

The serving image expects the built engine at `engine/build-sm89/bin`, the dashboard at `apps/web/dist`, and the base image `lovelace-engine-runtime:cuda13.1`. Models are mounted from `~/models`; they are not included in the image. Request logs and proxy captures go to `~/.local/state/lovelace` (override with `LOVELACE_STATE`).

Build the engine, dashboard and serving image, then start a profile:

```sh
deploy/build-engine.sh llama-server
(cd apps/web && bun install && bun run build)
docker build -f deploy/serve.Dockerfile -t lovelace-serve:latest .
deploy/serve.sh balanced
```

Available profiles are `balanced` (q5_0/q5_0 KV), `quality` (q8_0/q5_0 KV), and `vision` (balanced KV with the mmproj model). `CTX` overrides the context size; defaults are 327680 for balanced and quality and 262144 for vision. Use `deploy/serve.sh stop`, `status`, or `logs [engine|proxy]` to manage or inspect the services. `deploy/compose.yaml` is an alternative Compose configuration for the balanced profile; run `docker compose -f deploy/compose.yaml up -d` from the repository root.

## Benchmarks and evaluation

- `bench/replay/` replays recorded multi-agent traffic against the OpenAI-compatible API. See its README for extraction and replay commands.
- `bench/bakeoff/` runs and analyzes engine comparison matrices.
- `eval/kld/` measures quantization divergence on recorded reply windows. See its README for inputs and run instructions.

## Upstream

Pull a tagged llama.cpp release into the subtree with:

```sh
git subtree pull --prefix engine llama-upstream <tag> --squash
```

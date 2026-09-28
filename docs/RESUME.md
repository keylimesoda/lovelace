# Resume notes (paused 2026-09-28 ~12:40 for a reboot)

## State
- `main` (pushed): ReplaySSM, KV-budget scheduler + pinned async parking + message-boundary checkpoints,
  observability, sm_89 kernels (quantized-KV FA decode/verify, IQ GEMV small-batch, DeltaNet fusions),
  mode-specific draft ubatch (-ubd 32 for MTP; pool 327,680 cells), deploy profiles/image/capture proxy, docs.
  Gate: bench/results/main-gate/REPORT.md. 327k replay: bench/results/main-327k/REPORT.md.
- Decisions: GPU-5 IQ4_XS-3.84bpw weights, q5_0/q5_0 KV, MTP n=3, -c 327680, -np 6 (docs/quality-kld.md).
- Branch `feat/session-store` (worktree ~/Work/lovelace-wt/sessions, local only, NOT pushed): per-agent session
  store per docs/design/session-store.md. Steps done: identity + RAM/NVMe tiers, measurements
  (pinned 25.5 GB/s; 200k session = 4.66 GB + 1.88 GB checkpoints; zstd < 1%), step 2 append-only KV delta mirror
  (builds, unit tests pass per agent). Last commit: WIP (pause). Next: 2-agent deep smoke, then steps 3-6
  (restore prefetch, shared prefix or remove the flag, telemetry, 6-agent replays). Move the core to an Opus agent
  (user policy: switch to Opus where Luna shows degradation).
- Branch `feat/fattn-prefill` (local): parked capacity option (see docs/design/session-store.md).
- KLD weight-quant candidates (GPU-4 IQ3_S-3.23bpw, ISTA GSQ-RCO-IQ3_S at ~/models/ista) were queued, not run:
  resume with `cd eval/kld && ./run_kld.sh` (finished steps are skipped).
- Production: old `ninfer-qwen` container is stopped; cutover to lovelace is pending the session store.

## After the reboot
1. `docker ps -a`: make sure no lovelace/bench containers auto-started; ninfer-qwen stays stopped.
2. Recreate the tools used by scripts: `pip install uv` into /tmp/uvtool (it was a tmpfs venv), rebuild
   bench/replay/.venv on first `uv run`.
3. `git -C ~/Work/lovelace worktree list`: worktrees survive the reboot (they're on disk).
4. `rm -f /tmp/lovelace-gpu.lock; mkdir -p /tmp/lovelace-gpu.q` (tmpfs is cleared anyway).

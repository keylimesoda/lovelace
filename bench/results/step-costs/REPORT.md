# Step costs (branch feat/step-costs, from main 4b6187d5d)

Question: cut the remaining per-step costs of 6-sequence MTP decode (host gap, MTP draft head, MMQ at 24 rows)
without changing outputs.

## Commands

```sh
D=50000,50000,50000,50000,50000,50000
BIN=../step-base-bin bench/step_profile.sh bench/results/step-costs/base six=$D one=64000   # main (4b6187d5d)
BIN=../step-mask-bin bench/step_profile.sh bench/results/step-costs/mask six=$D one=64000   # item 1 (0c727e790)
bench/op_gates.sh bench/results/step-costs/gates/mask.txt ../step-mask-bin KQ_MASK       # + test-gdn-replay, test-kq-mask
bench/greedy_gate.sh bench/results/step-costs/greedy/mask-mtp ../step-base-bin ../step-mask-bin
NO_SPEC=1 bench/greedy_gate.sh bench/results/step-costs/greedy/mask-nospec ../step-base-bin ../step-mask-bin
# deterministic acceptance (temperature 0; six2 = the same six prompts again in the same server)
GREEDY=1 BIN=../step-base-bin bench/step_profile.sh bench/results/step-costs/accept/base six=$D one=64000 six2=$D
GREEDY=1 BIN=../step-mask-bin bench/step_profile.sh bench/results/step-costs/accept/mask six=$D one=64000 six2=$D
```

## Item 1: KQ mask built on the GPU (0c727e790) — measured

The host uploads one I32 per cell (24-bit position | 8-bit set of the batch sequences that see it, 1.1 MB at 286k
cells) and one (sequence bit, last visible position) pair per row; `GGML_OP_KQ_MASK` writes the f16 mask on the
device (~10 us). A cell in several batch sequences is handled by its visibility set, so no fallback is needed for
it. The host mask stays for ALiBi/SWA, > 8 sequences per stream, positions >= 2^24, and ubatches where the M-RoPE
2D rule hides a cell (image blocks); the choice is part of the graph reuse key.

Step profile, 20 steady steps (`base/`, `mask/` `*.table.txt`):

| | main | GPU mask |
|---|---:|---:|
| 6 x 50k step wall, ms | 57.85 | **54.94** (-5.0%) |
| 6 x 50k host gap, ms | 12.03 | **8.95** |
| 1 x 64k step wall, ms | 31.60 | **30.80** |
| 1 x 64k host gap, ms | 5.31 | 4.75 |
| 6 x 50k summed decode, tok/s (drive) | 312.4 | 308.9 |
| 6 x 50k draft acceptance (drive) | 0.664 | 0.605 |

The drive samples with production settings (temperature, fixed seeds, timing-dependent batching); acceptance
moved with the sampled continuations, not with the mask (the mask is bitwise equal, below), so summed decode in
this table is not a clean comparison; step wall is.

Correctness (`gates/mask.txt`): test-backend-ops KQ_MASK 22/22 (CUDA vs CPU), test-gdn-replay all passed,
test-kq-mask (committed, engine/tests/test-kq-mask.cpp): device mask bitwise equal to the host mask on CUDA and
CPU for 6 and 24 rows, n_pos 1 and 4 (M-RoPE), 286,720 cells, a prefix shared by two batch sequences, non-causal,
3 streams; the M-RoPE image block and 9-sequence cases are refused (host mask).

Greedy gate (`greedy/mask-mtp/COMPARE.txt`, `greedy/mask-nospec/COMPARE.txt`, six sessions at ~20k, 128 tokens):
**sequential 6/6 identical with MTP on and with MTP off: PASS.** With MTP on, the sequential draft counters are
identical too (main and GPU mask both 519 accepted / 724 drafted = 0.717, per-session equal). Concurrent: 1/6
identical on both (MTP and no-MTP), the same as main against itself (step-profile REPORT: timing-dependent batch
composition flips near-ties; margins 0.04-0.06 here), so it does not separate a code change.

### Deterministic draft acceptance: main vs GPU mask (`accept/`, temperature 0, 256 tokens, ignore_eos)

The stochastic drive's 0.664 -> 0.605 was a sampling artifact: at temperature 0 the draft counters are identical,
per request, on every capture, so the draft (MTP) context's mask is not implicated and nothing needed fixing.

| | main | GPU mask |
|---|---:|---:|
| 6 x 50k acceptance (six / six2) | 1062/1392 = 0.763 / 1061/1399 = 0.758 | **identical** (same per-request counts) |
| 1 x 64k acceptance | 178/230 = 0.774 | **identical** |
| 6 x 50k step wall, ms (six / six2) | 56.38 / 57.03 | **52.09 / 51.39** (-8.9%) |
| 6 x 50k host gap, ms | 11.69 / 12.07 | 7.68 / 6.94 |
| 6 x 50k summed decode, tok/s (six / six2) | 346.7 / 349.5 | **364.4 / 378.5** (+7%) |
| 1 x 64k step wall, ms | 30.82 | **28.52** |
| 1 x 64k decode, tok/s | 108.6 | **114.9** |

## Item 2: reduced-vocab MTP draft head — implemented, NOT yet measured

- `--spec-draft-vocab FILE` (MTP only): `llama_set_draft_vocab()` gathers the listed rows of the Q6_K output
  weights once (16384 rows, 69 MB) into a draft head; the Qwen3.5/Qwen3-Next MTP graphs compute logits over those
  rows and scatter them into a full-vocabulary row (-INF elsewhere), so samplers and logits readers are unchanged.
  The target's verification uses the full head: outputs stay exact; only acceptance can change.
- Token set: `bench/draft_vocab.py` (docstring has the regenerate command) → `deploy/draft-vocab/qwen3.8-27b.txt`.
  Ranked by frequency in the agents' rendered assistant turns + 0.25 x frequency in tool results/user messages.
  28 sessions, 2.06 M output tokens, 23,287 distinct: in-corpus coverage 99.41%, leave-one-session-out coverage
  **96.85%** (32k rows: 99.02%). The out-of-sample number matters: step_drive replays corpus sessions, so its
  acceptance is an in-sample (optimistic) measurement; 32k rows may be the safer choice for unseen traffic.
- Bench scripts mount deploy/draft-vocab at /draft-vocab (`step_profile.sh`, `greedy_gate.sh`: `-- --spec-draft-vocab
  /draft-vocab/qwen3.8-27b.txt`; `h2h.sh`: `SERVER_ARGS=...`).
- Missing: step profile (6 x 50k, 1 x 64k) with acceptance and summed decode, greedy gate 6/6, and the
  production profile flag (deploy/profiles/*.args) only after those pass.

## Not done

- Item 0 (from Main): target graph-launch gap regression after the restore-latency merge (1.67 -> 2.19 ms): not
  investigated; a `CPU_SAMPLE=1` step profile of main is the next step.
- Item 3 (MMQ at 24 rows): not started. Note for it: with 24 columns, ntiles_x is 1 for any mmq_x >= 24, so mmq_x
  changes keep the stream-k K partition (bitwise identical); mmq_y or grid changes do not.
- 6-agent replay vs main: not run.

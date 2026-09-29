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

## Item 2: reduced-vocab MTP draft head (0abc321bb) — measured: keep with the 32k set, not the 16k set

- `--spec-draft-vocab FILE` (MTP only): `llama_set_draft_vocab()` gathers the listed rows of the Q6_K output
  weights once (16384 rows, 69 MB) into a draft head; the Qwen3.5/Qwen3-Next MTP graphs compute logits over those
  rows and scatter them into a full-vocabulary row (-INF elsewhere), so samplers and logits readers are unchanged.
  The target's verification uses the full head: outputs stay exact; only acceptance can change.
- Token sets: `bench/draft_vocab.py` → `deploy/draft-vocab/`. Ranked by frequency in the agents' rendered assistant
  turns + 0.25 x frequency in tool results/user messages. 28 sessions, 2.06 M output tokens, 23,287 distinct.
  16k rows (`--n 16384`; the `vocab/v16` runs used it as `qwen3.8-27b.txt`): corpus coverage 99.41%,
  leave-one-session-out 96.85%. 32k rows (now `qwen3.8-27b.txt`, the default `--n`; `qwen3.8-27b-32k.txt` in the
  `vocab/v32` runs; every token the agents emitted plus the most frequent read ones): 100% / 99.02%.
- Held-out sets (`draft_vocab.py --exclude`, the six sessions step_drive uses at 50k): `heldout6-16k.txt`,
  `heldout6-32k.txt` (30,235 rows: all tokens of the other 22 sessions), built from 22 sessions without them;
  coverage of the six sessions' outputs 95.44% / 98.10%. The held-out drives use only those six sessions
  (`SESSIONS=...`, step_drive `--sessions`), at 20k, 35k and 50k (6 concurrent requests each) and 1 x 64k.

Commands (binary `../step-head-bin` = cf57614a5, GPU mask + draft vocab; temperature 0, ignore_eos):

```sh
H=2026-09-27T08-42-53,2026-09-27T08-18-54,2026-09-25T07-15-46,2026-09-26T02-32-58,2026-09-26T09-54-32,2026-09-27T09-11-19
uv run python ../draft_vocab.py --bin ../../engine/build-sm89/bin --model /models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf \
    --n 16384 --exclude $H --out ../../deploy/draft-vocab/heldout6-16k.txt          # and --n 32768 -> heldout6-32k.txt
V="-- --spec-draft-vocab /draft-vocab/heldout6-16k.txt"                              # none for the full head
GREEDY=1 BIN=../step-head-bin bench/step_profile.sh bench/results/step-costs/vocab/h16 six=$D one=64000 $V   # 256 tokens
GREEDY=1 MAX_TOKENS=1024 BIN=../step-head-bin bench/step_profile.sh bench/results/step-costs/vocab1k/h16 six=$D one=64000 $V
SESSIONS=$H GREEDY=1 MAX_TOKENS=1024 BIN=../step-head-bin bench/step_profile.sh bench/results/step-costs/vocab1k/h16-b \
    six20=20000,20000,20000,20000,20000,20000 six35=35000,35000,35000,35000,35000,35000 $V
```

Held-out acceptance, 1024 tokens per request (`vocab1k/`; accepted/drafted, pooled over the 18 six-way requests):

| | full head | 16k held-out | 32k held-out |
|---|---:|---:|---:|
| 6 x 50k | 4427/5125 = 0.864 | 0.856 (-0.78) | 0.858 (-0.56) |
| 6 x 35k | 4276/5575 = 0.767 | 0.752 (-1.53) | 0.774 (+0.67) |
| 6 x 20k | 4340/5388 = 0.806 | 0.793 (-1.25) | 0.807 (+0.11) |
| **pooled 18 requests** | **0.8107** (3.420 tok/step) | **0.7986 (-1.21 pts)** (3.385) | **0.8118 (+0.11 pts)** (3.424) |
| 1 x 64k (one request, noisy) | 661/1085 = 0.609 | 0.585 | 0.640 |

One request moves by +-10 accepted tokens from a single changed draft (it shifts every later draft window), so
single-request and 256-token numbers are noise-level (`vocab/`, 256 tokens: 6 x 50k full 0.763, in-sample 16k
0.764 / 32k 0.759, held-out 16k 0.763 / 32k 0.766). The pooled 16k loss is consistent at every depth; 32k is flat.

Cost (per step; the draft head is `mmvq q6_K`, 1109 us per draft step at 151,936 rows):

| | full head | 16k | 32k |
|---|---:|---:|---:|
| draft head per draft step, us (6 rows) | 1109 | 77 | 139 (30k rows) - 150 (32k) |
| MTP draft step, ms (6 x 50k / 1 x 64k) | 2.00 / 1.47 | 0.975 / 0.475 | 1.04 / 0.54 |
| 6 x 50k step wall, ms (`vocab1k/*/six`) | 49.13 | **46.05** | 50.17 measured, 46.2 expected [a] |
| 1 x 64k step wall, ms | 27.48 | **24.58** | 25.95 (median 25.10) |
| summed decode, 6 x 50k / 35k / 20k, tok/s | 442.6 / 427.1 / 465.0 | 462.7 / 457.0 / 507.3 (+6.9%) | 438.6 / 448.2 / 500.3 (+3.9%) |
| 1 x 64k decode, tok/s | 103.9 | 110.9 | 108.3 (+4.2%) |

[a] The draft vocabulary does not touch the target graph, but in the 32k 6 x 50k capture the target's own kernels
ran 8% slower (matmul 23.55 vs 21.78 ms, FA 10.01 vs 9.68) and the host gap was 1.5 ms higher: machine noise
between holds. Full step minus the measured draft saving (3 x 0.96 ms) gives 46.2 ms.

**Decision:** 16k fails the held-out criterion (-1.21 points > 0.5) despite the larger speedup; **32k passes**
(held-out +0.11 points, summed decode +3.9% over three depths, -2.9 ms of GPU time per 6-sequence step, -2.8 ms
single-stream). Greedy gate with the 32k set (`greedy/vocab32k-mtp/`, main vs `step-head-bin` +
`--spec-draft-vocab`): **sequential 6/6 identical**, draft counters equal (519/724 on both); concurrent 1/6, the
same flips as the mask gate. Enabled: `deploy/draft-vocab/qwen3.8-27b.txt` is the 32k set, `serve.Dockerfile`
copies it to /opt/lovelace/draft-vocab/ and all three deploy profiles pass `--spec-draft-vocab` (serve image not
rebuilt here; production stays stopped).

## Item 0: target graph-launch gap (1.67 -> 2.19 ms after the restore-latency merge) — not a regression

`CPU_SAMPLE=1 BIN=../step-base-bin bench/step_profile.sh bench/results/step-costs/cpu-main six=$D`, then
`bench/gap_samples.py` (new: the decode thread's CUDA runtime calls and CPU samples between the last GPU activity
and each target decode's first kernel) and `bench/step_table.py --gaps` (new: host gap split by the GPU work on
either side). Same window on every capture at hand (6 x 50k):

| capture | build | window, ms | cudaGraphLaunch, ms | cudaGraphExecUpdate, ms/step |
|---|---|---:|---:|---:|
| step-profile/v6 | pre-rebase branch | 1.894 | 1.380 | 0.145 |
| step-profile/final | rebased on restore-latency | 2.297 | 1.559 | 0.239 |
| accept/base | main 4b6187d5d | 2.155 | 1.358 | 0.238 |
| cpu-main | main 4b6187d5d, CPU sampling | **1.756** | 1.266 | 0.156 |
| accept/mask | GPU mask | 1.998 | 1.358 | 0.141 |

The window is the driver's `cudaGraphLaunch` of the 1,974-node target graph (same node count in v6, final and
main); 30% of the decode thread's samples in it are `pthread_mutex_unlock` inside `cuGraphLaunch`. Main spans
1.76-2.16 ms across two captures, as wide as the v6 -> final difference, and its low end is below v6: run-to-run
driver/host variance, not a code regression. The rest: every ~6 steps a graph is re-captured and updated as n_kv
crosses its 256-cell padding (`cudaGraphExecUpdate` ~0.95 ms each, 0.14-0.24 ms/step amortized). Levers, if
wanted: fewer target-graph nodes (fusion), or a coarser n_kv padding to update less often; not pursued here.

## Merge note

feat/resident-cost 896206c36 adds a per-ubatch window `kv_lo` (K/V views and the host mask read cells kv_lo + j,
reuse key checks `kq_mask->op_params[0] == kv_lo`); merged together, `llama_kv_mask_set_keys()` must read cells
kv_lo + j as well and the reuse key must carry kv_lo on the keys tensor. Whoever merges second resolves it.

## Not done

- Item 3 (MMQ at 24 rows): not started. Note for it: with 24 columns, ntiles_x is 1 for any mmq_x >= 24, so mmq_x
  changes keep the stream-k K partition (bitwise identical); mmq_y or grid changes do not.
- 6-agent replay vs main (GPU mask + 32k draft vocab): not run.

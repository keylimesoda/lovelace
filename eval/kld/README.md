# lovelace-kld

Quantization error of local GGUF quants and KV-cache types at long context (32k–250k tokens).
It works like llama.cpp's `llama-perplexity --kl-divergence-base` / `--kl-divergence`, but the base
file is sparse: for each scored token it keeps the top-k log-probs instead of the full vocabulary
(248,320 floats per token). That keeps base files small enough for 250k-token windows.

| file | |
|---|---|
| `build_corpus.py` | picks windows from the replay corpus → `windows.jsonl` (see its docstring) |
| `lovelace_kld.cpp`, `kld_math.h`, `CMakeLists.txt` | the `lovelace-kld` tool |
| `run_kld.sh` | both studies end to end: VRAM plan, base, compare, summary table |
| `plan_vram.py` | per-run VRAM plan from the GGUF tensor sizes → `-ot` FFN offload; time estimate |
| `summarize.py` | summary table over compare results |
| `candidates.jsonl`, `counts.jsonl`, `windows.jsonl` | corpus artifacts (gitignored, private content) |
| `results/` | base files and compare results (gitignored) |

## What is measured

A window is a recorded agent turn, rendered with the model's chat template: context (depth) +
the recorded assistant reply. The reply is the scored span, capped at 1024 tokens. The tool
tokenizes `text[:score_start_char]` and `text[score_start_char:]` separately
(`add_special=false`, `parse_special=true`), which is how the server sees a prompt and its
generated reply. It then decodes the window in `-b`-sized chunks and requests logits only for
the scored positions.

- `base` stores, for each scored position: the actual next token id and its log-prob, the
  log-sum-exp over the full vocabulary (accumulated in double), and the top-k `(id, log-prob)`
  pairs in descending order.
- `compare` recomputes the distribution Q at the same positions with the candidate model and KV
  types. For each position it computes:
  - `KL(P_base || Q)` over the base top-k ids plus one tail bucket:
    `p_tail = 1 - Σ p_topk` and `q_tail = 1 - Σ q(topk ids)`, both clamped to ≥ 1e-10. Merging
    the tail into one bucket can only lower KL, so this is a lower bound on the full-vocabulary
    KL. The gap is small when `p_tail` is small; `p_tail_mean` is reported so you can check.
  - top-1 agreement: `argmax Q` equals the base's top-1 token.
  - NLL of the actual next token under the base and under Q.

  Results are aggregated per window, per `depth_bucket`, and overall: mean, median, p99, and
  p99.9 KL, max KL, top-1 agreement %, mean NLL for both models, ΔNLL = NLL_q − NLL_base, and
  mean p_tail.

`compare` checks that each window tokenizes to the same tokens as in the base run (via a
token hash) and that the vocabulary size matches.

### Base file format (version 1, little-endian)

```
header   char[8] "LVKLDBAS" | u32 version | u32 topk | u32 n_windows | u32 n_vocab
         | u64 FNV-1a-64(model path) | u32 meta_len | meta JSON (model, n_ctx, ngl, ctk, ctv, fa, ...)
window   u32 id_len | id | i32 n_prefix | i32 n_scored | u64 FNV-1a-64(n_prefix, tokens)
         | n_scored × position
position i32 next_id | f32 next_logprob | f64 logsumexp | topk × (i32 id, f32 logprob)
```

With the default `--topk 64`, each position takes 528 bytes, so a base file for 30 windows is
about 10 MB.

## Build (Docker; the host has no CUDA toolkit)

Links against the CUDA build of llama.cpp b11223 in `~/Work/src/llama.cpp/build-sm89`:

```sh
docker run --rm -v /home/ric/Work:/home/ric/Work -w /home/ric/Work/lovelace/eval/kld \
  nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -c '
    set -e
    apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake >/dev/null
    mkdir -p /tmp/stubs && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stubs/libcuda.so.1
    nice -n 19 cmake -S . -B build -DLLAMA_ROOT=/home/ric/Work/src/llama.cpp \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/tmp/stubs
    nice -n 19 cmake --build build -j 8
    chown -R 1000:1000 build
    LD_LIBRARY_PATH=/home/ric/Work/src/llama.cpp/build-sm89/bin:/tmp/stubs ./build/lovelace-kld selftest'
```

`selftest` checks the math only and needs no GPU or model.

## Run

The llama.cpp libraries carry `RUNPATH=/src/build-sm89/bin`, which is the path from their own
build container, so the tool needs `LD_LIBRARY_PATH`. The runtime image comes from
`deploy/engines.Dockerfile` and provides cudart, cuBLAS, and libgomp; `--gpus all` provides
`libcuda`.

```sh
cd ~/Work/lovelace/eval/kld
DRY_RUN=1 ./run_kld.sh   # VRAM plans, time estimates, exact commands; no GPU
./run_kld.sh             # everything, about 3.7 h; finished steps are skipped on rerun
```

`run_kld.sh` refuses to start while more than 1 GiB of VRAM is in use (override with
`FORCE=1`). Each step writes `<out>.part` and renames it to `<out>` when it completes, and tees
its log to `<out>.log`. At the end it writes the summary to `results/summary.txt`. To run a
single step by hand, use the `KLD` docker command at the top of `run_kld.sh` followed by one of
the `+ lovelace-kld ...` lines that `DRY_RUN=1` prints.

### Windows (already done; rerun only when the corpus changes)

```sh
uv run build_corpus.py                                    # estimate pass, writes candidates.jsonl
docker run --rm --user 1000:1000 -e CUDA_VISIBLE_DEVICES= \
  -v /home/ric/Work:/home/ric/Work -v /home/ric/models:/home/ric/models:ro \
  -w /home/ric/Work/lovelace/eval/kld nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -c '
    mkdir -p /tmp/stubs && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stubs/libcuda.so.1
    LD_LIBRARY_PATH=/home/ric/Work/src/llama.cpp/build-sm89/bin:/tmp/stubs nice -n 19 \
      ./build/lovelace-kld tokcount --model /home/ric/models/ref/Qwen3.8-27B-Q8_0.gguf \
        --windows candidates.jsonl --out counts.jsonl'      # vocab only: no weights, no GPU, ~1 min
uv run build_corpus.py --counts counts.jsonl              # exact depths → windows.jsonl
```

The current `windows.jsonl` has 6 windows per bucket, with exact depths:

| bucket | depth (tokens) |
|---|---|
| 32k | 30,968–33,140 |
| 64k | 60,235–67,620 |
| 128k | 120,804–131,753 |
| 170k | 162,708–173,121 |
| 250k | 235,752–249,398 |

All six 250k windows are concatenations; the deepest real turn is 221k tokens. The longest
window, context plus scored tokens, is 249,866 tokens.

## Studies

| study | reference | candidates | buckets |
|---|---|---|---|
| (1) weight quant | Q8_0 + f16 KV | IQ4_XS-3.84bpw + f16 KV | 32k–170k |
| (2) KV quant | IQ4_XS-3.84bpw + f16 KV | same weights + KV q8_0, q4_0, q5_0, q4_1 | 32k–250k |

Study (1) stops at 170k: at 250k, Q8_0 weights (25.4 GiB on the GPU side) plus 15.3 GiB of f16
KV would stream almost every FFN.

Every run uses `-ngl 99 -fa on -b 2048 -ub 2048 --max-outputs 256 -t 16 --load-mode none`. So
all layers, all KV, and the recurrent state are on the GPU. `-ub`, `-fa`, and `--max-outputs`
(the number of logit rows per call) select kernels, so they are identical for base and compare;
`compare` warns if they differ from the base file's metadata.

### Fitting 24 GB: FFN weights on the host, streamed by op-offload

If a config doesn't fit, `plan_vram.py` moves the `ffn_up`/`ffn_gate`/`ffn_down` weights of some
layers to the host with
`-ot 'blk\.(<layers>)\.ffn_(up|gate|down)\.weight=CPU'`. It picks the layers with the largest
FFN tensors until the plan fits. With op-offload, llama.cpp copies those weights to the GPU and
runs the matmul there. Relevant settings:

- op-offload: `--op-offload`/`--no-op-offload` in `common/arg.cpp`, default on. It maps to
  `llama_context_params.op_offload`, which `llama_context_default_params()` sets to `true`;
  `lovelace-kld` leaves it on.
- `GGML_OP_OFFLOAD_MIN_BATCH=1`: the CUDA backend only offloads ops with a batch of at least
  `GGML_OP_OFFLOAD_MIN_BATCH` (default 32). `run_kld.sh` sets it to 1, so the small tail ubatches
  also run on the GPU, with the same kernels as GPU-resident weights. Streaming therefore
  shouldn't change the numbers [inferred from `ggml_backend_sched`'s offload path; not
  measured].
- `--load-mode none`: with mmap, the CPU-override weights stay in pageable `CPU_Mapped` memory,
  and llama.cpp warns about it. Without mmap, they go to the first host buffer type in
  llama.cpp's CPU list, which is pinned `CUDA_Host`. The log should show a
  `CUDA_Host model buffer size` line; the ~25 GB/s transfer assumption requires pinned memory.

The plan with `--vram-gib 22.5` (the 4090 has 23.99 GiB). Numbers are GiB, n_ctx comes from the
selected windows, and the full breakdown is in `DRY_RUN=1 ./run_kld.sh`:

| step | n_ctx | GPU weights | host FFN (layers) | KV | compute~ | total | est. time |
|---|---|---|---|---|---|---|---|
| (1) base Q8_0 f16 | 173,824 | 9.22 | 16.14 (61) | 10.61 | 1.51 | 22.24 | 33.5 min |
| (1) IQ4_XS f16 | 173,824 | 9.49 | 1.89 (15) | 10.61 | 1.38 | 22.38 | 21.3 min |
| (2) base IQ4_XS f16 | 250,112 | 4.86 | 6.52 (59) | 15.27 | 1.38 | 22.41 | 40.8 min |
| (2) IQ4_XS q8_0 | 250,112 | 11.12 | 0.26 (2) | 8.11 | 2.34 | 22.46 | 32.3 min |
| (2) IQ4_XS q4_0 | 250,112 | 11.38 | 0 | 4.29 | 2.20 | 18.77 | 31.9 min |
| (2) IQ4_XS q5_0 | 250,112 | 11.38 | 0 | 5.25 | 2.20 | 19.72 | 31.9 min |
| (2) IQ4_XS q4_1 | 250,112 | 11.38 | 0 | 4.77 | 2.20 | 19.25 | 31.9 min |

The estimated total is 224 min (about 3.7 h).

How the columns are computed:

- **Weights** are the summed GGUF tensor bytes of the 64 trunk layers plus `output.weight`.
  `token_embd` always stays on the CPU. IQ4_XS's MTP layer (`blk.64`) isn't loaded.
- **KV** is n_ctx × 16 attention layers × 4 KV heads × 256 × (K + V bytes). That is 64 KiB per
  token for f16, 34 for q8_0, 22 for q5_0, 20 for q4_1, and 18 for q4_0. All totals also include
  0.15 GiB of recurrent state (48 Gated-DeltaNet layers) and 0.75 GiB of fixed CUDA/cuBLAS
  overhead.
- **Compute** is an estimate, not a measurement. It covers logits for 256 rows (0.24 GiB), the
  activations of one 2048-token ubatch, and, when FFNs are streamed, staging for one layer.
  Quantized KV adds FA's f16 copy of K and V (0.95 GiB at 250k). The real sizes appear in each
  log as llama.cpp's `... buffer size` lines. If a run hits out-of-memory, or those lines exceed
  the plan, rerun with a lower budget, e.g. `VRAM_GIB=21.5 ./run_kld.sh`.
- **Time** is prefill tokens / 2000 tok/s, plus one stream of the host FFN bytes per ubatch at
  25 GB/s, plus the model load at 2 GB/s. The scored tail adds four small ubatches per window
  (`--max-outputs 256`), and each one streams the offloaded FFNs again.
  - Host-side post-processing is not included. It measured 0.9 ms per scored row for `base` and
    1.4 ms for `compare` (248k vocab, one core), which is under a minute per run.
- **Host RAM**: `--load-mode none` keeps the host FFNs in pinned memory, up to 16.1 GiB for the
  Q8_0 base.

## Output

Each `compare` prints a table (per window, then per bucket and overall) and writes the same
numbers to its JSON file (`overall`, `buckets`, `windows[].stats`). The JSON also records the
run parameters and the base file's metadata. `summarize.py` puts several results side by side:

```sh
python3 summarize.py results/wq-iq4xs-f16.json results/kv-iq4xs-*.json
```

## Options

`lovelace-kld base|compare` accept:

| option | meaning |
|---|---|
| `-c N` | context size; `0` (default) = the longest selected window, rounded up to 256 |
| `-ngl N` | layers on the GPU |
| `-ot REGEX=BUFT[,...]` | tensor buffer overrides (repeatable). BUFT is a device type (`CPU`, `CUDA0`) or a pinned host type (`CUDA_Host`) |
| `-ctk T`, `-ctv T` | KV cache types |
| `-fa on\|off\|auto` | flash attention |
| `-b N`, `-ub N` | batch and ubatch size |
| `--max-outputs N` | llama's `n_outputs_max`: logit rows per decode call. Bounds the logits part of the compute buffer and the host logits buffer |
| `--load-mode auto\|none\|mmap\|...` | model load mode |
| `--buckets B,...` | only these depth buckets. `compare` also accepts a base file that covers more buckets |
| `-t N` | threads |
| `--topk N` (base) | top-k entries stored per position |
| `--max-scored N` (base) | scored-token cap per window |
| `--verbose` | full llama.cpp log |

Notes:

- `base` skips windows that exceed n_ctx, with a warning; `compare` fails instead of skipping.
- Quantized V cache requires `-fa on`. This llama.cpp build has `GGML_CUDA_FA_ALL_QUANTS=ON`,
  so q5_0 and q4_1 have CUDA FA kernels.
- Noise floor: CPU and CUDA kernels round differently, as do different `-ub`/`-fa` settings. In
  a CPU test with a tiny model, changing `-ub` 32→64 or `-fa` gave a KL of about 0.003, while
  changing `-b` alone gave exactly 0. `run_kld.sh` holds all of these fixed, and every matmul
  runs on the GPU.

# lovelace-kld

Quantization error of local GGUF quants and KV-cache types at long context (32k–250k tokens).
It works like llama.cpp's `llama-perplexity --kl-divergence-base` / `--kl-divergence`, but the base
file is sparse: for each scored token it keeps the top-k log-probs instead of the full vocabulary
(248,320 floats per token). That keeps base files small enough for 250k-token windows.

| file | |
|---|---|
| `build_corpus.py` | picks windows from the replay corpus → `windows.jsonl` (see its docstring) |
| `lovelace_kld.cpp`, `kld_math.h`, `CMakeLists.txt` | the `lovelace-kld` tool |
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
cd ~/Work/lovelace/eval/kld && mkdir -p results
KLD='docker run --rm --gpus all --user 1000:1000 --ipc=host
  -v /home/ric/Work:/home/ric/Work -v /home/ric/models:/home/ric/models:ro
  -w /home/ric/Work/lovelace/eval/kld
  -e LD_LIBRARY_PATH=/home/ric/Work/src/llama.cpp/build-sm89/bin
  lovelace-engine-runtime:cuda13.1 ./build/lovelace-kld'
```

### 0. Windows (already done; rerun only when the corpus changes)

```sh
/tmp/uvtool/bin/uv run build_corpus.py                     # estimate pass, writes candidates.jsonl
docker run --rm --user 1000:1000 -e CUDA_VISIBLE_DEVICES= \
  -v /home/ric/Work:/home/ric/Work -v /home/ric/models:/home/ric/models:ro \
  -w /home/ric/Work/lovelace/eval/kld nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -c '
    mkdir -p /tmp/stubs && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stubs/libcuda.so.1
    LD_LIBRARY_PATH=/home/ric/Work/src/llama.cpp/build-sm89/bin:/tmp/stubs nice -n 19 \
      ./build/lovelace-kld tokcount --model /home/ric/models/ref/Qwen3.8-27B-Q8_0.gguf \
        --windows candidates.jsonl --out counts.jsonl'      # vocab only: no weights, no GPU, ~1 min
/tmp/uvtool/bin/uv run build_corpus.py --counts counts.jsonl   # exact depths → windows.jsonl
```

### 1. Base: Q8_0 with f16 KV, partial offload

```sh
$KLD base --model /home/ric/models/ref/Qwen3.8-27B-Q8_0.gguf --windows windows.jsonl \
  --out results/base-q8_0-f16.bin --topk 64 \
  -ngl 28 -ctk f16 -ctv f16 -fa on -b 2048 -ub 512 -t 16
```

### 2. Compare: IQ4_XS at KV f16, q8_0, q4_0

```sh
M=/home/ric/models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf
B=results/base-q8_0-f16.bin
$KLD compare --model $M --windows windows.jsonl --base $B --out results/iq4xs-kv-f16.json  -ngl 44 -ctk f16  -ctv f16  -fa on -b 2048 -ub 512 -t 16
$KLD compare --model $M --windows windows.jsonl --base $B --out results/iq4xs-kv-q8_0.json -ngl 60 -ctk q8_0 -ctv q8_0 -fa on -b 2048 -ub 512 -t 16
$KLD compare --model $M --windows windows.jsonl --base $B --out results/iq4xs-kv-q4_0.json -ngl 99 -ctk q4_0 -ctv q4_0 -fa on -b 2048 -ub 512 -t 16
```

Each command prints a table (per window, then per bucket and overall) and writes the same
numbers to the JSON file (`overall`, `buckets`, `windows[].stats`). The JSON also records the run
parameters and the base file's metadata.

### Noise floor (recommended)

CPU and CUDA kernels do not round identically, so `-ngl` alone shifts the logits slightly. So do
`-ub` (the kernel choice depends on the ubatch size) and `-fa`. Changing `-b` alone had no
effect in a CPU test.
Keep `-ub` and `-fa` the same for base and compare runs, and measure the floor by comparing the
reference against itself with a different offload split:

```sh
$KLD compare --model /home/ric/models/ref/Qwen3.8-27B-Q8_0.gguf --windows windows.jsonl --base $B \
  --out results/q8_0-kv-f16-floor.json -ngl 24 -ctk f16 -ctv f16 -fa on -b 2048 -ub 512 -t 16
```

## Notes

- `n_ctx`: `-c 0` (the default) sizes the context from the longest window, rounded up to 256.
  For the current `windows.jsonl` (longest window 249,866 tokens) that gives 250,112. The limit
  is the model's training context (262,144). `base` skips windows that don't fit, with a warning;
  `compare` fails instead of skipping.
- Starting `-ngl` values: Qwen3.8-27B (`qwen35`) has 64 layers, and one layer in four uses full
  attention (4 KV heads × 256 dims), so f16 KV at 250k tokens needs about 16 GB. The KV cache
  follows its layer, so partial offload puts the KV of CPU-side layers in host RAM. Each block
  of 4 layers costs about 1.7 GB of Q8_0 weights + 1.0 GB of f16 KV, or 0.8 GB of IQ4_XS
  weights + 1.0 / 0.53 / 0.28 GB of f16 / q8_0 / q4_0 KV. The compute buffer adds about 1.5 GB,
  including 0.5 GB for logits at `-ub 512`.
  - If you hit out-of-memory, lower `-ngl` or `-ub`. The results don't depend on it, apart from
    the noise floor above.
  - Quantized V cache requires `-fa on`.
- On the host, each output row gets a few passes over its 248k logits (max, log-sum-exp in
  double, a top-k heap). Only scored rows are copied to the host. A chunk never has more than
  `-ub` output rows, which bounds the host logits buffer at `-ub` × 1 MB.

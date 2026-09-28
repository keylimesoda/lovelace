#!/usr/bin/env bash
# lovelace-kld studies end to end (README "Studies"). Needs an otherwise idle RTX 4090.
#
#   ./run_kld.sh              run every step; steps whose output already exists are skipped
#   DRY_RUN=1 ./run_kld.sh    print the VRAM plans, time estimates and commands only (no GPU)
#   VRAM_GIB=21.5 ./run_kld.sh  plan against less VRAM (after an OOM, or if the printed
#                               llama.cpp buffer sizes exceed the plan)
#
# (1) weight quantization: base Q8_0 + f16 KV vs IQ4_XS + f16 KV, buckets 32k-170k
#     (Q8_0 weights + f16 KV beyond ~175k tokens would stream nearly every FFN)
# (2) KV quantization: base IQ4_XS + f16 KV vs IQ4_XS + q8_0 / q4_0 / q5_0 / q4_1 KV, all buckets
#
# Every layer and all KV live on the GPU (-ngl 99). When a config doesn't fit, plan_vram.py moves
# the FFN weights of some layers to pinned host memory (-ot ...=CPU with --load-mode none), and
# llama.cpp's op-offload (on by default) streams them to the GPU for every ubatch.
# GGML_OP_OFFLOAD_MIN_BATCH=1 streams even the small tail ubatches, so every matmul runs on the GPU
# with the same kernels whether its weights are resident or streamed.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

REF=/home/ric/models/ref/Qwen3.8-27B-Q8_0.gguf
IQ4=/home/ric/models/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf
WINDOWS=windows.jsonl
SHORT=32k,64k,128k,170k
ALL=32k,64k,128k,170k,250k
KV_CANDIDATES=(q8_0 q4_0 q5_0 q4_1)
VRAM_GIB=${VRAM_GIB:-22.5}
DRY_RUN=${DRY_RUN:-0}
UV=/tmp/uvtool/bin/uv
LLAMA_LIB=/home/ric/Work/src/llama.cpp/build-sm89/bin

# identical for every base and compare run: the kernel choice depends on -ub, -fa and --max-outputs
B=2048
UB=2048
MAX_OUT=256
CTX=(-ngl 99 -fa on -b "$B" -ub "$UB" --max-outputs "$MAX_OUT" -t 16 --load-mode none)

KLD=(docker run --rm --gpus all --user 1000:1000 --ulimit memlock=-1:-1
    -v /home/ric/Work:/home/ric/Work -v /home/ric/models:/home/ric/models:ro -w "$PWD"
    -e LD_LIBRARY_PATH="$LLAMA_LIB" -e GGML_OP_OFFLOAD_MIN_BATCH=1
    lovelace-engine-runtime:cuda13.1 ./build/lovelace-kld)

plan() { # model buckets kv-type label -> stdout: -ot value ("" = fits); stderr: plan + estimate
    "$UV" run -q --no-project --with numpy --with pyyaml python plan_vram.py \
        --model "$1" --windows "$WINDOWS" --buckets "$2" --ctk "$3" --ctv "$3" \
        -b "$B" -ub "$UB" --max-outputs "$MAX_OUT" --vram-gib "$VRAM_GIB" --label "$4"
}

step() { # base|compare out model buckets kv-type label [base-file]
    local cmd=$1 out=$2 model=$3 buckets=$4 kv=$5 label=$6 base=${7:-}
    local ot
    ot=$(plan "$model" "$buckets" "$kv" "$label")
    local args=("$cmd" --model "$model" --windows "$WINDOWS" --buckets "$buckets" --out "$out.part"
        -ctk "$kv" -ctv "$kv" "${CTX[@]}")
    [[ -n $base ]] && args+=(--base "$base")
    [[ -n $ot ]] && args+=(-ot "$ot")
    echo "+ lovelace-kld ${args[*]}"
    [[ $DRY_RUN == 1 ]] && { echo; return; }
    if [[ -s $out ]]; then
        echo "  $out exists, skipping"
        return
    fi
    local t0=$SECONDS
    # one GPU-mutex hold per step, so other GPU jobs interleave between steps
    "$(dirname "$0")/../../deploy/gpu-lock.sh" "${KLD[@]}" "${args[@]}" 2>&1 | tee "$out.log"
    mv "$out.part" "$out"
    echo "  $label: $(((SECONDS - t0) / 60)) min"
    echo
}

[[ -x build/lovelace-kld ]] || { echo "build/lovelace-kld missing: build it first (README)"; exit 1; }
if [[ $DRY_RUN != 1 ]]; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
    echo "note: ${used} MiB in use at start; each step waits for the GPU mutex (deploy/gpu-lock.sh)"
fi
mkdir -p results

echo "== (2) KV quantization, buckets $ALL"
step base results/kv-base-iq4xs-f16.bin "$IQ4" "$ALL" f16 kv-base-iq4xs-f16
for t in "${KV_CANDIDATES[@]}"; do
    step compare "results/kv-iq4xs-$t.json" "$IQ4" "$ALL" "$t" "kv-iq4xs-$t" results/kv-base-iq4xs-f16.bin
done

echo "== (1) weight quantization, buckets $SHORT"
step base results/wq-base-q8_0-f16.bin "$REF" "$SHORT" f16 wq-base-q8_0-f16
step compare results/wq-iq4xs-f16.json "$IQ4" "$SHORT" f16 wq-iq4xs-f16 results/wq-base-q8_0-f16.bin

[[ $DRY_RUN == 1 ]] && exit 0
results=(results/wq-iq4xs-f16.json)
for t in "${KV_CANDIDATES[@]}"; do results+=("results/kv-iq4xs-$t.json"); done
python3 summarize.py "${results[@]}" | tee results/summary.txt

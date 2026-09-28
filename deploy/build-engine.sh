#!/usr/bin/env bash
# Build the lovelace engine for this box (sm_89 RTX 4090, Zen 4/5 AVX-512) inside the CUDA devel image.
# Usage: deploy/build-engine.sh [target ...]
# env: JOBS (default 32), NICE (default 0), BUILD_DIR (default build-sm89), ENGINE_DIR (default <repo>/engine)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE_DIR=${ENGINE_DIR:-$ROOT/engine}
JOBS=${JOBS:-32}
NICE=${NICE:-0}
BUILD_DIR=${BUILD_DIR:-build-sm89}
TARGETS=("$@")
mkdir -p "$HOME/.cache/lovelace-ccache"
docker run --rm --user "$(id -u):$(id -g)" -v "$ENGINE_DIR:/src" -w /src -e HOME=/tmp \
  -v "$HOME/.cache/lovelace-ccache:/ccache" -e CCACHE_DIR=/ccache -e CCACHE_MAXSIZE=20G -e CCACHE_BASEDIR=/src \
  lovelace-engine-devel:cuda13.1 bash -lc "
    set -e
    # always (re)configure: CMake globs template-instance sources at configure time, so new files need it
    {
      cmake -B $BUILD_DIR -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DGGML_NATIVE=OFF -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON \
        -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_QUANTS=all -DLLAMA_BUILD_TESTS=ON \
        -DCMAKE_C_FLAGS=-march=znver4 -DCMAKE_CXX_FLAGS=-march=znver4 \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/cuda-stubs -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/opt/cuda-stubs
    }
    nice -n $NICE cmake --build $BUILD_DIR -j$JOBS ${TARGETS[*]:+--target ${TARGETS[*]}}
  "

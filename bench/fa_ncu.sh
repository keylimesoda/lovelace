#!/usr/bin/env bash
# Nsight Compute profile of one FlashAttention kernel launch from a test-backend-ops perf case, in one GPU-lock hold.
#   bench/fa_ncu.sh OUT_PREFIX BIN_DIR CASE_REGEX [KERNEL_REGEX]
#     e.g. bench/fa_ncu.sh bench/results/prefill-depth/ncu-pf engine/build-sm89/bin \
#            'kv=131072,nb=1024,type_K=q5_0,type_V=q5_0,.*causal' flash_attn_ext_mma_q_pf
# Writes OUT_PREFIX.ncu-rep and OUT_PREFIX.txt (details page: speed of light, workload, instruction and warp state
# statistics). KERNEL_REGEX default: flash_attn_ext (the first matching launch after 2 warm-up launches is profiled).
set -euo pipefail
OUT=$(realpath -m "$1") BIN=$(realpath "$2") CASE=$3 KREGEX=${4:-flash_attn_ext}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$OUT")"
"$HERE/../deploy/gpu-lock.sh" docker run --rm --gpus all --cap-add SYS_ADMIN -e LD_LIBRARY_PATH=/e -v "$BIN:/e:ro" \
  -v "$(dirname "$OUT"):/out" lovelace-engine-devel:cuda13.1 bash -c "
    ncu -f -o /out/$(basename "$OUT") --kernel-name regex:'$KREGEX' --launch-skip 2 --launch-count 1 \
      --section SpeedOfLight --section ComputeWorkloadAnalysis --section InstructionStats --section SchedulerStats \
      --section WarpStateStats --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section SourceCounters \
      --import-source no /e/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p '$CASE' > /out/$(basename "$OUT").run.txt 2>&1
    ncu -i /out/$(basename "$OUT").ncu-rep --page details > /out/$(basename "$OUT").txt
    chown $(id -u):$(id -g) /out/$(basename "$OUT")*"
grep -E "Duration|Compute \(SM\) Throughput|Memory Throughput|Registers|Achieved Occupancy|Issue Slots Busy|No Eligible|Warp Cycles Per Issued|Stall|Executed Ipc" "$OUT.txt" | head -40

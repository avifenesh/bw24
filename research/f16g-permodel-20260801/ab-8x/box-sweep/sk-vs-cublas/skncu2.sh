#!/usr/bin/env bash
set -u
cd ~/arc-sk
export CUDA_VISIBLE_DEVICES=2
export PATH=$HOME/cuda-13.3.1/bin:/opt/nvidia/nsight-compute/2025.4.1:$PATH
M=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PF=research/e2e/prompts/board-2048.txt
D=/tmp/skncu
NCU=/opt/nvidia/nsight-compute/2025.4.1/ncu
SEC="--section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats --section WarpStateStats --section LaunchStats"
echo "=== ncu f16g2 (sk)"
MEMRA_MOE_F16G=2 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PROMPT_FILE=$PF \
  $NCU --kernel-name regex:moe_f16g_sk_kernel --launch-skip 130 --launch-count 6 \
  $SEC -f -o $D/ncu-sk ./target/release/run-gen "$M" > $D/ncu-sk.log 2>&1
echo rc=$?
echo "=== ncu f16g1 (cublas cutlass grouped)"
MEMRA_MOE_F16G=1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PROMPT_FILE=$PF \
  $NCU --kernel-name "regex:cutlass_80_tensorop_f16_s16816gemm_f16_grouped" --launch-skip 130 --launch-count 6 \
  $SEC -f -o $D/ncu-cublas ./target/release/run-gen "$M" > $D/ncu-cublas.log 2>&1
echo rc=$?
echo "=== NCU DONE"

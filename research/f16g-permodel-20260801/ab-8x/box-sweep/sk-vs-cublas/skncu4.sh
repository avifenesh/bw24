#!/usr/bin/env bash
set -u
cd /home/ubuntu/arc-sk
M=/home/ubuntu/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PF=research/e2e/prompts/board-2048.txt
D=/tmp/skncu
NCU=/opt/nvidia/nsight-compute/2025.4.1/ncu
SEC="--section SpeedOfLight --section Occupancy --section SchedulerStats --section LaunchStats"
echo "=== ncu f16g2 (sk) trimmed"
sudo env CUDA_VISIBLE_DEVICES=2 MEMRA_MOE_F16G=2 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PROMPT_FILE=$PF \
  $NCU --kernel-name regex:moe_f16g_sk_kernel --launch-skip 131 --launch-count 4 \
  $SEC -f -o $D/ncu-sk ./target/release/run-gen "$M" > $D/ncu-sk.log 2>&1
echo rc=$?
echo "=== ncu f16g1 (cublas cutlass grouped) trimmed"
sudo env CUDA_VISIBLE_DEVICES=2 MEMRA_MOE_F16G=1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PROMPT_FILE=$PF \
  $NCU --kernel-name regex:cutlass --launch-skip 131 --launch-count 4 \
  $SEC -f -o $D/ncu-cublas ./target/release/run-gen "$M" > $D/ncu-cublas.log 2>&1
echo rc=$?
sudo chown ubuntu:ubuntu $D/ncu-sk.ncu-rep $D/ncu-cublas.ncu-rep 2>/dev/null
echo "=== NCU DONE"

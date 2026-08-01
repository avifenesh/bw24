#!/usr/bin/env bash
set -u
cd /home/ubuntu/arc-sk
M=/home/ubuntu/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PF=research/e2e/prompts/board-2048.txt
D=/tmp/skncu
NCU=/opt/nvidia/nsight-compute/2025.4.1/ncu
SEC="--section SpeedOfLight --section Occupancy --section SchedulerStats --section LaunchStats"
echo "=== ncu f16g1 take 3: kernel base name is Kernel2"
sudo env CUDA_VISIBLE_DEVICES=2 MEMRA_MOE_F16G=1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PROMPT_FILE=$PF \
  $NCU --kernel-name regex:Kernel2 --launch-skip 131 --launch-count 4 \
  $SEC -f -o $D/ncu-cublas ./target/release/run-gen "$M" > $D/ncu-cublas.log 2>&1
echo rc=$?
sudo chown ubuntu:ubuntu $D/ncu-cublas.ncu-rep 2>/dev/null
echo "=== DONE"

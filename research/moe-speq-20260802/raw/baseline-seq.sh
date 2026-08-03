#!/usr/bin/env bash
set -u
cd ~/receipts/moe-speq
for i in 2 3; do
  {
    echo "=== boot-run$i start $(date -u +%FT%TZ) ==="
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader -i 0
  } > boot-run$i.log 2>&1
  env CUDA_VISIBLE_DEVICES=0 \
    MEMRA_PROMPT_FILE=/home/ubuntu/memra/research/gemma4-bringup/depth-prompt-1736.txt \
    MEMRA_NGEN=128 \
    /home/ubuntu/memra/target/release/run-gen /opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime >> boot-run$i.log 2>&1
  echo "=== boot-run$i exit=$? $(date -u +%FT%TZ) ===" >> boot-run$i.log
done
echo BASELINE-SEQ-DONE

#!/bin/bash
set -u
LABEL=$1; MODEL=$2
OUT=/home/avifenesh/projects/wt-prefill/research/prefill-gemm-20260806
W=/home/avifenesh/projects/wt-prefill
mkdir -p $OUT/nsys
cd $OUT/nsys
nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader > $OUT/logs/nsys-$LABEL.gpu
MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
  /usr/local/cuda-13.1/bin/nsys profile -t cuda -o nsys-$LABEL -f true --stats=false \
  $W/target/release/run-gen "$MODEL" > $OUT/logs/nsys-$LABEL.log 2>&1
rc=$?
/usr/local/cuda-13.1/bin/nsys stats --report cuda_gpu_kern_sum --format csv \
  -o nsys-$LABEL nsys-$LABEL.nsys-rep >> $OUT/logs/nsys-$LABEL.log 2>&1
echo "rc=$rc"

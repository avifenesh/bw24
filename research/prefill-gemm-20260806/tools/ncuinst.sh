#!/bin/bash
set -u
LABEL=$1; MODEL=$2; KERN=$3
OUT=/home/avifenesh/projects/wt-prefill/research/prefill-gemm-20260806
W=/home/avifenesh/projects/wt-prefill
M="smsp__inst_executed_pipe_tensor.sum,\
smsp__inst_executed_pipe_fma.sum,\
smsp__inst_executed_pipe_alu.sum,\
smsp__inst_executed_pipe_lsu.sum,\
smsp__inst_executed.sum,\
smsp__inst_executed_pipe_fp16.sum,\
sm__cycles_elapsed.sum,\
sm__pipe_tensor_cycles_active.sum,\
sm__pipe_fma_cycles_active.sum,\
sm__pipe_alu_cycles_active.sum,\
sm__inst_executed_pipe_tensor_op_imma.sum"
MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 \
  /usr/local/cuda-13.1/bin/ncu -k "$KERN" -s 40 -c 2 --metrics "$M" \
  --csv --log-file $OUT/ncu/ncuinst-$LABEL.csv \
  $W/target/release/run-gen "$MODEL" > $OUT/logs/ncuinst-$LABEL.log 2>&1
echo "rc=$?"

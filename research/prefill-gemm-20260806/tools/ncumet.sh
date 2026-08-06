#!/bin/bash
set -u
LABEL=$1; MODEL=$2; KERN=$3
OUT=/home/avifenesh/projects/wt-prefill/research/prefill-gemm-20260806
W=/home/avifenesh/projects/wt-prefill
M="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum"
MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 \
  /usr/local/cuda-13.1/bin/ncu -k "$KERN" -s 40 -c 3 --metrics "$M" \
  --csv --log-file $OUT/ncu/ncumet-$LABEL.csv \
  $W/target/release/run-gen "$MODEL" > $OUT/logs/ncumet-$LABEL.log 2>&1
echo "rc=$?"

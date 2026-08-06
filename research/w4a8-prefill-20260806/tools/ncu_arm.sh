#!/bin/bash
# ncu on one arm's prefill GEMM: instruction mix (which pipe issues how much) + stall reasons.
# Usage: ncu_arm.sh <label> <model> <kernel-regex> [env...]
set -u
LABEL=$1; MODEL=$2; KERN=$3; shift 3
W=/home/avifenesh/projects/wt-w4a8
OUT=$W/research/w4a8-prefill-20260806
NCU=/usr/local/cuda-13.1/bin/ncu
mkdir -p "$OUT/ncu" "$OUT/logs"
INST="smsp__inst_executed_pipe_tensor.sum,\
smsp__inst_executed_pipe_fma.sum,\
smsp__inst_executed_pipe_alu.sum,\
smsp__inst_executed_pipe_lsu.sum,\
smsp__inst_executed.sum,\
sm__cycles_elapsed.sum,\
sm__pipe_tensor_cycles_active.sum,\
sm__pipe_fma_cycles_active.sum,\
sm__pipe_alu_cycles_active.sum"
MET="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum"
for kind in inst met; do
  [ "$kind" = inst ] && M="$INST" C=2 || { M="$MET"; C=3; }
  # ncu needs root for GPU perf counters on this box (ERR_NVGPUCTRPERM otherwise);
  # sudo -n env keeps the arm's env vars across the privilege boundary.
  sudo -n env "$@" MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 \
    HOME="$HOME" \
    $NCU -k "$KERN" -s 40 -c $C --metrics "$M" \
    --csv --log-file "$OUT/ncu/ncu$kind-$LABEL.csv" \
    "$W/target/release/run-gen" "$MODEL" > "$OUT/logs/ncu$kind-$LABEL.log" 2>&1
  echo "$kind rc=$?"
done

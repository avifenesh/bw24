#!/bin/bash
set -u
LABEL=$1; MODEL=$2; KERN=$3
OUT=/home/avifenesh/projects/wt-prefill/research/prefill-gemm-20260806
W=/home/avifenesh/projects/wt-prefill
mkdir -p $OUT/ncu
NCUSEC="--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis"
MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 \
  /usr/local/cuda-13.1/bin/ncu -k "$KERN" -s 40 -c 3 $NCUSEC \
  --csv --log-file $OUT/ncu/ncu-$LABEL.csv \
  $W/target/release/run-gen "$MODEL" > $OUT/logs/ncu-$LABEL.log 2>&1
echo "rc=$?"

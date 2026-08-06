#!/bin/bash
# Positive control: does MEMRA_MMQ_F8F4=1 actually swap the prefill GEMM kernel?
# Exports the kern-sum CSV, then DELETES the .nsys-rep (never committed — it captures process env).
set -u
LABEL=$1; MODEL=$2; shift 2
W=/home/avifenesh/projects/wt-w4a8
OUT=$W/research/w4a8-prefill-20260806
NSYS=/usr/local/cuda-13.1/bin/nsys
mkdir -p "$OUT/nsys" "$OUT/logs"
TMP=$(mktemp -d /tmp/w4a8-nsys-XXXX)
cd "$TMP" || exit 1
nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader > "$OUT/logs/nsys-$LABEL.gpu"
env "$@" MEMRA_PROMPT_FILE=$W/research/e2e/prompts/pp512.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
  $NSYS profile -t cuda -o "nsys-$LABEL" -f true --stats=false \
  "$W/target/release/run-gen" "$MODEL" > "$OUT/logs/nsys-$LABEL.log" 2>&1
rc=$?
$NSYS stats --report cuda_gpu_kern_sum --format csv -o "nsys-$LABEL" "nsys-$LABEL.nsys-rep" \
  >> "$OUT/logs/nsys-$LABEL.log" 2>&1
cp "$TMP"/nsys-$LABEL*_cuda_gpu_kern_sum.csv "$OUT/nsys/" 2>/dev/null
rm -rf "$TMP"   # .nsys-rep never leaves /tmp
echo "rc=$rc"

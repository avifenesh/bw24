#!/usr/bin/env bash
# moe-speq capture: one prompt class per device. Usage: capture-dev.sh <dev> <tag> <prompt> <chat01>
set -u
DEV=$1; TAG=$2; PROMPT=$3; CHAT=$4
BIN=/home/ubuntu/memra/target/release
ART=/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime
OUT=/home/ubuntu/receipts/moe-speq
CH=()
if [ "$CHAT" = "1" ]; then CH=(MEMRA_CHAT=1); fi

# 1. plain decode with route trace (per-step actual expert need)
rm -f $OUT/trace-$TAG.txt
{
  echo "=== trace-$TAG start $(date -u +%FT%TZ) dev=$DEV chat=$CHAT prompt=$PROMPT ==="
  nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader -i $DEV
} > $OUT/gen-trace-$TAG.log 2>&1
env CUDA_VISIBLE_DEVICES=$DEV "${CH[@]}" \
  MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 \
  MEMRA_MOE_TRACE=$OUT/trace-$TAG.txt \
  $BIN/run-gen $ART >> $OUT/gen-trace-$TAG.log 2>&1
echo "=== trace-$TAG exit=$? $(date -u +%FT%TZ) ===" >> $OUT/gen-trace-$TAG.log

# 2. spec runs K=1,2,4 with per-round debug (acceptance chain positions)
for K in 1 2 4; do
  {
    echo "=== spec-$TAG-k$K start $(date -u +%FT%TZ) dev=$DEV ==="
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader -i $DEV
  } > $OUT/spec-$TAG-k$K.log 2>&1
  env CUDA_VISIBLE_DEVICES=$DEV "${CH[@]}" \
    MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 \
    MEMRA_SPEC_K=$K MEMRA_DEBUG_SPEC=1 MEMRA_SPEC_STATS=1 \
    $BIN/run-spec $ART >> $OUT/spec-$TAG-k$K.log 2>&1
  echo "=== spec-$TAG-k$K exit=$? $(date -u +%FT%TZ) ===" >> $OUT/spec-$TAG-k$K.log
done
echo "CAPTURE-$TAG-DONE"

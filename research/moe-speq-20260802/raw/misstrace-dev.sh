#!/usr/bin/env bash
# moe-speq miss-trace pass: patched run-gen, MISS + ROUTE traces from the SAME forwards.
set -u
DEV=$1; TAG=$2; PROMPT=$3; CHAT=$4
BIN=/home/ubuntu/memra-moespeq/target/release
ART=/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime
OUT=/home/ubuntu/receipts/moe-speq
CH=()
if [ "$CHAT" = "1" ]; then CH=(MEMRA_CHAT=1); fi
rm -f $OUT/miss-$TAG.txt $OUT/route-$TAG.txt
{
  echo "=== miss-$TAG start $(date -u +%FT%TZ) dev=$DEV chat=$CHAT prompt=$PROMPT bin=patched(memra-moespeq) ==="
  nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader -i $DEV
} > $OUT/gen-miss-$TAG.log 2>&1
env CUDA_VISIBLE_DEVICES=$DEV "${CH[@]}" \
  MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 \
  MEMRA_MOE_MISS_TRACE=$OUT/miss-$TAG.txt \
  MEMRA_MOE_TRACE=$OUT/route-$TAG.txt \
  $BIN/run-gen $ART >> $OUT/gen-miss-$TAG.log 2>&1
echo "=== miss-$TAG exit=$? $(date -u +%FT%TZ) ===" >> $OUT/gen-miss-$TAG.log
echo "MISS-$TAG-DONE"

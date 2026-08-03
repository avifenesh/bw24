#!/usr/bin/env bash
# moe-speq spec-trace pass: run-spec K=1,2,4 with BOTH debug rounds and route trace.
# Waits for the miss pass on this device to finish first (single tenant per device).
set -u
DEV=$1; TAG=$2; PROMPT=$3; CHAT=$4
BIN=/home/ubuntu/memra/target/release
ART=/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime
OUT=/home/ubuntu/receipts/moe-speq
CH=()
if [ "$CHAT" = "1" ]; then CH=(MEMRA_CHAT=1); fi
while ! grep -q "MISS-$TAG-DONE" $OUT/misstrace-$TAG.out 2>/dev/null; do sleep 20; done
for K in 1 2 4; do
  rm -f $OUT/spectrace-$TAG-k$K.txt
  {
    echo "=== spectrace-$TAG-k$K start $(date -u +%FT%TZ) dev=$DEV ==="
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader -i $DEV
  } > $OUT/spec2-$TAG-k$K.log 2>&1
  env CUDA_VISIBLE_DEVICES=$DEV "${CH[@]}" \
    MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 \
    MEMRA_SPEC_K=$K MEMRA_DEBUG_SPEC=1 MEMRA_SPEC_STATS=1 \
    MEMRA_MOE_TRACE=$OUT/spectrace-$TAG-k$K.txt \
    $BIN/run-spec $ART >> $OUT/spec2-$TAG-k$K.log 2>&1
  echo "=== spec2-$TAG-k$K exit=$? $(date -u +%FT%TZ) ===" >> $OUT/spec2-$TAG-k$K.log
done
echo "SPECTRACE-$TAG-DONE"

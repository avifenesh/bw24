#!/usr/bin/env bash
# moe-speq FINAL capture: ONE patched run-spec process per (class,K) emits
#   - miss3-$TAG-k$K.txt  per-lookup H/M (MEMRA_MOE_MISS_TRACE, patched binary)
#   - route3-$TAG-k$K.txt route trace (MEMRA_MOE_TRACE)
#   - spec3-$TAG-k$K.log  oracle + spec + [R] debug + stats
# In-process pairing: oracle decode sweeps (denominator) + verify predictions (numerator).
set -u
DEV=$1; TAG=$2; PROMPT=$3; CHAT=$4
BIN=/home/ubuntu/memra-moespeq/target/release
ART=/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime
OUT=/home/ubuntu/receipts/moe-speq
CH=()
if [ "$CHAT" = "1" ]; then CH=(MEMRA_CHAT=1); fi
# wait for the running spectrace pass on this device
while ! grep -q "SPECTRACE-$TAG-DONE" $OUT/spectrace-$TAG.out 2>/dev/null; do sleep 20; done
for K in 1 2 4; do
  rm -f $OUT/miss3-$TAG-k$K.txt $OUT/route3-$TAG-k$K.txt
  {
    echo "=== spec3-$TAG-k$K start $(date -u +%FT%TZ) dev=$DEV bin=patched ==="
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader -i $DEV
  } > $OUT/spec3-$TAG-k$K.log 2>&1
  env CUDA_VISIBLE_DEVICES=$DEV "${CH[@]}" \
    MEMRA_PROMPT_FILE=$PROMPT MEMRA_NGEN=128 \
    MEMRA_SPEC_K=$K MEMRA_DEBUG_SPEC=1 MEMRA_SPEC_STATS=1 \
    MEMRA_MOE_MISS_TRACE=$OUT/miss3-$TAG-k$K.txt \
    MEMRA_MOE_TRACE=$OUT/route3-$TAG-k$K.txt \
    $BIN/run-spec $ART >> $OUT/spec3-$TAG-k$K.log 2>&1
  echo "=== spec3-$TAG-k$K exit=$? $(date -u +%FT%TZ) ===" >> $OUT/spec3-$TAG-k$K.log
done
echo "SPEC3-$TAG-DONE"

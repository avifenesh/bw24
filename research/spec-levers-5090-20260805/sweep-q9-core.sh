#!/bin/bash
# q9 (Qwen3.5-9B NVFP4+MTP, 5.3G — VRAM-viable beside the other lanes) lever check:
# does the stack transfer to the small artifact? Arms: default K3/B32, K3/B128,
# K3/B128+pmin (9B regime K=2-3 per FLAGS.md; keep the shipped K, vary the levers).
# 3 rounds, one flock hold per round, arms interleaved inside, order alternated.
set -u
cd "$(dirname "$0")"
FWD="d-K3B32:3:32:0 b-K3B128:3:128:0 s-K3B128pm:3:128:0.3"
REV="s-K3B128pm:3:128:0.3 b-K3B128:3:128:0 d-K3B32:3:32:0"
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then ARMS=$FWD; else ARMS=$REV; fi
  ./run-round.sh q9 "r$r" 1 4 $ARMS
done
echo Q9_CORE_DONE >> logs/driver.log
echo Q9_CORE_DONE

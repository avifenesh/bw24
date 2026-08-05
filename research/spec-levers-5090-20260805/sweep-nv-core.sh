#!/bin/bash
# CORE nv sweep on the 5090 (82 SM): 5 rounds, each round = ONE flock hold with all
# 5 arms interleaved inside it (A/B same thermal window), arm order alternated per
# round. Between rounds the lock is released so the 3 sibling lanes get the card.
# Arms:
#   d-K3B32    = today's shipped default (worker.rs K=3, burst=32)
#   b-K5B32    = pod stack baseline (serve-K optimum at default burst)
#   b-K5B64    = burst midpoint (pod: flat at 188 SM — the 82-SM question)
#   b-K5B128   = burst lever
#   s-K5B128pm = full stack (+ PMIN=0.3 PMIN0=1)
set -u
cd "$(dirname "$0")"
FWD="d-K3B32:3:32:0 b-K5B32:5:32:0 b-K5B64:5:64:0 b-K5B128:5:128:0 s-K5B128pm:5:128:0.3"
REV="s-K5B128pm:5:128:0.3 b-K5B128:5:128:0 b-K5B64:5:64:0 b-K5B32:5:32:0 d-K3B32:3:32:0"
for r in 1 2 3 4 5; do
  if [ $((r % 2)) -eq 1 ]; then ARMS=$FWD; else ARMS=$REV; fi
  ./run-round.sh nv "r$r" 1 4 $ARMS
done
echo NV_CORE_DONE >> logs/driver.log
echo NV_CORE_DONE

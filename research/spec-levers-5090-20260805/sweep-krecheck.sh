#!/bin/bash
# K re-check at the winning burst on 82 SM. The core sweep showed K=5 LOSES to K=3 at
# B32 here (pod had K=5 optimum) — so the K question is fully open at B128: K=3..6 at
# B128+pmin, plus K3/B128 without pmin (isolate whether pmin's win is K-dependent).
# 3 rounds, one flock hold per round, ladder alternated.
set -u
cd "$(dirname "$0")"
FWD="kchk-K3B128:3:128:0 kchk-K3B128pm:3:128:0.3 kchk-K4B128pm:4:128:0.3 kchk-K5B128pm:5:128:0.3 kchk-K6B128pm:6:128:0.3"
REV="kchk-K6B128pm:6:128:0.3 kchk-K5B128pm:5:128:0.3 kchk-K4B128pm:4:128:0.3 kchk-K3B128pm:3:128:0.3 kchk-K3B128:3:128:0"
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then ARMS=$FWD; else ARMS=$REV; fi
  ./run-round.sh nv "r$r" 1 4 $ARMS
done
echo KRECHECK_DONE >> logs/driver.log
echo KRECHECK_DONE

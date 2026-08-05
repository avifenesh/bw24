#!/bin/bash
# Guards on the 82-SM winner (K=3 + BURST=128, pmin off):
#   1. c=8 no-regression: def K3/B32 vs win K3/B128, 3 rounds, c=8 x 24 req.
#   2. long-gen shape check: max_tokens=512 (B128 = 4 bursts vs B32 = 16), c=1, 2 rounds —
#      proves the B128 win is not a "one burst per request" artifact of the 128-tok shape.
# One flock hold per round, order alternated.
set -u
cd "$(dirname "$0")"
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then ARMS="c8def:3:32:0 c8win:3:128:0"; else ARMS="c8win:3:128:0 c8def:3:32:0"; fi
  ./run-round.sh nv "r$r" 8 24 $ARMS
done
for r in 1 2; do
  if [ $((r % 2)) -eq 1 ]; then ARMS="lg512def:3:32:0 lg512win:3:128:0"; else ARMS="lg512win:3:128:0 lg512def:3:32:0"; fi
  MAXTOK=512 ./run-round.sh nv "r$r" 1 4 $ARMS
done
echo C8_GUARD_DONE >> logs/driver.log
echo C8_GUARD_DONE

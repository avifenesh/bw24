#!/bin/bash
# Burst ladder at the 82-SM K-optimum (K=3): B32/B64/B128/B256, c=1, max_tokens=512
# (long-gen shape so >1 burst boundary exists even at B128). 3 rounds, one hold per
# round, ladder alternated. Completes the "find the local optimum at 82 SM" question.
set -u
cd "$(dirname "$0")"
FWD="bl-B32:3:32:0 bl-B64:3:64:0 bl-B128:3:128:0 bl-B256:3:256:0"
REV="bl-B256:3:256:0 bl-B128:3:128:0 bl-B64:3:64:0 bl-B32:3:32:0"
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then ARMS=$FWD; else ARMS=$REV; fi
  MAXTOK=512 ./run-round.sh nv "r$r" 1 4 $ARMS
done
echo BLADDER_DONE >> logs/driver.log
echo BLADDER_DONE

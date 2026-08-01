#!/bin/bash
# m0-nccl extension battery: graphed + peer a2a, graphed ping-pongs.
# Pair tests on 0-3 (3 idle-verified below). a2a set built from idle GPUs among {0..4}.
set -u
cd ~/m0-nccl
R=~/m0-nccl/receipts
mkdir -p $R

run() {
  local name=$1; shift
  echo "--- $name : $* ---"
  timeout 300 ./commbench2 "$@" > $R/$name.jsonl 2> $R/$name.err
  local rc=$?
  echo "rc=$rc rows=$(wc -l < $R/$name.jsonl)"
  [ $rc -ne 0 ] && tail -5 $R/$name.err
}

# a2a set: GPU0 + any of 1-4 idle (0% util, <1GiB used)
A2A="0"
for g in 1 2 3 4; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$util" -eq 0 ] && [ "$mem" -lt 1024 ]; then A2A="$A2A $g"; fi
done
NA=$(echo $A2A | wc -w)
echo "ext a2a set: $A2A (n=$NA)" | tee $R/ext_a2a_set.txt
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> $R/ext_a2a_set.txt
if [ "$NA" -lt 2 ]; then echo "no idle partner, aborting a2a"; else
  run nccl_ga2a  ga2a  $A2A
  run peer_a2a   pa2a  $A2A
  run peer_ga2a  gpa2a $A2A
fi

# graphed pair tests: pick first idle partner from the set
PARTNER=$(echo $A2A | awk '{print $2}')
if [ -n "${PARTNER:-}" ]; then
  echo "pair partner: $PARTNER" >> $R/ext_a2a_set.txt
  run peer_gpp_0_$PARTNER gppp 0 $PARTNER
  run nccl_gpp_0_$PARTNER gpp  0 $PARTNER
fi

date -u +%Y-%m-%dT%H:%M:%SZ >> $R/ext_a2a_set.txt
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> $R/ext_a2a_set.txt
cat $R/nccl_ga2a.jsonl $R/peer_a2a.jsonl $R/peer_ga2a.jsonl $R/peer_gpp_*.jsonl $R/nccl_gpp_*.jsonl > $R/ext.jsonl 2>/dev/null
echo "DONE ext_rows=$(wc -l < $R/ext.jsonl)"

#!/bin/bash
# m0-nccl full battery. GPUs 0,3,4 only (0 = ours; 3,4 verified idle 0MiB).
# GPUs 5-7 never touched. 1,2 included in a2a ONLY if 0% util + <100MiB at check time.
set -u
cd ~/m0-nccl
R=~/m0-nccl/receipts
mkdir -p $R

echo "=== env receipts ==="
date -u +%Y-%m-%dT%H:%M:%SZ > $R/env.txt
nvidia-smi --query-gpu=index,name,driver_version,utilization.gpu,memory.used --format=csv >> $R/env.txt
/usr/local/cuda/bin/nvcc --version | tail -2 >> $R/env.txt
echo "libnccl: /opt/pytorch/cuda/lib/libnccl.so.2 (2.27.7+cuda13.0)" >> $R/env.txt
nvidia-smi topo -m > $R/topo.txt
nvidia-smi -q -i 0 | grep -A2 "Product Name" >> $R/env.txt

run() {
  local name=$1; shift
  echo "--- $name : $* ---"
  timeout 300 ./commbench "$@" > $R/$name.jsonl 2> $R/$name.err
  local rc=$?
  echo "rc=$rc rows=$(wc -l < $R/$name.jsonl)"
  [ $rc -ne 0 ] && tail -5 $R/$name.err
}

# NCCL pair tests
run nccl_pp_0_3    pp 0 3
run nccl_pp_0_4    pp 0 4
run nccl_uni_0_3   uni 0 3
run nccl_uni_3_0   uni 3 0
run nccl_uni_0_4   uni 0 4
run nccl_uni_4_0   uni 4 0
run nccl_bidir_0_3 bidir 0 3
run nccl_bidir_0_4 bidir 0 4

# Peer-copy control
run peer_pp_0_3    ppp 0 3
run peer_pp_0_4    ppp 0 4
run peer_uni_0_3   puni 0 3
run peer_uni_3_0   puni 3 0
run peer_uni_0_4   puni 0 4
run peer_uni_4_0   puni 4 0
run peer_bidir_0_3 pbidir 0 3
run peer_bidir_0_4 pbidir 0 4

# all-to-all: base set {0,3,4}; widen with 1/2 only if truly idle right now
A2A="0 3 4"
for g in 1 2; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$util" -eq 0 ] && [ "$mem" -lt 100 ]; then A2A="$A2A $g"; fi
done
echo "a2a set: $A2A" | tee $R/a2a_set.txt
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> $R/a2a_set.txt
run nccl_a2a a2a $A2A

date -u +%Y-%m-%dT%H:%M:%SZ >> $R/env.txt
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> $R/env.txt
cat $R/*.jsonl > $R/all.jsonl
echo "DONE total_rows=$(wc -l < $R/all.jsonl)"

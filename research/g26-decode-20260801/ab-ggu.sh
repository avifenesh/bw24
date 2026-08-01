#!/bin/bash
# g26 gelu gate/up geometry probe: base(0) vs j8 vs j8r2 — new binary, router w8 in all arms
set -u
cd ~/lane2
OUT=research/g26-decode-20260801
G26=$HOME/models/gemma-4-26B_q4_0-it.gguf
export CUDA_VISIBLE_DEVICES=2 MEMRA_NGEN=128
IDS=$(cat research/gemma4-bringup/depth-prompt-1736-ids.txt)
for i in 1 2 3; do
  MEMRA_MOE_DEVQ8_GGU=0    ./target/release/run-gen "$G26" $IDS > $OUT/ggu-base-$i.log 2>&1
  MEMRA_MOE_DEVQ8_GGU=j8   ./target/release/run-gen "$G26" $IDS > $OUT/ggu-j8-$i.log 2>&1
  MEMRA_MOE_DEVQ8_GGU=j8r2 ./target/release/run-gen "$G26" $IDS > $OUT/ggu-j8r2-$i.log 2>&1
done
echo GGU-DONE

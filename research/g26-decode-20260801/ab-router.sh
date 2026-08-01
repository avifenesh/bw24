#!/bin/bash
# g26 decode A/B: lone-warp router (default) vs w8 twin (MEMRA_ROUTER_V2=1), interleaved x3
set -u
cd ~/lane2
OUT=research/g26-decode-20260801
G26=$HOME/models/gemma-4-26B_q4_0-it.gguf
export CUDA_VISIBLE_DEVICES=2 MEMRA_NGEN=128
IDS=$(cat research/gemma4-bringup/depth-prompt-1736-ids.txt)
for i in 1 2 3; do
  ./target/release/run-gen "$G26" $IDS > $OUT/abr-depth-base-$i.log 2>&1
  MEMRA_ROUTER_V2=1 ./target/release/run-gen "$G26" $IDS > $OUT/abr-depth-w8-$i.log 2>&1
  MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt ./target/release/run-gen "$G26" > $OUT/abr-board-base-$i.log 2>&1
  MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt MEMRA_ROUTER_V2=1 ./target/release/run-gen "$G26" > $OUT/abr-board-w8-$i.log 2>&1
done
echo ABR-DONE

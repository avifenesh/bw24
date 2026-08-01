#!/bin/bash
# final gates + headline A/B: base binary (lone-warp router) vs final binary (w8 default), naked
set -u
cd ~/lane2
OUT=research/g26-decode-20260801
G26=$HOME/models/gemma-4-26B_q4_0-it.gguf
Q35=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
export CUDA_VISIBLE_DEVICES=2
./target/release/kernel-check > $OUT/kernel-check-final.log 2>&1
echo "kernel-check rc=$? tail: $(tail -1 $OUT/kernel-check-final.log)"
IDS=$(cat research/gemma4-bringup/depth-prompt-1736-ids.txt)
MEMRA_NGEN=16 ./target/release/run-gen "$G26" $IDS > $OUT/gate-final-g26-depth.log 2>&1
echo "g26-depth rc=$? $(grep -oE "(MATCH|MISMATCH)" $OUT/gate-final-g26-depth.log | head -1)"
MEMRA_NGEN=16 MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt ./target/release/run-gen "$G26" > $OUT/gate-final-g26-board.log 2>&1
echo "g26-board rc=$? $(grep -oE "(MATCH|MISMATCH)" $OUT/gate-final-g26-board.log | head -1)"
MEMRA_NGEN=16 MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt ./target/release/run-gen "$Q35" > $OUT/gate-final-q35-board.log 2>&1
echo "q35-board rc=$? $(grep -oE "(MATCH|MISMATCH)" $OUT/gate-final-q35-board.log | head -1)"
export MEMRA_NGEN=128
for i in 1 2 3; do
  /tmp/g26-base-run-gen "$G26" $IDS > $OUT/final-depth-base-$i.log 2>&1
  ./target/release/run-gen "$G26" $IDS > $OUT/final-depth-new-$i.log 2>&1
  MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt /tmp/g26-base-run-gen "$G26" > $OUT/final-board-base-$i.log 2>&1
  MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt ./target/release/run-gen "$G26" > $OUT/final-board-new-$i.log 2>&1
done
echo FINAL-DONE

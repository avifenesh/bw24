#!/bin/bash
# ragged token-tile A/B (lane2, GPU2, 2026-08-01)
# A=/tmp/lane2-base-run-gen (pre-change binary, fixed MMQ_X=128)
# B=target/release/run-gen (ragged {64,96,128} dispatch)
# C=target/release/run-gen + MEMRA_MMQ_IQEXP_RAGGED=0 (attribution: new binary, legacy tile)
# interleaved x3, MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 (median of 5 per run, warmup excluded)
set -u
cd ~/lane2
OUT=research/ragged-tiles-20260801
G26=$HOME/models/gemma-4-26B_q4_0-it.gguf
Q35=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
G26ARGS=$(cat research/gemma4-bringup/depth-prompt-1736-ids.txt)
export CUDA_VISIBLE_DEVICES=2
export MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5
for i in 1 2 3; do
  MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt /tmp/lane2-base-run-gen "$Q35" > $OUT/ab-q35-base-$i.log 2>&1
  MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt ./target/release/run-gen "$Q35" > $OUT/ab-q35-ragged-$i.log 2>&1
  MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt MEMRA_MMQ_IQEXP_RAGGED=0 ./target/release/run-gen "$Q35" > $OUT/ab-q35-ragged0-$i.log 2>&1
  /tmp/lane2-base-run-gen "$G26" $G26ARGS > $OUT/ab-g26-base-$i.log 2>&1
  ./target/release/run-gen "$G26" $G26ARGS > $OUT/ab-g26-ragged-$i.log 2>&1
done
echo AB-DONE

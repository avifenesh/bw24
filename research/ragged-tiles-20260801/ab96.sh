#!/bin/bash
# probe: tile-96 floor vs base vs ragged-default (q35, GPU2, interleaved x3)
set -u
cd ~/lane2
OUT=research/ragged-tiles-20260801
Q35=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
export CUDA_VISIBLE_DEVICES=2
export MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5
export MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt
for i in 1 2 3; do
  /tmp/lane2-base-run-gen "$Q35" > $OUT/ab96-q35-base-$i.log 2>&1
  ./target/release/run-gen "$Q35" > $OUT/ab96-q35-ragged-$i.log 2>&1
  MEMRA_MMQ_IQEXP_RAGGED=96 ./target/release/run-gen "$Q35" > $OUT/ab96-q35-r96-$i.log 2>&1
done
echo AB96-DONE

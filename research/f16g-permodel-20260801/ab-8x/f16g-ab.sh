#!/bin/bash
# f16g default arbitration — interleaved x5 pairs, board-2048, naked(def) vs MEMRA_MOE_F16G=0
# usage: f16g-ab.sh <gpu> <gguf> <tag>
GPU=$1; G=$2; TAG=$3
cd ~/fleet-v060
FOX=research/e2e/prompts/board-2048.txt
for i in 1 2 3 4 5; do
  for arm in def off; do
    E=""; [ $arm = off ] && E="MEMRA_MOE_F16G=0"
    out=$(env CUDA_VISIBLE_DEVICES=$GPU $E MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 target/release/run-gen "$G" 2>&1)
    p=$(echo "$out" | grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
    d=$(echo "$out" | grep -oE "= [0-9.]+ tok/s \((Stage|graph)" | tail -1 | grep -oE "[0-9.]+" | head -1)
    echo "$TAG pair$i $arm prefill=$p decode=$d"
  done
done
echo "$TAG AB_DONE"

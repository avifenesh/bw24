#!/bin/bash
# Three-arm q35 board-2048 prime probe, interleaved x5 round-robin, one lock hold
# (the caller wraps this whole script in flock /tmp/gpu-h100.lock).
# cublas = naked (MEMRA_MOE_F16G unset -> Hopper mode-1 default, batch router on)
# skfull = MEMRA_MOE_F16G=2 (sk visitor, direct-from-quant Q4_K/Q6_K/IQ4_XS/IQ3_S loaders
#          default ON + deep tail default ON — the full new form), cross=32 (H100 sweep winner)
# skref  = MEMRA_MOE_F16G=2 MEMRA_F16G_DIRECT=0 MEMRA_F16G_TAIL=0 (the round-51 reference
#          form: workspace dequant + 2-stage legacy tail), same cross=32
cd ~/memra
FOX=research/e2e/prompts/board-2048.txt
G=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
BW=target/release/run-gen
OUT=/tmp/probe-flipfull.log
: > $OUT
for i in 1 2 3 4 5; do
  echo "== iter $i arm=cublas ==" >> $OUT
  MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
  echo "== iter $i arm=skfull ==" >> $OUT
  MEMRA_MOE_F16G=2 MEMRA_F16G_SK_CROSS=32 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
  echo "== iter $i arm=skref ==" >> $OUT
  MEMRA_MOE_F16G=2 MEMRA_F16G_DIRECT=0 MEMRA_F16G_TAIL=0 MEMRA_F16G_SK_CROSS=32 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
done
echo PROBE-DONE >> $OUT

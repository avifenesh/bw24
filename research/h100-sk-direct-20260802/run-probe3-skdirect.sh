#!/bin/bash
# Three-arm q35 board-2048 prime probe, interleaved x5, same session (runs UNDER the gpu flock).
# cublas   = naked (MEMRA_MOE_F16G unset -> Hopper mode-1 default, batch router on)
# skdirect = MEMRA_MOE_F16G=2 (sk visitor + direct-from-quant Q4_K/Q6_K tile loaders, default ON),
#            MEMRA_F16G_SK_CROSS=32 (the H100 sweep winner, research/sk-bm128-20260801/h100/)
# skws     = MEMRA_MOE_F16G=2 MEMRA_F16G_DIRECT=0 (the v0.62 workspace sk form, same cross=32)
cd ~/memra
FOX=research/e2e/prompts/board-2048.txt
G=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
BW=target/release/run-gen
OUT=/tmp/probe-skdirect.log
: > $OUT
for i in 1 2 3 4 5; do
  echo "== iter $i arm=cublas ==" >> $OUT
  MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
  echo "== iter $i arm=skdirect ==" >> $OUT
  MEMRA_MOE_F16G=2 MEMRA_F16G_SK_CROSS=32 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
  echo "== iter $i arm=skws ==" >> $OUT
  MEMRA_MOE_F16G=2 MEMRA_F16G_DIRECT=0 MEMRA_F16G_SK_CROSS=32 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
done
echo PROBE-DONE >> $OUT

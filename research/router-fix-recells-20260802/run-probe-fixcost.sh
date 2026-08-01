#!/bin/bash
cd ~/memra
FOX=research/e2e/prompts/board-2048.txt
G=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
BW=target/release/run-gen
OUT=/tmp/probe-fixcost-q35.log
: > $OUT
for i in 1 2 3; do
  echo "== iter $i arm=naked ==" >> $OUT
  MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
  echo "== iter $i arm=exact0 ==" >> $OUT
  MEMRA_ROUTER_PREFILL_EXACT=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
done
echo PROBE-DONE >> $OUT

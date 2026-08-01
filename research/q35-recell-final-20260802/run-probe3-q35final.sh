#!/bin/bash
# Three-arm recovery probe x3 interleaved, same session (runs UNDER the gpu flock).
# naked = batch router (fast-exact) | batch0 = MEMRA_ROUTER_BATCH=0 (as-merged slow-exact)
# exact0 = MEMRA_ROUTER_PREFILL_EXACT=0 (pre-fix reference, ~8444 last session)
cd ~/memra
FOX=research/e2e/prompts/board-2048.txt
G=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
BW=target/release/run-gen
OUT=/tmp/probe-q35final.log
: > $OUT
for i in 1 2 3; do
  echo "== iter $i arm=naked ==" >> $OUT
  MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
  echo "== iter $i arm=batch0 ==" >> $OUT
  MEMRA_ROUTER_BATCH=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
  echo "== iter $i arm=exact0 ==" >> $OUT
  MEMRA_ROUTER_PREFILL_EXACT=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$FOX timeout 900 $BW $G >> $OUT 2>&1
done
echo PROBE-DONE >> $OUT

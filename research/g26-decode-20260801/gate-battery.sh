#!/bin/bash
# g26 router w8 re-arbitration: run-gen argmax gate, 6 real prompts x {default lone-warp, V2=1 w8}
set -u
cd ~/lane2
OUT=research/g26-decode-20260801
G26=$HOME/models/gemma-4-26B_q4_0-it.gguf
export CUDA_VISIBLE_DEVICES=2 MEMRA_NGEN=16
IDS=$(cat research/gemma4-bringup/depth-prompt-1736-ids.txt)
run_ids(){ # $1=arm-label $2=V2-env
  if [ -z "$2" ]; then ./target/release/run-gen "$G26" $IDS > $OUT/gate-depth1736-$1.log 2>&1
  else MEMRA_ROUTER_V2=$2 ./target/release/run-gen "$G26" $IDS > $OUT/gate-depth1736-$1.log 2>&1; fi
  echo "depth1736 $1 rc=$? $(grep -oE "(MATCH|MISMATCH)" $OUT/gate-depth1736-$1.log | head -1)"
}
run_txt(){ # $1=promptfile-stem $2=arm-label $3=V2-env
  if [ -z "$3" ]; then MEMRA_PROMPT_FILE=$OUT/prompt-$1.txt ./target/release/run-gen "$G26" > $OUT/gate-$1-$2.log 2>&1
  else MEMRA_PROMPT_FILE=$OUT/prompt-$1.txt MEMRA_ROUTER_V2=$3 ./target/release/run-gen "$G26" > $OUT/gate-$1-$2.log 2>&1; fi
  echo "$1 $2 rc=$? $(grep -oE "(MATCH|MISMATCH)" $OUT/gate-$1-$2.log | head -1)"
}
run_b2048(){ # board-2048 text
  if [ -z "$2" ]; then MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt ./target/release/run-gen "$G26" > $OUT/gate-board2048-$1.log 2>&1
  else MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt MEMRA_ROUTER_V2=$2 ./target/release/run-gen "$G26" > $OUT/gate-board2048-$1.log 2>&1; fi
  echo "board2048 $1 rc=$? $(grep -oE "(MATCH|MISMATCH)" $OUT/gate-board2048-$1.log | head -1)"
}
run_ids base ""
run_ids w8 1
run_b2048 base ""
run_b2048 w8 1
for p in readme contributing architecture flags; do
  run_txt $p base ""
  run_txt $p w8 1
done
echo GATE-BATTERY-DONE

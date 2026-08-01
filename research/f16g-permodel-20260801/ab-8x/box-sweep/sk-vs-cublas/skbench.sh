#!/usr/bin/env bash
set -u
cd ~/arc-sk
export CUDA_VISIBLE_DEVICES=2
export PATH=$HOME/cuda-13.3.1/bin:$PATH
M=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PF=research/e2e/prompts/board-2048.txt
D=/tmp/skncu; mkdir -p $D
run_pp() { # mode reps log
  MEMRA_MOE_F16G=$1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=$2 MEMRA_PROMPT_FILE=$PF \
    timeout 900 ./target/release/run-gen "$M" > "$3" 2>&1
}
echo "=== sanity r1 f16g2"; run_pp 2 3 $D/pp-f16g2-r1.log; grep "pp-only" $D/pp-f16g2-r1.log | tail -4
echo "=== sanity r1 f16g1"; run_pp 1 3 $D/pp-f16g1-r1.log; grep "pp-only" $D/pp-f16g1-r1.log | tail -4
echo "=== sanity r2 f16g2"; run_pp 2 3 $D/pp-f16g2-r2.log; grep "pp-only" $D/pp-f16g2-r2.log | tail -4
echo "=== sanity r2 f16g1"; run_pp 1 3 $D/pp-f16g1-r2.log; grep "pp-only" $D/pp-f16g1-r2.log | tail -4
echo "=== nsys f16g1"
MEMRA_MOE_F16G=1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PROMPT_FILE=$PF \
  nsys profile --stats=true -o $D/nsys-f16g1 --force-overwrite=true ./target/release/run-gen "$M" > $D/nsys-f16g1.log 2>&1
echo "=== nsys f16g2"
MEMRA_MOE_F16G=2 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PROMPT_FILE=$PF \
  nsys profile --stats=true -o $D/nsys-f16g2 --force-overwrite=true ./target/release/run-gen "$M" > $D/nsys-f16g2.log 2>&1
echo "=== ALL DONE"

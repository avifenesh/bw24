#!/bin/bash
# q4k-f16-mirrors gate battery (round 49, GPU 3): kernel-check + run-gen argmax (q27) +
# run-spec K=1..8 self-consistency (q27, HPOST=1 PMIN=0.3) + q35 argmax no-regression.
set -u
cd $HOME/arc4
D=$HOME/arc4/research/q4k-f16-mirrors-20260801
mkdir -p $D
Q27=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
Q35=/opt/dlami/nvme/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
export CUDA_VISIBLE_DEVICES=3

echo "== kernel-check (q27 model arg) =="
./target/release/kernel-check $Q27 > $D/kernel-check.log 2>&1
echo "kernel-check rc=$? fails=$(grep -c FAIL $D/kernel-check.log)"
grep -E "f16 T=" $D/kernel-check.log

echo "== run-gen argmax q27 board-2048 =="
MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt MEMRA_NGEN=64 \
  ./target/release/run-gen $Q27 > $D/rungen-argmax-q27.log 2>&1
echo "run-gen q27 rc=$?"
grep -E "argmax=.*(MATCH|MISMATCH)" $D/rungen-argmax-q27.log

echo "== run-spec K=1..8 q27 board-2048 (HPOST=1 PMIN=0.3) =="
MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 MEMRA_NGEN=256 \
  MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt \
  ./target/release/run-spec $Q27 > $D/runspec-k1-8-q27.log 2>&1
echo "run-spec q27 rc=$?"
grep -E "self-consistency|SELF-CONSISTENCY" $D/runspec-k1-8-q27.log

echo "== run-gen argmax q35 board-2048 (no-regression; zero Q4_K tensors) =="
MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt MEMRA_NGEN=64 \
  ./target/release/run-gen $Q35 > $D/rungen-argmax-q35.log 2>&1
echo "run-gen q35 rc=$?"
grep -E "argmax=.*(MATCH|MISMATCH)" $D/rungen-argmax-q35.log
echo GATES-DONE

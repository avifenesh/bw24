#!/usr/bin/env bash
# pro6000wk-validation: correctness gates on q9 (WK-edition exactness-transfer proof)
set -uo pipefail
cd /root/bw24
export PATH=/usr/local/cuda-13.1/bin:$HOME/.cargo/bin:$PATH
R=/root/receipts
mkdir -p "$R"
M=/root/models/qwen35-9b-nvfp4-gguf
Q9=$M/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=$M/draft-9b-owntrim-nvfp4head-q4blk.gguf
PF=research/e2e/prompts/pp512.txt

echo "=== GATE 1: kernel-check (naked) ===" | tee "$R/gate1-kernel-check.log"
target/release/kernel-check >> "$R/gate1-kernel-check.log" 2>&1
tail -3 "$R/gate1-kernel-check.log"

echo "=== GATE 2: run-gen argmax (q9, pp512 text) ===" | tee "$R/gate2-rungen-argmax.log"
MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$PF timeout 600 target/release/run-gen "$Q9" >> "$R/gate2-rungen-argmax.log" 2>&1
grep -E "argmax|MATCH|MISMATCH|tok/s" "$R/gate2-rungen-argmax.log" | head -6

echo "=== GATE 3: run-spec K=1..3 self-consistency (q9 + owntrim draft) ===" | tee "$R/gate3-runspec-k123.log"
for K in 1 2 3; do
  echo "=== K=$K ===" | tee -a "$R/gate3-runspec-k123.log"
  MEMRA_SPEC_K=$K MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=64 MEMRA_PROMPT_FILE=$PF \
    timeout 600 target/release/run-spec "$Q9" >> "$R/gate3-runspec-k123.log" 2>&1
done
grep -E "self-consistency|acceptance|tok/s" "$R/gate3-runspec-k123.log" | head -12
echo "GATES DONE"

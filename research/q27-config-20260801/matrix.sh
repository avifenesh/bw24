#!/bin/bash
# q27 shipping-config matrix (micro-sprint 2026-08-01, GPU 3): board-2048 prompt,
# MEMRA_NGEN=512, MEMRA_SPEC_K=3 HPOST=1 PMIN=0.3. Fresh process per run.
# Configs: A=default budget(32768)+KQRP on, B=43008+KQRP on, C=full-f16(50688)+KQRP off.
set -u
cd $HOME/arc4
D=$HOME/arc4/research/q27-config-20260801
M=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=3
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 3 -l 5 >> $D/vram-samples.log &
SMIPID=$!
run_cfg() {
  local name=$1 r=$2; shift 2
  echo "[$(date -u +%H:%M:%SZ)] pre  $name r$r vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 3)" >> $D/vram-guard.log
  env "$@" MEMRA_SPEC_K=3 MEMRA_NGEN=512 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 \
    MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt \
    ./target/release/run-spec $M > $D/matrix-$name-r$r.log 2>&1
  echo "$name r$r rc=$?"
  echo "[$(date -u +%H:%M:%SZ)] post $name r$r vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 3)" >> $D/vram-guard.log
}
run_cfg A-default-kqrp 1
run_cfg B-bud43008-kqrp 1 MEMRA_PP_F16_BUDGET_MB=43008
run_cfg C-fullf16-nokqrp 1 MEMRA_KQRP=0 MEMRA_PP_F16_BUDGET_MB=50688
kill $SMIPID 2>/dev/null
echo PHASE1-DONE

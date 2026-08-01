#!/bin/bash
# jstrip-exit perf A/B: base vs exit, interleaved x3 pairs, MEMRA_PP_ONLY x5 in-process reps.
# q35 board-2048 (the (4,252) gate/up + (16,256) down forms, ~65-pair groups) and
# g26 pp1736 (control: 147-236-pair groups, mostly full tiles — must be flat).
set -u
cd ~/lane4
D=research/jstrip-exit-20260801
Q35=$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
G26=$HOME/models/gemma-4-26B_q4_0-it.gguf
DEPTH=$(cat research/gemma4-bringup/depth-prompt-1736-ids.txt)
for rep in 1 2 3; do
  for arm in base exit; do
    BIN=$D/run-gen.$arm
    echo "=== rep$rep q35-board2048 $arm ==="
    CUDA_VISIBLE_DEVICES=4 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt \
      timeout 900 "$BIN" "$Q35" 2>&1 | grep -E "pp-only"
    echo "=== rep$rep g26-d1736 $arm ==="
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=4 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 \
      timeout 900 "$BIN" "$G26" $DEPTH 2>&1 | grep -E "pp-only"
  done
done
echo "=== PERF DONE ==="

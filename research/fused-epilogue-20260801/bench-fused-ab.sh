#!/bin/bash
# Lane 3 (GPU 3, darklanes-8x): fused act-epilogue A/B — interleaved pairs, N=3 per arm.
# Arms: fused (MEMRA_MOE_FUSE_ACTQ=1, the new default) vs twopass (=0 rollback seam).
# Cells: g26 pp1736 (depth-prompt-1736-ids, MEMRA_NGEN=4) and q35 board-2048 (MEMRA_NGEN=4).
# All params baked as literals (workflow args do not propagate to background fan-out).
set -u
cd "$HOME/lane3" || exit 1
BW="$HOME/lane3/target/release"
OUT="$HOME/lane3/research/fused-epilogue-20260801"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=3

G26="$HOME/models/gemma-4-26B_q4_0-it.gguf"
Q35="$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
G26IDS="research/gemma4-bringup/depth-prompt-1736-ids.txt"
Q35PF="research/e2e/prompts/board-2048.txt"

run_g26() { # label fuse rep
  local log="$OUT/g26-$1-rep$3.log"
  # shellcheck disable=SC2046
  MEMRA_MOE_FUSE_ACTQ=$2 MEMRA_NGEN=4 timeout 900 "$BW/run-gen" "$G26" $(cat "$G26IDS") >"$log" 2>&1
  echo "g26 $1 rep$3: $(grep -E '^prefill ' "$log" | tail -1)  argmax:$(grep -cE " MATCH$" "$log")/$(grep -cE "MISMATCH" "$log")miss"
}
run_q35() { # label fuse rep
  local log="$OUT/q35-$1-rep$3.log"
  MEMRA_MOE_FUSE_ACTQ=$2 MEMRA_NGEN=4 MEMRA_PROMPT_FILE="$Q35PF" timeout 900 "$BW/run-gen" "$Q35" >"$log" 2>&1
  echo "q35 $1 rep$3: $(grep -E '^prefill ' "$log" | tail -1)  argmax:$(grep -cE " MATCH$" "$log")/$(grep -cE "MISMATCH" "$log")miss"
}

if [ -f "$Q35" ]; then
  for rep in 1 2 3; do
    run_q35 fused 1 "$rep"
    run_q35 twopass 0 "$rep"
  done
fi
if [ -f "$G26" ]; then
  for rep in 1 2 3; do
    run_g26 fused 1 "$rep"
    run_g26 twopass 0 "$rep"
  done
fi
echo "DONE"

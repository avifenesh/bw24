#!/usr/bin/env bash
# agentworld: stage-2 onboarding gates on Qwen-AgentWorld-35B-A3B UD-Q4_K_M (5090).
# Mirrors research/onboard-ornith-20260801 gate shapes exactly:
#   1. run-gen argmax pp22  — MEMRA_CHAT=1, short prompt (prefill==decode argmax MATCH)
#   2. run-gen argmax pp302-class + chat sanity — MEMRA_CHAT=1 MEMRA_NGEN=250, the Rust
#      review prompt (ChatML tail render is the Bonsai-class risk; output read by hand)
# kernel-check ran once per branch build (kernel-check.log). Every GPU run under
# flock /tmp/gpu5090.lock. Params baked as literals (workflow-args-no-propagate).
set -u
W=/home/avifenesh/projects/bw24-agentworld
R=$W/research/agentworld-20260802
MODEL=/data/ai-ml/hf-models/agentworld-35b-gguf/Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf
mkdir -p "$R/gates"

echo "=== gates $(date -u +%FT%TZ) git=$(git -C "$W" rev-parse --short HEAD) ==="
echo "--- gate: argmax pp22 ---"
MEMRA_CHAT=1 MEMRA_PROMPT="$(cat "$R/prompts/pp22.txt")" \
  flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$MODEL" \
  > "$R/gates/agentworld-q4km-argmax-pp22.log" 2>&1
grep -E "argmax|resident-experts decision|prefill [0-9]+ tok" "$R/gates/agentworld-q4km-argmax-pp22.log"

echo "--- gate: argmax pp302-class + chat sanity (NGEN=250) ---"
MEMRA_CHAT=1 MEMRA_NGEN=250 MEMRA_PROMPT="$(cat "$R/prompts/rust-review-302.txt")" \
  flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$MODEL" \
  > "$R/gates/agentworld-q4km-chat-sanity.log" 2>&1
grep -E "argmax|resident-experts decision|prefill [0-9]+ tok|generated [0-9]+ tokens" \
  "$R/gates/agentworld-q4km-chat-sanity.log"
echo "GATES-DONE $(date -u +%FT%TZ)"

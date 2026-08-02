#!/usr/bin/env bash
# agentworld: one bounded own-gen corpus chunk (default 64 prompts) under the rig lock.
# Rerun until frspec-owngen stops printing PARTIAL and writes the ranks (line-count resume;
# greedy temp-0 makes chunked == single-run). Regime: docs/DRAFT-REGIME.md §build-one,
# 32768 protocol, canonical 254-prompt pack, --ngen 512, chat template ON (default).
# Params baked as literals (workflow-args-no-propagate).
set -euo pipefail
LIMIT=${1:-64}
W=/home/avifenesh/projects/bw24-agentworld
R=$W/research/agentworld-20260802
MODEL=/data/ai-ml/hf-models/agentworld-35b-gguf/Qwen-AgentWorld-35B-A3B-UD-Q4_K_M.gguf
RANKS=/data/ai-ml/hf-models/agentworld-35b-gguf/owngen-ranks-32768.gguf
PACK=$W/research/gemma4-bringup/corpus-prompts
IDS=$R/corpus/agentworld-owngen-ids.txt
mkdir -p "$R/corpus"
{
  echo "=== chunk start $(date -Is) limit=$LIMIT model=$MODEL"
  flock /tmp/gpu5090.lock "$W/target/release/frspec-owngen" "$MODEL" "$RANKS" 32768 \
    --ngen 512 --corpus-out "$IDS" --limit "$LIMIT" "$PACK"
  echo "=== chunk end $(date -Is) rc=$?"
} 2>&1 | tee -a "$R/corpus/agentworld-owngen.log"

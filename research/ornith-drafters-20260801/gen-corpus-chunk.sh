#!/usr/bin/env bash
# One bounded own-gen corpus chunk (64 prompts, ~5-10 min GPU) under the rig lock.
# Rerun the same command until frspec-owngen stops printing PARTIAL and writes the ranks
# (the resume is line-count-based; greedy temp-0 makes chunked == single-run). Params are
# baked as literals per model (workflow-args-no-propagate).
#
# usage: gen-corpus-chunk.sh <ornith9b|ornith35b|katcoder> [chunk-prompts]
set -euo pipefail
KEY=$1
LIMIT=${2:-64}
WT=/home/avifenesh/projects/wt-ornith-drafters
RD=$WT/research/ornith-drafters-20260801
PACK=$WT/research/gemma4-bringup/corpus-prompts   # canonical 254-prompt pack (RECIPE.md §1)
case $KEY in
  ornith9b)
    MODEL=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
    RANKS=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/owngen-ranks-32768.gguf ;;
  ornith35b)
    MODEL=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
    RANKS=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/owngen-ranks-32768.gguf ;;
  katcoder)
    MODEL=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
    RANKS=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/owngen-ranks-32768.gguf ;;
  *) echo "unknown model key: $KEY (ornith9b|ornith35b|katcoder)"; exit 2 ;;
esac
IDS=$RD/corpus/$KEY-owngen-ids.txt
LOG=$RD/corpus/$KEY-owngen.log
mkdir -p "$RD/corpus"
{
  echo "=== chunk start $(date -Is) limit=$LIMIT model=$MODEL"
  flock /tmp/gpu5090.lock "$WT/target/release/frspec-owngen" "$MODEL" "$RANKS" 32768 \
    --ngen 512 --corpus-out "$IDS" --limit "$LIMIT" "$PACK"
  echo "=== chunk end $(date -Is) rc=$?"
} 2>&1 | tee -a "$LOG"

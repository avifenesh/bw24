#!/usr/bin/env bash
# agentworld: donor-block trimmed drafter build (docs/DRAFT-REGIME.md donor variant,
# Ornith-35B recipe 1:1 — research/ornith-drafters-20260801/RECIPE.md).
#   donor  = Qwen3.6-35B-A3B-UD-IQ4_XS.gguf (blk.40 NextN block; byte-verbatim, law 2)
#   ranks  = AgentWorld's OWN generations (law 1; gen-corpus-chunk.sh output)
#   quant  AFTER trim: NVFP4 head + Q4_K_M block (law 3, hqmtp order)
# CPU-only; no GPU lock needed.
set -euo pipefail
W=/home/avifenesh/projects/bw24-agentworld
R=$W/research/agentworld-20260802
DONOR=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
RANKS=/data/ai-ml/hf-models/agentworld-35b-gguf/owngen-ranks-32768.gguf.txt
OUT=/data/ai-ml/hf-models/agentworld-35b-gguf/draft-agentworld-owntrim-nvfp4head-q4blk.gguf
{
  echo "=== build-draft $(date -Is) donor=$DONOR ranks=$RANKS"
  nice "$W/tools/make-trimmed-draft.sh" "$DONOR" "$RANKS" "$OUT" 32768
  echo "=== rc=$? out=$OUT"
  sha256sum "$OUT" "$RANKS"
} 2>&1 | tee "$R/build-agentworld-draft.log"

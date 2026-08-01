#!/usr/bin/env bash
# Emit the artifact manifest (sha256 + bytes) for one model's drafter set — weights stay
# on /data, the manifest is the committed receipt.
# usage: make-manifest.sh <ornith9b|ornith35b|katcoder>
set -euo pipefail
KEY=$1
WT=/home/avifenesh/projects/wt-ornith-drafters
RD=$WT/research/ornith-drafters-20260801
case $KEY in
  ornith9b)
    DIR=/data/ai-ml/hf-models/ornith-1.0-9b-gguf
    TARGET=$DIR/ornith-1.0-9b-Q8_0.gguf
    DONOR=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
    DRAFT=$DIR/draft-ornith9b-owntrim-nvfp4head-q4blk.gguf ;;
  ornith35b)
    DIR=/data/ai-ml/hf-models/ornith-1.0-35b-gguf
    TARGET=$DIR/ornith-1.0-35b-Q4_K_M.gguf
    DONOR=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
    DRAFT=$DIR/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf ;;
  katcoder)
    DIR=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf
    TARGET=$DIR/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
    DONOR=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
    DRAFT=$DIR/draft-katcoder-owntrim-nvfp4head-q4blk.gguf ;;
  *) echo "unknown model key: $KEY"; exit 2 ;;
esac
mkdir -p "$RD/manifests"
OUT=$RD/manifests/$KEY.manifest
{
  echo "# $KEY drafter manifest — $(date -Is), lane/ornith-drafters"
  echo "# role sha256 bytes path"
  for pair in "target:$TARGET" "donor:$DONOR" \
              "ranks-gguf:$DIR/owngen-ranks-32768.gguf" \
              "ranks-txt:$DIR/owngen-ranks-32768.gguf.txt" \
              "corpus-ids:$RD/corpus/$KEY-owngen-ids.txt" \
              "draft:$DRAFT"; do
    role=${pair%%:*}; f=${pair#*:}
    if [ -f "$f" ]; then
      printf '%s %s %s %s\n' "$role" "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c%s "$f")" "$f"
    else
      printf '%s MISSING - %s\n' "$role" "$f"
    fi
  done
} > "$OUT"
cat "$OUT"

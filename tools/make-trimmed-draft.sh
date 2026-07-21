#!/usr/bin/env bash
# One-command trimmed-draft builder — the winning draft-file recipe (2026-07-17):
#   extract the MTP block from the model GGUF (byte-verbatim = embedded-head parity)
#   -> trim the lm head to the model's OWN-generation top-N (frspec-owngen ranks)
#   -> requant: block Q4_K_M + head NVFP4 (hqmtp order: quantize AFTER trim; NVFP4
#      head measured zero acceptance cost, q4 block measured +acceptance vs q8).
# Measured on the unsloth 27B artifact: 101 tok/s @ 85.2% acceptance (K=3), beating
# the prior daily draft (99.6 @ 83.3%). Serve with BW24_MTP_DRAFT=<out>.
#
# usage: make-trimmed-draft.sh <model.gguf> <ranks.txt> <out-draft.gguf> [topN] [imatrix.gguf]
set -euo pipefail
MODEL=$1; RANKS=$2; OUT=$3; TOPN=${4:-32768}; IMATRIX=${5:-}
PY=${BW24_CONVERT_PY:-/data/projects/bench-engines/vllm-venv/bin/python}
GGUFPY=${BW24_GGUFPY:-/data/projects/llama.cpp/gguf-py}
QUANT=${BW24_QUANTIZE:-/data/projects/llama.cpp/build/bin/llama-quantize}
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

PYTHONPATH=$GGUFPY "$PY" "$HERE/extract_mtp_draft.py" "$MODEL" "$TMP/draft-full.gguf"
PYTHONPATH=$GGUFPY "$PY" "$HERE/trim_draft_head.py" "$TMP/draft-full.gguf" "$RANKS" \
    "$TMP/draft-trim.gguf" "$TOPN"
"$QUANT" --allow-requantize ${IMATRIX:+--imatrix "$IMATRIX"} \
    --output-tensor-type nvfp4 --token-embedding-type q5_k \
    "$TMP/draft-trim.gguf" "$OUT" Q4_K_M | tail -1
echo "draft ready: $OUT"
echo "serve with:  BW24_MTP_DRAFT=$OUT ..."

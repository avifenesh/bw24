#!/usr/bin/env bash
# eval.sh — one entrypoint for the bw24 training arm (arm 3) scoreboard.
#
# Run this after every ~10 new records land in research/tune-data/*.jsonl. It:
#   1. validates the whole corpus (all rig files) and prints a per-rig summary,
#   2. scores the retrieval baseline (BGE/TF-IDF) on the frozen leave-one-out folds,
#   3. scores the stage-b head (GBM + LogReg) on the SAME folds and prints one
#      comparison table vs (majority, retrieval-BGE, retrieval-TF-IDF) with CIs.
#
# CPU-ONLY (the GPU belongs to a kernel agent; CUDA is locked out in-script),
# fully offline, and < 5 min. Thread pools are capped so it does not hammer the
# rig that shares RAM/cores with the local LLM servers.
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# Prefer an interpreter that also has sentence-transformers so the BGE rows are
# real; else fall back to the system python (TF-IDF only, BGE rows show n/a).
COLBERT_PY="/home/avifenesh/projects/colbert-2/.venv/bin/python"
if [ -x "$COLBERT_PY" ] && "$COLBERT_PY" -c "import sentence_transformers" >/dev/null 2>&1; then
  PY="$COLBERT_PY"; BACKEND_NOTE="BGE (colbert-2 venv) + TF-IDF"
else
  PY="$(command -v python3)"; BACKEND_NOTE="TF-IDF only (no sentence-transformers on system python)"
fi

echo "=============================================================================="
echo "bw24 training-arm eval  —  interpreter: $PY"
echo "text/retrieval backend: $BACKEND_NOTE"
echo "=============================================================================="

echo; echo ">>> [1/3] dataset validate + per-rig summary"
"$PY" dataset.py

echo; echo ">>> [2/3] retrieval baseline — leave-one-out"
"$PY" baseline.py

echo; echo ">>> [3/3] stage-b head (GBM + LogReg) vs baselines — leave-one-out"
"$PY" train_gbm.py

echo; echo "done. (re-run after ~10 new tune-data records; append the stage-b table"
echo "as a corpus-meta row with: $PY train_gbm.py --emit-record )"

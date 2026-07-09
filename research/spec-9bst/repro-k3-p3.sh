#!/usr/bin/env bash
# Repro: 9B ST modelopt spec K=3 on p3-agentic-long — the 9bspec lane claimed "OOM, 24GB
# ceiling" with no captured error (runs died in ~30s, stderr swallowed by the parse pipe,
# sibling 27B lane was cycling the GPU). This re-runs the exact arm ALONE on an idle GPU
# with raw output preserved. Verdict updates the two 2026-07-09 9bst rows in rig5090.jsonl.
set -u
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-modelopt
TRIM=$MODEL/frspec-9bst-modelopt-32768.gguf
P3=/home/avifenesh/projects/bw24/research/e2e/prompts/p3-agentic-long.txt
BIN=/home/avifenesh/projects/bw24/target/release/run-spec
OUTDIR=/home/avifenesh/projects/bw24/research/spec-9bst
STAMP=$(date +%H%M%S)

echo "== preflight =="
free -g | head -2
nvidia-smi --query-gpu=clocks.sm,memory.used,temperature.gpu --format=csv,noheader
FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F, '$2+0 > 500')
if [ -n "$FOREIGN" ]; then echo "ABORT: foreign GPU process: $FOREIGN"; exit 1; fi

for run in 1 2; do
  LOG=$OUTDIR/repro-k3-p3-run$run-$STAMP.log
  echo "== K=3 pmin=0.3 trim p3 run=$run -> $LOG =="
  BW24_NGEN=256 BW24_SPEC_K=3 BW24_SPEC_PMIN=0.3 BW24_SPEC_STATS=1 \
    BW24_FRSPEC_TRIM=$TRIM BW24_PROMPT_FILE=$P3 \
    "$BIN" "$MODEL" >"$LOG" 2>&1
  RC=$?
  echo "exit=$RC"
  if [ $RC -ne 0 ]; then
    echo "-- failure tail --"; tail -20 "$LOG"
    echo "-- gpu state at failure --"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  else
    grep -E 'tok/s|acceptance|self-consistency' "$LOG" | tail -6
  fi
done

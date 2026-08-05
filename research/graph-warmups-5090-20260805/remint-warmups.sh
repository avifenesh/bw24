#!/bin/bash
# WARMUPS re-mint on the deployment rig (5090, 82 SM): MEMRA_GRAPH_WARMUPS default(2) vs 1,
# interleaved x5 (H100 lane law 1 — cross-run comparisons are clock-drift-invalid), per model.
# Pod receipt being re-minted: research/graph-allocfree-20260805/logs/warmup-lever-N5.txt
# (q27 recapture -38%, q9 -41%, decode +1.1% both — on the pod GPU).
#
# GPU: local 5090, every run inside flock /tmp/gpu5090.lock (shared with fp8-blk128 — the
# flock is held per-invocation, released between = short holds).
# Baked literals (workflow-args-no-propagate law): paths + reps are constants below.
set -uo pipefail
cd /home/avifenesh/projects/wt-warmups
OUT=research/graph-warmups-5090-20260805/logs/remint-warmups-N5.txt
Q27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release/graph-allocfree-probe

{
  echo "# re-mint: MEMRA_GRAPH_WARMUPS 2(default) vs 1, interleaved x5, 5090 laptop (82 SM)"
  echo "# probe medians are N=5 inside each rep (probe --reps 5); arms interleaved per rep."
  echo "# start: $(date -Is)  commit: $(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm --format=csv,noheader
} > "$OUT"

for model in "$Q27" "$Q9"; do
  name=$(basename "$model")
  for rep in 1 2 3 4 5; do
    for arm in 2 1; do
      echo "=== $name rep$rep warmups=$arm ===" >> "$OUT"
      flock /tmp/gpu5090.lock \
        env MEMRA_GRAPH_WARMUPS=$arm "$BIN" "$model" --reps 5 2>&1 \
        | grep -E "recapture|capture\+prime|decode tok/s|launch\(async\)|SUMMARY|error|Error" >> "$OUT"
    done
  done
done
echo "# end: $(date -Is)" >> "$OUT"
echo DONE

#!/bin/bash
# WARMUPS re-mint on the deployment rig (5090, 82 SM): MEMRA_GRAPH_WARMUPS default(2) vs 1,
# N=5 pairs per model. Pod receipt being re-minted:
# research/graph-allocfree-20260805/logs/warmup-lever-N5.txt (q27 recapture -38%, q9 -41%,
# decode +1.1% both — pod GPU).
#
# PAIR PROTOCOL (H100 lane law 1 + the fp8 lane's pairsweep refinement): the rig is shared
# with the fp8-blk128 lane through flock /tmp/gpu5090.lock, so per-invocation locking would
# let a ~4-min competitor prefill burn land BETWEEN the two arms of one rep — exactly the
# clock/thermal drift interleaving exists to kill. Each rep therefore holds the lock across
# BOTH arms (one pair = one clock window, still a short hold), and the arm ORDER ALTERNATES
# per rep (2,1 / 1,2 / ...) so position-in-pair effects cancel.
# Baked literals (workflow-args-no-propagate law).
set -uo pipefail
cd /home/avifenesh/projects/wt-warmups
OUT=research/graph-warmups-5090-20260805/logs/remint-warmups-N5.txt
Q27=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=$PWD/target/release/graph-allocfree-probe

one() { # one <warmups> <name> <model> <rep>
  echo "=== $2 rep$4 warmups=$1 ===" >> "$OUT"
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader >> "$OUT"
  MEMRA_GRAPH_WARMUPS=$1 "$BIN" "$3" --reps 5 2>&1 \
    | grep -E "recapture|capture\+prime|decode tok/s|launch\(async\)|SUMMARY|error|Error" >> "$OUT"
}
export -f one
export OUT BIN

{
  echo "# re-mint: MEMRA_GRAPH_WARMUPS 2(default) vs 1, N=5 ADJACENT pairs (lock held per"
  echo "# pair, arm order alternates per rep), 5090 laptop (82 SM). probe medians are N=5"
  echo "# inside each invocation (--reps 5)."
  echo "# start: $(date -Is)  commit: $(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm --format=csv,noheader
} > "$OUT"

for model in "$Q27" "$Q9"; do
  name=$(basename "$model")
  for rep in 1 2 3 4 5; do
    if [ $((rep % 2)) -eq 1 ]; then order="2 1"; else order="1 2"; fi
    # one flock hold = one adjacent pair (both arms in the same clock window)
    flock -w 7200 /tmp/gpu5090.lock bash -c "
      for arm in $order; do one \$arm '$name' '$model' $rep; done"
  done
done
echo "# end: $(date -Is)" >> "$OUT"
echo DONE

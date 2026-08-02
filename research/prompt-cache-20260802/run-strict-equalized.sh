#!/bin/bash
# strict per the ACTUAL protocol (gate1-recal-20260802 + validate-h100.sh, as executed by
# f16g-default-rearb/run-followup.sh): equalized FP composition MEMRA_MMVQ=0
# MEMRA_NO_FUSE_NORMQ=1, worst-draw seeds MEMRA_GATE_SEED q35=16 / q9j=0, --batch 4.
# Two earlier invocations were out-of-protocol (bare strict; batch-4 without the env) —
# kept as *-strict-MISFIRE*.log: the documented accepted FP-composition gap, not a
# regression (config-mode was ALL GREEN on both models same session).
set -u
W=/home/avifenesh/projects/wt-prompt-cache
R=$W/research/prompt-cache-20260802
BIN=$W/target/release
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q9J=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
env MEMRA_GATE_SEED=16 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
  flock /tmp/gpu5090.lock timeout 3600 "$BIN/decode-batch-gate" "$Q35" --batch 4 --mode strict \
  > "$R/battery-decode-batch-q35-strict-equalized.log" 2>&1
echo "q35 strict-equalized rc=$? $(tail -1 "$R/battery-decode-batch-q35-strict-equalized.log")"
env MEMRA_GATE_SEED=0 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
  flock /tmp/gpu5090.lock timeout 3600 "$BIN/decode-batch-gate" "$Q9J" --batch 4 --mode strict \
  > "$R/battery-decode-batch-q9j-strict-equalized.log" 2>&1
echo "q9j strict-equalized rc=$? $(tail -1 "$R/battery-decode-batch-q9j-strict-equalized.log")"

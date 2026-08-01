#!/bin/bash
# Post-flip verification: KAT naked x3 (new default), ctrl bit-identity guard, rollback seam.
set -u
W=/home/avifenesh/projects/wt-kat-anomaly
R=$W/research/kat-anomaly-20260802
PF=$W/research/e2e/prompts/pp512.txt
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
CTRL=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
report() { # log
  local tg thash
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$1" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  thash=$(grep -A1 "^generated" "$1" | grep "tokens:" | sha256sum | cut -c1-16)
  echo "$(basename "$1"): decode=$tg sha=$thash argmax=$(grep -c MATCH "$1") $(grep -oE '\[stop: [A-Za-z]+\]' "$1" | head -1)"
}
for rep in 1 2 3; do
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$PF flock /tmp/gpu5090.lock timeout 900 \
    "$W/target/release/run-gen" "$KAT" > "$R/post-default-rep$rep.log" 2>&1
  report "$R/post-default-rep$rep.log"
done
MEMRA_PRIME_TOKENWISE=1 MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$PF flock /tmp/gpu5090.lock timeout 900 \
  "$W/target/release/run-gen" "$CTRL" > "$R/post-ctrl-guard.log" 2>&1
report "$R/post-ctrl-guard.log"
MEMRA_IQ_FAST=0 MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$PF flock /tmp/gpu5090.lock timeout 900 \
  "$W/target/release/run-gen" "$KAT" > "$R/post-rollback-seam.log" 2>&1
report "$R/post-rollback-seam.log"
echo POSTFLIP-DONE

#!/usr/bin/env bash
# lane/rp-on-st — NO-REGRESSION battery for the mmq_fp8_blk.cu MMA form swap.
# The swap is bit-identical by construction (same operands/fragments, identity ue8m0 scale) and was
# proven so at both the kernel level (fp8-mmq-check, every cell byte-for-byte) and the prefill-logit
# level (0/993280 bytes differ). This battery is the LIVE-PATH receipt on real checkpoints.
set -u
cd /home/avifenesh/projects/wt-rpst
R=research/rp-on-st-20260806
G9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf
ST=/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4
BLK=/data/ai-ml/hf-models/qwen36-27b-blk128fp8
L=$R/blksc-noreg.log
: > "$L"
say() { echo "$*" | tee -a "$L"; }
say "=== kernel-check: 9B NVFP4 GGUF (whole-engine bitwise battery) ==="
G=$(ls $G9/*.gguf 2>/dev/null | head -1); say "gguf=$G"
timeout 3600 ./target/release/kernel-check "$G" >> "$L" 2>&1; say "kernel-check rc=$?"
say "$(grep -acE '\bFAIL\b' "$L" | tr -d '\n') FAIL lines so far"
for pair in "9B-NVFP4-GGUF:$G" "ST-27B-e4m3+nvfp4:$ST" "BLK128-27B:$BLK"; do
  n=${pair%%:*}; a=${pair#*:}
  say "=== run-gen argmax + verify-prefill: $n ==="
  env MEMRA_FP8_MMQ_STATS=1 timeout 2400 ./target/release/run-gen "$a" --verify-prefill --max-tokens 16 >> "$L" 2>&1
  say "$n rc=$?"
done
say "=== run-gen pp512 + PP_LOGITS (the class that CARRIES the fp8-mmq hook): BLK128 27B ==="
env MEMRA_FP8_MMQ_STATS=1 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_NLL=1 \
    MEMRA_PROMPT_FILE=research/e2e/prompts/pp512.txt \
    timeout 2400 ./target/release/run-gen "$BLK" >> "$L" 2>&1
say "pp-nll rc=$?"
say "=== SUMMARY ==="
say "MATCH lines:    $(grep -ac 'MATCH' "$L")"
say "MISMATCH lines: $(grep -ac 'MISMATCH' "$L")"
say "FAIL lines:     $(grep -acE '\bFAIL\b' "$L")"
say "maxdiff lines:  $(grep -a -oP 'logit maxdiff=\K[0-9.e+-]+' "$L" | tr '\n' ' ')"
say "dispatch lines: $(grep -a -oP 'fp8-mmq dispatches: \K\d+' "$L" | tr '\n' ' ')"

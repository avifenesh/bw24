#!/usr/bin/env bash
# 6-DRAW LOOP — the model-level exactness evidence for branch (b) of the exactness bar.
#
# The free-running greedy streams of the MMQ arm and the Q8_0 floor DIVERGE (first flip at step 1,
# margin 0.264). Branch (b) therefore applies: per-block-FP8 arithmetic is not Q8_0-requant
# arithmetic, so stream-identity is the wrong instrument. What matters instead:
#   * is every divergence a NEAR-TIE (an FP-composition-class flip), or does the arm pick tokens
#     the floor ranked far down (a broken-kernel signature)?
#   * does that hold across independent draws, not one lucky prompt?
#
# For each of 6 distinct prompts: run the floor free-running to make a tape, then run the MMQ arm
# TEACHER-FORCED on that tape so both arms see bit-identical inputs at every position. Every
# disagreement is then attributable to that position's arithmetic alone, and its margin is printed.
set -uo pipefail
CK=${CK:-/data/ai-ml/hf-models/qwen3-1.7b-blk128fp8-synth}
BIN=${BIN:-./target/release/fp8_mmq_stream}
R=${R:-research/fp8st-20260804/mmq}
N=${N:-32}
mkdir -p "$R/sixdraw"

# 6 draws, each >= 16 tokens so m clears GEMM_M_THRESHOLD from step 0. Mixed domains: chat opener,
# arithmetic, code, prose, list, and a repetition-prone tail.
D1="151643 9707 11 1879 30 33464 264 3766 315 279 1372 220 16 17 18 19"
D2="220 17 488 220 17 284 220 19 13 220 18 488 220 18 284 220 21 13 220 19 488 220 19 284"
D3="750 282 2075 8595 982 1648 262 470 308 353 220 17 271 750 342 2075 8595 982 1648"
D4="785 6722 315 9625 374 12095 11 323 279 6722 315 9856 374 7148 11 323 279"
D5="16 13 23245 198 17 13 40655 198 18 13 90513 198 19 13 60555 198 20 13"
D6="785 4021 4014 39956 34208 1975 279 16務 5562 13 576 4021 4014 39956 34208 1975"

for i in 1 2 3 4 5 6; do
  eval "P=\$D$i"
  env -u MEMRA_FP8_MMQ -u MEMRA_FP8_BLK_GPU -u MEMRA_FP8_FOLD \
    "$BIN" "$CK" "$N" $P > "$R/sixdraw/d$i-floor.log" 2>&1
  rc_f=$?
  MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=8192 \
    MEMRA_FP8_MMQ_TF="$R/sixdraw/d$i-floor.log" \
    "$BIN" "$CK" "$N" $P > "$R/sixdraw/d$i-mmq-tf.log" 2>&1
  rc_m=$?
  echo "draw $i floor_rc=$rc_f mmq_rc=$rc_m  $(grep -o 'disagreements: .*' "$R/sixdraw/d$i-mmq-tf.log")"
done

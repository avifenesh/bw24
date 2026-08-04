#!/usr/bin/env bash
# Slice-3 model-level battery for the per-block FP8 MMQ prefill kernel (lane/fp8-mmq).
#
# Checkpoint: the genuine block-128 FP8 safetensors dir ARM B' built from the local Qwen3-1.7B
# BF16 dir (research/fp8st-20260803/armb/make_blk128_fp8_ckpt.py) — 196 2-D Linear weights as
# F8_E4M3 codes + a BF16 weight_scale_inv [ceil(out/128), ceil(in/128)] grid, per-block
# s = amax/448. Dynamic range varies block to block: the property ARM A's global fold destroys.
#
# Four runs:
#   floor        free-running greedy, no FP8 flags   -> the Q8_0-requant reference stream
#   armbprime    free-running greedy, ARM B' device dequant (byte-identical to floor by its own
#                kernel-check arm) -> proves the tape is arm-stable, not a floor artifact
#   tf-floor     teacher-forced on the floor tape, floor arm -> self-reproduction control
#   tf-mmq       teacher-forced on the floor tape, MEMRA_FP8_MMQ=1 -> the drift measurement
#
# Teacher forcing is what makes the last two comparable: free-running streams that flip once at a
# near-tie are incomparable afterwards (every later position sees a different prefix), so a
# raw stream diff cannot distinguish "arithmetic drifted a lot" from "one 0.26-logit tie flipped".
set -uo pipefail
CK=${CK:-/data/ai-ml/hf-models/qwen3-1.7b-blk128fp8-synth}
BIN=${BIN:-./target/release/fp8_mmq_stream}
R=${R:-research/fp8st-20260804/mmq}
N=${N:-128}

run() { # name, then env assignments already exported by caller
  echo "--- $1"
  "$BIN" "$CK" "$N" > "$R/stream-$1.log" 2>&1
  echo "$1 rc=$?"
}

env -u MEMRA_FP8_MMQ -u MEMRA_FP8_BLK_GPU "$BIN" "$CK" "$N" > "$R/stream-floor.log" 2>&1
echo "floor rc=$?"

MEMRA_FP8_BLK_GPU=1 "$BIN" "$CK" "$N" > "$R/stream-armbprime.log" 2>&1
echo "armbprime rc=$?"

MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=8192 "$BIN" "$CK" "$N" > "$R/stream-fp8mmq.log" 2>&1
echo "fp8mmq rc=$?"

MEMRA_FP8_MMQ_TF="$R/stream-floor.log" MEMRA_FP8_MMQ_LOGITS="$R/logits-floor.bin" \
  "$BIN" "$CK" "$N" > "$R/stream-tf-floor.log" 2>&1
echo "tf-floor rc=$?"

MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=8192 \
  MEMRA_FP8_MMQ_TF="$R/stream-floor.log" MEMRA_FP8_MMQ_LOGITS="$R/logits-mmq.bin" \
  "$BIN" "$CK" "$N" > "$R/stream-tf-mmq.log" 2>&1
echo "tf-mmq rc=$?"

MEMRA_FP8_BLK_GPU=1 \
  MEMRA_FP8_MMQ_TF="$R/stream-floor.log" MEMRA_FP8_MMQ_LOGITS="$R/logits-armbprime.bin" \
  "$BIN" "$CK" "$N" > "$R/stream-tf-armbprime.log" 2>&1
echo "tf-armbprime rc=$?"

# ARM A on the SAME teacher-forced tape — the drift the owner already ruled not-shippable, as the
# calibration yardstick for the MMQ arm's drift. ARM A needs a per-tensor e4m3 consumer, so
# MEMRA_ST_E4M3=1 rides the folded bytes through try_fp8_gemm.
MEMRA_FP8_FOLD=1 MEMRA_ST_E4M3=1 \
  MEMRA_FP8_MMQ_TF="$R/stream-floor.log" MEMRA_FP8_MMQ_LOGITS="$R/logits-arma.bin" \
  "$BIN" "$CK" "$N" > "$R/stream-tf-arma.log" 2>&1
echo "tf-arma rc=$?"

MEMRA_FP8_FOLD=1 MEMRA_ST_E4M3=1 "$BIN" "$CK" "$N" > "$R/stream-arma.log" 2>&1
echo "arma rc=$?"

# --- QUALITY SANITY: mean token NLL on a frozen real-text window (nll-window.txt, held-out GSM8K
#     test prose scraped from the local parquet — no model-under-test output in it, so no arm is
#     favoured by construction). W = window length in tokens.
W=${W:-1024}
for arm in floor mmq armbprime arma; do
  case $arm in
    floor)     ENVS=() ;;
    mmq)       ENVS=(MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=8192) ;;
    armbprime) ENVS=(MEMRA_FP8_BLK_GPU=1) ;;
    arma)      ENVS=(MEMRA_FP8_FOLD=1 MEMRA_ST_E4M3=1) ;;
  esac
  env "${ENVS[@]}" MEMRA_FP8_MMQ_NLL="$R/nll-window.txt" \
    "$BIN" "$CK" "$W" > "$R/nll-$arm.log" 2>&1
  echo "nll-$arm rc=$? $(grep -o 'mean_nll=.*' "$R/nll-$arm.log")"
done

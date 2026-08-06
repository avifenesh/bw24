#!/usr/bin/env bash
# step37-p2: FALSIFIABLE knife-edge test of the closed-form arm model for step35 prefill.
#
# MODEL (derived from hybrid_forward.rs:6820-6844 + the PRIME_MIN_T tail merge at :470):
#   On SWA layers (33 of 45, win=512) a chunk [b,e) computes off=max(0,b-(win-1)), t_kv=e-off.
#   t_kv<=win -> fa_prefill_view_ws (hd128 dequant-once FA);  t_kv>win -> sdpa_naive_w_quantized_view.
#   A chunk with b<=win-1 has off=0, so t_kv=e and it is FA iff e<=win.
#   A chunk with b>=win has t_kv=t+511>win for any t>=2.
#   => the FA rows are always a contiguous PREFIX [0,P) with the CLOSED FORM
#          P = c*floor(win/c)   for c<=win ;   P = 0   for c>win
#   The verdict depends ONLY on P. Two chunk sizes with equal P must be BIT-IDENTICAL
#   regardless of how many chunks they use; two with different P must DIFFER.
#
# Already measured (chunkinv-profile-20260806T223524Z.log): P(4096)=P(2048)=0 agree EXACT;
# P(512)=P(64)=512 agree EXACT with each other (section B) yet both DIFFER from 4096 by the
# same 1.813e0 (section A); P(1024)=P(768)=P(600)=0 all EXACT (section C).
#
# THIS RUN tests the model where it is easiest to FALSIFY:
#   PRED-1  ref=4096 vs 513  -> P 0 vs 0     -> EXACT.  A ONE-TOKEN change from 512 (which
#                                              DIFFERs) must flip the verdict. Nothing about
#                                              reduction order or chunk count is discontinuous
#                                              at 512->513; only the arm predicate is.
#   PRED-2  ref=4096 vs 512  -> P 0 vs 512   -> DIFFER (re-confirm in the same process/clock)
#   PRED-3  ref=512  vs 384  -> P 512 vs 384 -> DIFFER. Both are "small chunks with an FA
#                                              prefix", so a hand-wave of "small chunks are
#                                              noisy" predicts EXACT here; the model predicts
#                                              DIFFER because the PREFIX LENGTHS differ.
#   PRED-4  ref=512  vs 256  -> P 512 vs 512 -> EXACT. 20 chunks vs 10 chunks, DOUBLE the
#                                              partial-sum count, SAME prefix. This is the
#                                              control that kills reduction-order noise: if
#                                              divergence tracked chunk count this MUST differ.
# PRED-1 and PRED-4 are the load-bearing ones. Either failing refutes the closed form.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
P=research/chunk-invariance-20260805
RAW=$HOME/step37/raw
mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$RAW/chunkinv-knife-$TS.log

{
echo "=== step37-p2 chunkinv KNIFE-EDGE (closed-form P = c*floor(512/c)) $TS"
echo "=== model predicts: 4096~513 EXACT | 4096!=512 DIFFER | 512!=384 DIFFER | 512~256 EXACT"
(
  flock -w 1800 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## PRED-1+2: ref=4096 vs 513 (P=0, expect EXACT) and 512 (P=512, expect DIFFER) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    ./target/release/concat-prime-probe "$M" chunkinv --prompt-a "@$P/prompt-pp6257.txt" \
    --chunks 4096,513,512 --steps 24
  echo "=== rc=$?"

  echo; echo "########## PRED-3+4: ref=512 vs 384 (P=384, expect DIFFER) and 256 (P=512, expect EXACT) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    ./target/release/concat-prime-probe "$M" chunkinv --prompt-a "@$P/prompt-pp6257.txt" \
    --chunks 512,384,256 --steps 24
  echo "=== rc=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

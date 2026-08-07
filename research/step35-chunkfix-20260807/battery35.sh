#!/usr/bin/env bash
# lane/step35-chunkfix: the FINDING LANE's OWN committed battery, re-run post-fix.
# Pre-fix results (research/step37-p2-20260806): long -> 512/64 DIFFER by 1.813e0;
# knife -> PRED-2 (4096 vs 512) and PRED-3 (512 vs 384) DIFFER by design.
# Post-fix EVERY arm in BOTH scripts must be EXACT (the fix makes P == 0 everywhere).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
P=research/chunk-invariance-20260805
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/battery35-$TS.log
{
echo "=== lane/step35-chunkfix: finding-lane battery re-run POST-FIX $TS commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 3600 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  # --- chunkinv-long-step35.sh's body (GAP-2 sweep): control T=402 + defect T=4883 ---
  for p in prompt-pp512.txt prompt-pp6257.txt; do
    echo; echo "########## LONG $p : chunks 4096,2048,512,64 (pre-fix: pp6257 512/64 DIFFER) ##########"
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
      ./target/release/concat-prime-probe "$M" chunkinv --prompt-a "@$P/$p" \
      --chunks 4096,2048,512,64 --steps 24
    echo "exit=$?"
  done

  # --- chunkinv-knife-step35.sh's body: the 4 pre-registered predictions ---
  echo; echo "########## KNIFE PRED-1+2: ref=4096 vs 513,512 (pre-fix: 513 EXACT, 512 DIFFER) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    ./target/release/concat-prime-probe "$M" chunkinv --prompt-a "@$P/prompt-pp6257.txt" \
    --chunks 4096,513,512 --steps 24
  echo "=== rc=$?"
  echo; echo "########## KNIFE PRED-3+4: ref=512 vs 384,256 (pre-fix: 384 DIFFER@row384, 256 EXACT) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    ./target/release/concat-prime-probe "$M" chunkinv --prompt-a "@$P/prompt-pp6257.txt" \
    --chunks 512,384,256 --steps 24
  echo "=== rc=$?"

  # --- WIDER sweep the fix makes newly meaningful: the closed form said P differs for 95.7% of
  #     T < 12000, smallest affected T = 513. Probe the boundary + a monolithic arm. ---
  echo; echo "########## BOUNDARY: T=4883 chunks 0(monolithic),4096,1024,600,128,32,16 ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    ./target/release/concat-prime-probe "$M" chunkinv --prompt-a "@$P/prompt-pp6257.txt" \
    --chunks 4096,1024,600,128,32,16 --steps 24
  echo "=== rc=$?"
  echo; echo "########## CONTROL T=402 (below window) chunks 4096,512,64,32 — must stay EXACT ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    ./target/release/concat-prime-probe "$M" chunkinv --prompt-a "@$P/prompt-pp512.txt" \
    --chunks 4096,512,64,32 --steps 24
  echo "=== rc=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

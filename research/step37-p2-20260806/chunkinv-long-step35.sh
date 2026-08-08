#!/usr/bin/env bash
# step37-p2 GAP 2 closure: chunk-invariance ACROSS step35's two-kernel prefill split.
#
# The default gate (G6a, PASS) used the pinned T=96/147 prompts. On step35 those are BELOW the
# 512-token SWA window, so `swa && t_kv > win` is false in every chunk and every arm took
# fa_prefill_view_ws — the gate compared one kernel against itself. The at-risk property is
# whether the OTHER arm (sdpa_naive_w_quantized_view, the f32 windowed floor taken when
# t_kv > 512) agrees with it, since chunk size is what decides which one runs:
#
#   hybrid_forward.rs:6820-6844   off = swa ? base_len-(win-1) : 0 ; t_kv = base_len+t-off
#                                 t_kv > win  -> sdpa_naive_w_quantized_view   (f32 floor)
#                                 else        -> fa_prefill_view_ws            (hd128 FA)
#
# So this sweep uses prompts that STRADDLE the boundary, at chunk sizes that produce different
# arm mixes for the same prompt:
#   pp512   (~600 tok): chunk 4096 = 1 chunk naive_w ; chunk 64/32 = many chunks, MIXED
#   pp6257  (~6.2K tok): every chunk size multi-chunk, mixes differ by size
# If the two kernels are the same numeric class (the code comment's claim), output stays
# byte-identical. If not, chunkinv reports CHUNK-DEPENDENT with a per-row maxdiff STEP at the
# boundary row rather than a flat band — the razor already built into the probe.
#
# This is the mechanism test the canary cannot provide on this arch (GAP 1): the arm split is
# reached by CHUNK SIZE alone, no env seam required.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
P=research/chunk-invariance-20260805
RAW=$HOME/step37/raw
mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$RAW/chunkinv-long-$TS.log

{
echo "=== step37-p2 chunkinv LONG-PROMPT sweep (GAP 2) $TS"
echo "=== trunk: $M"
echo "=== win=512 (step35 SWA); the arm split is t_kv>512"
(
  flock -w 1800 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"

  for p in prompt-pp512.txt prompt-pp6257.txt; do
    echo; echo "########## $p : chunks 4096,2048,512,64 ##########"
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
      ./target/release/concat-prime-probe "$M" chunkinv --prompt-a "@$P/$p" \
      --chunks 4096,2048,512,64 --steps 24
    echo "exit=$?"
  done

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "log: $LOG"

#!/usr/bin/env bash
# lane/tick-seg: BEFORE/AFTER serve-prime perf, INTERLEAVED, N=5, one lock hold.
#
# INSTRUMENT: concat-prime-probe `ppprime --budget B` — times the SERVE-SHAPED multi-call prime
# (the worker tick-loop replica), which IS the path this fix changes. Monolithic ppprime and
# run-gen's prefill line cannot see a multi-call change by construction.
#
# ARMS, same binary, one process each, alternating AFTER/BEFORE (interleaved law):
#   AFTER  = naked default (request-level seq_end via queued_after — the shipped path)
#   BEFORE = MEMRA_PRIME_CALLLOCAL=1 (per-call seq_end via the rollback seam; the gate battery
#            proves the seam is a faithful restoration of the pre-fix arithmetic)
# CELLS:
#   pp6257 budget=1024 — THE SHIPPED INTERACTIVE DEFAULT (MEMRA_PREFILL_TICK). Enumeration: at
#          budget 1024 every call's per-call seq_end already exceeded win=512, so the arm
#          sequence is IDENTICAL pre/post — prediction is ZERO delta; the measurement proves it.
#          THIS is the 1%-STOP-bar cell.
#   pp6257 budget=256 — the dark-lane default, where the fix genuinely changes arms (pre-fix
#          calls 1-2 rode FA, post-fix all calls take the windowed arm). Expected small cost,
#          bounded, recorded.
#   pp512  budget=1024 — null control BELOW the window: single call either way, both arms take
#          the same branch; honest expected delta 0 = the box's noise floor for this instrument.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
P=research/chunk-invariance-20260805
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/perf-tickseg-$TS.log
N=5
{
echo "=== lane/tick-seg serve-prime perf, interleaved N=$N $TS commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm --format=csv,noheader

  for cell in "pp6257:prompt-pp6257.txt:1024" "pp6257:prompt-pp6257.txt:256" \
              "pp512:prompt-pp512.txt:1024"; do
    name=${cell%%:*}; rest=${cell#*:}; pf=${rest%%:*}; bg=${rest##*:}
    echo; echo "########## CELL $name budget=$bg : interleaved AFTER/BEFORE x$N ##########"
    for i in $(seq 1 $N); do
      echo "--- rep $i AFTER (naked default, request-level seq_end) budget=$bg"
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 1800 \
        ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P/$pf" \
        --budget "$bg" --reps 3 --warmup 1 2>&1 | grep -E "MEDIAN|rep "
      echo "--- rep $i BEFORE (MEMRA_PRIME_CALLLOCAL=1, per-call seq_end) budget=$bg"
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PRIME_CALLLOCAL=1 timeout 1800 \
        ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P/$pf" \
        --budget "$bg" --reps 3 --warmup 1 2>&1 | grep -E "MEDIAN|rep "
      nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm --format=csv,noheader | tr '\n' ' '; echo
    done
  done

  nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

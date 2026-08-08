#!/usr/bin/env bash
# lane/step35-chunkfix: BEFORE/AFTER prefill perf, INTERLEAVED, N>=5, one lock hold.
#
# INSTRUMENT: concat-prime-probe `ppprime` — it times prime_cache, which IS the path the fix
# changes. (run-gen's "prefill tok/s" line times forward_last, the CACHELESS monolithic path,
# where seq_end == t by construction — it cannot see this change at all, so it is not the
# instrument. Recorded so nobody re-measures the wrong thing.)
#
# ARMS, same binary, one process each, alternating A/B/A/B... (the repo's interleaved law —
# cross-run comparison is clock-drift-invalid):
#   AFTER  = naked default (seq_end predicate, the shipped path)
#   BEFORE = MEMRA_STEP35_SWA_TKV=1 (the pre-fix t_kv predicate via the rollback seam) — this is
#            an EXACT restoration of the old arm selection, so it is a true before-arm without
#            needing a second build of c809181d^.
# CELLS: pp512 (T=402, ENTIRELY BELOW the window -> both arms identical by construction, the
#        null control) and pp6257 (T=4883, past the window -> where the predicate matters).
# CHUNKS: 4096 (the SHIPPED DEFAULT — enumeration says the arm sequence is IDENTICAL there, so
#        the prediction is ZERO delta and this measurement PROVES it) and 512 (where the fix
#        actually changes arms: 1 chunk moves FA -> naive_w).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
P=research/chunk-invariance-20260805
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/perf35-$TS.log
N=5
{
echo "=== lane/step35-chunkfix prefill perf, interleaved N=$N $TS commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 5400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm --format=csv,noheader

  for cell in "pp6257:prompt-pp6257.txt:4096" "pp6257:prompt-pp6257.txt:512" \
              "pp512:prompt-pp512.txt:4096"; do
    name=${cell%%:*}; rest=${cell#*:}; pf=${rest%%:*}; ck=${rest##*:}
    echo; echo "########## CELL $name chunk=$ck : interleaved AFTER/BEFORE x$N ##########"
    for i in $(seq 1 $N); do
      echo "--- rep $i AFTER (naked default, seq_end predicate) chunk=$ck"
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PRIME_CHUNK=$ck timeout 1800 \
        ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P/$pf" --reps 3 --warmup 1 \
        2>&1 | grep -E "MEDIAN|rep "
      echo "--- rep $i BEFORE (MEMRA_STEP35_SWA_TKV=1, t_kv predicate) chunk=$ck"
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PRIME_CHUNK=$ck MEMRA_STEP35_SWA_TKV=1 timeout 1800 \
        ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P/$pf" --reps 3 --warmup 1 \
        2>&1 | grep -E "MEDIAN|rep "
      nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm --format=csv,noheader | tr '\n' ' '; echo
    done
  done

  nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

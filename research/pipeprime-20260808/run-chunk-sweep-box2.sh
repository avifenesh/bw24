#!/usr/bin/env bash
# Exploratory PP-2 pipeline microchunk sweep, N=3 alternating order, one GPU-lock hold.
set -uo pipefail

REPO=${REPO:-"$HOME/memra"}
MODEL=${MODEL:-"/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
PROMPT4096=${PROMPT4096:-"/tmp/pipeprime-prompt-pp4096.txt"}
RAW=${RAW:-"/tmp/pipeprime-sweep"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/sweep-summary-$TS.log"
P512="$RAW/prompt-pp512-$TS.txt"
P2048="$RAW/prompt-pp2048-$TS.txt"

mkdir -p "$RAW"
head -c 2800 "$PROMPT4096" >"$P512"
head -c 11200 "$PROMPT4096" >"$P2048"
cd "$REPO"

thermal() {
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used \
    --format=csv,noheader
}

{
  echo "=== pipeprime microchunk sweep $TS commit=$(git rev-parse HEAD)"
  rc=0
  (
    flock -w 14400 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    thermal
    for cell in \
      "pp512:$P512:128" \
      "pp512:$P512:64" \
      "pp2048:$P2048:512" \
      "pp2048:$P2048:256" \
      "pp4096:$PROMPT4096:1024" \
      "pp4096:$PROMPT4096:512" \
      "pp4096:$PROMPT4096:256"
    do
      name=${cell%%:*}
      rest=${cell#*:}
      prompt=${rest%:*}
      chunk=${rest##*:}
      log="$RAW/$name-c$chunk-raw-$TS.log"
      echo "########## $name chunk=$chunk ##########"
      thermal
      env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PRIME_CHUNK="$chunk" \
        timeout 3600 ./target/release/concat-prime-probe "$MODEL" pppipeperf \
        --prompt-a "@$prompt" --reps 3 --warmup 1 >"$log" 2>&1
      cell_rc=$?
      cat "$log"
      echo "$name chunk=$chunk exit=$cell_rc raw=$log"
      if (( cell_rc != 0 )); then
        rc=1
      fi
      thermal
    done
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  sweep_rc=$?
  echo "=== sweep rc=$sweep_rc"
  echo "=== medians ==="
  grep -h "pppipeperf MEDIAN:" "$RAW"/*-raw-"$TS".log || true
  echo "=== done $(date -u +%FT%TZ)"
  exit "$sweep_rc"
} >"$SUMMARY" 2>&1

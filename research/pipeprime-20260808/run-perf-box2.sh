#!/usr/bin/env bash
# PP-2 serial-vs-pipeline perf, N=5 alternating order per shape, one GPU-lock hold.
set -uo pipefail

REPO=${REPO:-"$HOME/memra"}
MODEL=${MODEL:-"/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
PROMPT4096=${PROMPT4096:-"/tmp/pipeprime-prompt-pp4096.txt"}
RAW=${RAW:-"/tmp/pipeprime-perf"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/perf-summary-$TS.log"
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
  echo "=== pipeprime perf $TS commit=$(git rev-parse HEAD)"
  for p in "$P512" "$P2048" "$PROMPT4096"; do
    echo "prompt=$p bytes=$(wc -c <"$p") sha256=$(sha256sum "$p" | awk '{print $1}')"
  done
  rc=0
  (
    flock -w 14400 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    thermal
    for cell in \
      "pp512:$P512" \
      "pp2048:$P2048" \
      "pp4096:$PROMPT4096"
    do
      name=${cell%%:*}
      prompt=${cell#*:}
      log="$RAW/$name-raw-$TS.log"
      echo "########## $name naked auto geometry ##########"
      thermal
      env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
        timeout 3600 ./target/release/concat-prime-probe "$MODEL" pppipeperf \
        --prompt-a "@$prompt" --reps 5 --warmup 1 >"$log" 2>&1
      cell_rc=$?
      cat "$log"
      echo "$name exit=$cell_rc raw=$log"
      if (( cell_rc != 0 )); then
        rc=1
      fi
      thermal
    done
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  perf_rc=$?
  echo "=== perf rc=$perf_rc"
  echo "=== medians ==="
  grep -h "pppipeperf MEDIAN:" "$RAW"/*-raw-"$TS".log || true
  echo "=== done $(date -u +%FT%TZ)"
  exit "$perf_rc"
} >"$SUMMARY" 2>&1

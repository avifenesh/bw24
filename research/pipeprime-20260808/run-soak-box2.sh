#!/usr/bin/env bash
# 200-prime PP-2 pipeline exactness/liveness soak. Run from the box2 memra checkout.
set -uo pipefail

REPO=${REPO:-"$HOME/memra"}
MODEL=${MODEL:-"/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
PROMPT=${PROMPT:-"/tmp/pipeprime-prompt-pp4096.txt"}
RAW=${RAW:-"/tmp/pipeprime-soak"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/soak-summary-$TS.log"
PROBE_LOG="$RAW/soak-raw-$TS.log"

mkdir -p "$RAW"
cd "$REPO"

thermal() {
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used \
    --format=csv,noheader
}

{
  echo "=== pipeprime soak $TS commit=$(git rev-parse HEAD)"
  echo "model=$MODEL"
  echo "prompt=$PROMPT bytes=$(wc -c <"$PROMPT") sha256=$(sha256sum "$PROMPT" | awk '{print $1}')"
  (
    flock -w 14400 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    thermal
    env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
      timeout 14400 ./target/release/concat-prime-probe "$MODEL" ppsplit \
      --prompt-a "@$PROMPT" --chunks auto --steps 1 --soak 200 \
      >"$PROBE_LOG" 2>&1
    rc=$?
    cat "$PROBE_LOG"
    echo "probe exit=$rc"
    echo "--- parsed soak verdict ---"
    grep -E "ppsplit:|soak 200/200|soak pipe_primes=|ppsplit verdict:" "$PROBE_LOG" || true
    echo "--- fault scan ---"
    grep -Ei "CUDA_ERROR|illegal address|MMU fault|MISMATCH|NOT-LIVE" "$PROBE_LOG" || true
    thermal
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  soak_rc=$?
  echo "=== soak rc=$soak_rc"
  echo "=== done $(date -u +%FT%TZ)"
  exit "$soak_rc"
} >"$SUMMARY" 2>&1

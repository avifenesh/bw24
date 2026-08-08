#!/usr/bin/env bash
# Pipe-prime target-rig acceptance battery. Run from the box2 memra checkout.
set -uo pipefail

export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:/usr/local/cuda/bin:/usr/bin:/bin"

REPO=${REPO:-"$HOME/memra"}
MODEL=${MODEL:-"/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
DRAFT=${DRAFT:-"/data/models/step37/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf"}
RAW=${RAW:-"/tmp/pipeprime-acceptance"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/acceptance-summary-$TS.log"
PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
PP=(MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1)

mkdir -p "$RAW"
cd "$REPO"

thermal() {
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used \
    --format=csv,noheader
}

run_gate() {
  local label=$1
  shift
  local log="$RAW/$label-$TS.log"
  echo "########## $label ##########"
  echo "raw=$log"
  thermal
  "$@" >"$log" 2>&1
  local rc=$?
  cat "$log"
  echo "$label exit=$rc"
  return "$rc"
}

{
  echo "=== pipeprime acceptance $TS commit=$(git rev-parse HEAD)"
  rc=0
  (
    flock -w 14400 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    thermal

    run_gate kernel-check timeout 3600 \
      ./target/release/kernel-check "$MODEL" || rc=1

    run_gate run-gen env "${PP[@]}" MEMRA_NGEN=64 timeout 2400 \
      ./target/release/run-gen "$MODEL" --prompt "$PROMPT" || rc=1

    run_gate run-spec env "${PP[@]}" MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=32 \
      MEMRA_PROMPT="$PROMPT" timeout 5400 \
      ./target/release/run-spec "$MODEL" || rc=1

    thermal
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  battery_rc=$?
  echo "=== acceptance rc=$battery_rc"
  echo "=== done $(date -u +%FT%TZ)"
  exit "$battery_rc"
} >"$SUMMARY" 2>&1

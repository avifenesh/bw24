#!/usr/bin/env bash
# Box1 N=8 same-prefix TTFT A/B. Caller may wait on /tmp/memra-gpu.lock.
set -uo pipefail

REPO=${REPO:-"$HOME/memra-prefixdedup"}
BIN=${BIN:-"$REPO/target/release/memra-server"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
RAW=${RAW:-"$REPO/research/prefixdedup-20260808/raw/box1"}
PORT=${PORT:-18108}
BASE="http://127.0.0.1:$PORT"
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
K=${K:-1024}
N=${N:-8}
SUFFIX=${SUFFIX:-16}
SUMMARY="$RAW/fanout-summary-$TS.log"
RESULTS="$RAW/fanout-ttft-$TS.jsonl"
SERVER_PID=

mkdir -p "$RAW"
cd "$REPO"

thermal() {
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used,memory.total \
    --format=csv,noheader
}

stop_server() {
  if [[ -n ${SERVER_PID:-} ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}

boot() {
  local arm=$1
  local dedup=$2
  local server_log="$RAW/fanout-server-$arm-$TS.log"
  env \
    MEMRA_COMPAT=openai \
    MEMRA_MODELS="step35=$MODEL" \
    MEMRA_SERVE_SPEC=0 \
    MEMRA_REUSE_POOL=0 \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_PREFILL_TICK=1024 \
    MEMRA_PREFIX_CACHE_MB=4096 \
    MEMRA_PREFIX_DEDUP="$dedup" \
    MEMRA_ADDR="127.0.0.1:$PORT" \
    "$BIN" >"$server_log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 180); do
    if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
      echo "$arm ready in <=$((i * 2))s log=$server_log"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$arm server died"
      tail -100 "$server_log"
      return 1
    fi
    sleep 2
  done
  echo "$arm readiness timeout"
  return 1
}

run_arm() {
  local arm=$1
  local dedup=$2
  local expect=$3
  boot "$arm" "$dedup" || return 1
  python3 research/prefixdedup-20260808/fanout_ttft.py \
    --base "$BASE" \
    --model step35 \
    --label "$arm" \
    --out "$RESULTS" \
    --n "$N" \
    --k "$K" \
    --suffix "$SUFFIX" \
    --expect "$expect" \
    --warmup || return 1
  stop_server
  sleep 3
  thermal
}

locked_run() {
  trap stop_server EXIT
  thermal
  run_arm dedup-off 0 cold || return 1
  run_arm dedup-on 1 dedup || return 1
}

{
  echo "=== prefix fanout TTFT ts=$TS commit=$(git rev-parse HEAD)"
  echo "binary=$BIN sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "model=$MODEL bytes=$(stat -c %s "$MODEL")"
  echo "protocol=one warmup plus one N=$N barrier burst per arm; K=$K suffix=$SUFFIX"
  if [[ ${MEMRA_GPU_LOCK_HELD:-0} == 1 ]]; then
    echo "using caller-held GPU lock $(date -u +%FT%TZ)"
    locked_run
    rc=$?
  else
    (
      flock -w 14400 9 || {
        echo "LOCK TIMEOUT"
        exit 75
      }
      echo "lock acquired $(date -u +%FT%TZ)"
      locked_run
      rc=$?
      echo "lock released $(date -u +%FT%TZ)"
      exit "$rc"
    ) 9>/tmp/memra-gpu.lock
    rc=$?
  fi
  echo "=== prefix fanout TTFT rc=$rc"
  echo "results=$RESULTS"
  echo "=== done $(date -u +%FT%TZ)"
  exit "$rc"
} >"$SUMMARY" 2>&1

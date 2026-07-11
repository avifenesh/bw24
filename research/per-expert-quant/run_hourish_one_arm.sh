#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"

: "${ARM:?set ARM}"
: "${ARTIFACT:?set ARTIFACT}"
: "${RUN_ID:?set the shared RUN_ID}"
: "${SERVER_BIN:?set SERVER_BIN}"

OUT_ROOT=${OUT_ROOT:-$HERE/results-hourish}
CACHE_DIR=${CACHE_DIR:-$HERE/.cache}
ADDR=${ADDR:-127.0.0.1:8080}
BASE_URL="http://$ADDR/v1/completions"
SERVER_ROOT=${BASE_URL%/v1/completions}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-900}
HOURISH_SHARD_TIMEOUT_S=${HOURISH_SHARD_TIMEOUT_S:-43200}
PAGE_PREFETCH_WINDOW=${PAGE_PREFETCH_WINDOW:-8}
BW24_SPILL_PREAD_DEPTH=${BW24_SPILL_PREAD_DEPTH:-8}

die() {
  echo "error: $*" >&2
  exit 2
}

[[ "$ARM" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid ARM"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid RUN_ID"
[[ -f "$ARTIFACT/manifest.json" ]] || die "missing artifact manifest"
[[ -x "$SERVER_BIN" ]] || die "missing executable server"
[[ "$HEALTH_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || die "invalid health timeout"
[[ "$HOURISH_SHARD_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || die "invalid shard timeout"
[[ "$BW24_SPILL_PREAD_DEPTH" =~ ^([1-9]|[1-5][0-9]|6[0-4])$ ]] || die "invalid spill depth"

CONTROL_DIR="$OUT_ROOT/_control/$RUN_ID/$ARM"
[[ ! -e "$CONTROL_DIR/complete" ]] || die "$ARM is already complete"
mkdir -p "$CONTROL_DIR"
ATTEMPT="$CONTROL_DIR/attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$ATTEMPT"
SERVER_LOG="$ATTEMPT/server.log"
SERVER_PID=""

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in {1..100}; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

cleanup() {
  status=$?
  trap - EXIT
  stop_server
  printf '%s\n' "$status" > "$ATTEMPT/exit-code"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if curl -sS --connect-timeout 1 --max-time 2 "$SERVER_ROOT/health" >/dev/null 2>&1; then
  die "an HTTP server is already answering at $SERVER_ROOT"
fi

{
  printf 'BW24_COMPAT=openai\n'
  printf 'BW24_SERVE_SPEC=0\n'
  printf 'BW24_KV_REUSE=0\n'
  printf 'BW24_CTX=8192\n'
  printf 'BW24_FAST=1\n'
  printf 'BW24_MMVQ=1\n'
  printf 'BW24_MOE_CACHE=1\n'
  printf 'BW24_MOE_GROUPED=1\n'
  printf 'BW24_MOE_PREWARM=1\n'
  printf 'BW24_MOE_PREFETCH=1\n'
  printf 'BW24_MOE_PAGE_PREFETCH=1\n'
  printf 'BW24_MOE_PAGE_PREFETCH_WINDOW=%s\n' "$PAGE_PREFETCH_WINDOW"
  printf 'BW24_MOE_MMAP_ADVICE=normal\n'
  printf 'BW24_MOE_RESIDENT=1\n'
  printf 'BW24_MOE_VRAM_FRAC=0.85\n'
  printf 'BW24_SPILL_IO=worker\n'
  printf 'BW24_SPILL_PREAD_DEPTH=%s\n' "$BW24_SPILL_PREAD_DEPTH"
  printf 'BW24_SPILL_STATS=1\n'
  printf 'BW24_MODELS=%s=%s\n' "$ARM" "$ARTIFACT"
  printf 'BW24_ADDR=%s\n' "$ADDR"
} > "$ATTEMPT/server.env"

env \
  -u BW24_API_KEY \
  -u BW24_FULL_PREC \
  -u BW24_MOE_GATE \
  -u BW24_MOE_RESIDENT_GB \
  -u BW24_MOE_SLOTS \
  -u BW24_MOE_STATS \
  -u BW24_MOE_TRACE \
  BW24_COMPAT=openai \
  BW24_SERVE_SPEC=0 \
  BW24_KV_REUSE=0 \
  BW24_CTX=8192 \
  BW24_FAST=1 \
  BW24_MMVQ=1 \
  BW24_MOE_CACHE=1 \
  BW24_MOE_GROUPED=1 \
  BW24_MOE_PREWARM=1 \
  BW24_MOE_PREFETCH=1 \
  BW24_MOE_PAGE_PREFETCH=1 \
  BW24_MOE_PAGE_PREFETCH_WINDOW="$PAGE_PREFETCH_WINDOW" \
  BW24_MOE_MMAP_ADVICE=normal \
  BW24_MOE_RESIDENT=1 \
  BW24_MOE_VRAM_FRAC=0.85 \
  BW24_SPILL_IO=worker \
  BW24_SPILL_PREAD_DEPTH="$BW24_SPILL_PREAD_DEPTH" \
  BW24_SPILL_STATS=1 \
  BW24_MODELS="$ARM=$ARTIFACT" \
  BW24_ADDR="$ADDR" \
  "$SERVER_BIN" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$ATTEMPT/server.pid"

deadline=$((SECONDS + HEALTH_TIMEOUT_S))
while (( SECONDS < deadline )); do
  kill -0 "$SERVER_PID" 2>/dev/null || {
    tail -n 100 "$SERVER_LOG" >&2 || true
    die "server exited before becoming healthy"
  }
  if curl -fsS --max-time 5 "$SERVER_ROOT/health" > "$ATTEMPT/health.json.tmp" 2>/dev/null \
    && python3 "$HERE/validate_server_health.py" "$ATTEMPT/health.json.tmp" "$ARM" --exact
  then
    mv "$ATTEMPT/health.json.tmp" "$ATTEMPT/health.json"
    break
  fi
  sleep 1
done
[[ -f "$ATTEMPT/health.json" ]] || die "server health timeout"

ARM="$ARM" MODEL="$ARM" ARTIFACT="$ARTIFACT" RUN_ID="$RUN_ID" \
  SERVER_BIN="$SERVER_BIN" SERVER_LOG="$SERVER_LOG" \
  BW24_SPILL_IO=worker BW24_SPILL_PREAD_DEPTH="$BW24_SPILL_PREAD_DEPTH" \
  BW24_SPILL_STATS=1 BW24_SERVE_SPEC=0 BASE_URL="$BASE_URL" \
  OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" \
  HOURISH_SHARD_TIMEOUT_S="$HOURISH_SHARD_TIMEOUT_S" \
  "$HERE/run_hourish_arm.sh"

RUN_DIR="$OUT_ROOT/$ARM/$RUN_ID"
if [[ ! -e "$RUN_DIR/code-score.json" && ! -e "$RUN_DIR/code-score.receipt.json" ]]; then
  "$HERE/score_hourish_code_container.sh" "$RUN_DIR"
fi
[[ -f "$RUN_DIR/code-score.json" && -f "$RUN_DIR/code-score.receipt.json" ]] \
  || die "code score evidence is incomplete"

stop_server
date -u +%FT%TZ > "$CONTROL_DIR/complete"
echo "hourish arm complete: $ARM/$RUN_ID"

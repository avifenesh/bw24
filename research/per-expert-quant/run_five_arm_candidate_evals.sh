#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
PUBLIC_RUNNER=${PUBLIC_RUNNER:-$HERE/run_public_evals.sh}
SERVER_BIN=${SERVER_BIN:-$ROOT/target/release/bw24-server}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/scratch/artifacts}
OUT_ROOT=${OUT_ROOT:-$HERE/results}
CACHE_DIR=${CACHE_DIR:-$HERE/.cache}
RUN_ID=${RUN_ID:-candidate-l1-$(date -u +%Y%m%dT%H%M%SZ)-$$}
ADDR=${ADDR:-127.0.0.1:8080}
BASE_URL="http://$ADDR/v1/completions"
SERVER_ROOT=${BASE_URL%/v1/completions}
LIMIT=${LIMIT:-1}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-256}
EVAL_TIMEOUT_S=${EVAL_TIMEOUT_S:-14400}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-600}
PAGE_PREFETCH_WINDOW=${PAGE_PREFETCH_WINDOW:-8}

ARMS=(plain_quant plain_reap_quant plain_reap_mix_quant mix_quant mix_quant_prune25)
ARTIFACTS=(plain-quant plain-reap-quant plain-reap-mix-quant mix-quant mix-quant-prune25)

die() {
  echo "error: $*" >&2
  exit 2
}

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID may contain only letters, digits, dot, underscore, and dash"
positive_integer "$LIMIT" || die "LIMIT must be a positive integer"
positive_integer "$MAX_GEN_TOKS" || die "MAX_GEN_TOKS must be a positive integer"
positive_integer "$EVAL_TIMEOUT_S" || die "EVAL_TIMEOUT_S must be a positive integer"
positive_integer "$HEALTH_TIMEOUT_S" || die "HEALTH_TIMEOUT_S must be a positive integer"
positive_integer "$PAGE_PREFETCH_WINDOW" || die "PAGE_PREFETCH_WINDOW must be a positive integer"
[[ -x "$SERVER_BIN" ]] || die "missing executable server: $SERVER_BIN"
[[ -x "$PUBLIC_RUNNER" ]] || die "missing executable public-eval runner: $PUBLIC_RUNNER"
command -v curl >/dev/null || die "curl is required"

for artifact in "${ARTIFACTS[@]}"; do
  [[ -f "$ARTIFACT_ROOT/$artifact/manifest.json" ]] \
    || die "missing artifact manifest: $ARTIFACT_ROOT/$artifact/manifest.json"
done
for arm in "${ARMS[@]}"; do
  [[ ! -e "$OUT_ROOT/$arm/$RUN_ID" ]] \
    || die "refusing to reuse output: $OUT_ROOT/$arm/$RUN_ID"
done

CONTROL_PARENT="$OUT_ROOT/_runs"
CONTROL_DIR="$CONTROL_PARENT/$RUN_ID"
mkdir -p "$CONTROL_PARENT"
mkdir "$CONTROL_DIR" 2>/dev/null || die "refusing to reuse control output: $CONTROL_DIR"
printf 'arm\tstatus\n' > "$CONTROL_DIR/status.tsv"

SERVER_PID=""

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in {1..50}; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

cleanup() {
  local status=$?
  trap - EXIT
  stop_server
  exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_health() {
  local run_dir=$1
  local model=$2
  local deadline=$((SECONDS + HEALTH_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 80 "$run_dir/server.log" >&2 || true
      die "server exited before health check passed for $model"
    fi
    if curl -fsS --max-time 5 "$SERVER_ROOT/health" \
        > "$run_dir/server-health.json.tmp" 2>/dev/null \
      && python3 "$HERE/validate_server_health.py" \
        "$run_dir/server-health.json.tmp" "$model" --exact
    then
      mv "$run_dir/server-health.json.tmp" "$run_dir/server-health.json"
      return 0
    fi
    sleep 1
  done
  tail -n 80 "$run_dir/server.log" >&2 || true
  die "server health timeout after ${HEALTH_TIMEOUT_S}s for $model"
}

if curl -sS --connect-timeout 1 --max-time 2 "http://$ADDR/health" >/dev/null 2>&1; then
  die "an HTTP server is already answering on $ADDR"
fi

for index in "${!ARMS[@]}"; do
  arm=${ARMS[$index]}
  artifact="$ARTIFACT_ROOT/${ARTIFACTS[$index]}"
  run_dir="$OUT_ROOT/$arm/$RUN_ID"
  mkdir -p "$OUT_ROOT/$arm"
  mkdir "$run_dir"

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
    printf 'BW24_MODELS=%s=%s\n' "$arm" "$artifact"
    printf 'BW24_ADDR=%s\n' "$ADDR"
  } > "$run_dir/server.env"

  echo "[$arm] starting fresh server"
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
    BW24_MODELS="$arm=$artifact" \
    BW24_ADDR="$ADDR" \
    "$SERVER_BIN" > "$run_dir/server.log" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$run_dir/server.pid"
  wait_for_health "$run_dir" "$arm"

  eval_status=0
  env \
    ARM="$arm" \
    MODEL="$arm" \
    ARTIFACT="$artifact" \
    SUITE=candidate \
    LIMIT="$LIMIT" \
    MAX_GEN_TOKS="$MAX_GEN_TOKS" \
    EVAL_TIMEOUT_S="$EVAL_TIMEOUT_S" \
    BASE_URL="$BASE_URL" \
    OUT_ROOT="$OUT_ROOT" \
    CACHE_DIR="$CACHE_DIR" \
    RUN_ID="$RUN_ID" \
    "$PUBLIC_RUNNER" || eval_status=$?
  stop_server

  if (( eval_status != 0 )); then
    printf '%s\tfailed:%s\n' "$arm" "$eval_status" >> "$CONTROL_DIR/status.tsv"
    die "$arm evaluation failed with status $eval_status"
  fi
  printf '%s\tcomplete\n' "$arm" >> "$CONTROL_DIR/status.tsv"
  echo "[$arm] complete"
done

echo "$CONTROL_DIR"

#!/usr/bin/env bash
# Target-rig gate battery. The caller must hold /tmp/memra-gpu.lock.
set -uo pipefail

REPO=${REPO:-"$HOME/memra-prefixdedup"}
SMOKE_MODEL=${SMOKE_MODEL:-"$HOME/smoke-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"}
SMOKE_DRAFT=${SMOKE_DRAFT:-"$HOME/smoke-models/draft-9b-owntrim-nvfp4head-q4blk.gguf"}
STEP_MODEL=${STEP_MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
RAW=${RAW:-"$REPO/research/prefixdedup-20260808/raw/box1"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/gates-summary-$TS.log"
PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."

mkdir -p "$RAW"
cd "$REPO"

snapshot() {
  echo "snapshot $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used,memory.total \
    --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader || true
}

run_gate() {
  local label=$1
  shift
  local log="$RAW/$label-$TS.log"
  echo "########## $label ##########"
  echo "raw=$log"
  snapshot
  "$@" >"$log" 2>&1
  local rc=$?
  cat "$log"
  echo "$label exit=$rc"
  snapshot
  return "$rc"
}

main() {
  echo "=== prefix dedup gates $TS commit=$(git rev-parse HEAD)"
  echo "smoke_model=$SMOKE_MODEL bytes=$(stat -c %s "$SMOKE_MODEL")"
  echo "smoke_draft=$SMOKE_DRAFT bytes=$(stat -c %s "$SMOKE_DRAFT")"
  echo "step_model=$STEP_MODEL bytes=$(stat -c %s "$STEP_MODEL")"
  snapshot

  run_gate build-server timeout 7200 \
    cargo build --release -p memra-server --bin memra-server || return 1
  run_gate build-run-gen timeout 7200 \
    cargo build --release -p memra-engine --bin run-gen || return 1

  run_gate serve-smoke timeout 10800 \
    tools/serve-smoke.sh "$SMOKE_MODEL" "$SMOKE_DRAFT" || return 1
  grep -q "serve-smoke: 0 failed" "$RAW/serve-smoke-$TS.log" || return 1
  cp -- /tmp/serve-smoke.log "$RAW/serve-smoke-last-server-$TS.log"

  local api_out="$RAW/apikey-$TS"
  mkdir -p "$api_out"
  cp -- research/apikeys-20260805/apikey_gate.py "$api_out/apikey_gate.py"
  run_gate apikeys timeout 7200 \
    tools/apikeys-gate.sh "$SMOKE_MODEL" "$api_out" || return 1
  grep -q "apikey_gate: 0 failed" "$RAW/apikeys-$TS.log" || return 1

  run_gate run-gen env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_NGEN=64 \
    timeout 3600 target/release/run-gen "$STEP_MODEL" --prompt "$PROMPT" || return 1
  grep -qE "argmax=[0-9]+ +decode argmax=[0-9]+ .* MATCH$" \
    "$RAW/run-gen-$TS.log" || return 1

  run_gate fanout-ttft env MEMRA_GPU_LOCK_HELD=1 RAW="$RAW" \
    research/prefixdedup-20260808/run-box1.sh || return 1
  local fanout_summary
  fanout_summary=$(ls -1t "$RAW"/fanout-summary-*.log | head -1)
  cat "$fanout_summary"
  grep -q "prefix fanout TTFT rc=0" "$fanout_summary" || return 1

  snapshot
  echo "=== prefix dedup gates rc=0"
}

main >"$SUMMARY" 2>&1
rc=$?
echo "summary=$SUMMARY rc=$rc"
exit "$rc"

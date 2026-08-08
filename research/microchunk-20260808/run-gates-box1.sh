#!/usr/bin/env bash
# Dynamic microchunk target-rig exactness battery. Run from the pinned Box1 checkout.
set -uo pipefail

REPO=${REPO:-"$HOME/memra-cx-microchunk"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
DRAFT=${DRAFT:-"$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf"}
RAW=${RAW:-"/tmp/microchunk-gates"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/gates-summary-$TS.log"
PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
PP=("MEMRA_PP_STAGES=2" "MEMRA_PP_DEVICES=0,1")

mkdir -p "$RAW"
cd "$REPO" || exit 1

snapshot() {
  echo "snapshot $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used \
    --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader || true
}

copy_probe_log() {
  local label=$1
  local command_log=$2
  local probe_log
  probe_log=$(grep -oE '/tmp/[[:alnum:]_.-]+\.log' "$command_log" | tail -1)
  if [[ -n "$probe_log" && -f "$probe_log" ]]; then
    local retained="$RAW/$label-probe-$TS.log"
    cp -- "$probe_log" "$retained"
    echo "$label probe_raw=$retained source=$probe_log"
  fi
}

run_gate() {
  local label=$1
  shift
  local log="$RAW/$label-$TS.log"
  echo "########## $label ##########"
  echo "raw=$log"
  snapshot
  "$@" >"$log" 2>&1
  local gate_rc=$?
  cat "$log"
  copy_probe_log "$label" "$log"
  echo "$label exit=$gate_rc"
  snapshot
  return "$gate_rc"
}

main() {
  echo "=== dynamic microchunk gates $TS commit=$(git rev-parse HEAD)"
  echo "model=$MODEL"
  echo "draft=$DRAFT"
  (
    flock -w 21600 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    snapshot

    run_gate kernel-check timeout 3600 \
      ./target/release/kernel-check "$MODEL" || exit 1

    run_gate ppsplit env "${PP[@]}" timeout 7200 \
      tools/prime-split-gate.sh "$MODEL" --devices 0,1 --stages 2 \
      --chunks auto,513 --steps 8 || exit 1
    run_gate ppsplit-canary env "${PP[@]}" timeout 7200 \
      tools/prime-split-gate.sh "$MODEL" --devices 0,1 --stages 2 \
      --chunks auto,513 --steps 8 --canary || exit 1

    run_gate chunkinv35 env "${PP[@]}" timeout 7200 \
      tools/chunk-invariance-gate.sh "$MODEL" --label step35-swa \
      --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
      --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24 || exit 1
    run_gate chunkinv35-canary env "${PP[@]}" timeout 7200 \
      tools/chunk-invariance-gate.sh "$MODEL" --label step35-swa \
      --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
      --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24 \
      --canary || exit 1

    run_gate tickinv35 env "${PP[@]}" timeout 7200 \
      tools/tick-invariance-gate.sh "$MODEL" --label step35-tick \
      --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
      --budgets 0,1024,513,512,256,64 --splits 64,256,512 \
      --seam MEMRA_PRIME_CALLLOCAL --steps 24 || exit 1
    run_gate tickinv35-canary env "${PP[@]}" timeout 7200 \
      tools/tick-invariance-gate.sh "$MODEL" --label step35-tick \
      --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
      --budgets 0,1024,513,512,256,64 --splits 64,256,512 \
      --seam MEMRA_PRIME_CALLLOCAL --steps 24 --canary || exit 1

    run_gate run-gen env "${PP[@]}" MEMRA_NGEN=64 timeout 3600 \
      ./target/release/run-gen "$MODEL" --prompt "$PROMPT" || exit 1

    run_gate run-spec env "${PP[@]}" MEMRA_MTP_DRAFT="$DRAFT" MEMRA_NGEN=32 \
      MEMRA_PROMPT="$PROMPT" timeout 7200 \
      ./target/release/run-spec "$MODEL" || exit 1

    snapshot
    echo "lock released $(date -u +%FT%TZ)"
    exit 0
  ) 9>/tmp/memra-gpu.lock
  local battery_rc=$?
  echo "=== dynamic microchunk gates rc=$battery_rc"
  echo "=== done $(date -u +%FT%TZ)"
  return "$battery_rc"
}

main >"$SUMMARY" 2>&1
rc=$?
echo "summary=$SUMMARY rc=$rc"
exit "$rc"

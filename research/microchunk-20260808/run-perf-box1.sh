#!/usr/bin/env bash
# PP-2 dynamic-vs-fixed microchunk performance, N=5 interleaved under one GPU lock.
set -uo pipefail

REPO=${REPO:-"$HOME/memra-cx-microchunk"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
PROMPT4096=${PROMPT4096:-"$REPO/research/step-sku-20260807/prompt-pp4096.txt"}
RAW=${RAW:-"/tmp/microchunk-perf"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/perf-summary-$TS.log"
P512="$RAW/prompt-pp512-$TS.txt"
P2048="$RAW/prompt-pp2048-$TS.txt"
P4096="$RAW/prompt-pp4096-$TS.txt"
N=${N:-5}

mkdir -p "$RAW"
head -c 2800 "$PROMPT4096" >"$P512"
head -c 11200 "$PROMPT4096" >"$P2048"
cp -- "$PROMPT4096" "$P4096"
cd "$REPO" || exit 1

snapshot() {
  echo "snapshot $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used \
    --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader || true
}

require_idle() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader 2>/dev/null || true)
  if [[ -n "$apps" ]]; then
    echo "GPU NOT IDLE AFTER LOCK ACQUISITION"
    printf '%s\n' "$apps"
    return 76
  fi
}

main() {
  echo "=== dynamic microchunk perf $TS commit=$(git rev-parse HEAD)"
  echo "model=$MODEL"
  echo "protocol=N=$N in-process interleaved fixed/dynamic, one warmup, one GPU-lock hold"
  local prompt
  for prompt in "$P512" "$P2048" "$P4096"; do
    echo "prompt=$prompt bytes=$(wc -c <"$prompt") sha256=$(sha256sum "$prompt" | awk '{print $1}')"
  done
  local rc=0
  (
    flock -w 21600 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    snapshot
    require_idle || exit $?
    local cell shape log cell_rc
    for cell in \
      "pp512:$P512" \
      "pp2048:$P2048" \
      "pp4096:$P4096"
    do
      shape=${cell%%:*}
      prompt=${cell#*:}
      log="$RAW/$shape-raw-$TS.log"
      echo "########## $shape ##########"
      echo "raw=$log"
      snapshot
      env -u MEMRA_PRIME_CHUNK -u MEMRA_PRIME_CHUNK_SCHED \
        -u MEMRA_PRIME_PP -u MEMRA_PRIME_PIPE \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 3600 \
        ./target/release/concat-prime-probe "$MODEL" ppschedperf \
        --prompt-a "@$prompt" --reps "$N" --warmup 1 >"$log" 2>&1
      cell_rc=$?
      cat "$log"
      echo "$shape exit=$cell_rc"
      if (( cell_rc != 0 )); then
        rc=1
      fi
      snapshot
    done
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  local perf_rc=$?
  echo "=== medians ==="
  grep -h "ppschedperf MEDIAN:" "$RAW"/*-raw-"$TS".log || true
  echo "=== dynamic microchunk perf rc=$perf_rc"
  echo "=== done $(date -u +%FT%TZ)"
  return "$perf_rc"
}

main >"$SUMMARY" 2>&1
rc=$?
echo "summary=$SUMMARY rc=$rc"
exit "$rc"

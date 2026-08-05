#!/usr/bin/env bash
# memra vs llama daily-27B dogfood diagnostic — 2026-08-05, local 5090.
# NOT BOARD MATERIAL (llama board benching is doctrine-stopped; this is the owner-asked
# "I think I got better with llama" experience check).
#
# Interleaved by rep: rep N runs a memra server phase then a llama server phase.
# Each phase holds /tmp/gpu5090.lock (F5 lane priority: lock released between phases).
# Both servers run the OWNER'S EXACT daily configs (serve-qwen36-27b[-memra], half=128k)
# at gpu-full-power, serving the same artifact (inode-verified).
set -uo pipefail

RDIR=/home/avifenesh/projects/bw24/research/memra-vs-llama-daily-20260805
SCRIPTS=$RDIR/scripts
LOGS=$RDIR/logs
PDIR=$RDIR/prompts
OUT=$RDIR/runs.jsonl
LOCK=/tmp/gpu5090.lock

DIR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp
MODEL=$DIR/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
MEMRA_DRAFT=$DIR/draft-daily-owntrim-nvfp4head-q4blk.gguf
LLAMA_DRAFT=$DIR/mtp-Qwen3.6-27B-Q4_K_M.gguf
MEMRA_BIN=/home/avifenesh/tmp-dogfood/memra-server-c716954b   # md5-matched snapshot of train c716954b
LLAMA_BIN=/home/avifenesh/projects/llama.cpp/build/bin/llama-server
CTX=131072

gpustate() {
  echo "[gpu $(date -u +%FT%TZ)] $(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[apps] /'
}

wait_health() { # url auth_header timeout_s
  local url=$1 hdr=$2 t=$3 i=0
  while [ $i -lt "$t" ]; do
    if curl -sf -m 2 ${hdr:+-H "$hdr"} "$url" >/dev/null 2>&1; then return 0; fi
    sleep 2; i=$((i+2))
  done
  return 1
}

wait_vram_drain() { # wait until GPU used < 3000 MiB (embed stub ~330 MiB stays)
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 3000 ] && return 0
    sleep 2
  done
  echo "WARN: VRAM did not drain (used=${used} MiB)"; return 1
}

phase() { # server rep
  local server=$1 rep=$2
  local slog=$LOGS/server-$server-r$rep.log
  local dlog=$LOGS/driver-$server-r$rep.log
  local pid=""

  exec 9>"$LOCK"
  flock 9   # wait for F5 lane if it holds the card
  echo "=== phase $server rep $rep $(date -u +%FT%TZ) ===" | tee -a "$dlog"
  gpustate >> "$dlog"

  if [ "$server" = memra ]; then
    MEMRA_MODELS="qwen36-27b=$MODEL+$MEMRA_DRAFT" \
    MEMRA_ADDR="127.0.0.1:8002" \
    MEMRA_API_KEY=aviary-local \
    MEMRA_CTX=$CTX \
    MEMRA_MAX_SESSIONS=1 \
    MEMRA_REUSE_POOL=1 \
    MEMRA_PRIME_CHUNK=2048 \
    "$MEMRA_BIN" > "$slog" 2>&1 &
    pid=$!
    if ! wait_health "http://127.0.0.1:8002/health" "Authorization: Bearer aviary-local" 180; then
      echo "FATAL: memra health timeout r$rep" | tee -a "$dlog"; kill $pid 2>/dev/null; flock -u 9; return 1
    fi
    port=8002
  else
    "$LLAMA_BIN" \
      -m "$MODEL" \
      --model-draft "$LLAMA_DRAFT" \
      --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1 \
      --alias qwen36-27b \
      --ctx-size $CTX --ubatch-size 512 -ngl 999 -ngld 999 -fa on --parallel 1 \
      --cache-type-k q8_0 --cache-type-v q5_1 \
      --cache-ram 0 \
      --jinja \
      --host 127.0.0.1 --port 8001 --api-key aviary-local --metrics \
      > "$slog" 2>&1 &
    pid=$!
    if ! wait_health "http://127.0.0.1:8001/health" "" 180; then
      echo "FATAL: llama health timeout r$rep" | tee -a "$dlog"; kill $pid 2>/dev/null; flock -u 9; return 1
    fi
    port=8001
  fi

  sleep 3
  python3 "$SCRIPTS/driver.py" "$server" "$port" "$rep" "$OUT" "$PDIR" >> "$dlog" 2>&1
  rc=$?
  gpustate >> "$dlog"
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null
  wait_vram_drain >> "$dlog" 2>&1
  flock -u 9
  exec 9>&-
  echo "=== phase $server rep $rep done rc=$rc ===" | tee -a "$dlog"
  return $rc
}

# ---- battery ----
gpu-full-power on || echo "warn: gpu-full-power unavailable"
trap 'gpu-full-power off' EXIT

echo "battery start $(date -u +%FT%TZ)" > "$LOGS/battery.log"
gpustate >> "$LOGS/battery.log"

for rep in 1 2 3 4 5; do
  phase memra "$rep" || echo "phase memra r$rep FAILED" >> "$LOGS/battery.log"
  phase llama "$rep" || echo "phase llama r$rep FAILED" >> "$LOGS/battery.log"
done

gpustate >> "$LOGS/battery.log"
echo "battery done $(date -u +%FT%TZ)" >> "$LOGS/battery.log"

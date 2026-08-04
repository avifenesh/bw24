#!/usr/bin/env bash
# pro6000wk-validation: idle + single-stream serving power (colo electricity math input)
set -uo pipefail
cd /root/bw24
export PATH=/usr/local/cuda-13.1/bin:$HOME/.cargo/bin:$PATH
R=/root/receipts
mkdir -p "$R/serve"
M9=/root/models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
ADDR=127.0.0.1:8177
BASE=http://$ADDR

nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used \
  --format=csv -l 1 > "$R/serve/serve-1hz.csv" 2>&1 &
SMIPID=$!

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/serve/driver.log"; }

# Phase 0: bare-GPU idle (no process)
log "PHASE bare-idle 60s begin"
sleep 60
log "PHASE bare-idle end"

# Phase 1: memra-server up, model resident, no traffic
MEMRA_COMPAT=openai MEMRA_MODELS="q9=$M9" MEMRA_ADDR=$ADDR target/release/memra-server > "$R/serve/server.log" 2>&1 &
SPID=$!
for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && break; sleep 2; done
curl -sf $BASE/health >/dev/null || { log "SERVER FAILED"; kill $SMIPID $SPID 2>/dev/null; exit 1; }
log "PHASE loaded-idle 120s begin (model resident, zero traffic)"
sleep 120
log "PHASE loaded-idle end"

# Phase 2: single-stream sustained load, 5 sequential requests x 256 tok
log "PHASE single-stream begin"
for i in 1 2 3 4 5; do
  t0=$(date +%s.%N)
  out=$(curl -sf -m 600 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"q9","messages":[{"role":"user","content":"Write a detailed explanation of how PCIe link training works, covering every LTSSM state."}],"max_tokens":256,"temperature":0}')
  t1=$(date +%s.%N)
  ct=$(echo "$out" | jq -r '.usage.completion_tokens' 2>/dev/null || echo '?')
  log "req $i: ${ct} tok in $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')s"
done
log "PHASE single-stream end"

# Phase 3: post-load idle again (residency confirmed)
log "PHASE post-load-idle 60s begin"
sleep 60
log "PHASE post-load-idle end"

kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
sleep 5
kill $SMIPID 2>/dev/null
log "SERVE POWER DONE"

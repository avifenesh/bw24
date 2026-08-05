#!/bin/bash
# Full TTFT matrix: B32/B128 x fix-on/fix-off, N=5, q27 nv, K=3, stream:true greedy.
# One flock hold for ALL four cells (short: 4 boots x ~15s probe) — same thermal regime.
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)
NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
BIN=$TREE/target/release/memra-server
PROBE=$TREE/research/spec-levers-5090-20260805/ttft-probe.py
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/logs/full-driver.log"; }

exec 9>/tmp/gpu5090.lock
flock 9
log "full TTFT matrix hold acquired ($(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader))"

for CELL in "32 fixon" "32 fixoff" "128 fixon" "128 fixoff"; do
  set -- $CELL; B=$1; ARM=$2
  EXTRA=()
  [ "$ARM" = fixoff ] && EXTRA=(MEMRA_SSE_PER_BURST=1)
  env "${EXTRA[@]}" MEMRA_SPEC_K=3 MEMRA_SPEC_BURST=$B \
    MEMRA_MODELS="q=$NV+$DR" MEMRA_ADDR=$ADDR \
    "$BIN" > "$R/logs/full-ttft-B$B-$ARM.server.log" 2>&1 &
  SPID=$!
  up=0
  for _ in $(seq 150); do
    curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  [ "$up" -eq 1 ] || { log "NO-UP B$B-$ARM"; kill "$SPID" 2>/dev/null; continue; }
  OUT=$(python3 "$PROBE" --base $BASE --model q --label "nv-K3-B$B-$ARM" \
        --out "$R/logs/points-ttft-full.jsonl" --n 5 --max-tokens 256)
  log "ttft B$B $ARM: $OUT"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
done
log "FULL_TTFT_DONE ($(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader))"
echo FULL_TTFT_DONE

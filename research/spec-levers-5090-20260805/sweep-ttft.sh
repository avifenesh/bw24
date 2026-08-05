#!/bin/bash
# Streaming-cadence guard for the burst flip: B32 vs B128 at K=3 (nv), stream:true,
# 256 tok. The worker emits ONE SSE event per burst, so B128 quarters the chunk count
# and stretches first-chunk latency — the felt path the owner's dogfood scores.
# One flock hold, both arms inside.
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)
NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
BIN=$TREE/target/release/memra-server
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/driver.log"; }

exec 9>/tmp/gpu5090.lock
flock 9
log "TTFT probe hold acquired"

for B in 32 128; do
  MEMRA_SPEC_K=3 MEMRA_SPEC_BURST=$B MEMRA_MODELS="q=$NV+$DR" MEMRA_ADDR=$ADDR \
    "$BIN" > "$R/logs/ttft-B$B.server.log" 2>&1 &
  SPID=$!
  up=0
  for _ in $(seq 150); do
    curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  [ "$up" -eq 1 ] || { log "NO-UP ttft-B$B"; kill "$SPID" 2>/dev/null; continue; }
  OUT=$(python3 ttft-probe.py --base $BASE --model q --label "nv-K3-B$B-stream" \
        --out "$R/logs/points-ttft.jsonl" --n 3 --max-tokens 256)
  log "ttft B$B: $OUT"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
done
log "TTFT_DONE"
echo TTFT_DONE

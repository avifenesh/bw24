#!/bin/bash
# SSE cadence fix — first receipt: B128 fix-on vs fix-off (MEMRA_SSE_PER_BURST=1 rollback
# seam, same binary), q27 nv artifact, N=3, replicating spec-levers ttft-probe.py
# (time-to-first-SSE-content-chunk + chunk count). ONE flock hold, both arms inside.
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
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/logs/first-driver.log"; }

exec 9>/tmp/gpu5090.lock
flock 9
log "first-result hold acquired (B128 fix-on vs fix-off, N=3)"

for ARM in fixon fixoff; do
  EXTRA=()
  [ "$ARM" = fixoff ] && EXTRA=(MEMRA_SSE_PER_BURST=1)
  env "${EXTRA[@]}" MEMRA_SPEC_K=3 MEMRA_SPEC_BURST=128 \
    MEMRA_MODELS="q=$NV+$DR" MEMRA_ADDR=$ADDR \
    "$BIN" > "$R/logs/first-B128-$ARM.server.log" 2>&1 &
  SPID=$!
  up=0
  for _ in $(seq 150); do
    curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  [ "$up" -eq 1 ] || { log "NO-UP $ARM"; kill "$SPID" 2>/dev/null; continue; }
  OUT=$(python3 "$PROBE" --base $BASE --model q --label "nv-K3-B128-$ARM" \
        --out "$R/logs/points-first.jsonl" --n 3 --max-tokens 256)
  log "B128 $ARM: $OUT"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
done
log "FIRST_RESULT_DONE"
echo FIRST_RESULT_DONE

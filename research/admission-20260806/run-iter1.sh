#!/bin/bash
# ADMISSION-YIELD FIRST RECEIPT: contended first-text at B128, fix-on vs fix-off, N=3.
# Cell: server up (q27 nv + draft, MEMRA_SPEC_BURST=128), one 512-tok request streaming
# in the background, second request arrives -> measure its first-text. The fix ends the
# in-flight burst at the round boundary when PENDING_ADMITS > 0; MEMRA_ADMIT_YIELD=0 is
# the rollback seam (full-burst holds = the sse-cadence VERDICT's 1.67s class).
# One flock hold for both arms. Same probe/prompts as sse-cadence's run-contention.sh.
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)
NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8203
BASE=http://$ADDR
BIN=$TREE/target/release/memra-server
PROBE=$TREE/research/spec-levers-5090-20260805/ttft-probe.py
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/logs/iter1-driver.log"; }

exec 9>/tmp/gpu5090.lock
flock 9
log "first-result hold acquired"

for ARM in on off; do
  YIELD=1; [ "$ARM" = off ] && YIELD=0
  MEMRA_ADMIT_YIELD=$YIELD MEMRA_SPEC_K=3 MEMRA_SPEC_BURST=128 \
    MEMRA_MODELS="q=$NV+$DR" MEMRA_ADDR=$ADDR \
    "$BIN" > "$R/logs/iter1-B128-$ARM.server.log" 2>&1 &
  SPID=$!
  up=0
  for _ in $(seq 150); do
    curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  [ "$up" -eq 1 ] || { log "NO-UP $ARM"; kill "$SPID" 2>/dev/null; continue; }
  # background load: one long request kept alive for the whole probe window
  curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d \
    '{"model":"q","messages":[{"role":"user","content":"Write a very detailed essay on the history of GPU computing, at least 800 words."}],"max_tokens":512,"temperature":0.0,"stream":false}' \
    > "$R/logs/iter1-B128-$ARM.bg.json" &
  BGPID=$!
  sleep 2  # let the background request get past prefill into its burst loop
  OUT=$(python3 "$PROBE" --base $BASE --model q --label "nv-K3-B128-contended-coldfirst-yield$YIELD" \
        --out "$R/logs/points-iter1.jsonl" --n 3 --max-tokens 128)
  log "contended B128 yield=$YIELD: $OUT"
  wait "$BGPID" 2>/dev/null
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
done
log "ITER1_DONE"
echo ITER1_DONE

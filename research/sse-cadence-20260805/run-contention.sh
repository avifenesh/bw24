#!/bin/bash
# Default-flip decider: felt TTFT for a NEW streaming request that joins while another
# session is mid-generation (the round-robin admission cost a bigger burst adds — the
# one interactive risk the c=1 TTFT matrix cannot see). B32 vs B128, fix-on binary,
# N=3 probes each with a 512-tok background request in flight. One flock hold.
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
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/logs/contention-driver.log"; }

exec 9>/tmp/gpu5090.lock
flock 9
log "contention hold acquired"

for B in 32 128; do
  MEMRA_SPEC_K=3 MEMRA_SPEC_BURST=$B MEMRA_MODELS="q=$NV+$DR" MEMRA_ADDR=$ADDR \
    "$BIN" > "$R/logs/contention-B$B.server.log" 2>&1 &
  SPID=$!
  up=0
  for _ in $(seq 150); do
    curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  [ "$up" -eq 1 ] || { log "NO-UP B$B"; kill "$SPID" 2>/dev/null; continue; }
  # background load: one long request kept alive for the whole probe window
  curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d \
    '{"model":"q","messages":[{"role":"user","content":"Write a very detailed essay on the history of GPU computing, at least 800 words."}],"max_tokens":512,"temperature":0.0,"stream":false}' \
    > "$R/logs/contention-B$B.bg.json" &
  BGPID=$!
  sleep 2  # let the background request get past prefill into its burst loop
  OUT=$(python3 "$PROBE" --base $BASE --model q --label "nv-K3-B$B-contended" \
        --out "$R/logs/points-contention.jsonl" --n 3 --max-tokens 128)
  log "contended B$B: $OUT"
  wait "$BGPID" 2>/dev/null
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
done
log "CONTENTION_DONE"
echo CONTENTION_DONE

#!/bin/bash
# Throughput no-regression at B128: fix-on vs fix-off, alternating boots (2 rounds each
# arm), c=1 and c=8, plus a B32 fix-on reference so the +7% B128-over-B32 claim is
# re-anchored in THIS thermal window. load-serve.py = the spec-levers cell shape
# (stream:false temp 0.7 128tok — the worker emission path is shared, so per-round
# channel-send overhead shows here if it exists). One flock hold per boot (short holds).
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)
NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
BIN=$TREE/target/release/memra-server
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/logs/thru-driver.log"; }

cell() { # cell <B> <arm fixon|fixoff> <conc> <reqs> <tag>
  local B=$1 ARM=$2 CONC=$3 REQS=$4 TAG=$5
  local EXTRA=()
  [ "$ARM" = fixoff ] && EXTRA=(MEMRA_SSE_PER_BURST=1)
  exec 9>/tmp/gpu5090.lock
  flock 9
  env "${EXTRA[@]}" MEMRA_SPEC_K=3 MEMRA_SPEC_BURST=$B \
    MEMRA_MODELS="q=$NV+$DR" MEMRA_ADDR=$ADDR \
    "$BIN" > "$R/logs/thru-$TAG.server.log" 2>&1 &
  local SPID=$!
  local up=0
  for _ in $(seq 150); do
    curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" -ne 1 ]; then log "NO-UP $TAG"; kill "$SPID" 2>/dev/null; flock -u 9; return 1; fi
  for p in 1 2; do
    python3 "$TREE/tools/load-serve.py" --base $BASE --model q \
      --concurrency "$CONC" --requests "$REQS" --max-tokens 128 \
      --out "$R/logs/points-thru.jsonl" --label "$TAG-p$p" >> "$R/logs/thru-load.log" 2>&1
    ROW=$(tail -1 "$R/logs/points-thru.jsonl" | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print("agg=%.1f p50=%.3f err=%d" % (d["agg_tok_s"],d["lat_p50_s"],d["n_err"]))' 2>/dev/null || echo parse-fail)
    GPU=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)
    log "$TAG p$p: $ROW [$GPU]"
  done
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
  flock -u 9
  exec 9>&-
}

# c=1: alternate the arms, 2 boots each; B32 fixon reference first and last bracket.
cell 32  fixon  1 4 c1-B32-fixon-r1
cell 128 fixon  1 4 c1-B128-fixon-r1
cell 128 fixoff 1 4 c1-B128-fixoff-r1
cell 128 fixon  1 4 c1-B128-fixon-r2
cell 128 fixoff 1 4 c1-B128-fixoff-r2

# c=8: same alternation.
cell 128 fixon  8 16 c8-B128-fixon-r1
cell 128 fixoff 8 16 c8-B128-fixoff-r1
cell 128 fixon  8 16 c8-B128-fixon-r2
cell 128 fixoff 8 16 c8-B128-fixoff-r2
cell 32  fixon  8 16 c8-B32-fixon-r1

log "THRU_DONE"
echo THRU_DONE

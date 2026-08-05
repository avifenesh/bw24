#!/bin/bash
# One serve measurement cell = ONE flock hold on the shared 5090 (/tmp/gpu5090.lock):
# boot memra-server with the cell's spec-lever env, run 2 load passes (c=1, 4 req,
# 128 tok, temp 0.7 = the pod's exact load shape), capture /metrics step_p50 + gpu
# thermal state per pass, tear down, release the lock.
#
# Usage: run-cell.sh <art nv|q9> <K> <BURST> <PMIN 0|0.3> <tag> [concurrency] [requests]
# Writes: logs/<tag>.server.log, appends logs/points.jsonl (load rows) + driver.log.
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)

ART="${1:?art}" ; K="${2:?K}" ; B="${3:?burst}" ; PM="${4:?pmin}" ; TAG="${5:?tag}"
CONC="${6:-1}" ; REQS="${7:-4}"

NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
NVDRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q9DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
if [ "$ART" = nv ]; then M=$NV; DR=$NVDRAFT; else M=$Q9; DR=$Q9DRAFT; fi

ADDR=127.0.0.1:8199
BASE=http://$ADDR
BIN=$TREE/target/release/memra-server
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader; }

# ---- take the shared-GPU lock for exactly this cell
exec 9>/tmp/gpu5090.lock
flock 9

if [ "$PM" != 0 ]; then
  MEMRA_SPEC_PMIN=$PM MEMRA_SPEC_PMIN0=1 MEMRA_SPEC_K=$K MEMRA_SPEC_BURST=$B \
    MEMRA_MODELS="q=$M+$DR" MEMRA_ADDR=$ADDR "$BIN" > "$R/logs/$TAG.server.log" 2>&1 &
else
  MEMRA_SPEC_K=$K MEMRA_SPEC_BURST=$B \
    MEMRA_MODELS="q=$M+$DR" MEMRA_ADDR=$ADDR "$BIN" > "$R/logs/$TAG.server.log" 2>&1 &
fi
SPID=$!
cleanup() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }
trap cleanup EXIT

up=0
for _ in $(seq 150); do
  curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
  kill -0 "$SPID" 2>/dev/null || break
  sleep 2
done
if [ "$up" -ne 1 ]; then
  log "NO-UP $TAG; server log tail:"; tail -5 "$R/logs/$TAG.server.log" >> "$R/logs/driver.log"
  exit 1
fi

for p in 1 2; do
  python3 "$TREE/tools/load-serve.py" --base $BASE --model q \
    --concurrency "$CONC" --requests "$REQS" --max-tokens 128 \
    --out "$R/logs/points.jsonl" --label "$TAG-p$p" >> "$R/logs/load.log" 2>&1
  MET=$(curl -sf $BASE/metrics 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print("step_p50=%.2fms step_p99=%.2fms tok_out=%d" % (d["step_p50_ms"],d["step_p99_ms"],d["tokens_out"]))' 2>/dev/null || echo metrics-fail)
  ROW=$(tail -1 "$R/logs/points.jsonl" | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print("agg=%.1f p50=%.3f err=%d" % (d["agg_tok_s"],d["lat_p50_s"],d["n_err"]))' 2>/dev/null || echo parse-fail)
  log "$TAG p$p: $ROW | $MET | $(gpustate)"
done
exit 0

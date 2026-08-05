#!/bin/bash
# One flock hold = one full INTERLEAVED round over a list of arms (A/B inside the hold,
# per the lane's shared-GPU discipline: the arms of a comparison sit in the same thermal
# window; across-round repeats give N). Server restarts per arm (env is boot-time), the
# model stays page-cached so reload is fast.
#
# Usage: run-round.sh <art nv|q9> <round-tag> <conc> <reqs> <arm>...
#        arm = name:K:BURST:PMIN   (PMIN 0 = off)
# Appends logs/points.jsonl (labels <art>-<name>-<round-tag>-p<pass>), logs/driver.log.
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)

ART="${1:?art}"; RT="${2:?round-tag}"; CONC="${3:?conc}"; REQS="${4:?reqs}"; shift 4
MAXTOK="${MAXTOK:-128}"   # env override for long-gen cells

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

exec 9>/tmp/gpu5090.lock
flock 9
log "ROUND $ART-$RT hold acquired: $* | $(gpustate)"

for a in "$@"; do
  IFS=: read -r name K B PM <<< "$a"
  TAG="$ART-$name-$RT"
  if [ "$PM" != 0 ]; then
    MEMRA_SPEC_PMIN=$PM MEMRA_SPEC_PMIN0=1 MEMRA_SPEC_K=$K MEMRA_SPEC_BURST=$B \
      MEMRA_MODELS="q=$M+$DR" MEMRA_ADDR=$ADDR "$BIN" > "$R/logs/$TAG.server.log" 2>&1 &
  else
    MEMRA_SPEC_K=$K MEMRA_SPEC_BURST=$B \
      MEMRA_MODELS="q=$M+$DR" MEMRA_ADDR=$ADDR "$BIN" > "$R/logs/$TAG.server.log" 2>&1 &
  fi
  SPID=$!
  up=0
  for _ in $(seq 150); do
    curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" -ne 1 ]; then
    log "NO-UP $TAG; tail:"; tail -5 "$R/logs/$TAG.server.log" >> "$R/logs/driver.log"
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
    continue
  fi
  for p in 1 2; do
    python3 "$TREE/tools/load-serve.py" --base $BASE --model q \
      --concurrency "$CONC" --requests "$REQS" --max-tokens "$MAXTOK" \
      --out "$R/logs/points.jsonl" --label "$TAG-p$p" >> "$R/logs/load.log" 2>&1
    ROW=$(tail -1 "$R/logs/points.jsonl" | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print("agg=%.1f p50=%.3f err=%d shed=%d" % (d["agg_tok_s"],d["lat_p50_s"],d["n_err"],d["n_shed"]))' 2>/dev/null || echo parse-fail)
    ACC=$(grep -o 'cum=[0-9]*/[0-9]*=0\.[0-9]*' "$R/logs/$TAG.server.log" | tail -1 || true)
    log "$TAG p$p: $ROW | acc ${ACC:-none} | $(gpustate)"
  done
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
done
log "ROUND $ART-$RT done, releasing"
exit 0

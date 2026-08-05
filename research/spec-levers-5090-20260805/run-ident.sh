#!/bin/bash
# Greedy stream-identity gate: same fixed prompt, greedy 128 tok, byte-compare the
# completion text between a lever config and the default config. One flock hold per
# server boot. Usage: run-ident.sh <art nv|q9> <K> <BURST> <PMIN 0|0.3> <tag>
# Writes logs/ident-<tag>.txt; the caller cmp -s's two of these.
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)

ART="${1:?art}" ; K="${2:?K}" ; B="${3:?burst}" ; PM="${4:?pmin}" ; TAG="${5:?tag}"

NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
NVDRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q9DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
if [ "$ART" = nv ]; then M=$NV; DR=$NVDRAFT; else M=$Q9; DR=$Q9DRAFT; fi

ADDR=127.0.0.1:8199
BASE=http://$ADDR
BIN=$TREE/target/release/memra-server
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/driver.log"; }

exec 9>/tmp/gpu5090.lock
flock 9

if [ "$PM" != 0 ]; then
  MEMRA_SPEC_PMIN=$PM MEMRA_SPEC_PMIN0=1 MEMRA_SPEC_K=$K MEMRA_SPEC_BURST=$B \
    MEMRA_MODELS="q=$M+$DR" MEMRA_ADDR=$ADDR "$BIN" > "$R/logs/ident-$TAG.server.log" 2>&1 &
else
  MEMRA_SPEC_K=$K MEMRA_SPEC_BURST=$B \
    MEMRA_MODELS="q=$M+$DR" MEMRA_ADDR=$ADDR "$BIN" > "$R/logs/ident-$TAG.server.log" 2>&1 &
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
[ "$up" -eq 1 ] || { log "NO-UP ident-$TAG"; tail -5 "$R/logs/ident-$TAG.server.log" >> "$R/logs/driver.log"; exit 1; }

curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d \
  '{"model":"q","messages":[{"role":"user","content":"Explain how a CUDA graph reduces kernel launch overhead, in about 200 words."}],"max_tokens":128,"temperature":0.0,"stream":false}' \
  | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());c=d["choices"][0]["message"];print((c.get("reasoning") or "") + (c.get("content") or ""))' \
  > "$R/logs/ident-$TAG.txt" 2>&1
log "ident-$TAG captured ($(wc -c < "$R/logs/ident-$TAG.txt") bytes)"
exit 0

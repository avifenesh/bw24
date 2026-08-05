#!/bin/bash
# Greedy content byte-identity: fix-on vs fix-off (MEMRA_SSE_PER_BURST=1), B128 + B32.
# STREAMED capture — concatenate every delta (reasoning + content) so the comparison
# covers exactly the surface the fix touches (chunk boundaries move; bytes must not).
# One flock hold per boot.
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)
NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
BIN=$TREE/target/release/memra-server
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/logs/ident-driver.log"; }

capture() { # capture <B> <arm> <tag>
  local B=$1 ARM=$2 TAG=$3
  local EXTRA=()
  [ "$ARM" = fixoff ] && EXTRA=(MEMRA_SSE_PER_BURST=1)
  exec 9>/tmp/gpu5090.lock
  flock 9
  env "${EXTRA[@]}" MEMRA_SPEC_K=3 MEMRA_SPEC_BURST=$B \
    MEMRA_MODELS="q=$NV+$DR" MEMRA_ADDR=$ADDR \
    "$BIN" > "$R/logs/ident-$TAG.server.log" 2>&1 &
  local SPID=$!
  local up=0
  for _ in $(seq 150); do
    curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" -ne 1 ]; then log "NO-UP $TAG"; kill "$SPID" 2>/dev/null; flock -u 9; return 1; fi
  curl -sN $BASE/v1/chat/completions -H "Content-Type: application/json" -d \
    '{"model":"q","messages":[{"role":"user","content":"Explain how a CUDA graph reduces kernel launch overhead, in about 200 words."}],"max_tokens":128,"temperature":0.0,"stream":true}' \
    | python3 -c '
import sys, json
buf = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"): continue
    p = line[5:].strip()
    if p == "[DONE]": break
    try: d = json.loads(p)
    except json.JSONDecodeError: continue
    delta = d.get("choices", [{}])[0].get("delta", {})
    buf.append(delta.get("reasoning") or "")
    buf.append(delta.get("content") or "")
sys.stdout.write("".join(buf))' > "$R/logs/ident-$TAG.txt"
  log "ident-$TAG captured ($(wc -c < "$R/logs/ident-$TAG.txt") bytes)"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
  flock -u 9
  exec 9>&-
}

capture 128 fixon  B128-fixon
capture 128 fixoff B128-fixoff
capture 32  fixon  B32-fixon
capture 32  fixoff B32-fixoff

for pair in "B128-fixon B128-fixoff" "B32-fixon B32-fixoff" "B32-fixon B128-fixon"; do
  set -- $pair
  if cmp -s "$R/logs/ident-$1.txt" "$R/logs/ident-$2.txt"; then
    log "identity $1 vs $2: BYTE-IDENTICAL"
  else
    log "identity $1 vs $2: MISMATCH"
  fi
done
log "IDENT_DONE"
echo IDENT_DONE

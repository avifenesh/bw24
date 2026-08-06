#!/bin/bash
# ADMISSION-YIELD FULL EVAL (lane/admission-latency, 2026-08-06). Phases:
#  1. contended first-text: B32+B128 x fix-on/off, N=5 (one hold, 4 boots)
#  2. solo TTFT: B32-on, B128-on, B128-off, N=5 (no admit waiting => on==off expected)
#  3. throughput: c=1 + c=8, both bursts, on/off, alternating boots x2/arm (interleaved)
#  4. greedy content byte-identity: solo B32/B128 on/off + CONTENDED B128 on/off
#     (the risk cell: yields + cold-first change burst boundaries only under contention)
#  5. gates: run-spec K=1..8 one arm (nv+draft B128), serve-smoke, decode-batch-gate
#     config B=8 + strict B=4, cargo test -p memra-server
# fix-off = MEMRA_ADMIT_YIELD=0 (rollback seam: full-burst holds + index-order spec phase).
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)
NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
DBG9B=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
P512=$TREE/research/e2e/prompts/pp512.txt
ADDR=127.0.0.1:8203
BASE=http://$ADDR
BIN=$TREE/target/release/memra-server
PROBE=$TREE/research/spec-levers-5090-20260805/ttft-probe.py
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/logs/full-eval-driver.log"; }

boot() { # boot <B> <yield 0|1> <tag> -> sets SPID; returns 1 on no-up
  local B=$1 Y=$2 TAG=$3
  MEMRA_ADMIT_YIELD=$Y MEMRA_SPEC_K=3 MEMRA_SPEC_BURST=$B \
    MEMRA_MODELS="q=$NV+$DR" MEMRA_ADDR=$ADDR \
    "$BIN" > "$R/logs/$TAG.server.log" 2>&1 &
  SPID=$!
  local up=0
  for _ in $(seq 150); do
    curl -sf $BASE/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 2
  done
  [ "$up" -eq 1 ] || { log "NO-UP $TAG"; kill "$SPID" 2>/dev/null; return 1; }
  return 0
}
down() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }

exec 9>/tmp/gpu5090.lock

# ---- Phase 1: contended first-text, N=5, one hold
flock 9
log "P1 contended hold acquired ($(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader))"
for CELL in "32 1" "32 0" "128 1" "128 0"; do
  set -- $CELL; B=$1; Y=$2
  boot "$B" "$Y" "cont-B$B-y$Y" || continue
  curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d \
    '{"model":"q","messages":[{"role":"user","content":"Write a very detailed essay on the history of GPU computing, at least 1500 words. Do not stop early."}],"max_tokens":2048,"temperature":0.0,"stream":false}' \
    > "$R/logs/cont-B$B-y$Y.bg.json" &
  BGPID=$!
  sleep 2
  OUT=$(python3 "$PROBE" --base $BASE --model q --label "contended-B$B-y$Y" \
        --out "$R/logs/points-contended.jsonl" --n 5 --max-tokens 128)
  log "P1 contended B$B yield=$Y: $OUT"
  kill "$BGPID" 2>/dev/null; wait "$BGPID" 2>/dev/null
  down
done
flock -u 9
log "P1_DONE"

# ---- Phase 2: solo TTFT, N=5, one hold
flock 9
log "P2 solo hold acquired"
for CELL in "32 1" "128 1" "128 0"; do
  set -- $CELL; B=$1; Y=$2
  boot "$B" "$Y" "solo-B$B-y$Y" || continue
  OUT=$(python3 "$PROBE" --base $BASE --model q --label "solo-B$B-y$Y" \
        --out "$R/logs/points-solo.jsonl" --n 5 --max-tokens 256)
  log "P2 solo B$B yield=$Y: $OUT"
  down
done
flock -u 9
log "P2_DONE"

# ---- Phase 3: throughput, alternating boots x2/arm, 2 passes per boot
thru() { # thru <B> <yield> <conc> <reqs> <tag>
  local B=$1 Y=$2 CONC=$3 REQS=$4 TAG=$5
  flock 9
  boot "$B" "$Y" "thru-$TAG" || { flock -u 9; return 1; }
  for p in 1 2; do
    python3 "$TREE/tools/load-serve.py" --base $BASE --model q \
      --concurrency "$CONC" --requests "$REQS" --max-tokens 128 \
      --out "$R/logs/points-thru.jsonl" --label "$TAG-p$p" >> "$R/logs/thru-load.log" 2>&1
    ROW=$(tail -1 "$R/logs/points-thru.jsonl" | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print("agg=%.1f p50=%.3f err=%d" % (d["agg_tok_s"],d["lat_p50_s"],d["n_err"]))' 2>/dev/null || echo parse-fail)
    log "P3 $TAG p$p: $ROW [$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)]"
  done
  down
  flock -u 9
}
# c=1 (solo decode rate; alternate on/off within each burst size)
thru 128 1 1 4 c1-B128-y1-r1
thru 128 0 1 4 c1-B128-y0-r1
thru 128 1 1 4 c1-B128-y1-r2
thru 128 0 1 4 c1-B128-y0-r2
thru 32  1 1 4 c1-B32-y1-r1
thru 32  0 1 4 c1-B32-y0-r1
# c=8 (the risk cell: early-burst-exit under load must not cost saturation throughput)
thru 128 1 8 16 c8-B128-y1-r1
thru 128 0 8 16 c8-B128-y0-r1
thru 128 1 8 16 c8-B128-y1-r2
thru 128 0 8 16 c8-B128-y0-r2
thru 32  1 8 16 c8-B32-y1-r1
thru 32  0 8 16 c8-B32-y0-r1
log "P3_DONE"

# ---- Phase 4: greedy content byte-identity (solo + CONTENDED)
sse_cat() { # concatenate every SSE delta to stdout
  python3 -c '
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
sys.stdout.write("".join(buf))'
}
IDENT_REQ='{"model":"q","messages":[{"role":"user","content":"Explain how a CUDA graph reduces kernel launch overhead, in about 200 words."}],"max_tokens":128,"temperature":0.0,"stream":true}'
ident() { # ident <B> <yield> <contended 0|1> <tag>
  local B=$1 Y=$2 C=$3 TAG=$4
  flock 9
  boot "$B" "$Y" "ident-$TAG" || { flock -u 9; return 1; }
  local BGPID=""
  if [ "$C" = 1 ]; then
    curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d \
      '{"model":"q","messages":[{"role":"user","content":"Write a very detailed essay on the history of GPU computing, at least 800 words."}],"max_tokens":512,"temperature":0.0,"stream":false}' \
      > "$R/logs/ident-$TAG.bg.json" &
    BGPID=$!
    sleep 2
  fi
  curl -sN $BASE/v1/chat/completions -H "Content-Type: application/json" -d "$IDENT_REQ" \
    | sse_cat > "$R/logs/ident-$TAG.txt"
  log "P4 ident-$TAG captured ($(wc -c < "$R/logs/ident-$TAG.txt") bytes)"
  [ -n "$BGPID" ] && { wait "$BGPID" 2>/dev/null;
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.stdout.write(d["choices"][0]["message"].get("reasoning") or "");sys.stdout.write(d["choices"][0]["message"]["content"] or "")' \
      "$R/logs/ident-$TAG.bg.json" > "$R/logs/ident-$TAG.bgtext.txt" 2>/dev/null \
      || log "P4 ident-$TAG bg parse FAIL"; }
  down
  flock -u 9
}
ident 128 1 0 solo-B128-y1
ident 128 0 0 solo-B128-y0
ident 32  1 0 solo-B32-y1
ident 32  0 0 solo-B32-y0
ident 128 1 1 cont-B128-y1
ident 128 0 1 cont-B128-y0
for pair in "solo-B128-y1 solo-B128-y0" "solo-B32-y1 solo-B32-y0" \
            "solo-B32-y1 solo-B128-y1" "cont-B128-y1 cont-B128-y0" \
            "solo-B128-y1 cont-B128-y1"; do
  set -- $pair
  if cmp -s "$R/logs/ident-$1.txt" "$R/logs/ident-$2.txt"; then
    log "P4 identity $1 vs $2: BYTE-IDENTICAL"
  else
    log "P4 identity $1 vs $2: MISMATCH"
  fi
done
if cmp -s "$R/logs/ident-cont-B128-y1.bgtext.txt" "$R/logs/ident-cont-B128-y0.bgtext.txt"; then
  log "P4 identity BG(cont-y1) vs BG(cont-y0): BYTE-IDENTICAL"
else
  log "P4 identity BG(cont-y1) vs BG(cont-y0): MISMATCH"
fi
log "P4_DONE"

# ---- Phase 5: gates
flock 9
MEMRA_MTP_DRAFT=$DR MEMRA_SPEC_BURST=128 \
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 \
  "$TREE/target/release/run-spec" "$NV" > "$R/logs/gate-runspec.log" 2>&1
log "P5 run-spec nv+draft B128 rc=$? PASS=$(grep -c PASS "$R/logs/gate-runspec.log")"
flock -u 9

flock 9
( cd "$TREE" && timeout 1800 tools/serve-smoke.sh > "$R/logs/gate-serve-smoke.log" 2>&1 )
log "P5 serve-smoke rc=$? failed=$(grep -c FAIL "$R/logs/gate-serve-smoke.log")"
flock -u 9

flock 9
out=$("$TREE/target/release/decode-batch-gate" "$DBG9B" --steps 32 --batch 8 --mode config 2>&1)
echo "$out" > "$R/logs/gate-dbg-config.log"
echo "$out" | grep -q "ALL GREEN" && log "P5 decode-batch-gate config B=8: ALL GREEN" \
  || log "P5 decode-batch-gate config B=8: FAIL"
out=$(MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 "$TREE/target/release/decode-batch-gate" \
  "$DBG9B" --steps 32 --batch 4 --mode strict 2>&1)
echo "$out" > "$R/logs/gate-dbg-strict.log"
echo "$out" | grep -q "ALL GREEN" && log "P5 decode-batch-gate strict B=4: ALL GREEN" \
  || log "P5 decode-batch-gate strict B=4: FAIL"
flock -u 9
exec 9>&-

( cd "$TREE" && cargo test -p memra-server --release > "$R/logs/gate-cargo-test.log" 2>&1 )
log "P5 cargo test -p memra-server rc=$? $(grep -E 'test result' "$R/logs/gate-cargo-test.log" | tail -2 | tr '\n' ' ')"
log "FULL_EVAL_DONE"
echo FULL_EVAL_DONE

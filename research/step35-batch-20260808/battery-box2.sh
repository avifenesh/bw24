#!/usr/bin/env bash
# Box2 (13.59.112.147, 2x PRO 6000, us-east-2) — serve-level gates + the c-scaling perf
# sweep, in parallel with box1's engine-level bit-identity battery. Box2 is this lane's
# own battery box; flock kept anyway (discipline, and future co-tenants).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step35-batch"
M=/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=/data/models/step37/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
RAW=$HOME/step35-batch/research/step35-batch-20260808/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/battery-box2-$TS.log
PORT=8095
BASE=http://127.0.0.1:$PORT

thermal() { nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader; }

{
echo "=== step35-batch box2 battery $TS tip=$(git rev-parse --short HEAD)"

echo; echo "########## S1: b2geo35 naked (GREEN expected) ##########"
MEMRA_STEP37_GGUF="$M" bash tools/step35-b2-geometry-gate.sh --port $PORT
echo "b2geo35 exit=$?"
echo; echo "########## S1c: b2geo35 canary (teeth) ##########"
MEMRA_STEP37_GGUF="$M" bash tools/step35-b2-geometry-gate.sh --canary --port $PORT
echo "b2geo35c exit=$?"

echo; echo "########## S2: serve c=1..8 byte-vs-serial (c=8 arm on top of the gate's 2/4) ##########"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal
  env MEMRA_MODELS="step35=${M}+${D}" MEMRA_SERVE_SPEC=0 MEMRA_SERVE_B1FAST=0 \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR=127.0.0.1:$PORT \
      ./target/release/memra-server > "$RAW/s2-server-$TS.log" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
  for i in $(seq 1 120); do
    sleep 5; curl -sf "$BASE/readyz" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo SERVER DIED; exit 1; }
  done
  BODY='{"model":"step35","messages":[{"role":"user","content":"List the first eight prime numbers, comma separated, then explain in two sentences why 1 is not prime."}],"max_tokens":48,"temperature":0.0}'
  ask() { curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
    | python3 -c 'import json,sys
r=json.load(sys.stdin); c=r.get("choices")
if c:
    m=c[0]["message"]; print(json.dumps({"reasoning": m.get("reasoning"), "content": m.get("content")}))
else:
    print("ERROR", json.dumps(r.get("error")))'; }
  ask > /tmp/s2-ref.txt; echo "ref: $(cat /tmp/s2-ref.txt | head -c 120)"
  FAILS=0
  for C in 8; do
    PIDS=(); for i in $(seq 1 $C); do ask > /tmp/s2-c$C-$i.txt & PIDS+=($!); done
    wait "${PIDS[@]}"
    for i in $(seq 1 $C); do
      if cmp -s /tmp/s2-ref.txt /tmp/s2-c$C-$i.txt; then echo "c$C[$i] == ref"
      else echo "c$C[$i] != ref"; cat /tmp/s2-c$C-$i.txt; FAILS=$((FAILS+1)); fi
    done
  done
  grep -m1 "\[step35-batch\] first B>1" "$RAW/s2-server-$TS.log" || { echo "no batched-walk line"; FAILS=$((FAILS+1)); }
  kill $SRV; wait $SRV 2>/dev/null; trap - EXIT
  echo "S2 verdict: $([ $FAILS -eq 0 ] && echo PASS || echo "FAIL ($FAILS)")"
  thermal
  echo "lock released $(date -u +%FT%TZ)"
  [ $FAILS -eq 0 ]
) 9>/tmp/memra-gpu.lock
echo "=== s2 rc=$?"

echo; echo "########## P1: decode aggregate c-sweep, DEFAULT batched serve, N=3 ##########"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal
  env MEMRA_MODELS="step35=${M}+${D}" MEMRA_SERVE_SPEC=0 \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR=127.0.0.1:$PORT \
      ./target/release/memra-server > "$RAW/perf-server-$TS.log" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
  for i in $(seq 1 120); do
    sleep 5; curl -sf "$BASE/readyz" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo SERVER DIED; exit 1; }
  done
  grep -m1 "decode chunk cap" "$RAW/perf-server-$TS.log" || true
  for c in 1 2 4 8; do
    for rep in 1 2 3; do
      echo "--- P1 c=$c rep=$rep ---"
      python3 tools/load-serve.py --base "$BASE" --model step35 --concurrency "$c" \
        --requests $((4 * c)) --max-tokens 128 --warmup 1 \
        --label "sb2-c${c}-r${rep}" --out "$RAW/perf-points-$TS.jsonl" --timeout 1800
      thermal
    done
  done
  kill $SRV; wait $SRV 2>/dev/null; trap - EXIT
  thermal
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== perf rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

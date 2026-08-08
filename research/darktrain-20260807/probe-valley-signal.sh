#!/usr/bin/env bash
# probe-valley-signal — deliverable-1 receipt (lane/darklane-training, 2026-08-07):
# the /metrics serve_idle_seconds valley signal flips to 0 under load and accrues back
# when the box quiets down.
#
# Asserts:
#   1. field exists and grows monotonically while idle (two samples 1.5s apart);
#   2. DURING an in-flight generation the signal reads 0.0;
#   3. after completion the signal accrues again from ~0 (a fresh valley).
#
# GPU: caller wraps in `flock /tmp/gpu5090.lock` per the local convention.
# Raw log: research/darktrain-20260807/raw/valley-signal.log (tee'd by the caller).
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
[ -f "$MODEL" ] || { echo "probe-valley: SKIP (no model at $MODEL)"; exit 0; }
ADDR=127.0.0.1:8188
BASE=http://$ADDR
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

[ -x target/release/memra-server ] || cargo build --release -p memra-server

MEMRA_MODELS="smoke=$MODEL" MEMRA_ADDR=$ADDR MEMRA_COMPAT=openai \
  target/release/memra-server > /tmp/probe-valley-server.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && break; sleep 2; done
curl -sf $BASE/health >/dev/null || { echo "server did not come up"; exit 1; }

idle() { curl -sf $BASE/metrics | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["serve_idle_seconds"])'; }

# 1. idle: the signal exists and accrues.
sleep 2
A=$(idle); sleep 1.5; B=$(idle)
echo "idle samples: A=$A B=$B"
python3 -c "assert float('$A') > 0, 'A not accruing'; assert float('$B') > float('$A'), 'not monotonic'" \
  && PASS "idle signal accrues monotonically ($A -> $B)" \
  || FAIL "idle signal did not accrue ($A -> $B)"

# 2. under load: fire a generation, sample the signal MID-FLIGHT. Budget sized to the
#    measured rate (first probe run: 258 tok in 1.2s spec — a 256 budget finished before
#    a 2s sample; 2000 tokens ≈ 9s keeps the sample well inside the window).
curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"smoke","messages":[{"role":"user","content":"Count slowly from one to fifty in words."}],
       "max_tokens":2000,"temperature":0,"stream":false}' > /tmp/probe-valley-gen.json &
GPID=$!
sleep 2   # well inside the generation window
L=$(idle)
echo "under-load sample: $L"
python3 -c "assert float('$L') == 0.0, 'expected 0.0 under load'" \
  && PASS "signal reads 0.0 while a generation is in flight" \
  || FAIL "signal nonzero ($L) under load"
wait $GPID || FAIL "generation request failed"
python3 -c 'import json; r=json.load(open("/tmp/probe-valley-gen.json")); assert r["usage"]["completion_tokens"]>0' \
  || FAIL "generation produced no tokens"

# 3. back to valley: the signal accrues again from a fresh epoch.
sleep 2
C=$(idle)
echo "post-load sample (after 2s quiet): $C"
python3 -c "assert 0 < float('$C') < 10, 'expected a fresh small idle age'" \
  && PASS "signal accrues again after load ($C s, fresh epoch)" \
  || FAIL "signal did not restart cleanly after load ($C)"

echo "probe-valley: $FAILS failed"
[ $FAILS -eq 0 ]

#!/usr/bin/env bash
# Functional test: dark-lane batch-prime (#17 increment).
# Fires N concurrent short harvest-lane completions at a plain (no-draft) server,
# confirms "[prime-batch dark]" fires and outputs are coherent + exact vs sequential.
set -u
MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
BIN="${2:-target/release/bw24-server}"
ADDR=127.0.0.1:18091
LOG="${3:-/tmp/darkbatch-server.log}"
N="${N:-3}"
GEN="${GEN:-32}"

pkill -f "bw24-server" 2>/dev/null; sleep 1
BW24_SERVE_SPEC=${SPEC:-0} BW24_COMPAT=openai BW24_MODELS="m=$MODEL" BW24_ADDR=$ADDR "$BIN" > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 60); do curl -sf "http://$ADDR/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://$ADDR/health" >/dev/null || { echo "FAIL: server no boot"; tail -5 "$LOG"; kill $SRV; exit 1; }

# ~40-token prompts (>= PRIME_MIN_T=16), distinct per slot
mkdir -p /tmp/darkbatch-out
prompt() { echo "Question $1: List the first ten prime numbers, one per line, then state their sum and briefly explain why two is the only even prime number in mathematics. Answer:"; }

# --- sequential reference (greedy, one at a time, interactive lane) ---
for i in $(seq 1 $N); do
  curl -s "http://$ADDR/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"m\",\"prompt\":$(prompt $i | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),\"max_tokens\":$GEN,\"temperature\":0}" \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"])' > /tmp/darkbatch-out/seq-$i.txt
done
SEQ_BATCH_LINES=$(grep -c "prime-batch dark" "$LOG" || true)

# --- concurrent harvest-lane fire ---
T0=$(date +%s.%N)
for i in $(seq 1 $N); do
  curl -s "http://$ADDR/v1/completions" -H 'content-type: application/json' -H 'x-lane: harvest' \
    -d "{\"model\":\"m\",\"prompt\":$(prompt $i | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),\"max_tokens\":$GEN,\"temperature\":0}" \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"])' > /tmp/darkbatch-out/dark-$i.txt &
done
# NOT bare `wait`: that also waits the backgrounded server job and hangs until it dies.
wait $(jobs -pr | grep -v "^$SRV\$")
T1=$(date +%s.%N)
kill -9 $SRV 2>/dev/null

BATCH_LINES=$(grep -c "prime-batch dark" "$LOG" || true)
echo "== [prime-batch dark] lines: $BATCH_LINES (during-seq: $SEQ_BATCH_LINES) =="
grep "prime-batch dark" "$LOG" | tail -3
FAILS=0
for i in $(seq 1 $N); do
  if ! cmp -s /tmp/darkbatch-out/seq-$i.txt /tmp/darkbatch-out/dark-$i.txt; then
    echo "TEXT MISMATCH slot $i:"; diff /tmp/darkbatch-out/seq-$i.txt /tmp/darkbatch-out/dark-$i.txt | head -5
    FAILS=$((FAILS+1))
  fi
done
echo "== exactness: $((N-FAILS))/$N match sequential greedy; wall $(echo "$T1 $T0" | awk '{printf "%.2f", $1-$2}')s =="
[ "$BATCH_LINES" -gt "$SEQ_BATCH_LINES" ] && [ "$FAILS" -eq 0 ] && echo "PASS" || echo "CHECK"

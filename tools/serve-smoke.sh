#!/usr/bin/env bash
# Normal-usage serving battery: boots bw24-server like a user would and exercises the
# real API surface — chat (stream + non-stream), plain completions, concurrency, greedy
# determinism, lane header, and spec-vs-plain greedy identity (the exactness contract at
# the SERVING level, not just the kernel gates).
#
# Usage: tools/serve-smoke.sh [model.gguf [draft.gguf]]
# Defaults to the E4B QAT pair (fast load). Exits nonzero on any failed check.
set -uo pipefail
cd "$(dirname "$0")/.."

# Default = the 9B NVFP4 + its regime draft (full serving support; E4B's serve path is
# first-light only — dc/graph/spec unwired — and gemma assistant drafts use BW24_DRAFT,
# not the '+draft' NextN attach).
MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
DRAFT="${2:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}"
[ -f "$MODEL" ] || { echo "serve-smoke: SKIP (no model at $MODEL)"; exit 0; }
ADDR=127.0.0.1:8177
BASE=http://$ADDR
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

[ -x target/release/bw24-server ] || cargo build --release -p bw24-server

start_server() {  # extra env via prefix, e.g. start_server "smoke=/path.gguf"
  # BW24_COMPAT=openai: this battery tests the OpenAI-compatible surface the README
  # sells (the default native /v1/completions shape is a different contract).
  BW24_COMPAT=openai BW24_MODELS="$1" BW24_ADDR=$ADDR target/release/bw24-server > /tmp/serve-smoke.log 2>&1 &
  SPID=$!
  for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up; log tail:"; tail -5 /tmp/serve-smoke.log; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; }
trap stop_server EXIT

chat() {  # prompt max_tokens [stream] [lane] -> body to stdout
  local prompt=$1 maxtok=$2 stream=${3:-false} lane=${4:-}
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    ${lane:+-H "x-lane: $lane"} \
    -d "{\"model\":\"smoke\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],
         \"max_tokens\":$maxtok,\"temperature\":0,\"stream\":$stream}"
}

echo "== serve-smoke: plain serving =="
start_server "smoke=$MODEL" || exit 1

# 1. health + models
curl -sf $BASE/models | grep -q smoke && PASS "/models lists the model" || FAIL "/models"

# 2. non-stream chat: 200, non-empty content, usage populated
R=$(chat "Name three primary colors, comma-separated." 48)
echo "$R" | python3 -c '
import json,sys
r = json.load(sys.stdin)
c = r["choices"][0]["message"]["content"]
assert c.strip(), "empty content"
assert r["usage"]["completion_tokens"] > 0, "no completion tokens"
assert r["choices"][0]["finish_reason"] in ("stop","length"), "bad finish_reason"
' && PASS "chat non-stream (content + usage + finish_reason)" || FAIL "chat non-stream"

# 3. streaming: data: chunks then [DONE]
S=$(chat "Count from one to five in words." 48 true)
echo "$S" | grep -q '^data: ' && echo "$S" | grep -q 'data: \[DONE\]' \
  && PASS "chat stream (SSE chunks + [DONE])" || FAIL "chat stream"

# 4. plain /v1/completions
curl -sf -m 300 $BASE/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"smoke","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["choices"][0]["text"].strip()' \
  && PASS "/v1/completions" || FAIL "/v1/completions"

# 5. greedy determinism: same prompt twice -> identical text
A=$(chat "Explain what a mutex is in one sentence." 64 | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
B=$(chat "Explain what a mutex is in one sentence." 64 | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
[ -n "$A" ] && [ "$A" = "$B" ] && PASS "greedy determinism (2 runs identical)" || FAIL "greedy determinism"

# 6. concurrency: 3 parallel chats all complete non-empty
pids=(); outs=()
for i in 1 2 3; do
  o=/tmp/serve-smoke-conc-$i.json
  outs+=("$o")
  ( chat "Write one fact about the number $i." 32 > "$o" ) & pids+=($!)
done
okc=0
for i in 0 1 2; do
  wait "${pids[$i]}" 2>/dev/null
  python3 -c "import json;assert json.load(open('${outs[$i]}'))['choices'][0]['message']['content'].strip()" 2>/dev/null && okc=$((okc+1))
done
[ $okc -eq 3 ] && PASS "3 concurrent chats" || FAIL "concurrency ($okc/3)"

# 7. lane header accepted (judge lane; response still served or shed with 429, never 5xx)
code=$(curl -s -m 300 -o /dev/null -w '%{http_code}' $BASE/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'x-lane: judge' \
  -d '{"model":"smoke","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}')
{ [ "$code" = 200 ] || [ "$code" = 429 ]; } && PASS "x-lane header ($code)" || FAIL "x-lane header ($code)"

# 8. long generation (exercises the graph door at budget >= 256)
R=$(chat "Tell a short story about a lighthouse keeper." 300)
echo "$R" | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["usage"]["completion_tokens"] >= 100' \
  && PASS "long generation (>=100 tok)" || FAIL "long generation"

PLAIN_MUTEX=$A
stop_server

# 9. spec serving: same greedy prompt must produce IDENTICAL text to plain serving
if [ -f "$DRAFT" ]; then
  echo "== serve-smoke: spec serving (draft attached) =="
  start_server "smoke=$MODEL+$DRAFT" || exit 1
  SA=$(chat "Explain what a mutex is in one sentence." 64 | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
  [ -n "$SA" ] && [ "$SA" = "$PLAIN_MUTEX" ] && PASS "spec == plain greedy text (serving exactness)" \
    || FAIL "spec-vs-plain text mismatch"
  stop_server
else
  echo "== serve-smoke: spec arm SKIP (no draft at $DRAFT)"
fi

echo "serve-smoke: $FAILS failed"
[ $FAILS -eq 0 ]

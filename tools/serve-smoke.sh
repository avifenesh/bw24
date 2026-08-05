#!/usr/bin/env bash
# Normal-usage serving battery: boots memra-server like a user would and exercises the
# real API surface — chat (stream + non-stream), plain completions, concurrency, greedy
# determinism, and spec-vs-plain greedy identity (the exactness contract at
# the SERVING level, not just the kernel gates).
#
# Usage: tools/serve-smoke.sh [model.gguf [draft.gguf]]
# Defaults to the E4B QAT pair (fast load). Exits nonzero on any failed check.
set -uo pipefail
cd "$(dirname "$0")/.."

# Default = the 9B NVFP4 + its regime draft (full serving support; E4B's serve path is
# first-light only — dc/graph/spec unwired — and gemma assistant drafts use MEMRA_DRAFT,
# not the '+draft' NextN attach).
MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
DRAFT="${2:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}"
[ -f "$MODEL" ] || { echo "serve-smoke: SKIP (no model at $MODEL)"; exit 0; }
ADDR=127.0.0.1:8177
BASE=http://$ADDR
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

[ -x target/release/memra-server ] || cargo build --release -p memra-server

start_server() {  # extra env via prefix, e.g. start_server "smoke=/path.gguf"
  # MEMRA_COMPAT=openai: this battery tests the OpenAI-compatible surface the README
  # sells (the default native /v1/completions shape is a different contract).
  MEMRA_COMPAT=openai MEMRA_MODELS="$1" MEMRA_ADDR=$ADDR target/release/memra-server > /tmp/serve-smoke.log 2>&1 &
  SPID=$!
  for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up; log tail:"; tail -5 /tmp/serve-smoke.log; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; }
trap stop_server EXIT

chat() {  # prompt max_tokens [stream] -> body to stdout
  local prompt=$1 maxtok=$2 stream=${3:-false}
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"smoke\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],
         \"max_tokens\":$maxtok,\"temperature\":0,\"stream\":$stream}"
}
# THE EMITTED STREAM = reasoning + content (gate-rot fix, 2026-08-04). The default smoke model
# (q9 Qwen3.5) is a REASONING model: at the small budgets this battery uses, every emitted token
# is still inside the thinking block, so `message.content` is legitimately "" with
# finish_reason=length. Asserting on `content` alone made checks 2/5/6/8 structurally
# unpassable — measured identically red on the pre-lane binary at 0a7349f6 (v0.68.0), i.e. this
# gate had been rotted in main, not broken by a lane. What the checks actually mean is "the
# server emitted deterministic non-empty output" and "spec emits the same text as plain", and
# that is the reasoning+content concatenation. Kept budget-independent on purpose: raising
# max_tokens until the model happens to close its thinking block would make the gate a
# model-verbosity coin flip.
say() { python3 -c '
import json,sys
m = json.load(sys.stdin)["choices"][0]["message"]
print((m.get("reasoning") or "") + (m.get("content") or ""))'; }

echo "== serve-smoke: plain serving =="
start_server "smoke=$MODEL" || exit 1

# 1. health + models
curl -sf $BASE/models | grep -q smoke && PASS "/models lists the model" || FAIL "/models"

# 2. non-stream chat: 200, non-empty content, usage populated
R=$(chat "Name three primary colors, comma-separated." 48)
echo "$R" | python3 -c '
import json,sys
r = json.load(sys.stdin)
m = r["choices"][0]["message"]
# reasoning OR content — a thinking model at a small budget emits only the former (see say()).
assert ((m.get("reasoning") or "") + (m.get("content") or "")).strip(), "empty emitted text"
assert r["usage"]["completion_tokens"] > 0, "no completion tokens"
assert r["choices"][0]["finish_reason"] in ("stop","length"), "bad finish_reason"
' && PASS "chat non-stream (text + usage + finish_reason)" || FAIL "chat non-stream"

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
A=$(chat "Explain what a mutex is in one sentence." 64 | say)
B=$(chat "Explain what a mutex is in one sentence." 64 | say)
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
  python3 -c "
import json
m = json.load(open('${outs[$i]}'))['choices'][0]['message']
assert ((m.get('reasoning') or '') + (m.get('content') or '')).strip()" 2>/dev/null && okc=$((okc+1))
done
[ $okc -eq 3 ] && PASS "3 concurrent chats" || FAIL "concurrency ($okc/3)"

# 7. long generation (exercises the graph door at budget >= 256)
R=$(chat "Tell a short story about a lighthouse keeper." 300)
echo "$R" | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["usage"]["completion_tokens"] >= 100' \
  && PASS "long generation (>=100 tok)" || FAIL "long generation"

PLAIN_MUTEX=$A
stop_server

# 8. spec serving: same greedy prompt must produce IDENTICAL text to plain serving
if [ -f "$DRAFT" ]; then
  echo "== serve-smoke: spec serving (draft attached) =="
  start_server "smoke=$MODEL+$DRAFT" || exit 1
  SA=$(chat "Explain what a mutex is in one sentence." 64 | say)
  [ -n "$SA" ] && [ "$SA" = "$PLAIN_MUTEX" ] && PASS "spec == plain greedy text (serving exactness)" \
    || FAIL "spec-vs-plain text mismatch"

  # 9. SAMPLED TRUNCATION MATRIX (added 2026-08-05, lane/sampler-truncation-fix).
  # THE GAP THIS CLOSES: every check above runs temperature 0. A greedy-only serve battery
  # cannot see any sampled-path defect, and it did not see this one — memra-server shipped a
  # top_p/min_p bug that spliced token id 0 ("!") mid-word on the published OpenAI surface
  # (found by the memra-vs-llama head-to-head arm, not by a gate:
  # research/memra-vs-llama-daily-20260805/logs/posthoc-lsampler.txt). Root cause and the
  # unit-level gate arms: research/sampfix-20260805/.
  #
  # The assertion is deliberately NOT "text equals a golden": sampled output legitimately
  # varies with the model, and a golden here would rot. It is a DIFFERENTIAL test against a
  # baseline that is STRUCTURALLY IMMUNE to this defect class, which makes it model- and
  # prompt-independent with no threshold to tune.
  #
  # Why untruncated t=0.8 is a sound baseline: the whole defect class is bad masking in the
  # filtered device path. With no truncation the filter threshold `th` is exactly 0, so nothing
  # is ever masked, the row can never go all-(-inf), and the argmax can never fall through to
  # its smallest-index tie-break. The untruncated arm therefore CANNOT exhibit the bug, and the
  # rate of low-id/odd characters it produces on this prompt is the model's honest baseline.
  #
  # The check (per truncated arm, same prompt, same seed, same temperature):
  #   (a) seeded reproducibility — same seed + same shape must reproduce the text exactly
  #   (b) non-empty text
  #   (c) the truncated arm's count of '!' must not EXCEED the immune baseline's count.
  #       Truncation removes low-probability tail tokens, so a correct truncated draw can only
  #       be *less* surprising than the untruncated one — more '!' after truncating is the
  #       signature, and it is what every corrupt arm did (measured on the pre-fix binary:
  #       baseline 0, top_p 12, llama-shape 12, min_p 2).
  #   (d) plus the raw structural forms that are corruption regardless of any baseline: a '!!'
  #       run, or a '!' spliced directly before an alphanumeric (`!bash`, `gpu-r!ig`) — a token
  #       boundary no healthy tokenizer emits.
  # Run through the SPEC server on purpose: truncation interacts with the rejection-sampling
  # verify, and the bug lived in the spec full-accept bonus path specifically.
  echo "== serve-smoke: sampled truncation matrix (spec server) =="
  samp() { # $1 = extra json sampling fields
    curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"smoke\",\"messages\":[{\"role\":\"user\",\"content\":\"List three shell commands that inspect a file, one per line.\"}],
           \"max_tokens\":64,\"temperature\":0.8,\"seed\":7${1:+,$1}}" | say
  }
  nbang() { printf '%s' "$1" | tr -cd '!' | wc -c; }
  # the immune reference: temp 0.8, NO truncation => th==0 => nothing masked => cannot corrupt
  TRUNC_BASE=$(samp '')
  TRUNC_BASE_N=$(nbang "$TRUNC_BASE")
  echo "  (immune baseline: untruncated t0.8 seed=7, bangs=$TRUNC_BASE_N)"
  check_trunc() { # $1 = json fields, $2 = label
    local t1 t2 bangs
    t1=$(samp "$1"); t2=$(samp "$1")
    if [ -z "$t1" ]; then FAIL "trunc $2: empty text"; return; fi
    if [ "$t1" != "$t2" ]; then FAIL "trunc $2: not reproducible at a fixed seed"; return; fi
    bangs=$(nbang "$t1")
    case "$t1" in
      *'!!'*) FAIL "trunc $2: '!!' run = id-0 fallthrough ($bangs '!')"; return ;;
    esac
    if printf '%s' "$t1" | grep -qE '![[:alnum:]]'; then
      FAIL "trunc $2: '!' spliced before an alphanumeric = id-0 injection ($bangs '!')"; return
    fi
    if [ "$bangs" -gt "$TRUNC_BASE_N" ]; then
      FAIL "trunc $2: $bangs '!' vs immune-baseline $TRUNC_BASE_N — truncation cannot ADD tail tokens"
      return
    fi
    PASS "trunc $2 (reproducible, bangs=$bangs <= baseline $TRUNC_BASE_N)"
  }
  check_trunc '"top_k":40'                                  'top_k=40'
  check_trunc '"top_p":0.95'                                'top_p=0.95'
  check_trunc '"min_p":0.05'                                'min_p=0.05'
  check_trunc '"top_k":40,"top_p":0.95,"min_p":0.05'        'llama-default k40+p0.95+m0.05'
  stop_server
else
  echo "== serve-smoke: spec arm SKIP (no draft at $DRAFT)"
fi

echo "serve-smoke: $FAILS failed"
[ $FAILS -eq 0 ]

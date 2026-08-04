#!/usr/bin/env bash
# fp8-ship item B, phases 2-5 — serving gates + serve-path perf cells for the OFFICIAL
# Qwen3.6-27B-FP8 checkpoint on the vast 2x5090. GPU 1 (phase 1 load gates own GPU 0).
# Train HEAD ac99e675: #68 closed, ST spec quarantine LIFTED — default greedy serve on an
# MTP dir checkpoint spec-bursts. Arms:
#   plain  = MEMRA_SERVE_SPEC=0 (tokenwise decode; the row comparable to the GGUF plain cell)
#   spec   = default env (embedded mtp.safetensors head, if the ST loader takes it)
#   spec-drafter = fp8=<dir>+<q27 own-trim draft gguf> (regime-draft replacement arm)
#   e4m3   = MEMRA_ST_E4M3=1 + SERVE_SPEC=0 — EXPECTED FLAT on this ckpt: every F8 tensor is
#            block-128 (407 weight_scale_inv, 0 scalar), and the QT_F8_E4M3 arm requires
#            blk=None, so all fall through to the same Q8_0 re-encode. Receipt confirms.
set -uo pipefail
cd /root/memra
export CUDA_VISIBLE_DEVICES=1
OUT=research/fp8ship-20260804/official
mkdir -p "$OUT"
CKPT=/root/models/qwen36-27b-fp8-official
DRAFT=/root/models/bench/drafts/qwen36-27b-nvfp4/draft-owntrim-nvfp4head-q4blk.gguf
P512=research/e2e/prompts/pp512.txt
ADDR=127.0.0.1:8199
BASE=http://$ADDR
DLOG=$OUT/phase2-driver.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }
snap(){ nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader -i 1; }
FAILS=0
PASS(){ log "  ok: $1"; }
FAIL(){ log "  FAIL: $1"; FAILS=$((FAILS+1)); }

start_server(){ # $1 extra env string, $2 log tag, $3 models spec (default fp8=$CKPT)
  local spec="${3:-fp8=$CKPT}"
  env $1 MEMRA_MODELS="$spec" MEMRA_ADDR=$ADDR \
    target/release/memra-server > $OUT/server-$2.log 2>&1 &
  SPID=$!
  for _ in $(seq 600); do
    curl -sf $BASE/health >/dev/null 2>&1 && return 0
    kill -0 $SPID 2>/dev/null || { log "server ($2) DIED; tail:"; tail -8 $OUT/server-$2.log | tee -a "$DLOG"; return 1; }
    sleep 2
  done
  log "server ($2) did not come up in 20min; tail:"; tail -8 $OUT/server-$2.log | tee -a "$DLOG"; return 1
}
stop_server(){ kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }
trap stop_server EXIT

log "== PHASE 2/3: serving gates, DEFAULT env (spec-on for MTP dir ckpts per ac99e675) =="
log "server start pre: $(snap)"
T0=$(date +%s.%N)
start_server "" default || exit 1
T1=$(date +%s.%N)
log "server (default env) up in $(echo "$T1 $T0" | awk '{printf "%.1f", $1-$2}')s"
grep -aE "spec|mtp|draft" $OUT/server-default.log | head -5 | tee -a "$DLOG"

curl -sf $BASE/models | grep -q '"fp8"' && PASS "/models lists fp8" || FAIL "/models"

PROMPT="What is the capital of France? Answer in one short sentence."
R=$(curl -sf -m 600 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"fp8\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":400,\"temperature\":0}")
echo "$R" > $OUT/chat-correctness.json
echo "$R" | python3 -c '
import json,sys
r=json.load(sys.stdin)
m=r["choices"][0]["message"]
text=(m.get("content") or "")+(m.get("reasoning") or "")
assert text.strip(), "empty content+reasoning"
assert "paris" in text.lower(), f"incoherent: {text[:200]!r}"
assert r["usage"]["completion_tokens"]>0
print("coherent:", text.strip()[:120].replace(chr(10)," "))
' >> "$DLOG" 2>&1 && PASS "chat completion coherent (Paris) via ckpt template" || FAIL "chat coherence"

for i in 1 2 3; do
  curl -sf -m 600 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"fp8\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":128,\"temperature\":0,\"cache_salt\":\"det-$i\"}" \
    | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("content") or "")+(m.get("reasoning") or ""))' \
    > /tmp/det-$i.txt
done
if diff -q /tmp/det-1.txt /tmp/det-2.txt >/dev/null && diff -q /tmp/det-2.txt /tmp/det-3.txt >/dev/null; then
  PASS "greedy determinism x3 (default/spec server, fresh compute each)"
else
  FAIL "greedy determinism x3"; diff /tmp/det-1.txt /tmp/det-2.txt | head -5 | tee -a "$DLOG"
fi
cp /tmp/det-1.txt $OUT/greedy-det-default.txt
stop_server
trap - EXIT

log "== PHASE 3b: serve-st-gate.sh on the official ckpt (spec-ON item 4 per ac99e675) =="
bash tools/serve-st-gate.sh "$CKPT" > $OUT/serve-st-gate.log 2>&1
GRC=$?
tail -14 $OUT/serve-st-gate.log | tee -a "$DLOG"
[ $GRC -eq 0 ] && PASS "serve-st-gate.sh" || FAIL "serve-st-gate.sh rc=$GRC"

log "== waiting for phase 1 (GPU0 load gates) to finish before perf cells =="
for _ in $(seq 720); do
  grep -q "PHASE 1 DONE" $OUT/phase1-driver.log 2>/dev/null && break
  sleep 10
done
grep -q "PHASE 1 DONE" $OUT/phase1-driver.log && log "phase 1 done — box quiet for perf" \
  || log "WARN: phase 1 not done after 2h wait; proceeding (state it in the receipt)"

log "== PHASE 4: serve-path perf cells, pp512-class prompt, N=5, single card (GPU1) =="
trap stop_server EXIT

# --- arm 1: PLAIN (SERVE_SPEC=0) ---
start_server "MEMRA_SERVE_SPEC=0" plain || exit 1
log "perf ST-plain pre: $(snap)"
python3 /root/serve-perf.py $BASE fp8 "$P512" 5 st-plain $OUT/serve-perf.jsonl \
  > $OUT/perf-st-plain.log 2>&1 && PASS "perf cell st-plain" || FAIL "perf cell st-plain: $(tail -2 $OUT/perf-st-plain.log)"
log "perf ST-plain post: $(snap) | $(grep '^SUMMARY' $OUT/perf-st-plain.log)"
grab_text(){ # $1 outfile $2 salt
  curl -sf -m 600 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"fp8\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":128,\"temperature\":0,\"cache_salt\":\"$2\"}" \
    | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("content") or "")+(m.get("reasoning") or ""))' > "$1"
}
grab_text $OUT/greedy-plain.txt plaincmp
stop_server

# --- arm 2: SPEC default (embedded MTP head from mtp.safetensors, if loaded) ---
start_server "" spec || exit 1
grep -aE "spec|mtp|draft|nextn" $OUT/server-spec.log | head -5 | tee -a "$DLOG"
log "perf ST-spec pre: $(snap)"
python3 /root/serve-perf.py $BASE fp8 "$P512" 5 st-spec-default $OUT/serve-perf.jsonl \
  > $OUT/perf-st-spec.log 2>&1 && PASS "perf cell st-spec-default" || FAIL "perf cell st-spec-default: $(tail -2 $OUT/perf-st-spec.log)"
log "perf ST-spec post: $(snap) | $(grep '^SUMMARY' $OUT/perf-st-spec.log)"
grab_text $OUT/greedy-spec.txt speccmp
stop_server
# spec-vs-plain text: burst overshoot tolerated (prefix rule, same as serve-st-gate item 4)
python3 - "$OUT/greedy-plain.txt" "$OUT/greedy-spec.txt" <<'EOF' | tee -a "$DLOG"
import sys
a, b = open(sys.argv[1]).read(), open(sys.argv[2]).read()
if a == b: print("spec-vs-plain greedy text: IDENTICAL")
elif a.startswith(b) or b.startswith(a): print(f"spec-vs-plain greedy text: PREFIX-MATCH (lens {len(a)}/{len(b)} — burst overshoot class)")
else:
    div = next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
    print(f"spec-vs-plain greedy text: DIVERGE at char {div}: plain={a[div:div+40]!r} spec={b[div:div+40]!r}")
EOF

# --- arm 3: SPEC with the q27 own-trim regime drafter (NVFP4-head gguf) ---
if [ -f "$DRAFT" ]; then
  if start_server "" spec-drafter "fp8=$CKPT+$DRAFT"; then
    grep -aE "spec|mtp|draft" $OUT/server-spec-drafter.log | head -5 | tee -a "$DLOG"
    log "perf ST-spec-drafter pre: $(snap)"
    python3 /root/serve-perf.py $BASE fp8 "$P512" 5 st-spec-drafter $OUT/serve-perf.jsonl \
      > $OUT/perf-st-spec-drafter.log 2>&1 && PASS "perf cell st-spec-drafter" || FAIL "perf cell st-spec-drafter: $(tail -2 $OUT/perf-st-spec-drafter.log)"
    log "perf ST-spec-drafter post: $(snap) | $(grep '^SUMMARY' $OUT/perf-st-spec-drafter.log)"
    grab_text $OUT/greedy-spec-drafter.txt draftcmp
    stop_server
  else
    log "FINDING: spec-drafter server failed to load (draft built against the NVFP4 ckpt) — captured, arm skipped"
    stop_server
  fi
else
  log "drafter gguf absent at $DRAFT — arm skipped"
fi

# --- arm 4: e4m3-direct (EXPECTED FLAT on block-128-only ckpt; receipt-confirm) ---
start_server "MEMRA_ST_E4M3=1 MEMRA_SERVE_SPEC=0" e4m3 || exit 1
log "perf ST-e4m3 pre: $(snap)"
python3 /root/serve-perf.py $BASE fp8 "$P512" 5 st-e4m3 $OUT/serve-perf.jsonl \
  > $OUT/perf-st-e4m3.log 2>&1 && PASS "perf cell st-e4m3" || FAIL "perf cell st-e4m3: $(tail -2 $OUT/perf-st-e4m3.log)"
log "perf ST-e4m3 post: $(snap) | $(grep '^SUMMARY' $OUT/perf-st-e4m3.log)"
grab_text $OUT/greedy-e4m3.txt e4m3cmp
if diff -q $OUT/greedy-plain.txt $OUT/greedy-e4m3.txt >/dev/null; then
  log "e4m3 vs plain greedy text: IDENTICAL (expected — block-128 tensors fall through to the same Q8_0 path)"
else
  log "e4m3 vs plain greedy text: DIFFERS (UNEXPECTED on this ckpt — capture)"
  diff $OUT/greedy-plain.txt $OUT/greedy-e4m3.txt | head -10 | tee -a "$DLOG"
fi
stop_server
trap - EXIT

log "PHASE 2-5 DONE: $FAILS failed"
exit $FAILS

#!/usr/bin/env bash
# Constrained-decoding FULL battery (lane/constrained-full, 2026-08-03).
#   Phase A — NO-OP EXACTNESS: unconstrained requests, BASELINE binary (merged v1 HEAD
#             c2ed69ed, /tmp/memra-server-c2ed69ed) vs the FULL binary — byte-identical
#             (greedy + seeded-sampled), the v1 lane's 6/6 protocol.
#   Phase B — CONSTRAINED CORRECTNESS on the FULL binary: json_object/json_schema parse +
#             validate on EVERY path: spec burst (default), plain batched (SPEC=0),
#             graphed (SPEC=0 + GS_MIN=32 + long budget), host oracle
#             (MEMRA_CONSTRAIN_HOST=1), sampled (temp>0, plain path).
#   Phase C — PERF, three-way x {plain, spec}: unconstrained vs constrained-v1-path
#             (MEMRA_CONSTRAIN_HOST=1 approximates v1: host mask + no device sample; v1
#             also lost graph — cited from merged receipts 117 vs 194) vs constrained-full.
#             N=3 interleaved same-session per arm.
# GPU serialized via flock /tmp/gpu5090.lock (call site), shared rig.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
ADDR=127.0.0.1:8195
BASE=http://$ADDR
OUT=research/constrained-full-20260803
BASELINE_BIN=/tmp/memra-server-c2ed69ed
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

start_server() { # $1 = binary, $2 = log, rest = extra env (VAR=val ...)
  local bin=$1 log=$2; shift 2
  env MEMRA_COMPAT=openai MEMRA_MODELS="q9=$MODEL" MEMRA_ADDR=$ADDR "$@" "$bin" > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 150); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up; log tail:"; tail -5 "$log"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; }
trap stop_server EXIT

req() { # $1=prompt $2=max_tokens $3=temperature $4=seed $5=extra-json ("" for none)
  local extra=""
  [ -n "$5" ] && extra=",$5"
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"q9\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\
\"max_tokens\":$2,\"temperature\":$3,\"seed\":$4$extra}"
}

content() { python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"], end="")'; }
fullmsg() { python3 -c '
import sys,json
r=json.load(sys.stdin); m=r["choices"][0]["message"]
print(json.dumps({"reasoning": m.get("reasoning"), "content": m["content"],
                  "n": r["usage"]["completion_tokens"]}, sort_keys=True))'; }

SCHEMA='{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0},"tags":{"type":"array","items":{"type":"string"},"minItems":2}},"required":["name","age","tags"],"additionalProperties":false}'
RF_SCHEMA="\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"person\",\"schema\":$SCHEMA}}"
RF_OBJ='"response_format":{"type":"json_object"}'

check_json_obj() { # $1=file $2=label
  if python3 -c '
import json,sys
v=json.load(open(sys.argv[1]))
assert isinstance(v,dict), f"not an object: {type(v)}"
' "$1"; then PASS "$2 parses as object"; else FAIL "$2 invalid: $(head -c 160 "$1")"; fi
}
check_schema() { # $1=file $2=label
  if python3 -c "
import json,sys,jsonschema
v=json.load(open(sys.argv[1]))
jsonschema.validate(v, json.loads('$SCHEMA'))
" "$1"; then PASS "$2 parses AND validates"; else FAIL "$2 invalid: $(head -c 160 "$1")"; fi
}

# ---------- Phase A: no-op exactness (baseline v1-HEAD vs full binary, unconstrained) ----------
echo "== Phase A: unconstrained byte-identity (c2ed69ed baseline vs constrained-full) =="
PROMPTS=("Explain PCIe lanes in two sentences." "List three prime numbers." "What is a mutex?")
for side in baseline new; do
  BIN=$([ $side = baseline ] && echo $BASELINE_BIN || echo target/release/memra-server)
  start_server "$BIN" "/tmp/cfull-$side.log" || exit 1
  for i in 0 1 2; do
    req "${PROMPTS[$i]}" 96 0 0 ""            | fullmsg > "$OUT/exact-$side-greedy-$i.txt"
    req "${PROMPTS[$i]}" 96 0.8 42 ""         | fullmsg > "$OUT/exact-$side-temp-$i.txt"
  done
  stop_server
done
for i in 0 1 2; do
  for kind in greedy temp; do
    if cmp -s "$OUT/exact-baseline-$kind-$i.txt" "$OUT/exact-new-$kind-$i.txt"; then
      PASS "unconstrained $kind #$i byte-identical"
    else
      FAIL "unconstrained $kind #$i DIFFERS"
    fi
  done
done

# ---------- Phase B: constrained correctness on EVERY path ----------
echo "== Phase B1: spec path (default env) =="
start_server target/release/memra-server /tmp/cfull-spec.log || exit 1
req "Describe a dog as a small JSON object with three keys." 512 0 0 "$RF_OBJ" \
  | content > "$OUT/b1-spec-obj.txt"
check_json_obj "$OUT/b1-spec-obj.txt" "spec json_object"
req "Invent a person named Rex, age 7, with hobby tags. Reply as JSON." 220 0 0 "$RF_SCHEMA" \
  | content > "$OUT/b1-spec-schema.txt"
check_schema "$OUT/b1-spec-schema.txt" "spec json_schema"
grep -c "spec-acc" /tmp/cfull-spec.log >/dev/null && PASS "spec bursts confirmed in log" \
  || FAIL "no spec bursts in log (constrained did not ride spec)"
stop_server

echo "== Phase B2: plain batched path (MEMRA_SERVE_SPEC=0) + sampled =="
start_server target/release/memra-server /tmp/cfull-plain.log MEMRA_SERVE_SPEC=0 || exit 1
req "Describe a dog as a small JSON object with three keys." 512 0 0 "$RF_OBJ" \
  | content > "$OUT/b2-plain-obj.txt"
check_json_obj "$OUT/b2-plain-obj.txt" "plain json_object"
req "Invent a person named Rex, age 7, with hobby tags. Reply as JSON." 220 0 0 "$RF_SCHEMA" \
  | content > "$OUT/b2-plain-schema.txt"
check_schema "$OUT/b2-plain-schema.txt" "plain json_schema"
req "Invent a person. Reply as JSON." 220 0.8 7 "$RF_SCHEMA" \
  | content > "$OUT/b2-sampled-schema.txt"
check_schema "$OUT/b2-sampled-schema.txt" "sampled json_schema (device gumbel + mask)"
stop_server

echo "== Phase B3: graphed path (SPEC=0, GS_MIN=32) =="
# Promotion engages on prefill-done-at-admit sessions (prefix-cache hit / continuation
# resume) — cold single requests decode eager-batched on THIS codebase (baseline binary
# behaves identically; verified 2026-08-03). Two same-prefix requests: request 1 seeds
# the prefix cache, request 2 admits prefill-done and promotes WITH the grammar mask.
start_server target/release/memra-server /tmp/cfull-graph.log MEMRA_SERVE_SPEC=0 MEMRA_GS_MIN=32 MEMRA_GRAPH_CENSUS=1 || exit 1
LONGSYS="You are a meticulous data-entry assistant for a veterinary clinic. Always answer with well-formed structured data. Never add commentary, markdown fences, or explanations. The clinic records include species, breed, age in whole years, weight in kilograms, vaccination status, and a list of behavioral tags observed during visits. Accuracy matters more than speed."
graph_req() {
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"q9\",\"messages\":[{\"role\":\"system\",\"content\":\"$LONGSYS\"},{\"role\":\"user\",\"content\":\"Invent a person named Rex, age 7, with hobby tags. Reply as JSON.\"}],\"max_tokens\":220,\"temperature\":0,\"seed\":0,$RF_SCHEMA}"
}
graph_req | content > "$OUT/b3-graph-schema-r1.txt"   # seeds the prefix cache (eager)
graph_req | content > "$OUT/b3-graph-schema.txt"      # prefix hit -> graph promotion
check_schema "$OUT/b3-graph-schema.txt" "graphed json_schema"
grep -q "graph-census" /tmp/cfull-graph.log && PASS "graph capture confirmed in log" \
  || FAIL "no graph capture in log (constrained did not promote)"
if cmp -s "$OUT/b3-graph-schema-r1.txt" "$OUT/b3-graph-schema.txt"; then
  PASS "graphed constrained == eager constrained (byte-identical, same prompt)"
else
  FAIL "graphed vs eager constrained DIFFER"
fi
stop_server

echo "== Phase B4: host oracle (MEMRA_CONSTRAIN_HOST=1, SPEC=0) =="
start_server target/release/memra-server /tmp/cfull-host.log MEMRA_SERVE_SPEC=0 MEMRA_CONSTRAIN_HOST=1 || exit 1
req "Invent a person named Rex, age 7, with hobby tags. Reply as JSON." 220 0 0 "$RF_SCHEMA" \
  | content > "$OUT/b4-host-schema.txt"
check_schema "$OUT/b4-host-schema.txt" "host-oracle json_schema"
stop_server
# device vs host-oracle greedy identity (same grammar, same prompt)
if cmp -s "$OUT/b2-plain-schema.txt" "$OUT/b4-host-schema.txt"; then
  PASS "device-mask greedy == host-oracle greedy (byte-identical)"
else
  FAIL "device-mask vs host-oracle greedy DIFFER"
fi
# spec vs plain constrained identity
if cmp -s "$OUT/b1-spec-schema.txt" "$OUT/b2-plain-schema.txt"; then
  PASS "spec constrained == plain constrained (byte-identical)"
else
  FAIL "spec vs plain constrained DIFFER"
fi

# ---------- Phase C: three-way perf, plain AND spec ----------
echo "== Phase C: perf (N=3 interleaved same-session per arm) =="
PROMPT="Write a long JSON object describing a spacecraft with at least ten keys."
perf_arm() { # $1=label $2=extra-json -> prints per-run rows, appends jsonl
  local t0 t1 n
  for k in 1 2 3; do
    t0=$(date +%s.%N)
    resp=$(req "$PROMPT" 256 0 0 "$2")
    t1=$(date +%s.%N)
    n=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["usage"]["completion_tokens"])')
    python3 -c "
import json
dt=$t1-$t0; n=$n
row={'arm':'$1','run':$k,'tokens':n,'wall_s':round(dt,3),'tok_s':round(n/dt,1)}
print(f\"  $1 run$k: {n} tok in {dt:.2f}s = {n/dt:.1f} tok/s\")
open('$OUT/perf.jsonl','a').write(json.dumps(row)+'\n')
"
  done
}
: > "$OUT/perf.jsonl"

echo "-- plain decode (MEMRA_SERVE_SPEC=0) --"
start_server target/release/memra-server /tmp/cfull-perf-plain.log MEMRA_SERVE_SPEC=0 || exit 1
perf_arm "plain-unconstrained" ""
stop_server
# v1-approx arm: host mask + host sample + full-row D2H (v1 additionally lost graph;
# graph is off in this SPEC=0 eager comparison anyway, so this isolates the mask path).
start_server target/release/memra-server /tmp/cfull-perf-host.log MEMRA_SERVE_SPEC=0 MEMRA_CONSTRAIN_HOST=1 || exit 1
perf_arm "plain-constr-v1host" "$RF_OBJ"
stop_server
start_server target/release/memra-server /tmp/cfull-perf-full.log MEMRA_SERVE_SPEC=0 || exit 1
perf_arm "plain-constr-full" "$RF_OBJ"
stop_server

echo "-- spec decode (default env) --"
start_server target/release/memra-server /tmp/cfull-perf-spec.log || exit 1
perf_arm "spec-unconstrained" ""
perf_arm "spec-constr-full" "$RF_OBJ"
stop_server

grep '\[constrained\]' /tmp/cfull-perf-full.log /tmp/cfull-perf-spec.log 2>/dev/null | tail -8
grep 'spec-acc' /tmp/cfull-perf-spec.log 2>/dev/null | tail -6

cp /tmp/cfull-spec.log "$OUT/serve-spec.log" 2>/dev/null || true
cp /tmp/cfull-perf-spec.log "$OUT/serve-perf-spec.log" 2>/dev/null || true
echo; echo "battery: $FAILS failure(s)"
exit $((FAILS > 0))

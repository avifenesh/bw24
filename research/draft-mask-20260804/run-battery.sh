#!/usr/bin/env bash
# DRAFT-SIDE GRAMMAR MASKING battery (lane/draft-mask, 2026-08-04).
#   Phase A — DRAFT-MASK EXACTNESS: masking ON vs OFF (MEMRA_DRAFT_MASK=0), SAME binary.
#             The mask changes which tokens get PROPOSED; verify-side truncation + target
#             sampling decide what's EMITTED, so the emitted stream must be BYTE-IDENTICAL
#             for greedy AND seeded-sampled, json_object AND json_schema.
#   Phase B — UNCONSTRAINED REGRESSION: 6/6 byte-identical vs the pre-lane binary (the
#             standard protocol from research/constrained-full-20260803).
#   Phase C — CONSTRAINED CORRECTNESS: schema/object outputs still parse+validate with
#             masking on (the mask must never produce illegal JSON).
#   Phase D — PERF: bounded tight schema (the GATE cell) + unbounded tight + json_object +
#             loose control, ON vs OFF, N=3 same-session per arm; acceptance + tok/s.
# Not gated: the UNBOUNDED tight cell's byte identity — its degenerate whitespace tail is
# draft-chain-SHAPE dependent in the PRE-LANE binary as well (probe-shape.sh). Reported as info.
# GPU serialized via flock /tmp/gpu5090.lock (call site), shared rig.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8197
BASE=http://$ADDR
OUT=research/draft-mask-20260804
BASELINE_BIN=${BASELINE_BIN:-/tmp/memra-server-prelane-draftmask}
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

start_server() { # $1 = binary, $2 = log, rest = extra env (VAR=val ...)
  local bin=$1 log=$2; shift 2
  env MEMRA_COMPAT=openai MEMRA_MODELS="q9=$MODEL+$DRAFT" MEMRA_ADDR=$ADDR "$@" "$bin" > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 150); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up; log tail:"; tail -5 "$log"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 1; }
trap stop_server EXIT

req() { # $1=prompt $2=max_tokens $3=temperature $4=seed $5=extra-json ("" for none)
  local extra=""
  [ -n "$5" ] && extra=",$5"
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"q9\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\
\"max_tokens\":$2,\"temperature\":$3,\"seed\":$4$extra}"
}
fullmsg() { python3 -c '
import sys,json
r=json.load(sys.stdin); m=r["choices"][0]["message"]
print(json.dumps({"reasoning": m.get("reasoning"), "content": m["content"],
                  "n": r["usage"]["completion_tokens"]}, sort_keys=True))'; }
content() { python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"], end="")'; }

SCHEMA='{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0},"tags":{"type":"array","items":{"type":"string"},"minItems":2}},"required":["name","age","tags"],"additionalProperties":false}'
RF_SCHEMA="\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"person\",\"schema\":$SCHEMA}}"
RF_OBJ='"response_format":{"type":"json_object"}'
# OBJ cell: the merged battery's perf prompt (json_object, long output). NOTE (measured
# 2026-08-04): json_object is NOT a tight grammar for this drafter — masking OFF logs
# gram_cuts=0/13, i.e. the drafter already proposes legal tokens. Kept as the merged-battery
# comparison point, not as the tight cell.
TIGHT="Write a long JSON object describing a spacecraft with at least ten keys."
# LOOSE control: the Rex cell — free-form prose under json_object.
LOOSE="Explain in three sentences how a Rex-class rocket engine gimbal works."
# TIGHT SCHEMA cell (the exactness GATE cell): the regime where verify-side truncation actually
# fires — masking OFF logs gram_cuts 3/12, 3/15, 1/10, 1/25 on this cell; masking ON logs 0/N in
# every round. An array of fully-required objects keeps the legal set near-singleton.
# BOUNDED ON PURPOSE (minItems==maxItems, tags maxItems): the model closes the JSON and the
# request ends at finish_reason=stop, ~241 tokens inside the 320 budget, so the run never enters
# the degenerate unbounded-whitespace tail. See the UNBOUNDED cell below for why that matters.
F3_SCHEMA='{"type":"object","properties":{"fleet":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"object","properties":{"id":{"type":"string"},"class":{"type":"string","enum":["scout","hauler","interceptor"]},"crew":{"type":"integer","minimum":1},"mass_t":{"type":"integer","minimum":1},"active":{"type":"boolean"},"tags":{"type":"array","minItems":2,"maxItems":2,"items":{"type":"string"}}},"required":["id","class","crew","mass_t","active","tags"],"additionalProperties":false}}},"required":["fleet"],"additionalProperties":false}'
RF_F3="\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"fleet3\",\"schema\":$F3_SCHEMA}}"
F3S="List three spacecraft in a fleet with their class, crew, mass and tags."
# UNBOUNDED TIGHT cell — INFORMATIONAL, NOT A GATE. minItems=6 with no maxItems anywhere: the
# 9B runs out of distinct fleet entries around char 600, then degenerates into unbounded
# whitespace (the JSON grammar permits arbitrary whitespace between tokens) and rides the 400-tok
# cap. That tail sits in a near-tie logit regime, and spec.rs:2151 documents that the verify batch
# shape T "changes FP summation order and can flip argmax at tight logit margins" — so the tail is
# draft-CHAIN-SHAPE dependent, not draft-MASK dependent. probe-shape.sh proves it on the PRE-LANE
# binary (no draft-mask code at all): K3 != K2 != K1 on this cell, all three diverging at the SAME
# char 603, while at fixed K=1 pre-lane == mask-ON byte-identical. Reported here, never gated.
FLEET_SCHEMA='{"type":"object","properties":{"fleet":{"type":"array","minItems":6,"items":{"type":"object","properties":{"id":{"type":"string"},"class":{"type":"string","enum":["scout","hauler","interceptor"]},"crew":{"type":"integer","minimum":1},"mass_t":{"type":"integer","minimum":1},"active":{"type":"boolean"},"tags":{"type":"array","minItems":2,"items":{"type":"string"}}},"required":["id","class","crew","mass_t","active","tags"],"additionalProperties":false}}},"required":["fleet"],"additionalProperties":false}'
RF_FLEET="\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"fleet\",\"schema\":$FLEET_SCHEMA}}"
TIGHTS="List six spacecraft in a fleet with their class, crew, mass and tags."

check_json_obj() { # $1=file $2=label
  if python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); assert isinstance(v,dict)' "$1"
  then PASS "$2 parses as object"; else FAIL "$2 invalid: $(head -c 160 "$1")"; fi
}
check_schema() { # $1=file $2=label
  if python3 -c "
import json,sys,jsonschema
jsonschema.validate(json.load(open(sys.argv[1])), json.loads('$SCHEMA'))" "$1"
  then PASS "$2 parses AND validates"; else FAIL "$2 invalid: $(head -c 160 "$1")"; fi
}

# ---------- Phase A: draft-mask ON vs OFF emitted-stream identity ----------
echo "== Phase A: draft-mask ON vs OFF byte-identity (constrained, spec path) =="
for arm in on off; do
  ENVX=(); [ $arm = off ] && ENVX=(MEMRA_DRAFT_MASK=0)
  start_server target/release/memra-server "/tmp/dm-$arm.log" "${ENVX[@]}" || exit 1
  req "$TIGHT" 256 0 0 "$RF_OBJ"        | fullmsg > "$OUT/dm-$arm-obj-greedy.txt"
  req "Give me a person record." 128 0 0 "$RF_SCHEMA"   | fullmsg > "$OUT/dm-$arm-schema-greedy.txt"
  # seeded-sampled CONSTRAINED (worker routes sampled constrained to plain decode — the
  # draft mask must not perturb it either).
  req "Give me a person record." 128 0.8 42 "$RF_SCHEMA" | fullmsg > "$OUT/dm-$arm-schema-temp.txt"
  # constrained on the LOOSE prompt (json_object over a prose-ish request)
  req "$LOOSE" 192 0 0 "$RF_OBJ"        | fullmsg > "$OUT/dm-$arm-loose-obj.txt"
  # TIGHT SCHEMA (GATE) — the cell where truncation actually fires and the JSON still closes.
  req "$F3S" 320 0 0 "$RF_F3"           | fullmsg > "$OUT/dm-$arm-tight3.txt"
  req "$F3S" 320 0.8 42 "$RF_F3"        | fullmsg > "$OUT/dm-$arm-tight3-temp.txt"
  # UNBOUNDED TIGHT (INFO only — degenerate whitespace tail, verify-shape dependent, see above).
  req "$TIGHTS" 400 0 0 "$RF_FLEET"     | fullmsg > "$OUT/dm-$arm-tightschema.txt"
  req "$TIGHTS" 400 0.8 42 "$RF_FLEET"  | fullmsg > "$OUT/dm-$arm-tightschema-temp.txt"
  cp "/tmp/dm-$arm.log" "$OUT/serve-dm-$arm.log"
  stop_server
done
for cell in obj-greedy schema-greedy schema-temp loose-obj tight3 tight3-temp tightschema-temp; do
  if cmp -s "$OUT/dm-on-$cell.txt" "$OUT/dm-off-$cell.txt"; then
    PASS "draft-mask ON == OFF: $cell (byte-identical emitted stream)"
  else
    FAIL "draft-mask ON != OFF: $cell"
  fi
done
# INFO cell: report, never gate (draft-chain-shape dependent in the PRE-LANE binary too).
if cmp -s "$OUT/dm-on-tightschema.txt" "$OUT/dm-off-tightschema.txt"; then
  echo "  info: tightschema (unbounded) ON == OFF"
else
  echo "  info: tightschema (unbounded) ON != OFF — degenerate whitespace tail, expected;"
  echo "        probe-shape.sh shows pre-lane K3!=K2!=K1 on this same cell (no mask code)."
fi
echo "  -- gram_cuts per session, ON vs OFF (the mechanism receipt) --"
for a in on off; do
  echo -n "  $a: "; grep -o 'gram_cuts=[0-9]*/[0-9]*' "$OUT/serve-dm-$a.log" | tr '\n' ' '; echo
done

# ---------- Phase B: unconstrained 6/6 vs the pre-lane binary ----------
echo "== Phase B: unconstrained byte-identity vs pre-lane binary =="
if [ ! -x "$BASELINE_BIN" ]; then
  FAIL "baseline binary $BASELINE_BIN missing (build pre-lane HEAD first)"
else
  PROMPTS=("Explain PCIe lanes in two sentences." "List three prime numbers." "What is a mutex?")
  for side in baseline new; do
    BIN=$([ $side = baseline ] && echo "$BASELINE_BIN" || echo target/release/memra-server)
    start_server "$BIN" "/tmp/dm-unc-$side.log" || exit 1
    for i in 0 1 2; do
      req "${PROMPTS[$i]}" 96 0 0 ""    | fullmsg > "$OUT/unc-$side-greedy-$i.txt"
      req "${PROMPTS[$i]}" 96 0.8 42 "" | fullmsg > "$OUT/unc-$side-temp-$i.txt"
    done
    stop_server
  done
  for i in 0 1 2; do
    for m in greedy temp; do
      if cmp -s "$OUT/unc-baseline-$m-$i.txt" "$OUT/unc-new-$m-$i.txt"; then
        PASS "unconstrained $m-$i byte-identical"
      else
        FAIL "unconstrained $m-$i DIFFERS"
      fi
    done
  done
fi

# ---------- Phase C: constrained correctness with masking ON ----------
echo "== Phase C: constrained correctness (masking ON) =="
start_server target/release/memra-server /tmp/dm-corr.log || exit 1
req "$TIGHT" 256 0 0 "$RF_OBJ" | content > "$OUT/c-obj.json"
req "Give me a person record." 128 0 0 "$RF_SCHEMA" | content > "$OUT/c-schema.json"
stop_server
check_json_obj "$OUT/c-obj.json" "json_object (mask ON, spec)"
check_schema   "$OUT/c-schema.json" "json_schema (mask ON, spec)"

# ---------- Phase D: perf, tight + loose, ON vs OFF ----------
echo "== Phase D: perf (N=3 interleaved same-session per arm) =="
: > "$OUT/perf.jsonl"
perf_arm() { # $1=label $2=prompt $3=extra-json
  local t0 t1 n resp
  for k in 1 2 3; do
    t0=$(date +%s.%N)
    resp=$(req "$2" 256 0 0 "$3")
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
for arm in on off; do
  ENVX=(); [ $arm = off ] && ENVX=(MEMRA_DRAFT_MASK=0)
  start_server target/release/memra-server "/tmp/dm-perf-$arm.log" "${ENVX[@]}" || exit 1
  perf_arm "tight3-constr-mask$arm" "$F3S" "$RF_F3"
  perf_arm "tightschema-constr-mask$arm" "$TIGHTS" "$RF_FLEET"
  perf_arm "obj-constr-mask$arm" "$TIGHT" "$RF_OBJ"
  perf_arm "loose-constr-mask$arm" "$LOOSE" "$RF_OBJ"
  perf_arm "unconstrained-mask$arm" "$TIGHT" ""
  cp "/tmp/dm-perf-$arm.log" "$OUT/serve-perf-$arm.log"
  stop_server
done
echo "-- acceptance (last spec-acc cum per arm; per-request rows are in the serve logs) --"
for a in on off; do echo -n "  $a: "; grep 'spec-acc' "$OUT/serve-perf-$a.log" | tail -1; done
echo "-- gram_cuts per request, in Phase D arm order (tight3, tightschema, obj, loose, unconstr) --"
for a in on off; do
  echo -n "  $a: "; grep -o 'gram_cuts=[0-9]*/[0-9]*' "$OUT/serve-perf-$a.log" | tr '\n' ' '; echo
done
echo "-- clone cost --"
grep 'clone_per_round' "$OUT/serve-perf-on.log" | tail -3
grep '^\[draft-mask\] [a-z]' "$OUT/serve-perf-on.log" | tail -3
echo "-- tok/s medians --"
python3 -c "
import json,statistics,collections
rows=[json.loads(l) for l in open('$OUT/perf.jsonl')]
by=collections.defaultdict(list)
for r in rows: by[r['arm']].append(r['tok_s'])
for a in sorted(by): print(f'  {a:34s} N={len(by[a])} med={statistics.median(by[a]):7.1f} tok/s  {by[a]}')
"

echo; echo "battery: $FAILS failure(s)"
exit $((FAILS > 0))

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
#   Phase D — PERF: tight schema (the "spacecraft, ten keys" json_object cell) + loose
#             control, ON vs OFF, N=3 interleaved same-session per arm; acceptance + tok/s.
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
# TIGHT cell: the merged battery's perf prompt (json_object, long output).
TIGHT="Write a long JSON object describing a spacecraft with at least ten keys."
# LOOSE control: the Rex cell — free-form prose, no grammar.
LOOSE="Explain in three sentences how a Rex-class rocket engine gimbal works."

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
  cp "/tmp/dm-$arm.log" "$OUT/serve-dm-$arm.log"
  stop_server
done
for cell in obj-greedy schema-greedy schema-temp loose-obj; do
  if cmp -s "$OUT/dm-on-$cell.txt" "$OUT/dm-off-$cell.txt"; then
    PASS "draft-mask ON == OFF: $cell (byte-identical emitted stream)"
  else
    FAIL "draft-mask ON != OFF: $cell"
  fi
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
  perf_arm "tight-constr-mask$arm" "$TIGHT" "$RF_OBJ"
  perf_arm "loose-constr-mask$arm" "$LOOSE" "$RF_OBJ"
  perf_arm "unconstrained-mask$arm" "$TIGHT" ""
  cp "/tmp/dm-perf-$arm.log" "$OUT/serve-perf-$arm.log"
  stop_server
done
echo "-- acceptance --"
grep 'spec-acc' "$OUT/serve-perf-on.log"  | tail -6
grep 'spec-acc' "$OUT/serve-perf-off.log" | tail -6
echo "-- clone cost --"
grep 'draft-mask' "$OUT/serve-perf-on.log" | tail -6

echo; echo "battery: $FAILS failure(s)"
exit $((FAILS > 0))

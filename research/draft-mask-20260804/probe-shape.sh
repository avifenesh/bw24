#!/usr/bin/env bash
# CONTROL PROBE (lane/draft-mask, 2026-08-04): is the constrained-spec emitted stream on the
# tightschema greedy cell invariant to DRAFT-CHAIN SHAPE, independent of this lane?
#
# The lane's exactness bar is "draft-mask ON == OFF, byte-identical". Phase A's tightschema cell
# fails it. Both arms end up in a degenerate unbounded-whitespace tail (the JSON grammar allows
# arbitrary whitespace), i.e. the near-tie regime where spec.rs:2151 documents that verify batch
# shape T "changes FP summation order and can flip argmax at tight logit margins".
#
# This probe changes draft-chain shape WITHOUT the lane: it runs the PRE-LANE binary (which has
# no draft-mask code at all) at MEMRA_SPEC_K=3 (default), 2 and 1. If the pre-lane outputs
# already differ across K on this cell, then the emitted stream is not shape-invariant here and
# ON-vs-OFF byte-identity is unachievable through a shape-varying verify — a property of the
# cell, not of draft masking. The same sweep on the LANE binary (mask ON) is the twin.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8198
BASE=http://$ADDR
OUT=research/draft-mask-20260804
PRELANE=${BASELINE_BIN:-/tmp/memra-server-prelane-draftmask}

FLEET_SCHEMA='{"type":"object","properties":{"fleet":{"type":"array","minItems":6,"items":{"type":"object","properties":{"id":{"type":"string"},"class":{"type":"string","enum":["scout","hauler","interceptor"]},"crew":{"type":"integer","minimum":1},"mass_t":{"type":"integer","minimum":1},"active":{"type":"boolean"},"tags":{"type":"array","minItems":2,"items":{"type":"string"}}},"required":["id","class","crew","mass_t","active","tags"],"additionalProperties":false}}},"required":["fleet"],"additionalProperties":false}'
RF_FLEET="\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"fleet\",\"schema\":$FLEET_SCHEMA}}"
TIGHTS="List six spacecraft in a fleet with their class, crew, mass and tags."

start_server() { local bin=$1 log=$2; shift 2
  env MEMRA_COMPAT=openai MEMRA_MODELS="q9=$MODEL+$DRAFT" MEMRA_ADDR=$ADDR "$@" "$bin" > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 150); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up"; tail -5 "$log"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 1; }
trap stop_server EXIT
fullmsg() { python3 -c '
import sys,json
r=json.load(sys.stdin); m=r["choices"][0]["message"]
print(json.dumps({"reasoning": m.get("reasoning"), "content": m["content"],
                  "n": r["usage"]["completion_tokens"]}, sort_keys=True))'; }
ask() { curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"q9\",\"messages\":[{\"role\":\"user\",\"content\":\"$TIGHTS\"}],\
\"max_tokens\":400,\"temperature\":0,\"seed\":0,$RF_FLEET}"; }

for kk in 3 2 1; do
  start_server "$PRELANE" "/tmp/dm-probe-prelane-k$kk.log" MEMRA_SPEC_K=$kk || exit 1
  ask | fullmsg > "$OUT/probe-prelane-k$kk.txt"
  cp "/tmp/dm-probe-prelane-k$kk.log" "$OUT/probe-prelane-k$kk.serve.log"
  stop_server
done
for kk in 3 2 1; do
  start_server target/release/memra-server "/tmp/dm-probe-mask-k$kk.log" MEMRA_SPEC_K=$kk || exit 1
  ask | fullmsg > "$OUT/probe-mask-k$kk.txt"
  cp "/tmp/dm-probe-mask-k$kk.log" "$OUT/probe-mask-k$kk.serve.log"
  stop_server
done

echo "== pre-lane binary, tightschema greedy, K sweep (no draft-mask code in this binary) =="
for kk in 2 1; do
  cmp -s "$OUT/probe-prelane-k3.txt" "$OUT/probe-prelane-k$kk.txt" \
    && echo "  K3 == K$kk (shape-invariant)" || echo "  K3 != K$kk (SHAPE-DEPENDENT, pre-lane)"
done
echo "== lane binary, draft-mask ON, same sweep =="
for kk in 2 1; do
  cmp -s "$OUT/probe-mask-k3.txt" "$OUT/probe-mask-k$kk.txt" \
    && echo "  K3 == K$kk (shape-invariant)" || echo "  K3 != K$kk (SHAPE-DEPENDENT, mask ON)"
done
echo "== K=1 cross-arm (shortest chain: pre-lane vs mask ON) =="
cmp -s "$OUT/probe-prelane-k1.txt" "$OUT/probe-mask-k1.txt" \
  && echo "  prelane-K1 == mask-K1" || echo "  prelane-K1 != mask-K1"

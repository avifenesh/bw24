#!/usr/bin/env bash
# Tight-cell calibration (lane/draft-mask, 2026-08-04): find a tight-grammar cell that
#   (a) fires the verify-side grammar cut with masking OFF (gram_cuts > 0), i.e. is genuinely
#       tight for this drafter, AND
#   (b) COMPLETES its JSON inside the token budget, so the run never enters the degenerate
#       unbounded-whitespace tail where the emitted stream is verify-shape dependent even in
#       the pre-lane binary (see probe-shape.sh: pre-lane K3 != K2 != K1 on the 6-item cell).
# Reports ON-vs-OFF byte identity + the per-arm gram_cuts for each candidate.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
OUT=research/draft-mask-20260804

F3='{"type":"object","properties":{"fleet":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"object","properties":{"id":{"type":"string"},"class":{"type":"string","enum":["scout","hauler","interceptor"]},"crew":{"type":"integer","minimum":1},"mass_t":{"type":"integer","minimum":1},"active":{"type":"boolean"},"tags":{"type":"array","minItems":2,"maxItems":2,"items":{"type":"string"}}},"required":["id","class","crew","mass_t","active","tags"],"additionalProperties":false}}},"required":["fleet"],"additionalProperties":false}'
RF3="\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"fleet3\",\"schema\":$F3}}"
P3="List three spacecraft in a fleet with their class, crew, mass and tags."

start_server() { local log=$1; shift
  env MEMRA_COMPAT=openai MEMRA_MODELS="q9=$MODEL+$DRAFT" MEMRA_ADDR=$ADDR "$@" \
    target/release/memra-server > "$log" 2>&1 &
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
                  "n": r["usage"]["completion_tokens"], "fin": r["choices"][0]["finish_reason"]},
                 sort_keys=True))'; }

for arm in on off; do
  ENVX=(); [ $arm = off ] && ENVX=(MEMRA_DRAFT_MASK=0)
  start_server "/tmp/dm-t3-$arm.log" "${ENVX[@]}" || exit 1
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"q9\",\"messages\":[{\"role\":\"user\",\"content\":\"$P3\"}],\
\"max_tokens\":320,\"temperature\":0,\"seed\":0,$RF3}" | fullmsg > "$OUT/probe-t3-$arm.txt"
  curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"q9\",\"messages\":[{\"role\":\"user\",\"content\":\"$P3\"}],\
\"max_tokens\":320,\"temperature\":0.8,\"seed\":42,$RF3}" | fullmsg > "$OUT/probe-t3-$arm-temp.txt"
  cp "/tmp/dm-t3-$arm.log" "$OUT/probe-t3-$arm.serve.log"
  stop_server
done

echo "== fleet3 (minItems=maxItems=3, tags maxItems=2), 320 tok =="
for m in "" "-temp"; do
  cmp -s "$OUT/probe-t3-on$m.txt" "$OUT/probe-t3-off$m.txt" \
    && echo "  identity$m: ON == OFF" || echo "  identity$m: ON != OFF"
done
python3 -c "
import json
for a in ('on','off'):
    for m in ('','-temp'):
        d=json.load(open('$OUT/probe-t3-'+a+m+'.txt'))
        print(f'  {a}{m}: n={d[\"n\"]} fin={d[\"fin\"]} len={len(d[\"content\"])}')
"
echo "  -- gram_cuts (verify-side cut rounds / total rounds) --"
for a in on off; do echo -n "  $a: "; grep -o 'gram_cuts=[0-9]*/[0-9]*' "$OUT/probe-t3-$a.serve.log" | tr '\n' ' '; echo; done

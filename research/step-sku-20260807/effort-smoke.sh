#!/usr/bin/env bash
# effort-smoke: serve-smoke proving the reasoning_effort surface changes behavior on step35.
#
# Boots memra-server with the Step-3.7-Flash trunk + MTP drafter over PP-2 (spec OFF per the
# #87 quarantine — the boot-and-serve shape arm G of lane/step-draft proved), then sends the
# SAME messages with reasoning_effort absent / low / medium / high plus the OpenRouter object
# form, and receipts:
#   1. usage.prompt_tokens moves when a level is supplied (the rendered system turn gains
#      'Reasoning: {level}\n\n' — with no client system turn, a whole system turn appears);
#   2. absent == the template default (lowest prompt_tokens, no Reasoning: line);
#   3. an invalid level 400s with the OpenAI error object;
#   4. [worker] caps line prints effort_levels=true for this model.
#
# Run ON THE BOX under flock (GPU window):
#   flock -w 1800 9 || exit; bash effort-smoke.sh 9>/tmp/memra-gpu.lock
set -u
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=~/step37/raw/effort-smoke-$TS.log
BIN=~/tokparity-memra/target/release/memra-server
TRUNK=$(ls ~/step37/models/step-3.7-flash/IQ4_XS/*00001-of-00003.gguf)
DRAFT=~/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
PORT=8091

exec > >(tee "$OUT") 2>&1
echo "=== effort-smoke $TS ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

MEMRA_MODELS="step35=${TRUNK}+${DRAFT}" MEMRA_SERVE_SPEC=0 \
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR=127.0.0.1:$PORT \
"$BIN" > ~/step37/raw/effort-smoke-server-$TS.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT

# wait for readiness (model load over PP-2 takes a while: 105 GB off page cache)
for i in $(seq 1 180); do
  sleep 5
  if curl -sf "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then echo "ready after ~$((i*5))s"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED"; tail -30 ~/step37/raw/effort-smoke-server-$TS.log; exit 1; fi
done

grep "template caps" ~/step37/raw/effort-smoke-server-$TS.log

req() { # req <label> <extra-json-fields>
  local label=$1 extra=$2
  local body
  body=$(python3 - "$extra" <<'PY'
import json, sys
extra = json.loads(sys.argv[1])
base = {"model": "step35",
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 8, "temperature": 0.0}
base.update(extra)
print(json.dumps(base))
PY
)
  echo "--- $label ---"
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
       -H 'Content-Type: application/json' -d "$body" \
    | python3 -c '
import json, sys
r = json.load(sys.stdin)
if "error" in r:
    print("ERROR:", json.dumps(r["error"]))
else:
    u = r.get("usage", {})
    c = r["choices"][0]["message"].get("content", "")
    print("prompt_tokens=%s completion_tokens=%s content=%r"
          % (u.get("prompt_tokens"), u.get("completion_tokens"), c))'
}

req "absent (template default)" '{}'
req "low"     '{"reasoning_effort": "low"}'
req "medium"  '{"reasoning_effort": "medium"}'
req "high"    '{"reasoning_effort": "high"}'
req "none -> clamps to low" '{"reasoning_effort": "none"}'
req "openrouter object form high" '{"reasoning": {"effort": "high"}}'
req "invalid level (expect 400)" '{"reasoning_effort": "extreme"}'

echo "=== smoke done; killing server ==="
kill $SRV; wait $SRV 2>/dev/null
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== end effort-smoke $TS ==="

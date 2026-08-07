#!/usr/bin/env bash
# think-smoke: serve-smoke proving the cross-model thinking surface changes behavior on the
# qwen ChatML class AND the gemma4 class (local 5090; step35 smoke = effort-smoke.sh on the
# box; hy3 has no local servable artifact — its arm is golden-pinned in chat.rs tests).
#
# Assertions per arch, all via usage.prompt_tokens deltas + response fields on ONE server:
#   qwen (thinking default ON):
#     absent == high (default IS thinking-on; identical prompt_tokens),
#     none -> prompt grows (closed <think>\n\n</think>\n\n tail = +3 tokens vs open),
#     reasoning field populated at default, none suppresses reasoning content.
#   gemma4 (thinking default OFF):
#     absent == none (default IS off), low/high -> prompt CHANGES (a <|think|> system turn
#     appears; with no client system turn that is a whole new turn).
#
# Run LOCALLY under the gpu lock: bash think-smoke.sh
set -uo pipefail
cd /home/avifenesh/projects/wt-stepsku
TS=$(date -u +%Y%m%dT%H%M%SZ)
RAW=research/step-sku-20260807/raw
LOG=$RAW/think-smoke-$TS.log
QWEN=$HOME/models/qwen3.5-9b-judge-q8_0.gguf
GEMMA=$HOME/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf
PORT=8094
BASE=http://127.0.0.1:$PORT

exec > >(tee "$LOG") 2>&1
echo "=== think-smoke $TS (local 5090) ==="
(
  flock -w 1800 9 || { echo "LOCK TIMEOUT"; exit 75; }
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  # MEMRA_SERVE_BATCH=0: gemma4 has NO batched decode arm (decode_batch.rs:553 asserts
  # non-gemma4, and the B=1 fast path also excludes gemma4), so a plain gemma4 chat on the
  # default batched scheduler panics the worker — a PRE-EXISTING serving gap this smoke
  # found (raw/think-smoke-20260807T093918Z.log), not a thinking-surface change. Legacy
  # round-robin serves both models; the thinking assertions are mode-independent.
  MEMRA_MODELS="qwen=$QWEN,gemma=$GEMMA" MEMRA_ADDR=127.0.0.1:$PORT MEMRA_SERVE_BATCH=0 \
    systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=48G --collect \
    ./target/release/memra-server > $RAW/think-smoke-server-$TS.log 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
  for i in $(seq 1 60); do
    sleep 3
    curl -sf "$BASE/readyz" >/dev/null 2>&1 && { echo "ready after ~$((i*3))s"; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -20 $RAW/think-smoke-server-$TS.log; exit 1; }
  done
  grep "template caps" $RAW/think-smoke-server-$TS.log

  req() { # req <model> <label> <extra-json>
    local model=$1 label=$2 extra=$3
    local body
    body=$(python3 - "$model" "$extra" <<'PY'
import json, sys
base = {"model": sys.argv[1],
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 24, "temperature": 0.0}
base.update(json.loads(sys.argv[2]))
print(json.dumps(base))
PY
)
    echo "--- $model / $label ---"
    curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" \
      | python3 -c '
import json, sys
r = json.load(sys.stdin)
if "error" in r:
    print("ERROR:", json.dumps(r["error"]))
else:
    u = r.get("usage", {})
    m = r["choices"][0]["message"]
    print("prompt_tokens=%s reasoning=%r content=%r"
          % (u.get("prompt_tokens"),
             (m.get("reasoning") or "")[:40], (m.get("content") or "")[:40]))'
  }

  echo; echo "########## qwen class (thinking default ON) ##########"
  req qwen "absent (default=thinking ON)" '{}'
  req qwen "high (same open think as default)" '{"reasoning_effort": "high"}'
  req qwen "low (thinking stays ON)" '{"reasoning_effort": "low"}'
  req qwen "none (thinking OFF, closed think block)" '{"reasoning_effort": "none"}'
  req qwen "openrouter enabled:false" '{"reasoning": {"enabled": false}}'

  echo; echo "########## gemma4 class (thinking default OFF) ##########"
  req gemma "absent (default=thinking OFF)" '{}'
  req gemma "none (same as default)" '{"reasoning_effort": "none"}'
  req gemma "low (thinking ON: <|think|> system turn appears)" '{"reasoning_effort": "low"}'
  req gemma "high (same ON shape)" '{"reasoning_effort": "high"}'

  kill $SRV; wait $SRV 2>/dev/null; trap - EXIT
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
) 9>/tmp/memra-gpu.lock
echo "=== end think-smoke $TS ==="

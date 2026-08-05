#!/usr/bin/env bash
# lane/accept-telemetry GPU validation: serve q9 NVFP4 + draft (spec), run requests,
# capture /metrics spec block + a usage.spec envelope. ONE short lock hold.
set -uo pipefail
cd /home/avifenesh/projects/wt-acctele
OUT=research/accept-telemetry-20260805
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8188
BASE=http://$ADDR
MEMRA_COMPAT=openai MEMRA_MODELS="q9=$MODEL+$DRAFT" MEMRA_ADDR=$ADDR \
  target/release/memra-server > "$OUT/serve.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && break; sleep 2; done
curl -sf $BASE/health >/dev/null || { echo "server did not come up"; tail -5 "$OUT/serve.log"; exit 1; }

echo "== baseline /metrics (no requests yet — spec block must be ABSENT) =="
curl -sf $BASE/metrics | python3 -m json.tool | tee "$OUT/metrics-baseline.json"

chat() { curl -sf -m 300 $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"q9\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":$3}"; }

echo "== requests: 3 greedy + 2 sampled (t=0.8), varied prompts =="
R1=$(chat "Explain what a mutex is in one sentence." 96 0)
R2=$(chat "Write a haiku about spring rain." 96 0)
R3=$(chat "List three prime numbers greater than 100, comma-separated." 96 0)
R4=$(chat "Describe a lighthouse at dusk in two sentences." 96 0.8)
R5=$(chat "Name three chemical elements and their symbols." 96 0.8)
echo "$R1" > "$OUT/resp1-greedy.json"
echo "$R4" > "$OUT/resp4-sampled.json"
echo "-- usage of request 1 (greedy):"
echo "$R1" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["usage"], indent=2))'
echo "-- usage of request 4 (sampled):"
echo "$R4" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["usage"], indent=2))'

sleep 1
echo "== /metrics after 5 spec requests =="
curl -sf $BASE/metrics | python3 -m json.tool | tee "$OUT/metrics-after.json"
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; trap - EXIT

echo "== serve-smoke (same lock hold — must stay 0 failed) =="
tools/serve-smoke.sh 2>&1 | tee "$OUT/serve-smoke.log" | tail -30
echo "== done =="

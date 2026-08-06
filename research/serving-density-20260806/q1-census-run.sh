#!/usr/bin/env bash
# Q1 empirical anchor: c=8 sessions sharing an identical ~4k system prefix + unique short
# tails against served q9 NVFP4 (+draft, spec default-ON) — measure the real VRAM footprint
# of 8 concurrent right-size-ladder sessions vs the post-load idle baseline, and capture the
# worker's own "observed session VRAM cost" line. The shared prefix is the dogfood ctx4k
# system+log text (research/memra-vs-llama-daily-20260805/prompts/ctx4k.txt shape).
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
OUT=research/serving-density-20260806/logs
ADDR=127.0.0.1:8178
BASE=http://$ADDR

MEMRA_COMPAT=openai MEMRA_CTX=8192 MEMRA_MODELS="q9=$MODEL+$DRAFT" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$OUT/q1-server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && break; sleep 2; done
curl -sf $BASE/health >/dev/null || { echo "server did not come up"; exit 1; }
sleep 3
IDLE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "idle-after-load memory.used: $IDLE MiB"

# VRAM sampler at 200ms during the burst
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done ) \
    > "$OUT/q1-vram-samples.txt" &
SAMPLER=$!

python3 research/serving-density-20260806/q1-clients.py --base $BASE \
    --prefix research/memra-vs-llama-daily-20260805/prompts/ctx4k.txt \
    --concurrency 8 --max-tokens 192 --out "$OUT/q1-requests.jsonl" \
    2>&1 | tee "$OUT/q1-clients.log"
RC=$?

kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
PEAK=$(sort -n "$OUT/q1-vram-samples.txt" | tail -1)
echo "peak memory.used during c=8: $PEAK MiB (delta over idle: $((PEAK-IDLE)) MiB)"
grep -E "observed session VRAM cost|right-sized|spec pool" "$OUT/q1-server.log" | head -20
curl -sf $BASE/metrics | python3 -m json.tool | head -30
exit $RC

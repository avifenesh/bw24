#!/usr/bin/env bash
# One-request smoke of each server phase before the full battery.
set -uo pipefail
RDIR=/home/avifenesh/projects/bw24/research/memra-vs-llama-daily-20260805
LOGS=$RDIR/logs
DIR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp
MODEL=$DIR/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf

exec 9>/tmp/gpu5090.lock
flock 9
echo "smoke start $(date -u +%FT%TZ)"

# --- memra ---
MEMRA_MODELS="qwen36-27b=$MODEL+$DIR/draft-daily-owntrim-nvfp4head-q4blk.gguf" \
MEMRA_ADDR="127.0.0.1:8002" MEMRA_API_KEY=aviary-local MEMRA_CTX=131072 \
MEMRA_MAX_SESSIONS=1 MEMRA_REUSE_POOL=1 MEMRA_PRIME_CHUNK=2048 \
/home/avifenesh/tmp-dogfood/memra-server-c716954b > "$LOGS/smoke-memra-server.log" 2>&1 &
MPID=$!
for i in $(seq 1 90); do
  curl -sf -m 2 -H "Authorization: Bearer aviary-local" http://127.0.0.1:8002/health >/dev/null 2>&1 && break
  sleep 2
done
curl -s -m 120 -H "Authorization: Bearer aviary-local" -H "Content-Type: application/json" \
  -d '{"model":"qwen36-27b","prompt":"<|im_start|>user\nSay OK.<|im_end|>\n<|im_start|>assistant\n","max_tokens":16,"temperature":0.8,"seed":42,"stream":true,"stream_options":{"include_usage":true}}' \
  http://127.0.0.1:8002/v1/completions > "$LOGS/smoke-memra-request.txt" 2>&1
echo "memra smoke rc=$?"
kill $MPID 2>/dev/null; sleep 3; kill -9 $MPID 2>/dev/null
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 3000 ] && break; sleep 2
done
echo "post-memra vram=$used"

# --- llama ---
/home/avifenesh/projects/llama.cpp/build/bin/llama-server \
  -m "$MODEL" --model-draft "$DIR/mtp-Qwen3.6-27B-Q4_K_M.gguf" \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1 \
  --alias qwen36-27b --ctx-size 131072 --ubatch-size 512 -ngl 999 -ngld 999 -fa on --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q5_1 --cache-ram 0 --jinja \
  --host 127.0.0.1 --port 8001 --api-key aviary-local --metrics \
  > "$LOGS/smoke-llama-server.log" 2>&1 &
LPID=$!
for i in $(seq 1 90); do
  curl -sf -m 2 http://127.0.0.1:8001/health >/dev/null 2>&1 && break
  sleep 2
done
curl -s -m 120 -H "Authorization: Bearer aviary-local" -H "Content-Type: application/json" \
  -d '{"model":"qwen36-27b","prompt":"<|im_start|>user\nSay OK.<|im_end|>\n<|im_start|>assistant\n","max_tokens":16,"seed":42,"stream":true,"stream_options":{"include_usage":true},"timings_per_token":true}' \
  http://127.0.0.1:8001/v1/completions > "$LOGS/smoke-llama-request.txt" 2>&1
echo "llama smoke rc=$?"
kill $LPID 2>/dev/null; sleep 3; kill -9 $LPID 2>/dev/null
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 3000 ] && break; sleep 2
done
echo "post-llama vram=$used"
flock -u 9
echo "smoke done $(date -u +%FT%TZ)"

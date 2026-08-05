#!/usr/bin/env bash
# Post-hoc confirm: memra + llama-shape truncation sampling (top_k 40, top_p 0.95,
# min_p 0.05, t=0.8) produced n=7 outputs on short-agentic in ALL 5 battery reps
# (identical 24-char length, seeds 1001-1005). Capture the actual TEXT across 4 seeds
# + the same request untruncated + greedy, to classify: seed-invariant collapse?
set -uo pipefail
RDIR=/home/avifenesh/projects/bw24/research/memra-vs-llama-daily-20260805
LOGS=$RDIR/logs
DIR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp

exec 9>/tmp/gpu5090.lock
flock 9
MEMRA_MODELS="qwen36-27b=$DIR/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf+$DIR/draft-daily-owntrim-nvfp4head-q4blk.gguf" \
MEMRA_ADDR="127.0.0.1:8002" MEMRA_API_KEY=aviary-local MEMRA_CTX=131072 \
MEMRA_MAX_SESSIONS=1 MEMRA_REUSE_POOL=1 MEMRA_PRIME_CHUNK=2048 \
/home/avifenesh/tmp-dogfood/memra-server-c716954b > "$LOGS/posthoc-memra-server.log" 2>&1 &
MPID=$!
for i in $(seq 1 90); do
  curl -sf -m 2 -H "Authorization: Bearer aviary-local" http://127.0.0.1:8002/health >/dev/null 2>&1 && break
  sleep 2
done

PROMPT=$(python3 -c "
import json
p = open('$RDIR/prompts/short-agentic.txt').read()
print(json.dumps(p), end='')")

out=$LOGS/posthoc-lsampler.txt
: > "$out"
for seed in 7 999 31337 424242; do
  echo "=== lsampler seed=$seed ===" >> "$out"
  curl -s -m 120 -H "Authorization: Bearer aviary-local" -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen36-27b\",\"prompt\":$PROMPT,\"max_tokens\":160,\"temperature\":0.8,\"top_k\":40,\"top_p\":0.95,\"min_p\":0.05,\"seed\":$seed}" \
    http://127.0.0.1:8002/v1/completions >> "$out" 2>&1
  echo >> "$out"
done
echo "=== untruncated t0.8 seed=7 ===" >> "$out"
curl -s -m 120 -H "Authorization: Bearer aviary-local" -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen36-27b\",\"prompt\":$PROMPT,\"max_tokens\":160,\"temperature\":0.8,\"seed\":7}" \
  http://127.0.0.1:8002/v1/completions >> "$out" 2>&1
echo >> "$out"
echo "=== greedy ===" >> "$out"
curl -s -m 120 -H "Authorization: Bearer aviary-local" -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen36-27b\",\"prompt\":$PROMPT,\"max_tokens\":160,\"temperature\":0}" \
  http://127.0.0.1:8002/v1/completions >> "$out" 2>&1
echo >> "$out"
# min_p alone / top_k alone / top_p alone — isolate which truncation knob causes it
for knob in '"top_k":40' '"top_p":0.95' '"min_p":0.05'; do
  echo "=== t0.8 seed=7 $knob only ===" >> "$out"
  curl -s -m 120 -H "Authorization: Bearer aviary-local" -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen36-27b\",\"prompt\":$PROMPT,\"max_tokens\":160,\"temperature\":0.8,$knob,\"seed\":7}" \
    http://127.0.0.1:8002/v1/completions >> "$out" 2>&1
  echo >> "$out"
done

kill $MPID 2>/dev/null; sleep 3; kill -9 $MPID 2>/dev/null
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 3000 ] && break; sleep 2
done
flock -u 9
echo "posthoc done vram=$used"

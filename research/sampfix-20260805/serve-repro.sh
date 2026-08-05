#!/usr/bin/env bash
# Gate (b): serve-level repro of the top_p/min_p id-0 injection bug, A/B on the SAME tree.
#
# Method mirrors research/memra-vs-llama-daily-20260805/scripts/posthoc-lsampler.sh exactly:
# same artifact, same daily serve env, same prompt (short-agentic), same seeds
# (7 / 999 / 31337 / 424242), same truncation shapes. Only the binary differs.
#
#   PREFIX = d1dc79b8~1 (col_stats.last() bonus-column stats)  -> expect "!" (id 0) injection
#   FIXED  = d1dc79b8   (filter_stats on the bonus column)     -> expect clean text
#
# The dogfood snapshot ~/tmp-dogfood/memra-server-c716954b is gone, so BOTH arms are built
# from this worktree: an A/B on one tree is the stronger control anyway (only the fix differs).
set -uo pipefail
RDIR=/home/avifenesh/projects/wt-sampfix/research/sampfix-20260805
DAILY=/home/avifenesh/projects/bw24/research/memra-vs-llama-daily-20260805
DIR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp

BIN=${1:?usage: serve-repro.sh <server-binary> <tag>}
TAG=${2:?usage: serve-repro.sh <server-binary> <tag>}
out=$RDIR/serve-repro-$TAG.txt
srv=$RDIR/serve-repro-$TAG-server.log

exec 9>/tmp/gpu5090.lock
flock 9
MEMRA_MODELS="qwen36-27b=$DIR/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf+$DIR/draft-daily-owntrim-nvfp4head-q4blk.gguf" \
MEMRA_ADDR="127.0.0.1:8002" MEMRA_API_KEY=aviary-local MEMRA_CTX=131072 \
MEMRA_MAX_SESSIONS=1 MEMRA_REUSE_POOL=1 MEMRA_PRIME_CHUNK=2048 \
"$BIN" > "$srv" 2>&1 &
MPID=$!
for i in $(seq 1 90); do
  curl -sf -m 2 -H "Authorization: Bearer aviary-local" http://127.0.0.1:8002/health >/dev/null 2>&1 && break
  sleep 2
done

PROMPT=$(python3 -c "
import json
p = open('$DAILY/prompts/short-agentic.txt').read()
print(json.dumps(p), end='')")

: > "$out"
req() { # $1 = label, $2 = extra json fields
  echo "=== $1 ===" >> "$out"
  curl -s -m 180 -H "Authorization: Bearer aviary-local" -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen36-27b\",\"prompt\":$PROMPT,\"max_tokens\":160,$2}" \
    http://127.0.0.1:8002/v1/completions >> "$out" 2>&1
  echo >> "$out"
}

# the failing matrix, verbatim from the head-to-head lane
for seed in 7 999 31337 424242; do
  req "lsampler seed=$seed" "\"temperature\":0.8,\"top_k\":40,\"top_p\":0.95,\"min_p\":0.05,\"seed\":$seed"
done
req 'untruncated t0.8 seed=7' '"temperature":0.8,"seed":7'
req 'greedy'                  '"temperature":0'
for seed in 7 999 31337 424242; do
  req "t0.8 seed=$seed top_k:40 only" "\"temperature\":0.8,\"top_k\":40,\"seed\":$seed"
  req "t0.8 seed=$seed top_p:0.95 only" "\"temperature\":0.8,\"top_p\":0.95,\"seed\":$seed"
  req "t0.8 seed=$seed min_p:0.05 only" "\"temperature\":0.8,\"min_p\":0.05,\"seed\":$seed"
done

kill $MPID 2>/dev/null; sleep 3; kill -9 $MPID 2>/dev/null
used=0
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 3000 ] && break; sleep 2
done
flock -u 9
echo "serve-repro $TAG done vram=$used -> $out"

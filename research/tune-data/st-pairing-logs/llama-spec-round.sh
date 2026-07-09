#!/usr/bin/env bash
# llama serve-best spec round: start server, query p1/p2/p3 per run-e2e.sh timing, kill.
# Usage: llama-spec-round.sh <9b|27b> <logfile>
set -euo pipefail
MODE=$1; LOG=$2
DIR=/home/avifenesh/projects/bw24/research/e2e/prompts
export GGML_CUDA_GRAPH_OPT=1
BIN=/home/avifenesh/projects/llama.cpp/build/bin/llama-server
if [ "$MODE" = 27b ]; then
  MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
  DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-Q4_K_M.gguf
else
  MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
  DRAFT=""
fi
ARGS=(-m "$MODEL" --ctx-size 16384 -ngl 999 -fa on --cache-type-k q8_0 --cache-type-v q5_1 \
      --host 127.0.0.1 --port 8899 --parallel 1)
[ -n "$DRAFT" ] && ARGS+=(--model-draft "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1 -ngld 999)
"$BIN" "${ARGS[@]}" > /tmp/e2e-llama-server.log 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null" EXIT
for i in $(seq 240); do curl -sf http://127.0.0.1:8899/health >/dev/null 2>&1 && break; sleep 2; done
{
echo "### llama $MODE spec round $(date -Is)"
echo "gpu-pre: $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)"
for P in p1-code-short p2-code-medium p3-agentic-long; do
  echo "=== llama $MODE $P ==="
  python3 - "$DIR/$P.txt" << 'PY'
import json,sys,urllib.request
prompt=open(sys.argv[1]).read()
req=urllib.request.Request('http://127.0.0.1:8899/completion',
  data=json.dumps({'prompt':prompt,'n_predict':256,'temperature':0,'cache_prompt':False}).encode(),
  headers={'Content-Type':'application/json'})
r=json.loads(urllib.request.urlopen(req, timeout=600).read())
t=r['timings']
print(f"prompt: {t['prompt_n']} tok @ {t['prompt_per_second']:.1f} tok/s | gen: {t['predicted_n']} tok @ {t['predicted_per_second']:.2f} tok/s")
PY
  echo "gpu: $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)"
done
} 2>&1 | tee "$LOG"
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null || true

#!/usr/bin/env bash
# E2E real-prompt bench: bw24 best config vs llama serve-script config, SAME prompts.
# Protocol (user, 2026-07-04): real coding+agentic prompts, few sizes, llama at its best
# serve setup, missing features stated explicitly, overall image every few steps.
# Usage: run-e2e.sh <9b|27b> <bw24|llama>
set -euo pipefail
MODE=$1; ENGINE=$2
DIR="$(dirname "$0")/prompts"
NGEN=256
if [ "$MODE" = 27b ]; then
  MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
  DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-Q4_K_M.gguf
  TRIM=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-Q4_K_M-frspec-code75-32768.gguf
  # daily config 2026-07-06: post-norm h_seed (HPOST), code75 trim (p2 +2.9 vs balanced, JSONL 2026-07-06), pmin 0.15 (config-sweep JSONL)
  BW_ENV="BW24_FRSPEC_TRIM=$TRIM BW24_SPEC_PMIN=0.15 BW24_SPEC_HPOST=1"
  BK=3   # 27B optimum (re-confirmed under HPOST: K=3 > K=2/K=4 both domains)
else
  MODEL=/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
  DRAFT=""
  BW_ENV="BW24_SPEC_PMIN=0.3 BW24_SPEC_HPOST=1"
  BK=3   # 9B optimum under HPOST 2026-07-06 (K=3 wins p1+p3, K=2 edges p2 by 3)
fi
if [ "$ENGINE" = bw24 ]; then
  for P in p1-code-short p2-code-medium p3-agentic-long; do
    echo "=== bw24 $MODE $P (K=$BK) ==="
    env $BW_ENV BW24_PROMPT="$(cat "$DIR/$P.txt")" \
      BW24_SPEC_K=$BK BW24_NGEN=$NGEN /home/avifenesh/projects/bw24/target/release/run-spec "$MODEL" 2>/dev/null \
      | grep -E "text prompt|generate\]|K=$BK\]" | head -3
  done
else
  # llama at the serve-script best config (MTP spec draft, n-max 3, p-min 0.1, KV q8_0/q5_1, fa, graphs)
  BIN=/data/projects/llama.cpp/build/bin/llama-server
  ARGS=(-m "$MODEL" --ctx-size 16384 -ngl 999 -fa on --cache-type-k q8_0 --cache-type-v q5_1 \
        --host 127.0.0.1 --port 8899 --parallel 1)
  [ -n "$DRAFT" ] && ARGS+=(--model-draft "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.1 -ngld 999)
  "$BIN" "${ARGS[@]}" > /tmp/e2e-llama.log 2>&1 &
  SPID=$!
  trap "kill $SPID 2>/dev/null" EXIT
  for i in $(seq 240); do curl -sf http://127.0.0.1:8899/health >/dev/null 2>&1 && break; sleep 2; done
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
  done
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null || true
fi

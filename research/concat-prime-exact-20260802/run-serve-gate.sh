#!/usr/bin/env bash
# Serve-level greedy isolation gate (lane/concat-prime-exact): c=1 sequential references
# vs c=16 concurrent, byte-identity per prompt, per model x per MEMRA_ROUTER_PREFILL_EXACT arm.
# Protocol == research/ornith-serve-20260801 §2 (tools/check-batch-exact.py, 16 prompts,
# 96 max_tokens, temperature 0 + seed 0). Every server run holds the 5090 lock for its
# whole lifetime so the co-resident lane never overlaps a measurement.
#
# Workflow-args law: every parameter is a literal here, nothing is passed in.
set -uo pipefail
cd /home/avifenesh/projects/wt-concat-prime-exact
OUT=research/concat-prime-exact-20260802
PORT=8121

run_one() {
  local label="$1" path="$2" arm="$3"
  local tag="${label}-exact${arm}"
  echo "=== serve gate ${tag} ==="
  MEMRA_ROUTER_PREFILL_EXACT="$arm" MEMRA_ADDR="127.0.0.1:${PORT}" \
    MEMRA_MODELS="m=${path}" \
    ./target/release/memra-server > "${OUT}/server-${tag}.log" 2>&1 &
  local pid=$!
  # wait for readiness (model load can take a minute on the 35B class)
  for _ in $(seq 1 180); do
    sleep 2
    if curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "server died early: see server-${tag}.log"; return 1; fi
  done
  python3 tools/check-batch-exact.py --base "http://127.0.0.1:${PORT}" --model m \
    --n 16 --max-tokens 96 --label "${tag}" \
    --out "${OUT}/greedy-hash-${tag}.jsonl" \
    --ref "${OUT}/greedy-refs-${tag}.json" 2>&1 | tee "${OUT}/greedy-hash-${tag}.log"
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  sleep 3
}

# Only this script's own servers are ever killed by pid — never pkill (the co-resident
# llama-server --embedding must survive).
gate() {
  local label="$1" path="$2"
  for arm in 1 0; do
    flock /tmp/gpu5090.lock bash -c "$(declare -f run_one); OUT='${OUT}'; PORT=${PORT}; cd /home/avifenesh/projects/wt-concat-prime-exact; run_one '${label}' '${path}' '${arm}'"
  done
}

gate o35b   /data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
gate kat    /data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
gate o9b    /data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
gate q35ctrl /data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf

echo "=== summary ==="
for f in "${OUT}"/greedy-hash-*.jsonl; do
  python3 -c "
import json,sys
for line in open('$f'):
    r=json.loads(line)
    print('%-22s %-6s %2d/%2d  errs=%d' % (r['label'], r['verdict'], r['n_match'], r['n'], r['n_err']))
"
done

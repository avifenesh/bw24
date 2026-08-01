#!/usr/bin/env bash
# fast-router serve gate: greedy c=1 vs c=16 byte identity on Ornith-35B at NAKED defaults
# (batch twins live). This is the contract the concat-prime fix bought — it must hold
# unchanged, because the batch twins are bit-identical to the kernels that bought it
# (kernel-check m-sweep). Protocol == research/concat-prime-exact-20260802/run-serve-gate.sh
# (tools/check-batch-exact.py, 16 prompts, 96 max_tokens, temperature 0 + seed 0). The
# server holds the 5090 lock for its whole lifetime; killed by pid, never pkill (the
# co-resident llama-server --embedding must survive). Workflow-args law: literals only.
set -uo pipefail
cd /home/avifenesh/projects/wt-fast-router
OUT=research/fast-router-20260802
PORT=8123

run_one() {
  local tag="o35b-batch-default"
  echo "=== serve gate ${tag} ==="
  MEMRA_ADDR="127.0.0.1:${PORT}" \
    MEMRA_MODELS="m=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf" \
    ./target/release/memra-server > "${OUT}/server-${tag}.log" 2>&1 &
  local pid=$!
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

flock /tmp/gpu5090.lock bash -c "$(declare -f run_one); OUT='${OUT}'; PORT=${PORT}; cd /home/avifenesh/projects/wt-fast-router; run_one"

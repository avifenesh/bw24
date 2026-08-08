#!/bin/bash
# 122B guard verification — default env, the previously-failing shapes.
set -u
cd /root/bw24-122b
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
MODEL=/dev/shm/122b/Qwen3.5-122B-A10B-UD-IQ4_XS.gguf
OUT=/root/receipts-122b/logs
exec 9>/tmp/gpu5090.lock
flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 1; }

PROMPT="Here is a Rust HTTP server source file: bw24-server (BASE-4): a minimal OpenAI-ish HTTP server that serves 2-4 concurrent agents across DIFFERENT models on one endpoint via a single GPU worker thread + step-interleave scheduler. Architecture: axum runs on a tokio runtime; ONE dedicated std thread owns the Engine and every loaded HybridModel because the CUDA context is thread-affine. Handlers submit commands over a std mpsc channel and receive tokens back over a per-request tokio mpsc channel. Endpoints include health, models, and v1 completions with streaming SSE support."

echo "=== FIX r1: default env, 110-tok prompt, NGEN=32 ==="
MEMRA_NGEN=32 ./target/release/run-gen "$MODEL" --prompt "$PROMPT" > "$OUT/fix-default-r1.log" 2>&1
echo "exit=$? verdict=$(grep -o 'MATCH\|MISMATCH' "$OUT/fix-default-r1.log" | head -1) guard=$(grep -c 'v4 decode family disabled' "$OUT/fix-default-r1.log")"

echo "=== FIX r2: identical rerun (x2 self-consistency) ==="
MEMRA_NGEN=32 ./target/release/run-gen "$MODEL" --prompt "$PROMPT" > "$OUT/fix-default-r2.log" 2>&1
echo "exit=$?"
if diff <(grep '^tokens:' "$OUT/fix-default-r1.log") <(grep '^tokens:' "$OUT/fix-default-r2.log") > /dev/null; then
  echo "ARGMAX-X2-IDENTICAL"
else
  echo "ARGMAX-X2-DIVERGED"
fi

echo "=== FIX 4k: the original failing 4k-class prompt (argmax-run1 repro), NGEN=16 ==="
# reuse the architecture doc prompt from the prior lane via a file
if [ -f /root/receipts-122b/prompt-4k.txt ]; then
  MEMRA_NGEN=16 MEMRA_PROMPT_FILE=/root/receipts-122b/prompt-4k.txt ./target/release/run-gen "$MODEL" > "$OUT/fix-4k.log" 2>&1
  echo "exit=$? verdict=$(grep -o 'MATCH\|MISMATCH' "$OUT/fix-4k.log" | head -1)"
else
  echo "SKIP: no 4k prompt file"
fi
echo "ALL VERIFY DONE"

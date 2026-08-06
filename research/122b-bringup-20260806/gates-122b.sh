#!/bin/bash
# 122B gate battery under the gqa guard — lane/122b-bringup 2026-08-06.
set -u
cd /root/bw24-122b
export PATH=/root/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
MODEL=/dev/shm/122b/Qwen3.5-122B-A10B-UD-IQ4_XS.gguf
OUT=/root/receipts-122b/logs
exec 9>/tmp/gpu5090.lock
flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 1; }

echo "=== GATE 1: kernel-check (model-backed) ==="
./target/release/kernel-check "$MODEL" > "$OUT/gate-kernel-check.log" 2>&1
echo "exit=$? tail: $(tail -1 "$OUT/gate-kernel-check.log")"

echo "=== GATE 2: run-spec probe (MTP ships? K=1..3) ==="
for K in 1 2 3; do
  MEMRA_SPEC_K=$K timeout 900 ./target/release/run-spec "$MODEL" > "$OUT/gate-runspec-k$K.log" 2>&1
  echo "K=$K exit=$? tail: $(tail -2 "$OUT/gate-runspec-k$K.log" | head -1)"
done

echo "=== GATE 3: chunkinv ==="
timeout 1800 bash tools/chunk-invariance-gate.sh "$MODEL" > "$OUT/gate-chunkinv.log" 2>&1
echo "exit=$? tail: $(tail -1 "$OUT/gate-chunkinv.log")"

echo "=== GATE 4: serve-smoke ==="
timeout 2400 bash tools/serve-smoke.sh "$MODEL" /nonexistent-no-draft > "$OUT/gate-serve-smoke.log" 2>&1
echo "exit=$? tail: $(tail -3 "$OUT/gate-serve-smoke.log" | tr '\n' ' | ')"

echo "ALL GATES DONE"

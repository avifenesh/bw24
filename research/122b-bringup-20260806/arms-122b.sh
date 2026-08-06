#!/bin/bash
# 122B NaN arm isolation — lane/122b-bringup 2026-08-06.
# Hypothesis: fa_v4_smem q_ints[8][64] bound (gqa<=8) overflows at 122B gqa=16 (32 qh / 2 kvh).
# Each arm: same >96-token prompt (decode t_kv crosses FA_VEC_MIN=96 -> vec dispatch), NGEN=8.
# Verdict = the run_gen prefill-vs-decode argmax gate (MATCH/MISMATCH) + NaN line.
set -u
cd /root/bw24-122b
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
MODEL=/dev/shm/122b/Qwen3.5-122B-A10B-UD-IQ4_XS.gguf
OUT=/root/receipts-122b/logs
mkdir -p "$OUT"

PROMPT="Here is a Rust HTTP server source file: bw24-server (BASE-4): a minimal OpenAI-ish HTTP server that serves 2-4 concurrent agents across DIFFERENT models on one endpoint via a single GPU worker thread + step-interleave scheduler. Architecture: axum runs on a tokio runtime; ONE dedicated std thread owns the Engine and every loaded HybridModel because the CUDA context is thread-affine. Handlers submit commands over a std mpsc channel and receive tokens back over a per-request tokio mpsc channel. Endpoints include health, models, and v1 completions with streaming SSE support."

run_arm() {
  local name="$1"; shift
  echo "=== ARM $name : $* ==="
  env MEMRA_NGEN=8 "$@" ./target/release/run-gen "$MODEL" --prompt "$PROMPT" \
    > "$OUT/arm-$name.log" 2>&1
  local rc=$?
  local verdict=$(grep -o "MATCH\|MISMATCH" "$OUT/arm-$name.log" | head -1)
  local nan=$(grep -c "NaN" "$OUT/arm-$name.log")
  echo "ARM $name exit=$rc verdict=${verdict:-none} nan_lines=$nan"
}

flock -w 7200 /tmp/gpu5090.lock -c '
  cd /root/bw24-122b
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
  true
' || { echo "LOCK TIMEOUT"; exit 1; }

# serialize the whole battery under the box lock
exec 9>/tmp/gpu5090.lock
flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 1; }

run_arm a0-default
run_arm a1-deep0    MEMRA_FA_DEEP=0
run_arm a2-v4off    MEMRA_FA_V4=0
run_arm a3-v43off   MEMRA_FA_V4=0 MEMRA_FA_V3=0
run_arm a4-v432off  MEMRA_FA_V4=0 MEMRA_FA_V3=0 MEMRA_FA_V2=0
run_arm a5-novec    MEMRA_NO_FA_VEC=1
run_arm a6-oracle   MEMRA_FAST=0
echo "ALL ARMS DONE"

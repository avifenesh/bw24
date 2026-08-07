#!/usr/bin/env bash
# pp2spec-crash STEP 3 — triangulate the corrupted operand by moving the draft arm.
# Round 2 named the faulting kernel under the DEFAULT (graph) draft: embed_gather_u32,
# Warp MMU Fault, dev0 (stage 1 = primary), VA 0x484_c6b3c500 (same page as round 1's Xid 31).
# That kernel appears (T=1, grid 16) in the spec path ONLY as the draft-graph replay's first
# node (mtp_head_forward_cap -> embed_gather_device, spec.rs:1281).
# The finding lane's F2 said NOGRAPH fails identically — if so, an eager-arm dump must name a
# DIFFERENT kernel reading the same class of operand. Which kernel that is discriminates:
#   - embed_gather_u32_t (eager chain op A)  -> the corrupted operand is the token id / table
#   - something in attn/commit               -> the corruption is upstream of the embed
# E1: NOGRAPH=1 + coredump, exact A sequence. E2: default graph + coredump (2nd dump,
# reproducibility of the round-2 kernel). Fresh server per arm; both dumps analyzed.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2crash
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
ADDR=127.0.0.1:8123
BASE=http://$ADDR

exec 9>/tmp/memra-gpu.lock
flock -w 1800 9 || { echo "FAIL: gpu lock timeout"; exit 1; }
echo "gpu lock acquired $(date -u +%FT%TZ)"

wait_up() {
  for _ in $(seq 1 "$1"); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
  echo "FAIL: something already serving $ADDR"; exit 1
fi

run_arm() { # $1=arm-name $2=extra-env (string, may be empty)
  local arm=$1; shift
  local extra=("$@")
  echo "=== ARM $arm (exact A sequence under coredump env) ==="
  env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
    MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
    CUDA_COREDUMP_FILE="$OUT/core-$arm-%p.nvcudmp" \
    "${extra[@]}" \
    $BIN/memra-server > "$OUT/E-$arm-server.log" 2>&1 &
  local PID=$!
  if ! wait_up 180; then echo "FAIL: $arm server never came up"; tail -20 "$OUT/E-$arm-server.log"; kill $PID 2>/dev/null; return 1; fi
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
    --requests 8 --max-tokens 96 --greedy --warmup 1 --label E-$arm-c2 \
    --out "$OUT/E-points.jsonl" > "$OUT/E-$arm-c2.log" 2>&1
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
    --requests 16 --max-tokens 96 --greedy --warmup 0 --label E-$arm-c4 \
    --out "$OUT/E-points.jsonl" > "$OUT/E-$arm-c4.log" 2>&1
  sleep 20
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
  echo "--- $arm hits ---"
  grep -n -i "illegal\|abort\|alloc failed" "$OUT/E-$arm-server.log" | head -4
  ls -la "$OUT"/core-$arm-*.nvcudmp 2>/dev/null || echo "($arm: no dump)"
}

# NOTE: spec.rs gates graph draft on `env::var("MEMRA_SPEC_NOGRAPH").is_err()` — ANY set
# value (even =0) forces the eager chain, so the graph arm passes a no-op var instead.
run_arm nograph MEMRA_SPEC_NOGRAPH=1
run_arm graph2 MEMRA_PP2CRASH_ARM=graph2

for core in "$OUT"/core-nograph-*.nvcudmp "$OUT"/core-graph2-*.nvcudmp; do
  [ -e "$core" ] || continue
  echo "=== cuda-gdb: $core ==="
  /usr/local/cuda-13.2/bin/cuda-gdb --batch \
    -ex "target cudacore $core" \
    -ex "info cuda kernels" \
    -ex "info registers" \
    > "$OUT/E-cudagdb-$(basename "$core" .nvcudmp).log" 2>&1
  head -30 "$OUT/E-cudagdb-$(basename "$core" .nvcudmp).log"
done
nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/E-gpu-post.csv"
echo PP2CRASH_ROUND3_DONE

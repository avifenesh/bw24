#!/usr/bin/env bash
# pp2spec-crash STEP 4 — FULL coredump (param + global memory) to read the faulting kernel's
# actual operands. Round 3: both draft arms fault in the draft-chain embed gather at the SAME
# VA 0x484_c6b3c500 (graph arm embed_gather_u32, eager arm embed_gather_u32_t). Lightweight
# dumps omit param memory (CUDBG_ERROR_INVALID_MEMORY_ACCESS); the full dump reads:
#   params: embd*, token_d*, x_out*, n_embd, qtype, row_bytes  (offsets 0x0/0x8/0x10/0x18/0x1c/0x20)
#   global: token_d[0] content, and whether the fault VA falls inside [embd, embd+vocab*rb).
# That splits garbage-token-index vs stale/unmapped-table definitively.
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

rm -f "$OUT"/core-full-*.nvcudmp
echo "=== ARM F: FULL coredump, exact A sequence (graph draft default) ==="
env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
  MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
  CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
  CUDA_COREDUMP_FILE="$OUT/core-full-%p.nvcudmp" \
  $BIN/memra-server > "$OUT/F-server.log" 2>&1 &
PID=$!
if ! wait_up 180; then echo "FAIL: server never came up"; tail -20 "$OUT/F-server.log"; kill $PID 2>/dev/null; exit 1; fi
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
  --requests 8 --max-tokens 96 --greedy --warmup 1 --label F-c2 \
  --out "$OUT/F-points.jsonl" > "$OUT/F-c2.log" 2>&1
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
  --requests 16 --max-tokens 96 --greedy --warmup 0 --label F-c4 \
  --out "$OUT/F-points.jsonl" > "$OUT/F-c4.log" 2>&1
# a FULL dump of ~10GB VRAM takes a while to write — wait generously for the file to settle
for _ in $(seq 1 60); do
  sz1=$(stat -c%s "$OUT"/core-full-*.nvcudmp 2>/dev/null | head -1 || echo 0)
  sleep 10
  sz2=$(stat -c%s "$OUT"/core-full-*.nvcudmp 2>/dev/null | head -1 || echo 0)
  [ -n "$sz1" ] && [ "$sz1" = "$sz2" ] && [ "$sz1" != "0" ] && break
done
kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
echo "--- hits ---"
grep -n -i "illegal\|abort\|alloc failed" "$OUT/F-server.log" | head -4
ls -la "$OUT"/core-full-*.nvcudmp 2>/dev/null || { echo "NO FULL DUMP"; exit 1; }

CORE=$(ls "$OUT"/core-full-*.nvcudmp | head -1)
echo "=== cuda-gdb param + operand readout: $CORE ==="
/usr/local/cuda-13.2/bin/cuda-gdb --batch \
  -ex "target cudacore $CORE" \
  -ex "info cuda kernels" \
  -ex "echo \n--- params (embd*, token_d*, x_out*, n_embd|qtype, row_bytes) ---\n" \
  -ex "x/5gx (@parameter unsigned long long*)0x0" \
  -ex "echo \n--- token_d[0] ---\n" \
  -ex "x/4wx *(@parameter unsigned long long*)0x8" \
  -ex "echo \n--- embd base bytes ---\n" \
  -ex "x/4gx *(@parameter unsigned long long*)0x0" \
  -ex "echo \n--- fault-page probe (0x484c6b3c500) ---\n" \
  -ex "x/2gx 0x484c6b3c500" \
  > "$OUT/F-cudagdb-params.log" 2>&1
cat "$OUT/F-cudagdb-params.log"
nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/F-gpu-post.csv"
echo PP2CRASH_ROUND4_DONE

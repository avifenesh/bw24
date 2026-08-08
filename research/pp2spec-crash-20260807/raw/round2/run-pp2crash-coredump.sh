#!/usr/bin/env bash
# pp2spec-crash STEP 2 — name the faulting kernel via CUDA coredump-on-exception.
# Round 1 (receipts ~/receipts/pp2crash, mirrored research/pp2spec-crash-20260807/raw/round1):
#   A bare:        c=2 1/8 ok, c=4 0/16 — REPRODUCES, Xid 31 FAULT_PDE VIRT_READ on GPU0
#   L launch-block: 28/28 clean — timing-dependent
#   B memcheck:     16/16 clean, 0 findings — sanitizer serialization hides it
# Coredump-on-exception keeps async timing until the fault, then dumps the faulting kernel.
# Lightweight dump (no global memory) keeps it fast and small.
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

rm -f "$OUT"/core-*.nvcudmp
echo "=== PHASE C: coredump-on-exception (dev10 spec ON, c=4) ==="
env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_PP2SPEC_UNQUARANTINE=1 \
  MEMRA_SPEC_GATE=0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
  CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
  CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
  CUDA_COREDUMP_FILE="$OUT/core-%h-%p.nvcudmp" \
  $BIN/memra-server > "$OUT/C-server.log" 2>&1 &
PID=$!
if ! wait_up 180; then echo "FAIL: phase C server never came up"; tail -20 "$OUT/C-server.log"; kill $PID 2>/dev/null; exit 1; fi
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
  --requests 16 --max-tokens 96 --greedy --warmup 0 --label C-c4-coredump \
  --out "$OUT/C-points.jsonl" > "$OUT/C-c4.log" 2>&1
# the dump is written when the exception fires; give the driver time to finish writing
sleep 20
kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
echo "--- phase C server log hits ---"
grep -n -i "illegal\|panic\|step error\|alloc failed" "$OUT/C-server.log" | head -10
ls -la "$OUT"/core-*.nvcudmp 2>/dev/null || { echo "NO COREDUMP PRODUCED"; }

for core in "$OUT"/core-*.nvcudmp; do
  [ -e "$core" ] || continue
  echo "=== cuda-gdb analysis: $core ==="
  /usr/local/cuda-13.2/bin/cuda-gdb --batch \
    -ex "target cudacore $core" \
    -ex "info cuda kernels" \
    -ex "bt" \
    -ex "info cuda threads" \
    > "$OUT/C-cudagdb-$(basename "$core" .nvcudmp).log" 2>&1
  head -60 "$OUT/C-cudagdb-$(basename "$core" .nvcudmp).log"
done
nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/C-gpu-post.csv"
echo PP2CRASH_COREDUMP_DONE

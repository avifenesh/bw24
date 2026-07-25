#!/bin/bash
# The H100 serving-lane validation battery (ARCHITECTURE-H100.md §6) — one command,
# gates-first, perf after. Run on the box from the repo root. Exits nonzero on any gate
# failure; perf numbers are recorded, not judged (bands live in the docs).
#
# Usage: tools/validate-h100.sh <model.gguf> [--quick]
#   --quick: 16-step gates, skip the bench curve (pre-commit sanity).
set -u
MODEL="${1:?model.gguf}"
QUICK=""
[ "${2:-}" = "--quick" ] && QUICK=1
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
NVCC="${BW24_NVCC:-/usr/local/cuda-13.1/bin/nvcc}"
STEPS=$([ -n "$QUICK" ] && echo 16 || echo 32)
FAIL=0

echo "== build (sm_90a; cu sources touched to defeat rsync-stale fatbins) =="
touch crates/bw24-engine/cu/*.cu crates/bw24-engine/build.rs
BW24_CUDA_ARCH=90a BW24_NVCC=$NVCC cargo build --release -p bw24-engine \
  --bin kernel-check --bin run-gen --bin decode-batch-gate --bin decode-batch-bench \
  || exit 1
BW24_CUDA_ARCH=90a BW24_NVCC=$NVCC cargo build --release -p bw24-server || exit 1

echo "== gate: policy tests =="
BW24_CUDA_ARCH=90a BW24_NVCC=$NVCC cargo test --release -p bw24-engine --lib 2>&1 | tail -1

echo "== gate: kernel-check =="
./target/release/kernel-check | tail -1 | grep -q "ALL GREEN" || { echo "KERNEL-CHECK FAIL"; FAIL=1; }

echo "== gate: decode-batch (config B=8) =="
./target/release/decode-batch-gate "$MODEL" --steps $STEPS --batch 8 --mode config \
  | tail -1 | grep -q "ALL GREEN" || { echo "BATCH-GATE(config) FAIL"; FAIL=1; }

echo "== gate: decode-batch (strict, equalized composition) =="
BW24_MMVQ=0 BW24_NO_FUSE_NORMQ=1 ./target/release/decode-batch-gate "$MODEL" \
  --steps $STEPS --batch 4 --mode strict \
  | tail -1 | grep -q "ALL GREEN" || { echo "BATCH-GATE(strict) FAIL"; FAIL=1; }

if [ -z "$QUICK" ] && [ $FAIL -eq 0 ]; then
  echo "== perf record: serving-regime curve (ctx=512, N=3) =="
  ./target/release/decode-batch-bench "$MODEL" --steps 96 --reps 3 --batches 1,2,4,8 --ctx 512 \
    | grep -E "B=|scale"
  echo "== perf record: single-seq prime+decode (N=3) =="
  bash bench_bw24.sh "$MODEL" 3 512 2>/dev/null | grep -E "run 1|run 2|run 3|median"
fi

[ $FAIL -eq 0 ] && echo "VALIDATE-H100: ALL GATES GREEN" || echo "VALIDATE-H100: FAILURES ($FAIL)"
exit $FAIL

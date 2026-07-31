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
NVCC="${MEMRA_NVCC:-/usr/local/cuda-13.1/bin/nvcc}"
STEPS=$([ -n "$QUICK" ] && echo 16 || echo 32)
FAIL=0

echo "== build (sm_90a; cu sources touched to defeat rsync-stale fatbins) =="
touch crates/memra-engine/cu/*.cu crates/memra-engine/build.rs
MEMRA_CUDA_ARCH=90a MEMRA_NVCC=$NVCC cargo build --release -p memra-engine \
  --bin kernel-check --bin run-gen --bin decode-batch-gate --bin decode-batch-bench \
  || exit 1
MEMRA_CUDA_ARCH=90a MEMRA_NVCC=$NVCC cargo build --release -p memra-server || exit 1

echo "== gate: policy tests =="
MEMRA_CUDA_ARCH=90a MEMRA_NVCC=$NVCC cargo test --release -p memra-engine --lib 2>&1 | tail -1

echo "== gate: kernel-check =="
./target/release/kernel-check | tail -1 | grep -q "ALL GREEN" || { echo "KERNEL-CHECK FAIL"; FAIL=1; }

echo "== gate: decode-batch (config B=8) =="
./target/release/decode-batch-gate "$MODEL" --steps $STEPS --batch 8 --mode config \
  | tail -1 | grep -q "ALL GREEN" || { echo "BATCH-GATE(config) FAIL"; FAIL=1; }

echo "== gate: decode-batch (strict, equalized composition) =="
MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 ./target/release/decode-batch-gate "$MODEL" \
  --steps $STEPS --batch 4 --mode strict \
  | tail -1 | grep -q "ALL GREEN" || { echo "BATCH-GATE(strict) FAIL"; FAIL=1; }

# GRAPH LANE gates (round 35: graph-decode-gate rotted OUTSIDE this battery for weeks —
# an emission off-by-one in the gate masqueraded as 171/256 stream corruption. Everything
# guarding a live lane belongs HERE.)
echo "== gate: decode-dc (device counters, bit-identity) =="
MEMRA_CUDA_ARCH=90a MEMRA_NVCC=$NVCC cargo build --release -p memra-engine \
  --bin decode-dc-gate --bin graph-decode-gate --bin graph-session-gate >/dev/null 2>&1
./target/release/decode-dc-gate "$MODEL" 2>&1 | tail -1 | grep -q "PASS" \
  || { echo "DC-GATE FAIL"; FAIL=1; }
echo "== gate: graph-decode (capture/replay bit-identity) =="
./target/release/graph-decode-gate "$MODEL" 2>&1 | tail -1 | grep -q "PASS" \
  || { echo "GRAPH-DECODE-GATE FAIL"; FAIL=1; }
echo "== gate: graph-session (serving GraphSession vs generate_graph) =="
./target/release/graph-session-gate "$MODEL" 2>&1 | tail -1 | grep -q "ALL GREEN" \
  || { echo "GRAPH-SESSION-GATE FAIL"; FAIL=1; }

if [ -z "$QUICK" ] && [ $FAIL -eq 0 ]; then
  echo "== perf record: serving-regime curve (ctx=512, N=3) =="
  ./target/release/decode-batch-bench "$MODEL" --steps 96 --reps 3 --batches 1,2,4,8 --ctx 512 \
    | grep -E "B=|scale"
  echo "== perf record: single-seq prime+decode (N=3) =="
  # tee the raw log; never let the pipe swallow error output (evidence discipline)
  bash tools/bench_memra_protocol.sh "$MODEL" 3 512 2>&1 | tee memra-single.log \
    | grep -E "run [0-9]|median" || echo "single-seq bench produced no readings — see memra-single.log"
fi

[ $FAIL -eq 0 ] && echo "VALIDATE-H100: ALL GATES GREEN" || echo "VALIDATE-H100: FAILURES ($FAIL)"
exit $FAIL

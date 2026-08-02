#!/usr/bin/env bash
# box-aug2 PHASE 1 round 3 — pdl per-context fix + mem-pool access grant (pp2-crossdev-fixes.patch
# = both fixes on top of BOX-COMMIT c041f70e). Full M1 gate list + boundary interleave + quick
# battery, all on the FINAL binary.
set -uo pipefail
cd "$HOME/memra"
export PATH=$HOME/.cargo/bin:$PATH
export MEMRA_NVCC=/opt/pytorch/cuda/bin/nvcc
export LIBRARY_PATH=/opt/pytorch/cuda/lib:/home/ubuntu/cuda-13.3.1/lib/stubs
M=$HOME/models/Qwen3.5-9B-Q8_0.gguf
B=./target/release/pp2-gate
G=./target/release/run-gen
R=$HOME/receipts/m1-pp2
LOCK="flock /tmp/gpu-box.lock"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "=== round3 start $(stamp) ==="

MEMRA_CUDA_ARCH=90a cargo build --release -p memra-engine \
  --bin pp2-gate --bin pp-transport-smoke --bin run-gen --bin kernel-check \
  2>&1 | tee "$R/rebuild-r3.log" | tail -3
grep -q "^error" "$R/rebuild-r3.log" && { echo "BUILD FAILED r3 — aborting"; exit 1; }

# ---- the full M1-finish list (mission §1b), final binary ----
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./target/release/pp-transport-smoke 2>&1 | tee "$R/transport-smoke-r3.log"
$LOCK env CUDA_VISIBLE_DEVICES=0                                    "$B" "$M"          2>&1 | tee "$R/q9-singledev-r3.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_DEVICES=0,1             "$B" "$M"          2>&1 | tee "$R/q9-dev01-r3.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_DEVICES=0,1             "$B" "$M" 16 32 5  2>&1 | tee "$R/q9-dev01-split5-r3.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_DEVICES=0,1 MEMRA_PP_OVERLAP=1 "$B" "$M"   2>&1 | tee "$R/q9-dev01-overlap-r3.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MEMRA_PP_DEVICES=0,7 "$B" "$M"          2>&1 | tee "$R/q9-dev07-r3.log"

# ---- boundary cost, interleaved x5, final binary (arms differ ONLY by pp2 door/placement) ----
for pass in 1 2 3 4 5; do
  $LOCK env CUDA_VISIBLE_DEVICES=0   MEMRA_QWEN_DC=0 MEMRA_NGEN=128                                        "$G" "$M" 55 2>&1 | tee "$R/boundary/naked-r3-p$pass.log"
  $LOCK env CUDA_VISIBLE_DEVICES=0   MEMRA_QWEN_DC=0 MEMRA_NGEN=128 MEMRA_PP_STAGES=2                      "$G" "$M" 55 2>&1 | tee "$R/boundary/pp2-samedev-r3-p$pass.log"
  $LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_QWEN_DC=0 MEMRA_NGEN=128 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 "$G" "$M" 55 2>&1 | tee "$R/boundary/pp2-dev01-r3-p$pass.log"
done
grep -H "tok/s" "$R"/boundary/*-r3-p*.log | grep "gen-only" | tee "$R/boundary/rates-r3.txt"

# ---- final quick battery on the final binary ----
CUDA_VISIBLE_DEVICES=0 bash tools/validate-h100.sh "$M" --quick 2>&1 | tee "$HOME/receipts/battery/validate-q9-quick-r3.log"

echo "=== [$(stamp)] round-3 verdicts ==="
{
  grep -H "pp2 gate PASS\|pp2 gate FAIL\|Error" "$R"/q9-*-r3.log
  grep -H "pp-transport-smoke PASS\|pp-transport-smoke FAIL" "$R/transport-smoke-r3.log"
  echo "-- boundary r3 --"; cat "$R/boundary/rates-r3.txt"
  echo "-- battery r3 --"; tail -1 "$HOME/receipts/battery/validate-q9-quick-r3.log"
} 2>&1 | tee "$HOME/receipts/phase1-round3-verdicts.txt"
echo "=== round3 done $(stamp) ==="

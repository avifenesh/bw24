#!/usr/bin/env bash
# box-aug2 PHASE 1 round 2 — after the pdl per-context fix (pdl-perctx.patch applied on top
# of BOX-COMMIT c041f70e). Reruns: cross-device M1 gates, full boundary interleave x5 on the
# PATCHED binary, a2a with the -l:libnccl.so.2 link fix, post-patch quick battery.
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
echo "=== round2 start $(stamp) ==="

# ---- rebuild the patched engine bins (Rust-only change; fatbins cached) ----
MEMRA_CUDA_ARCH=90a cargo build --release -p memra-engine \
  --bin pp2-gate --bin pp-transport-smoke --bin run-gen --bin kernel-check \
  2>&1 | tee "$R/rebuild-patched.log" | tail -3
grep -q "error\[" "$R/rebuild-patched.log" && { echo "BUILD FAILED — aborting round 2"; exit 1; }

# ---- M1 gates, patched binary (postfix names; the failing pre-fix logs stay as evidence) ----
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./target/release/pp-transport-smoke 2>&1 | tee "$R/transport-smoke-postfix.log"
$LOCK env CUDA_VISIBLE_DEVICES=0                                    "$B" "$M"          2>&1 | tee "$R/q9-singledev-postfix.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_DEVICES=0,1             "$B" "$M"          2>&1 | tee "$R/q9-dev01-postfix.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_DEVICES=0,1             "$B" "$M" 16 32 5  2>&1 | tee "$R/q9-dev01-split5-postfix.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_DEVICES=0,1 MEMRA_PP_OVERLAP=1 "$B" "$M"   2>&1 | tee "$R/q9-dev01-overlap-postfix.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MEMRA_PP_DEVICES=0,7 "$B" "$M"          2>&1 | tee "$R/q9-dev07-postfix.log"

# ---- boundary cost, full interleave x5 on the PATCHED binary (single-window arms) ----
for pass in 1 2 3 4 5; do
  $LOCK env CUDA_VISIBLE_DEVICES=0   MEMRA_QWEN_DC=0 MEMRA_NGEN=128                                        "$G" "$M" 55 2>&1 | tee "$R/boundary/naked-fix-p$pass.log"
  $LOCK env CUDA_VISIBLE_DEVICES=0   MEMRA_QWEN_DC=0 MEMRA_NGEN=128 MEMRA_PP_STAGES=2                      "$G" "$M" 55 2>&1 | tee "$R/boundary/pp2-samedev-fix-p$pass.log"
  $LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_QWEN_DC=0 MEMRA_NGEN=128 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 "$G" "$M" 55 2>&1 | tee "$R/boundary/pp2-dev01-fix-p$pass.log"
done
grep -H "tok/s" "$R"/boundary/*-fix-p*.log | grep "gen-only" | tee "$R/boundary/rates-fix.txt"

# ---- M0 a2a: link against the versioned soname (no libnccl.so symlink on this DLAMI) ----
echo "=== [$(stamp)] M0 a2a (round 2) ==="
RA=$HOME/receipts/m0-a2a
cd "$HOME/m0-nccl"
NVCC=/opt/pytorch/cuda/bin/nvcc
$NVCC -O3 commbench.cu  -o commbench  -I/opt/pytorch/cuda/include -L/opt/pytorch/cuda/lib -l:libnccl.so.2 2>> "$RA/env.txt" || echo "COMMBENCH BUILD FAIL(r2)" | tee -a "$RA/env.txt"
$NVCC -O3 commbench2.cu -o commbench2 -I/opt/pytorch/cuda/include -L/opt/pytorch/cuda/lib -l:libnccl.so.2 2>> "$RA/env.txt" || echo "COMMBENCH2 BUILD FAIL(r2)" | tee -a "$RA/env.txt"
export LD_LIBRARY_PATH=/opt/pytorch/cuda/lib:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader > "$RA/a2a_set.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >> "$RA/a2a_set.txt"
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./commbench a2a 0 1               > "$RA/a2a_n2.jsonl" 2> "$RA/a2a_n2.err"
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./commbench a2a 0 1 2 3           > "$RA/a2a_n4.jsonl" 2> "$RA/a2a_n4.err"
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./commbench a2a 0 1 2 3 4 5 6 7   > "$RA/a2a_n8.jsonl" 2> "$RA/a2a_n8.err"
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./commbench2 ga2a 0 1             > "$RA/ga2a_n2.jsonl" 2> "$RA/ga2a_n2.err"
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./commbench2 ga2a 0 1 2 3         > "$RA/ga2a_n4.jsonl" 2> "$RA/ga2a_n4.err"
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./commbench2 ga2a 0 1 2 3 4 5 6 7 > "$RA/ga2a_n8.jsonl" 2> "$RA/ga2a_n8.err"
nvidia-smi topo -m > "$RA/topo.txt"

# ---- post-patch battery (the patch touches every pdl launch path — regate it) ----
echo "=== [$(stamp)] battery postfix (GPU 0, --quick) ==="
cd "$HOME/memra"
CUDA_VISIBLE_DEVICES=0 bash tools/validate-h100.sh "$M" --quick 2>&1 | tee "$HOME/receipts/battery/validate-q9-quick-postfix.log"

echo "=== [$(stamp)] round-2 verdicts ==="
{
  grep -H "pp2 gate PASS\|pp2 gate FAIL" "$R"/q9-*postfix.log
  grep -H "pp-transport-smoke PASS\|pp-transport-smoke FAIL" "$R/transport-smoke-postfix.log"
  echo "-- boundary (fix) --"; cat "$R/boundary/rates-fix.txt"
  echo "-- a2a --"; for f in "$RA"/a2a_n*.jsonl "$RA"/ga2a_n*.jsonl; do echo "$f: $(wc -l < "$f") rows"; done
  echo "-- battery --"; tail -1 "$HOME/receipts/battery/validate-q9-quick-postfix.log"
} 2>&1 | tee "$HOME/receipts/phase1-round2-verdicts.txt"
echo "=== round2 done $(stamp) ==="

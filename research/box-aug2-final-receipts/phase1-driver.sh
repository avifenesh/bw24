#!/usr/bin/env bash
# box-aug2 PHASE 1 driver (lane/box-phase1) — M1-finish + boundary cost + M0 a2a re-confirm + battery.
# Receipts: ~/receipts/{m1-pp2,m0-a2a,battery}/ (NEVER /tmp). tee raw first, parse second.
# GPU discipline: every run pins CUDA_VISIBLE_DEVICES; multi-GPU holds via flock /tmp/gpu-box.lock.
set -uo pipefail
cd "$HOME/memra"
export PATH=$HOME/.cargo/bin:$PATH
M=$HOME/models/Qwen3.5-9B-Q8_0.gguf
B=./target/release/pp2-gate
G=./target/release/run-gen
R=$HOME/receipts/m1-pp2
LOCK="flock /tmp/gpu-box.lock"
mkdir -p "$R/boundary" "$HOME/receipts/m0-a2a" "$HOME/receipts/battery"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "=== phase1 driver start $(stamp) — BOX-COMMIT: $(cat $HOME/memra/BOX-COMMIT.txt) ==="

# ---- pre-state: all GPUs must be idle before multi-GPU work ----
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tee "$R/gpu-state-pre.txt"
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{exit 1}'; then
  echo "gpu-state: all idle (<1GiB)"
else
  echo "WARNING: non-idle GPU detected at start — recorded above, proceeding on assigned devices only"
fi

# =====================================================================================
# 1b. M1-FINISH gate list (verbatim set from tools/box-aug2-mission.md §1b)
# =====================================================================================
echo "=== [$(stamp)] M1 gates ==="
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./target/release/pp-transport-smoke 2>&1 | tee "$R/transport-smoke.log"
$LOCK env CUDA_VISIBLE_DEVICES=0                                    "$B" "$M"          2>&1 | tee "$R/q9-singledev.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_DEVICES=0,1             "$B" "$M"          2>&1 | tee "$R/q9-dev01.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_DEVICES=0,1             "$B" "$M" 16 32 5  2>&1 | tee "$R/q9-dev01-split5.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_DEVICES=0,1 MEMRA_PP_OVERLAP=1 "$B" "$M"   2>&1 | tee "$R/q9-dev01-overlap.log"
$LOCK env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MEMRA_PP_DEVICES=0,7 "$B" "$M"          2>&1 | tee "$R/q9-dev07.log"

# =====================================================================================
# 1b+. Boundary-cost receipt: interleaved x5, three arms, eager host-logits loop forced
# (MEMRA_QWEN_DC=0 — the pp2 door lives in decode_step; dc/graph loops are NOT pp2-wired,
# so all arms share the same eager loop and differ ONLY by the pp2 door/placement).
# Arm A: naked eager. Arm B: pp2 same-device (pure seam cost — the M0-comparable number).
# Arm C: pp2 cross-device 0,1 (correctness-mode placement: stage-1 weights are PEER-READ).
# =====================================================================================
echo "=== [$(stamp)] boundary-cost interleaved x5 ==="
for pass in 1 2 3 4 5; do
  $LOCK env CUDA_VISIBLE_DEVICES=0   MEMRA_QWEN_DC=0 MEMRA_NGEN=128                                        "$G" "$M" 55 2>&1 | tee "$R/boundary/naked-p$pass.log"
  $LOCK env CUDA_VISIBLE_DEVICES=0   MEMRA_QWEN_DC=0 MEMRA_NGEN=128 MEMRA_PP_STAGES=2                      "$G" "$M" 55 2>&1 | tee "$R/boundary/pp2-samedev-p$pass.log"
  $LOCK env CUDA_VISIBLE_DEVICES=0,1 MEMRA_QWEN_DC=0 MEMRA_NGEN=128 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 "$G" "$M" 55 2>&1 | tee "$R/boundary/pp2-dev01-p$pass.log"
done
grep -H "tok/s" "$R"/boundary/*.log | grep "gen-only" | tee "$R/boundary/rates.txt"

# =====================================================================================
# 1c. M0 a2a re-confirm (mission §1c verbatim + the graph-captured set the 19.7/30.8/55.8
# reference curve actually comes from: commbench2 ga2a — nccl_ga2a_n{2,4,8} @64KiB/peer).
# NCCL per the Jul-31 env receipt: /opt/pytorch/cuda/lib/libnccl.so.2 (re-verified).
# =====================================================================================
echo "=== [$(stamp)] M0 a2a ==="
RA=$HOME/receipts/m0-a2a
mkdir -p "$HOME/m0-nccl" && cd "$HOME/m0-nccl"
cp "$HOME"/memra/research/m0-nccl-20260801/src/{commbench.cu,commbench2.cu,runall.sh,runext.sh} .
NVCC=/opt/pytorch/cuda/bin/nvcc
ls -la /opt/pytorch/cuda/lib/libnccl.so.2 > "$RA/env.txt" 2>&1
$NVCC --version >> "$RA/env.txt" 2>&1
$NVCC -O3 commbench.cu  -o commbench  -I/opt/pytorch/cuda/include -L/opt/pytorch/cuda/lib -lnccl 2>> "$RA/env.txt" || echo "COMMBENCH BUILD FAIL" | tee -a "$RA/env.txt"
$NVCC -O3 commbench2.cu -o commbench2 -I/opt/pytorch/cuda/include -L/opt/pytorch/cuda/lib -lnccl 2>> "$RA/env.txt" || echo "COMMBENCH2 BUILD FAIL" | tee -a "$RA/env.txt"
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

# =====================================================================================
# 1d. Validation battery — GPU 0 (bench-only), quick. validate-h100.sh rebuilds itself
# (touches .cu to defeat rsync-stale fatbins). Box toolchain: MEMRA_NVCC + LIBRARY_PATH.
# =====================================================================================
echo "=== [$(stamp)] battery (GPU 0, --quick) ==="
cd "$HOME/memra"
export MEMRA_NVCC=/opt/pytorch/cuda/bin/nvcc
export LIBRARY_PATH=/opt/pytorch/cuda/lib:/home/ubuntu/cuda-13.3.1/lib/stubs
CUDA_VISIBLE_DEVICES=0 bash tools/validate-h100.sh "$M" --quick 2>&1 | tee "$HOME/receipts/battery/validate-q9-quick.log"

echo "=== [$(stamp)] verdict roll-up ==="
{
  echo "== M1 gates =="
  grep -H "pp2 gate PASS\|pp2 gate FAIL" "$R"/q9-*.log
  grep -H "pp-transport-smoke PASS\|pp-transport-smoke FAIL" "$R/transport-smoke.log"
  grep -c "= 1" "$R/transport-smoke.log" | sed 's/^/CanAccessPeer=1 pair count: /'
  echo "== boundary rates (gen-only tok/s) =="
  cat "$R/boundary/rates.txt"
  echo "== battery =="
  tail -3 "$HOME/receipts/battery/validate-q9-quick.log"
} 2>&1 | tee "$HOME/receipts/phase1-verdicts.txt"
echo "=== phase1 driver done $(stamp) ==="

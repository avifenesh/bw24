#!/usr/bin/env bash
# lane/fp8-decode-v1 e2e battery on the LOCAL 27B FP8-ST checkpoint (nvidia MIXED_PRECISION:
# 208 F8_E4M3 per-tensor-scale projections + 193 NVFP4 MLP). ONE lock hold, ONE binary.
#
# A = slab  : the 208 F8 tensors host-re-encoded to a resident Q8_0 slab (1.0625 B/w) -- today's default
# B = e4m3  : MEMRA_ST_E4M3=1, checkpoint-native e4m3 residency (1.0 B/w) + the fused trunk (e65ef35b)
#
# Interleaved A,B,A,B,... so both arms share one thermal/clock window (the H100 lane's law:
# cross-run comparisons are clock-drift-invalid).
set -u
cd /home/avifenesh/projects/wt-fp8dec
R=research/fp8dec-20260805
CK=/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4
P=research/e2e/prompts/pp512.txt
G() { nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm --format=csv,noheader; }
: > "$R/ab27b-gpustate.txt"

# ---- phase 1: interleaved decode A/B, N=5 pairs
for r in 1 2 3 4 5; do
  echo "r$r A-pre  $(G)" >> "$R/ab27b-gpustate.txt"
  env MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" MEMRA_RESIDENCY_CENSUS=1 \
      timeout 2400 target/release/run-gen "$CK" > "$R/ab27b-A-slab-r$r.log" 2>&1
  echo "A r$r rc=$?"
  echo "r$r B-pre  $(G)" >> "$R/ab27b-gpustate.txt"
  env MEMRA_ST_E4M3=1 MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" MEMRA_RESIDENCY_CENSUS=1 \
      timeout 2400 target/release/run-gen "$CK" > "$R/ab27b-B-e4m3-r$r.log" 2>&1
  echo "B r$r rc=$?"
done

# ---- phase 2: branch-(b) exactness. e4m3 arithmetic != Q8_0 arithmetic, so bit-identity is the
# WRONG question between CONTAINERS. Protocol (v2's, reused): take the SLAB arm's greedy tape as
# the reference, teacher-force BOTH arms on it -> identical inputs at every position -> count
# argmax disagreements and measure the tape's NLL under each arm.
TAPE="$R/tf-tape-slab.txt"
grep -oP '^tokens: \[\K[^]]+' "$R/ab27b-A-slab-r1.log" | tr -d ' ' > "$TAPE"
echo "tape ids=$(tr ',' '\n' < "$TAPE" | grep -c .)"
env MEMRA_FORCE_TOKENS_FILE="$TAPE" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" \
    timeout 2400 target/release/run-gen "$CK" > "$R/tf-A-slab.log" 2>&1
echo "tf-A rc=$?"
env MEMRA_ST_E4M3=1 MEMRA_FORCE_TOKENS_FILE="$TAPE" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" \
    timeout 2400 target/release/run-gen "$CK" > "$R/tf-B-e4m3.log" 2>&1
echo "tf-B rc=$?"

# ---- phase 3: rollback seam -- e4m3 residency with the launch fusion OFF. Isolates how much of
# arm B is the BYTES (native residency) vs the LAUNCH FUSION (e65ef35b), and proves the seam works.
env MEMRA_ST_E4M3=1 MEMRA_E4M3_DUAL=0 MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" \
    MEMRA_RESIDENCY_CENSUS=1 timeout 2400 target/release/run-gen "$CK" > "$R/ab27b-C-e4m3-nofuse.log" 2>&1
echo "C nofuse rc=$?"
echo "final $(G)" >> "$R/ab27b-gpustate.txt"

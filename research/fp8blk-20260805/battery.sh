#!/usr/bin/env bash
# lane/fp8-blk128-decode e2e battery on the 27B BLOCK-128 FP8-ST checkpoint
# (/data/ai-ml/hf-models/qwen36-27b-blk128fp8: 208 block-128 F8_E4M3 projections + the source's 193
# NVFP4 MLP planes byte-identical). ONE lock hold, ONE binary (md5 in BINARY-md5-battery.txt).
#
# A = slab : MEMRA_ST_E4M3_BLK=0 -> the 208 block-128 tensors dequant to a resident Q8_0 slab
#            (1.0625 B/w) -- ARM B', the path this class took before this lane. The narrow seam is
#            used so the per-tensor arm (already shipped + receipted) stays on its default and the
#            A/B is single-variable.
# B = blk  : naked default -> checkpoint-native e4m3 residency (1.0 B/w) + per-k128 in-kernel scale
#            fold (qmatvec_e4m3_blk_mmvq).
#
# Interleaved A,B,A,B,... so both arms share ONE thermal/clock window (the H100 lane's law:
# cross-run comparisons are clock-drift-invalid, including the denominator).
set -u
cd /home/avifenesh/projects/wt-fp8blk
R=research/fp8blk-20260805
CK=/data/ai-ml/hf-models/qwen36-27b-blk128fp8
P=research/e2e/prompts/pp512.txt
G() { nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm --format=csv,noheader; }
: > "$R/ab27b-gpustate.txt"

# ---- phase 1: interleaved decode A/B, N=5 pairs
for r in 1 2 3 4 5; do
  echo "r$r A-pre  $(G)" >> "$R/ab27b-gpustate.txt"
  env MEMRA_ST_E4M3_BLK=0 MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" MEMRA_RESIDENCY_CENSUS=1 \
      timeout 2400 target/release/run-gen "$CK" > "$R/ab27b-A-slab-r$r.log" 2>&1
  echo "A r$r rc=$?"
  echo "r$r B-pre  $(G)" >> "$R/ab27b-gpustate.txt"
  env MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" MEMRA_RESIDENCY_CENSUS=1 \
      timeout 2400 target/release/run-gen "$CK" > "$R/ab27b-B-blk-r$r.log" 2>&1
  echo "B r$r rc=$?"
done

# ---- phase 2: branch-(b) exactness. In-kernel per-k128 e4m3 dequant is NOT the same arithmetic as
# the Q8_0 re-encode, so bit-identity is the WRONG question between CONTAINERS. Protocol (the
# decode-v1 protocol, verbatim): take the SLAB arm's greedy tape as the reference, teacher-force
# BOTH arms on it -> identical inputs at every position -> count argmax disagreements and measure
# the tape's NLL under each arm (lower = better model of the SAME sequence).
TAPE="$R/tf-tape-slab.txt"
grep -oP '^tokens: \[\K[^]]+' "$R/ab27b-A-slab-r1.log" | tr -d ' ' > "$TAPE"
echo "tape ids=$(tr ',' '\n' < "$TAPE" | grep -c .)"
env MEMRA_ST_E4M3_BLK=0 MEMRA_FORCE_TOKENS_FILE="$TAPE" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" \
    timeout 2400 target/release/run-gen "$CK" > "$R/tf-A-slab.log" 2>&1
echo "tf-A rc=$?"
env MEMRA_FORCE_TOKENS_FILE="$TAPE" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P" \
    timeout 2400 target/release/run-gen "$CK" > "$R/tf-B-blk.log" 2>&1
echo "tf-B rc=$?"

# ---- phase 3: PREFILL must not regress. The block-128 class routes m>=16 through
# try_e4m3_blk_prefill (dequant-per-call to the Q8_0 slab), so prefill arithmetic is the FLOOR's
# bit-for-bit; this measures whether the per-call dequant costs pp512 anything.
for r in 1 2 3; do
  env MEMRA_ST_E4M3_BLK=0 MEMRA_PP_ONLY=1 MEMRA_PROMPT_FILE="$P" \
      timeout 2400 target/release/run-gen "$CK" > "$R/pp27b-A-slab-r$r.log" 2>&1
  echo "ppA r$r rc=$?"
  env MEMRA_PP_ONLY=1 MEMRA_PROMPT_FILE="$P" \
      timeout 2400 target/release/run-gen "$CK" > "$R/pp27b-B-blk-r$r.log" 2>&1
  echo "ppB r$r rc=$?"
done

# ---- phase 4: the SHARED rollback seam. MEMRA_ST_E4M3=0 must take BOTH e4m3 classes back to the
# Q8_0 slab (flags doctrine: one seam that reverts all native e4m3 residency).
env MEMRA_ST_E4M3=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE="$P" MEMRA_RESIDENCY_CENSUS=1 \
    timeout 2400 target/release/run-gen "$CK" > "$R/census-27b-rollback.log" 2>&1
echo "rollback rc=$?"
echo "final $(G)" >> "$R/ab27b-gpustate.txt"

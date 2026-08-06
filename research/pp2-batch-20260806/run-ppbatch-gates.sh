#!/usr/bin/env bash
# pp2-batch STEP 4 — the batched stage-split gate battery. 2x RTX PRO 6000, 2026-08-06.
#
# What this proves (and what it does NOT): `decode_step_batch_ppn` runs each stage's layer
# range on its own engine/stream with a [B, n_embd] boundary copy. That copy is exact and
# every stage runs the same kernels on the same bytes in the same order, so PP-N adds ZERO
# deviation — batched PP-N must be BIT-IDENTICAL to the unsplit batched body at the same B,
# in BOTH placement orders. Anything less is a seam bug, not "acceptable drift".
#
# Receipts to ~/receipts/pp2batch/gates. tee FIRST, parse the log SECOND (a pipe into a
# parser eats the failure text). Params baked as literals — workflow args do not propagate.
# GPU window held under flock /tmp/memra-gpu.lock (box shared with the step37-p2 lane).
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2batch/gates
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q27=/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
BIN=target/release
FAILS=0

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-state-pre.txt" 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> "$OUT/gpu-state-pre.txt" 2>&1

run() { local log="$OUT/$1"; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "=== $log: env[${envs[*]:-}] $*"
  if ! env "${envs[@]}" "$@" 2>&1 | tee "$log"; then echo "FAIL: $log"; FAILS=$((FAILS+1)); fi
}

# ---- THE LANE GATE: batched PP-2 bit-identity, both placement orders ----------------
# --batch 1,4,8 in one process => all widths measured against the SAME loaded weights.
# reps=3 on the split arm: the shared-Engine scratch race this design avoids was a 35%
# FLAKE (2026-08-02), so one green replay is not evidence of absence.
run ppbatch-q9-dev01.log   MEMRA_PP_DEVICES=0,1 -- \
  $BIN/decode-batch-gate "$Q9" --mode pp --stages 2 --steps 32 --batch 1,4,8 --reps 3
run ppbatch-q9-dev10.log   MEMRA_PP_DEVICES=1,0 -- \
  $BIN/decode-batch-gate "$Q9" --mode pp --stages 2 --steps 32 --batch 1,4,8 --reps 3
# single-device split (no placement): isolates the SEAM from the transport — a failure here
# with dev01 green would be a stage-engine/pointer-table bug, not a peer-copy bug.
run ppbatch-q9-singledev.log -- \
  $BIN/decode-batch-gate "$Q9" --mode pp --stages 2 --steps 32 --batch 1,4,8 --reps 3
# uneven cut: the even-split default hides off-by-one fence bugs (both stages same length).
run ppbatch-q9-split5.log  MEMRA_PP_DEVICES=0,1 MEMRA_PP_SPLITS=5 -- \
  $BIN/decode-batch-gate "$Q9" --mode pp --stages 2 --steps 32 --batch 1,4,8 --reps 3
# N=4 over the pair (2 stages per card): the general-N wiring, not just the N=2 case.
run ppbatch-q9-n4-dev0011.log MEMRA_PP_DEVICES=0,0,1,1 -- \
  $BIN/decode-batch-gate "$Q9" --mode pp --stages 4 --steps 32 --batch 1,4,8 --reps 2
# EXACT-16 tier under the split (B=12/16 > cap 8): the ExactScopeN find — verify_exact is
# per-Engine, so a per-stage numeric split would show here and NOWHERE else.
run ppbatch-q9-dev01-b16.log MEMRA_PP_DEVICES=0,1 -- \
  $BIN/decode-batch-gate "$Q9" --mode pp --stages 2 --steps 16 --batch 12,16 --reps 2
# q27 (the bigger arch, Q4_K_M + NVFP4 mixed): same bar, one placement.
run ppbatch-q27-dev01.log  MEMRA_PP_DEVICES=0,1 -- \
  $BIN/decode-batch-gate "$Q27" --mode pp --stages 2 --steps 24 --batch 1,4,8 --reps 2

# ---- REFUSAL STILL BITES on the residue (BATCH_PP=0 = unsplit walk, no override) ----
# Expect a nonzero exit with the refusal text: MEMRA_BATCH_PP=0 sends the batched path back
# through the unsplit body, which under a sharded cross-device placement must FAIL CLOSED.
echo "=== refusal-residue (expected FAIL with the refusal text) ==="
if env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_BATCH_PP=0 \
     $BIN/decode-batch-bench "$Q9" --steps 8 --reps 1 --batches 4 \
     > "$OUT/refusal-residue.log" 2>&1; then
  echo "FAIL: MEMRA_BATCH_PP=0 unsplit walk was NOT refused"; FAILS=$((FAILS+1))
else
  grep -q "refused with the ppN door open" "$OUT/refusal-residue.log" \
    && echo "refusal-residue: refused as designed" \
    || { echo "FAIL: died WITHOUT the refusal text (cause unknown — see log)"; FAILS=$((FAILS+1)); }
fi

# ---- STANDING BATTERY (door SHUT — the split must not move single-device behavior) ----
run kernel-check.log        -- $BIN/kernel-check
run dbg-q9-config.log       -- $BIN/decode-batch-gate "$Q9" --steps 32 --batch 4
run dbg-q9-strict.log       MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 -- \
  $BIN/decode-batch-gate "$Q9" --steps 32 --batch 4 --mode strict
run run-gen-q9-naked.log    MEMRA_NGEN=8 -- $BIN/run-gen "$Q9" 55
# the eager arm's own gate, on this tree: the split must not have moved decode_step_h
run ppn-q9-n2-dev01.log     MEMRA_PP_DEVICES=0,1 -- $BIN/ppn-gate "$Q9" 2 16 32

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-state-post.txt" 2>&1

echo; echo "==== verdicts ===="
grep -H "pp gate PASS\|pp gate FAIL\|pp mode verdict\|ALL GREEN" $OUT/ppbatch-*.log | sed "s|$OUT/||"
grep -H "ppn gate PASS\|ppn gate FAIL" $OUT/ppn-*.log | sed "s|$OUT/||"
grep -H "ALL GREEN\|FAILED\|gate1 (\|gate2 (\|gate3 (" $OUT/dbg-*.log | sed "s|$OUT/||"
grep -H "ALL GREEN\|FAIL" $OUT/kernel-check.log | sed "s|$OUT/||" | tail -3
grep -H "MATCH\|MISMATCH" $OUT/run-gen-q9-naked.log | sed "s|$OUT/||" | tail -3
echo "script-detected failures: $FAILS"
exit $FAILS

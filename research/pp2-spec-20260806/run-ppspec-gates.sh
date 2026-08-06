#!/usr/bin/env bash
# pp2-spec STEP 3 — the spec-verify stage-split gate battery. 2x RTX PRO 6000, 2026-08-06.
#
# What this proves: `decode_step_t_core_ppn` runs each stage's layer range of the T=K+1 VERIFY
# forward on its own engine/stream with a [T, n_embd] boundary copy. That copy is exact and every
# stage runs the same kernels (`verify_layers` — the one range-scoped body the unsplit trunk also
# calls) on the same bytes in the same order, so the split MUST be BIT-IDENTICAL to the unsplit
# verify at the same T, in BOTH placement orders. Anything less is a seam bug, not drift.
#
# Then the DERIVED end-to-end bar: run-spec K=1..8 over the split. Greedy spec decode is exact by
# construction, so self-consistency PASS is necessary but weak; the sharp check is that the
# ACCEPTANCE COUNTS are IDENTICAL split vs door-shut (acceptance is a deterministic function of
# the verify logits under greedy accept, so bit-identical logits force identical counts — a
# different count means the split moved something the bit gate did not see).
#
# Receipts to ~/receipts/pp2spec/gates. tee FIRST, parse the log SECOND (a pipe into a parser
# eats the failure text). Params baked as literals — workflow args do not propagate.
# GPU window held under flock /tmp/memra-gpu.lock by the caller (box shared with step37-p2).
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2spec/gates
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

# ---- THE LANE GATE: verify PP-2 bit-identity, both placement orders ------------------
# --ts 2,5,9 in one process => all verify widths against the SAME loaded weights.
# T=2 is the K=1 minimum (and the t==2 spec_m2 window), T=9 crosses the t>=3 batched-linear
# window. reps=3 on the split arm: the scratch race this design avoids was a 35% FLAKE.
run ppspec-q9-dev01.log   MEMRA_PP_DEVICES=0,1 -- \
  $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
run ppspec-q9-dev10.log   MEMRA_PP_DEVICES=1,0 -- \
  $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
# single-device split (no placement): isolates the SEAM from the transport — a failure here with
# dev01 green would be a stage-engine bug, not a peer-copy bug.
run ppspec-q9-singledev.log -- \
  $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
# uneven cut: the even-split default hides off-by-one fence bugs (both stages same length).
run ppspec-q9-split5.log  MEMRA_PP_DEVICES=0,1 MEMRA_PP_SPLITS=5 -- \
  $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
# N=4 over the pair (2 stages per card): general-N wiring, not just N=2.
run ppspec-q9-n4-dev0011.log MEMRA_PP_DEVICES=0,0,1,1 -- \
  $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 4 --steps 16 --ts 2,5,9 --reps 2
# HPOST: the h_seed arm reads the POST-norm column instead of pre-norm — a different buffer
# crossing the same boundary, and the seed is what the drafter is re-seeded from.
run ppspec-q9-dev01-hpost.log MEMRA_PP_DEVICES=0,1 MEMRA_SPEC_HPOST=1 -- \
  $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 12 --ts 5 --reps 2
# q27 (the bigger arch, Q4_K_M + NVFP4 mixed): same bar, one placement.
run ppspec-q27-dev01.log  MEMRA_PP_DEVICES=0,1 -- \
  $BIN/decode-batch-gate "$Q27" --mode ppspec --stages 2 --steps 12 --ts 2,5,9 --reps 2

# ---- REFUSAL STILL BITES on the residue (SPEC_PP=0 = unsplit trunk, no override) -----
# MEMRA_SPEC_PP=0 sends verify back through the unsplit trunk, which under a sharded
# cross-device placement must FAIL CLOSED. Expect nonzero exit WITH the refusal text.
echo "=== refusal-residue SPEC_PP=0 (expected FAIL with the refusal text) ==="
if env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_SPEC_PP=0 MEMRA_NGEN=8 MEMRA_SPEC_K=2 \
     $BIN/run-spec "$Q9" 55 > "$OUT/refusal-specpp0.log" 2>&1; then
  echo "FAIL: MEMRA_SPEC_PP=0 unsplit verify was NOT refused"; FAILS=$((FAILS+1))
else
  grep -q "refused with the ppN door open" "$OUT/refusal-specpp0.log" \
    && echo "refusal-specpp0: refused as designed" \
    || { echo "FAIL: died WITHOUT the refusal text (cause unknown — see log)"; FAILS=$((FAILS+1)); }
fi

# ---- THE END-TO-END ARM: run-spec K=1..8 over the split ------------------------------
# Two invocations, same prompt/NGEN, so the acceptance counts are directly comparable. The
# self-consistency PASS is the exactness gate; the count equality is the sharper check.
#
# MEMRA_QWEN_DC=0 on the SPLIT arms: run-spec's ORACLE is plain `generate`, whose default Qwen
# route is the DEVICE-COUNTER loop `decode_step_dc` — a path with no pp stage split, so
# pp2-hardening makes it fail closed under a sharded cross-device placement. The refusal fires
# in the reference arm before spec runs at all (measured 2026-08-06: the first battery's four
# run-spec arms all died on that quoted refusal, NOT on anything in the verify trunk). =0 takes
# the eager `decode_step` -> `decode_step_h` route, which IS stage-split, so BOTH arms of the
# comparison walk split paths. The door-shut denominators keep the default dc route, which is
# also the honest baseline: dc is what single-device serving actually runs.
# dc-over-PP is the ONE remaining unsplit hole (dc + the graph capture that wraps it).
DC0=MEMRA_QWEN_DC=0
run runspec-q9-dev01.log  MEMRA_PP_DEVICES=0,1 MEMRA_PP_STAGES=2 $DC0 MEMRA_NGEN=64 -- \
  $BIN/run-spec "$Q9" 55
run runspec-q9-doorshut.log MEMRA_NGEN=64 -- $BIN/run-spec "$Q9" 55
# door-shut WITH the same DC0 seam: the exact denominator for the split arm's acceptance counts
# (dc vs eager oracle changes nothing under greedy, and this arm proves it rather than assuming).
run runspec-q9-doorshut-dc0.log $DC0 MEMRA_NGEN=64 -- $BIN/run-spec "$Q9" 55
run runspec-q9-dev10.log  MEMRA_PP_DEVICES=1,0 MEMRA_PP_STAGES=2 $DC0 MEMRA_NGEN=64 -- \
  $BIN/run-spec "$Q9" 55
run runspec-q27-dev01.log MEMRA_PP_DEVICES=0,1 MEMRA_PP_STAGES=2 $DC0 MEMRA_NGEN=48 -- \
  $BIN/run-spec "$Q27" 55
run runspec-q27-doorshut.log MEMRA_NGEN=48 -- $BIN/run-spec "$Q27" 55
run runspec-q27-doorshut-dc0.log $DC0 MEMRA_NGEN=48 -- $BIN/run-spec "$Q27" 55

# ---- STANDING BATTERY (door SHUT — the split must not move single-device behavior) ----
run kernel-check.log      -- $BIN/kernel-check
run dbg-q9-config.log     -- $BIN/decode-batch-gate "$Q9" --steps 32 --batch 4
run dbg-q9-strict.log     MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 -- \
  $BIN/decode-batch-gate "$Q9" --steps 32 --batch 4 --mode strict
run run-gen-q9-naked.log  MEMRA_NGEN=8 -- $BIN/run-gen "$Q9" 55
# the predecessor lane's own gate, on this tree: verify_layers extraction must not have moved
# the batched split either (it shares nothing, but the extraction touched the shared file).
run ppbatch-q9-dev01.log  MEMRA_PP_DEVICES=0,1 -- \
  $BIN/decode-batch-gate "$Q9" --mode pp --stages 2 --steps 16 --batch 1,4,8 --reps 2

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-state-post.txt" 2>&1

echo; echo "==== verdicts ===="
grep -H "pp gate PASS\|pp gate FAIL\|ppspec mode verdict\|pp mode verdict\|ALL GREEN" \
  $OUT/ppspec-*.log $OUT/ppbatch-*.log 2>/dev/null | sed "s|$OUT/||"
echo "-- run-spec self-consistency + acceptance --"
grep -H "SELF-CONSISTENCY\|acceptance:" $OUT/runspec-*.log | sed "s|$OUT/||"
grep -H "ALL GREEN\|FAILED\|gate1 (\|gate2 (\|gate3 (" $OUT/dbg-*.log | sed "s|$OUT/||"
grep -H "ALL GREEN\|FAIL" $OUT/kernel-check.log | sed "s|$OUT/||" | tail -3
grep -H "MATCH\|MISMATCH" $OUT/run-gen-q9-naked.log | sed "s|$OUT/||" | tail -3
echo "script-detected failures: $FAILS"
exit $FAILS

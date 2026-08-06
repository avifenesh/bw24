#!/usr/bin/env bash
# pp2-batch STEP 4b — the two things the gate battery does NOT cover:
#
#   (1) serve-smoke OVER THE SPLIT. The gate battery proves `decode_step_batch_ppn` is
#       bit-identical in-process; it does not prove memra-server can BOOT and serve across
#       two cards. That is the actual Step-SKU deliverable (105GB fits only across the pair),
#       and it exercises the paths the gate cannot reach: session-cache alloc (the
#       `pp::new_cache` fix in this lane), prefix restore, eviction retry, streaming,
#       concurrency. Run twice — door SHUT for the baseline fail set, door OPEN dev01 for
#       the split — because serve-smoke has a KNOWN non-empty fail set on the q9 pair
#       (research/serve-st-20260803: 4 checks, small-max_tokens routing condition). The
#       verdict is "the split does not ADD failures", so the baseline must be measured on
#       THIS binary, not quoted from an old log.
#
#   (2) THE WIDE-WIDTH SEAM at B=12/16. The battery's b16 arm was an INVALID ARM: it panicked
#       inside the door-OFF reference at decode_batch.rs:474 with
#         "decode_step_batch: B=12 > cap 8 with no exact tier (Q8_0 m>8 needs the q8rp
#          mirror's b16 class; m>16 crosses GEMM/dp4a numeric configs) — refused"
#       That is the PRE-EXISTING width policy, not a pp bug: `decode_batch_exact16_ok`
#       admits only Q4_0/Q6_K/F8_E4M3/Q8_0+rp4 weights, and both box models are NVFP4
#       (q27 also Q4_K_M), so neither has an exact-16 tier at all. The split path carries the
#       same assert (decode_batch.rs:610), so it does not bypass the policy.
#       The substitute puts BOTH sides on the measurement door (MEMRA_DECODE_BATCH_CAP=16),
#       which tests something strictly better for this lane: the m>=16 GEMM-tier kernel
#       family crossing a stage boundary. Bit-identity is still the bar — the door changes
#       WHICH tier both arms use, and the split must not perturb whichever one that is.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2batch/serve
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
FAILS=0

# The wide-width arm already PASSED (B=12/16 bit-identical under the cap16 door); kept in the
# script so the receipt is reproducible.
# ---- (2) wide-width seam, both arms on the non-exact measurement tier -----------------
echo "=== wide-width B=12,16 under MEMRA_DECODE_BATCH_CAP=16 (dev01) ==="
if ! env MEMRA_PP_DEVICES=0,1 MEMRA_DECODE_BATCH_CAP=16 \
     $BIN/decode-batch-gate "$Q9" --mode pp --stages 2 --steps 16 --batch 12,16 --reps 2 \
     2>&1 | tee "$OUT/ppbatch-q9-dev01-b16-cap16.log"; then
  echo "FAIL: wide-width split arm"; FAILS=$((FAILS+1))
fi

# ---- (1) serve-smoke: baseline (door shut) then over the split -----------------------
# No draft arg: the box stages the 27B daily draft, not q9's, so serve-smoke's spec arm
# SKIPs by design (`[ -f "$DRAFT" ]`). The spec-over-PP2 path is explicitly NOT this lane.
echo "=== serve-smoke A: door SHUT (baseline fail set on THIS binary) ==="
bash tools/serve-smoke.sh "$Q9" /nonexistent-draft.gguf 2>&1 | tee "$OUT/serve-smoke-doorshut.log"
A_EXIT=${PIPESTATUS[0]}
sleep 5

echo "=== serve-smoke B: door OPEN stages=2 devices=0,1 (THE Step-SKU config) ==="
# MEMRA_SERVE_SPEC=0 IS LOAD-BEARING, and finding out why is a result of this lane. q9 carries
# an embedded MTP head, so serving self-specs by DEFAULT and every request funnels through
# `decode_step_t`, which still fails closed (spec.rs:1332). Measured, not inferred — the
# repro's HTTP 400 body:
#   "step error: decode_step_t (spec verify): refused with the ppN door open across 2+ devices"
# with the server log confirming the split loaded ([pp] cross-device transport banner, 33
# layers). So batched PP-2 serving is real TODAY only on the non-spec path; spec-over-PP2 needs
# the verify trunk stage-split (the T=K+1 batched forward), which this lane's seam now makes
# reachable but does not itself do. Arm A runs spec-on because that is its own default baseline;
# the set-difference is still apples-to-apples on every check spec does not gate.
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_SERVE_SPEC=0 \
  bash tools/serve-smoke.sh "$Q9" /nonexistent-draft.gguf 2>&1 | tee "$OUT/serve-smoke-pp2-dev01.log"
B_EXIT=${PIPESTATUS[0]}
# The transport banner is printed by the SERVER, whose stdout serve-smoke.sh redirects to a
# fixed /tmp/serve-smoke.log — which arm C then overwrites. Snapshot it here, while it is still
# arm B's (the first version of this script checked it at the end and reported a liveness
# failure on a run that was in fact split-live; that was a harness bug, not a serving finding).
cp /tmp/serve-smoke.log "$OUT/server-stdout-pp2-dev01.log" 2>/dev/null || true
# Control: the SAME non-spec config with the door SHUT. Without it, arm B's fail set conflates
# "the split broke it" with "MEMRA_SERVE_SPEC=0 broke it" — one variable per comparison.
echo "=== serve-smoke C: door SHUT, MEMRA_SERVE_SPEC=0 (the non-spec control) ==="
MEMRA_SERVE_SPEC=0 \
  bash tools/serve-smoke.sh "$Q9" /nonexistent-draft.gguf 2>&1 | tee "$OUT/serve-smoke-nospec-doorshut.log"
C_EXIT=${PIPESTATUS[0]}

echo; echo "==== serve-smoke deltas ===="
echo "A door-shut spec-on: $A_EXIT failed | B pp2 nospec: $B_EXIT failed | C door-shut nospec: $C_EXIT failed"
echo "-- A (door-shut, spec on) fail set:"; grep -h "  FAIL:" "$OUT/serve-smoke-doorshut.log" | sort > "$OUT/failset-doorshut.txt"; cat "$OUT/failset-doorshut.txt"
echo "-- B (pp2, nospec) fail set:";        grep -h "  FAIL:" "$OUT/serve-smoke-pp2-dev01.log" | sort > "$OUT/failset-pp2.txt";     cat "$OUT/failset-pp2.txt"
echo "-- C (door-shut, nospec) CONTROL fail set:"; grep -h "  FAIL:" "$OUT/serve-smoke-nospec-doorshut.log" | sort > "$OUT/failset-nospec.txt"; cat "$OUT/failset-nospec.txt"
# THE verdict: B vs C, the single-variable comparison (both non-spec; only the split differs).
echo "-- ADDED BY THE SPLIT, B minus C (must be empty):"
comm -13 "$OUT/failset-nospec.txt" "$OUT/failset-pp2.txt" | tee "$OUT/failset-added.txt"
[ -s "$OUT/failset-added.txt" ] && { echo "FAIL: the split ADDED serve-smoke failures"; FAILS=$((FAILS+1)); }
# Proof the split was actually LIVE in arm B (not silently door-shut): the server log must
# carry the cross-device transport banner. A green run without it proves nothing.
# Written as an if/else, not a `||`/`&&` chain — that chain's precedence made the fallback
# branch fire on a run that WAS split-live.
cp /tmp/serve-smoke.log "$OUT/server-stdout-nospec-doorshut.log" 2>/dev/null || true
if grep -q "cross-device transport" "$OUT/server-stdout-pp2-dev01.log" 2>/dev/null; then
  echo "pp2 arm: split CONFIRMED live — $(grep -m1 'cross-device transport' "$OUT/server-stdout-pp2-dev01.log")"
else
  echo "FAIL: no pp transport banner in arm B's server stdout — may have served single-device"
  FAILS=$((FAILS+1))
fi
# Control: the door-SHUT arm must NOT show it (proof the check can read zero where zero is).
if grep -q "cross-device transport" "$OUT/server-stdout-nospec-doorshut.log" 2>/dev/null; then
  echo "FAIL: door-SHUT arm showed a transport banner — the door leaks"; FAILS=$((FAILS+1))
else
  echo "control: door-shut arm shows NO transport banner (the liveness check reads zero)"
fi

nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv > "$OUT/gpu-post.csv"
echo "script-detected failures: $FAILS"
exit $FAILS

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
MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
  bash tools/serve-smoke.sh "$Q9" /nonexistent-draft.gguf 2>&1 | tee "$OUT/serve-smoke-pp2-dev01.log"
B_EXIT=${PIPESTATUS[0]}

echo; echo "==== serve-smoke deltas ===="
echo "door-shut failed checks: $A_EXIT | pp2-dev01 failed checks: $B_EXIT"
echo "-- door-shut fail set:"; grep -h "  FAIL:" "$OUT/serve-smoke-doorshut.log" | sort > "$OUT/failset-doorshut.txt"; cat "$OUT/failset-doorshut.txt"
echo "-- pp2 fail set:";      grep -h "  FAIL:" "$OUT/serve-smoke-pp2-dev01.log" | sort > "$OUT/failset-pp2.txt";     cat "$OUT/failset-pp2.txt"
echo "-- ADDED BY THE SPLIT (must be empty):"
comm -13 "$OUT/failset-doorshut.txt" "$OUT/failset-pp2.txt" | tee "$OUT/failset-added.txt"
[ -s "$OUT/failset-added.txt" ] && { echo "FAIL: the split ADDED serve-smoke failures"; FAILS=$((FAILS+1)); }
# Proof the split was actually LIVE in arm B (not silently door-shut): the server log must
# carry the cross-device transport banner. A green run without it proves nothing.
grep -q "cross-device transport: stage0=dev0 stage1=dev1" "$OUT/serve-smoke-pp2-dev01.log" \
  || grep -q "cross-device transport" /tmp/serve-smoke.log \
  && echo "pp2 arm: split CONFIRMED live (transport banner present)" \
  || { echo "FAIL: no pp transport banner — arm B may have served single-device"; FAILS=$((FAILS+1)); }

nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv > "$OUT/gpu-post.csv"
echo "script-detected failures: $FAILS"
exit $FAILS

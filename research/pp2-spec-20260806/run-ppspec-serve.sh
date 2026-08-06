#!/usr/bin/env bash
# pp2-spec STEP 3b — serve-smoke WITH SPEC ON over the split. 2x RTX PRO 6000, 2026-08-06.
#
# THE deliverable check: the predecessor lane's PP-2 serving worked only with
# MEMRA_SERVE_SPEC=0, because q9 carries an embedded MTP head so the server self-specs by
# default and every request funneled through `decode_step_t`, which failed closed:
#   "step error: decode_step_t (spec verify): refused with the ppN door open across 2+ devices"
# This script's arm B is that exact configuration with spec left at its DEFAULT (on). The HTTP
# 400 must be gone, and the fail set must not grow.
#
# Four arms, because a two-arm comparison would conflate two variables:
#   A  door shut, spec ON   — the baseline fail set on THIS binary (serve-smoke has a KNOWN
#                             non-empty fail set on the q9 pair: research/serve-st-20260803,
#                             4 checks, small-max_tokens routing). Never quote an old log.
#   B  pp2 dev01, spec ON   — THE new configuration. Verdict = B minus A is empty.
#   C  pp2 dev01, spec OFF  — the predecessor lane's shipped config, re-measured here so
#                             "spec over the split costs no checks" is a within-run claim.
#   D  door shut, spec OFF  — the control that proves C's fail set is not the door's doing.
#
# Receipts to ~/receipts/pp2spec/serve. GPU window held by the caller under flock.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2spec/serve
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
FAILS=0

# No draft arg: the box stages the 27B daily draft, not q9's, so serve-smoke's EXTERNAL-draft
# arm SKIPs by design (`[ -f "$DRAFT" ]`). The spec path under test is the EMBEDDED q9 MTP
# head, which is what the server self-specs with — that is the vehicle, not the external draft.
smoke() { # $1 = label, $2... = env words
  local label="$1"; shift
  echo "=== serve-smoke $label: env[$*] ==="
  env "$@" bash tools/serve-smoke.sh "$Q9" /nonexistent-draft.gguf 2>&1 \
    | tee "$OUT/serve-smoke-$label.log"
  local ex=${PIPESTATUS[0]}
  # The pp transport banner is printed by the SERVER, whose stdout serve-smoke.sh redirects to
  # a FIXED /tmp/serve-smoke.log that the next arm overwrites. Snapshot per arm (the
  # predecessor lane hit exactly this and reported a harness failure as a serving one).
  cp /tmp/serve-smoke.log "$OUT/server-stdout-$label.log" 2>/dev/null || true
  grep -h "  FAIL:" "$OUT/serve-smoke-$label.log" | sort > "$OUT/failset-$label.txt"
  echo "$label: $ex failed checks"
  return 0
}

smoke A-doorshut-speconn
sleep 5
smoke B-pp2-speconn MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1
sleep 5
smoke C-pp2-nospec  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_SERVE_SPEC=0
sleep 5
smoke D-doorshut-nospec MEMRA_SERVE_SPEC=0

echo; echo "==== fail sets ===="
for a in A-doorshut-speconn B-pp2-speconn C-pp2-nospec D-doorshut-nospec; do
  echo "-- $a:"; cat "$OUT/failset-$a.txt"
done

# THE verdict: B vs A — both spec-ON, only the split differs.
echo "-- ADDED BY THE SPLIT WITH SPEC ON, B minus A (must be empty):"
comm -13 "$OUT/failset-A-doorshut-speconn.txt" "$OUT/failset-B-pp2-speconn.txt" \
  | tee "$OUT/failset-added-spec.txt"
if [ -s "$OUT/failset-added-spec.txt" ]; then
  echo "FAIL: the split with spec ON ADDED serve-smoke failures"; FAILS=$((FAILS+1))
fi
# Secondary: B vs C — spec ON vs OFF over the SAME split.
echo "-- ADDED BY SPEC OVER THE SPLIT, B minus C:"
comm -13 "$OUT/failset-C-pp2-nospec.txt" "$OUT/failset-B-pp2-speconn.txt" \
  | tee "$OUT/failset-spec-over-split.txt"

# THE REFUSAL MUST BE GONE. Its text is what arm B used to die with, on every request.
echo "-- refusal text anywhere in arm B (must be absent):"
if grep -h "refused with the ppN door open" \
     "$OUT/serve-smoke-B-pp2-speconn.log" "$OUT/server-stdout-B-pp2-speconn.log" 2>/dev/null; then
  echo "FAIL: arm B still carries the verify refusal — the door did not open for serving"
  FAILS=$((FAILS+1))
else
  echo "(absent — the verify refusal is lifted on the serving path)"
fi

# Split liveness: arm B's server stdout must carry the cross-device banner. A green run
# without it proves nothing. if/else, not a `||`/`&&` chain (precedence bug precedent).
if grep -q "cross-device transport" "$OUT/server-stdout-B-pp2-speconn.log" 2>/dev/null; then
  echo "arm B: split CONFIRMED live — $(grep -m1 'cross-device transport' "$OUT/server-stdout-B-pp2-speconn.log")"
else
  echo "FAIL: no pp transport banner in arm B's server stdout — may have served single-device"
  FAILS=$((FAILS+1))
fi
# And the door-shut control must NOT show it (proof the check reads zero where zero is).
if grep -q "cross-device transport" "$OUT/server-stdout-A-doorshut-speconn.log" 2>/dev/null; then
  echo "FAIL: door-SHUT arm showed a transport banner — the door leaks"; FAILS=$((FAILS+1))
else
  echo "control: door-shut arm shows NO transport banner (the liveness check reads zero)"
fi
# Spec liveness in arm B: a server that silently fell back to plain decode would pass every
# check and prove nothing about spec-over-split. Spec bursts emit ONE Token event per accepted
# run, and the worker logs the spec session path — grep the server's own spec marker.
echo "-- spec liveness markers in arm B's server stdout:"
grep -h -m5 -i "spec" "$OUT/server-stdout-B-pp2-speconn.log" 2>/dev/null || echo "(none found)"

nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv > "$OUT/gpu-post.csv"
echo "script-detected failures: $FAILS"
exit $FAILS

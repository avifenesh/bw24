#!/usr/bin/env bash
# lane/step-draft — the ON-BOX assertion, on the real Step-3.7-Flash artifact.
#
# Runs on the 2x RTX PRO 6000 box (96 GB each). Step-3.7-Flash IQ4_XS is ~105 GB, so it does
# NOT fit one card: every arm here is a PP-2 arm. That is the whole point — PP-2 is precisely
# the regime where spec is quarantined (#87), and the two things this lane owes the operator
# are (1) a step35 model served WITHOUT a drafter says so, and (2) a step35 model served WITH
# a drafter over PP-2 with spec armed REFUSES rather than booting into a context-killing bug.
#
# Three arms, all needing the real artifact and the real `arch.is_step35()` bit — none of them
# can run on the 5090 (no artifact) and none of them can be faked by a unit test (the arch bit
# comes off the loaded GGUF):
#   E: step35 over PP-2, NO drafter            -> WARNS (the silent class, now audible)
#   F: step35 over PP-2, drafter, spec ARMED   -> REFUSES with the #87 pointer
#   G: step35 over PP-2, drafter, MEMRA_SERVE_SPEC=0 -> boots and SERVES (the quarantine
#      configuration is not collateral damage: attaching a drafter must not brick PP-2 serving)
#
# Usage (on the box): bash ~/memra/research/step-draft-20260807/run-box-assert.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
R=research/step-draft-20260807/raw
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$R/box-assert-$STAMP.log
ADDR=127.0.0.1:8181
mkdir -p "$R"

[ -f "$M" ] || { echo "SKIP: no Step trunk at $M"; exit 0; }
[ -f "$D" ] || { echo "SKIP: no Step MTP head at $D"; exit 0; }
[ -x target/release/memra-server ] || { echo "FAIL: build memra-server first"; exit 1; }

# PP-2 across both cards, exactly as the step37 lane booted Step (research/step37-p2-20260806:
# `MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1`, nothing else) — which is ALSO the #87 regime.
#
# The placement matters and an earlier draft of this script had it wrong: `MEMRA_PP_SHARD=0` or
# `MEMRA_PP_STREAMS=0` both make `pp_sharded_cross_device()` FALSE — those seams bring every
# weight home to the primary, so nothing is remote and the quarantine legitimately does not
# bind. Setting either would have produced a script whose arm F "passed" by never entering the
# regime under test. Defaults (shard on, streams on) are the regime that measured c=4 -> 0/48.
PP=(MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1)

FAILF=$(mktemp /tmp/step-draft-box-fails.XXXXXX)
PASS() { echo "  ok: $1" | tee -a "$LOG"; }
FAIL() { echo "  FAIL: $1" | tee -a "$LOG"; echo x >> "$FAILF"; }
SKIP() { echo "  SKIP: $1" | tee -a "$LOG"; }

{ echo "=== lane/step-draft ON-BOX assertion $STAMP"
  echo "=== host: $(hostname)  rig: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' '/')"
  echo "=== trunk: $M"; echo "=== draft: $D"
  echo "=== pp: ${PP[*]}"
} > "$LOG"

# Step is 105 GB over PP-2: the load is minutes, not seconds. 40 min ceiling.
boot() {  # $1 = MEMRA_MODELS spec, $2 = logfile, rest = extra env
  local spec=$1 out=$2; shift 2
  SPID=; ST=
  env "${PP[@]}" "$@" MEMRA_MODELS="$spec" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$out" 2>&1 &
  SPID=$!
  for _ in $(seq 1200); do
    curl -sf http://$ADDR/health >/dev/null 2>&1 && { ST=UP; return 0; }
    kill -0 "$SPID" 2>/dev/null || { ST=DOWN; wait "$SPID" 2>/dev/null; SPID=; return 0; }
    sleep 2
  done
  ST=TIMEOUT; return 0
}
# `kill 0` signals the whole PROCESS GROUP — never default SPID to 0.
kill_server() {
  [ -n "${SPID:-}" ] || return 0
  kill "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null || true
  SPID=
  for _ in $(seq 60); do
    curl -sf http://$ADDR/health >/dev/null 2>&1 || return 0
    sleep 1
  done
}
trap kill_server EXIT

exec 3>&1
(
flock 9 || { echo "could not take the GPU lock" >&3; exit 1; }
echo "lock acquired $(date -Is)" | tee -a "$LOG" >&3

# ---- ARM E: step35 + PP-2 + NO drafter -> the WARNING fires ----
# THE arm the 5090 cannot run: it needs `arch.is_step35()` off a real Step GGUF. Before this
# lane, this configuration was the silent defect — served plain decode, no error, no log line.
echo "########## ARM E: step35, no drafter -> WARNS ##########" | tee -a "$LOG" >&3
E=$R/box-armE-warn-$STAMP.log
boot "step=$M" "$E"
if [ "$ST" = UP ]; then
  grep -q "no MTP drafter attached" "$E" \
    && PASS "E: the warning fired on a real step35 load" \
    || FAIL "E: step35 served plain decode SILENTLY — the defect is still live"
  grep -q "does NOT mean the model has no drafter" "$E" \
    && PASS "E: warning explains nextn=0 is expected for this arch" \
    || FAIL "E: warning missing the nextn=0 explanation"
  grep -q "+/path/to/Step3.7-flash-mtp-Q8_0.gguf" "$E" \
    && PASS "E: warning carries the actionable attach spelling" \
    || FAIL "E: warning does not say HOW to attach"
else
  FAIL "E: server did not come up ($ST)"; tail -30 "$E" | tee -a "$LOG" >&3
fi
kill_server

# ---- ARM F: step35 + PP-2 + drafter + spec ARMED -> REFUSE (#87) ----
# The load-time refusal. Booting green here means the second concurrent spec session takes the
# whole CUDA context down (measured c=4 -> 0/48, research/pp2-spec-20260806).
echo "########## ARM F: step35 + drafter + spec armed over PP-2 -> REFUSE ##########" | tee -a "$LOG" >&3
F=$R/box-armF-refuse-pp2spec-$STAMP.log
boot "step=$M+$D" "$F"
if [ "$ST" = DOWN ]; then
  PASS "F: refused to start (did not boot into the #87 regime)"
  grep -q "REFUSING to start" "$F" \
    && PASS "F: refusal is explicit" || FAIL "F: no REFUSING line"
  grep -q "#87" "$F" && PASS "F: refusal points at the quarantine issue" \
    || FAIL "F: refusal does not cite #87"
  grep -q "research/pp2-spec-20260806" "$F" \
    && PASS "F: refusal points at the receipts" || FAIL "F: no receipts pointer"
  grep -q "MEMRA_SERVE_SPEC=0" "$F" \
    && PASS "F: refusal names the fix" || FAIL "F: refusal offers no fix"
  # The refusal must beat the 105 GB load: burning 20 minutes of PP-2 weight streaming to
  # then refuse on a decision knowable from env alone is a bug, not a gate pass. `pp_cuts`/
  # `pp_sharded_cross_device` are env-only and callable pre-runtime for exactly this reason.
  grep -q "Engine ready" "$F" \
    && FAIL "F: refused only AFTER the engine/model load — decide from env, before the load" \
    || PASS "F: refused BEFORE the 105 GB load"
elif [ "$ST" = UP ]; then
  FAIL "F: BOOTED with spec armed over PP-2 — #87 quarantine is not enforced at load"
  tail -30 "$F" | tee -a "$LOG" >&3
else
  FAIL "F: neither up nor down ($ST)"; tail -30 "$F" | tee -a "$LOG" >&3
fi
kill_server

# ---- ARM G: step35 + PP-2 + drafter + MEMRA_SERVE_SPEC=0 -> boots and SERVES ----
# The quarantine must not be collateral. An operator who attaches the drafter today (so the
# config is ready for when #87 lifts) must still get a serving server under the standing flag.
echo "########## ARM G: drafter attached + spec disarmed -> boots and serves ##########" | tee -a "$LOG" >&3
G=$R/box-armG-quarantine-serves-$STAMP.log
boot "step=$M+$D" "$G" MEMRA_SERVE_SPEC=0
if [ "$ST" = UP ]; then
  PASS "G: booted with the drafter attached under the standing quarantine flag"
  grep -q "REFUSING to start" "$G" \
    && FAIL "G: refused despite spec being disarmed" || PASS "G: no spurious refusal"
  grep -q "no MTP drafter attached" "$G" \
    && FAIL "G: warned despite an attached drafter" || PASS "G: no spurious warning"
  curl -sf -m 600 http://$ADDR/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"step","prompt":"Name three prime numbers.","max_tokens":32,"temperature":0}' \
    > "$R/box-armG-gen-$STAMP.json" 2>&1
  grep -q '"text"' "$R/box-armG-gen-$STAMP.json" \
    && PASS "G: served a completion over PP-2 with the drafter loaded" \
    || FAIL "G: generation failed"
else
  FAIL "G: server did not come up ($ST)"; tail -30 "$G" | tee -a "$LOG" >&3
fi
kill_server

echo "lock released $(date -Is)" | tee -a "$LOG" >&3
) 9> /tmp/memra-gpu.lock

FAILS=$(wc -l < "$FAILF" | tr -d ' ')
rm -f "$FAILF"
echo "=== FAILS=$FAILS" | tee -a "$LOG"
echo "=== log: $LOG"
[ "$FAILS" -eq 0 ]

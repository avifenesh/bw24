#!/usr/bin/env bash
# lane/step-draft — the LOUD drafter semantics, asserted against a REAL booted server.
#
# The unit tests pin `draft_verdict`'s decision and its message text GPU-free. What they
# cannot pin is that the load path actually CALLS it, on a real model, with the real
# `mtp.is_some()` and the real env reads. That is what this does: four server boots on the
# 5090 with the 9B pair, each asserting one branch of the semantics from the server's own
# stderr and exit behavior.
#
# The 9B stands in for step35 on three of the four arms deliberately — there is no Step
# artifact on this rig (105 GB), and the arms that do not need step35's arch bit are
# arch-independent by construction (the refusal/quoted-error paths). Arm B is the one that
# needs `arch.is_step35()` and is therefore ON-BOX only; it is asserted here in the negative
# (a non-step35 model must NOT warn — the no-noise half of the contract).
#
# Usage: research/step-draft-20260807/run-loud-gates.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
D9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
R=research/step-draft-20260807/raw
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$R/loud-gates-5090-$STAMP.log
ADDR=127.0.0.1:8179
mkdir -p "$R"

[ -f "$Q9" ] || { echo "SKIP: no model at $Q9"; exit 0; }
[ -x target/release/memra-server ] || cargo build --release -p memra-server || exit 1

# FAILS lives in a FILE, not a variable: every arm runs inside the `flock` subshell, and a
# shell variable incremented there cannot reach the exit check outside it — the script would
# have exited 0 with failing arms. (The second of this gate's own two bugs.)
FAILF=$(mktemp /tmp/step-draft-fails.XXXXXX)
PASS() { echo "  ok: $1" | tee -a "$LOG"; }
FAIL() { echo "  FAIL: $1" | tee -a "$LOG"; echo x >> "$FAILF"; }
# SKIP is neither: the arm did not run, so it gets no verdict. Scoring an unrun arm green is
# the dishonesty this whole lane exists to remove; scoring it red is a false alarm that trains
# the operator to ignore the gate.
SKIP() { echo "  SKIP: $1" | tee -a "$LOG"; }

# A "refused" arm must have refused for OUR reason. A server that dies of trunk OOM on a
# shared card also exits nonzero, and the first run of this gate scored exactly that as a
# PASS on arm C — the refusal assertions grepped a log whose only FATAL was
# CUDA_ERROR_OUT_OF_MEMORY, from a load that never reached the draft path. Any arm that
# expects DOWN must first prove the death was not incidental.
#
# Arm C's incidental OOM is a SKIP (the trunk load never reached the draft path — nothing was
# tested). Arm D's is a FAIL: D asserts the refusal lands at PARSE time, so a trunk load
# happening at all means the parse-time check did not fire, which is exactly the regression.
not_incidental() {  # $1 = logfile, $2 = arm label, $3 = skip|fail on incidental OOM
  if grep -q "CUDA_ERROR_OUT_OF_MEMORY" "$1"; then
    if [ "${3:-fail}" = skip ]; then
      SKIP "$2: card busy (trunk OOM before the draft path) — arm not run"
    else
      FAIL "$2: reached a trunk GPU load at all — the parse-time refusal did not fire"
    fi
    return 1
  fi
  return 0
}

{ echo "=== lane/step-draft loud-drafter gates $STAMP"
  echo "=== commit: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
  echo "=== rig: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "=== model: $Q9"; echo "=== draft: $D9"
} > "$LOG"

# Boot a server and set ST=UP|DOWN|TIMEOUT plus SPID. NOT a command substitution: `ST=$(boot)`
# runs the function in a SUBSHELL, so SPID would never propagate to the caller and every
# later arm would silently health-check the PREVIOUS arm's still-running server. (Found by
# this script failing after arm A — see RESULTS.md §"the gate's own two bugs".)
boot() {  # $1 = MEMRA_MODELS spec, $2 = logfile, rest = extra env
  local spec=$1 out=$2; shift 2
  SPID=; ST=
  env "$@" MEMRA_MODELS="$spec" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$out" 2>&1 &
  SPID=$!
  for _ in $(seq 90); do
    curl -sf http://$ADDR/health >/dev/null 2>&1 && { ST=UP; return 0; }
    kill -0 "$SPID" 2>/dev/null || { ST=DOWN; wait "$SPID" 2>/dev/null; SPID=; return 0; }
    sleep 2
  done
  ST=TIMEOUT; return 0
}
# NEVER `kill "${SPID:-0}"`: `kill 0` signals the ENTIRE PROCESS GROUP, i.e. this script.
# An unset SPID means "nothing to kill", not "kill everything".
kill_server() {
  [ -n "${SPID:-}" ] || return 0
  kill "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null || true
  SPID=
  # The next arm rebinds $ADDR, so make sure the port is actually released.
  for _ in $(seq 30); do
    curl -sf http://$ADDR/health >/dev/null 2>&1 || return 0
    sleep 1
  done
}
trap kill_server EXIT

exec 3>&1  # so the flock subshell's output still reaches the terminal

(
flock 9 || { echo "could not take the 5090 lock" >&3; exit 1; }
echo "lock acquired $(date -Is)" | tee -a "$LOG" >&3

# ---- ARM A: drafter attached over a '+draft' spec -> attaches, quiet, spec live ----
echo "########## ARM A: '+draft' attaches, no warning ##########" | tee -a "$LOG" >&3
A=$R/armA-attached-$STAMP.log
boot "m=$Q9+$D9" "$A"
if [ "$ST" = UP ]; then
  ARM_A_RAN=1
  grep -q "regime draft attached" "$A" \
    && PASS "A: drafter attached line present" || FAIL "A: no attach line"
  grep -q "no MTP drafter attached" "$A" \
    && FAIL "A: warned despite an attached drafter" || PASS "A: no spurious warning"
  # spec must actually RUN, not merely be eligible: [spec-acc] is the liveness receipt.
  curl -sf -m 120 http://$ADDR/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"m","prompt":"Name three primes.","max_tokens":48,"temperature":0}' \
    > "$R/armA-gen-$STAMP.json" 2>&1
  grep -q '"text"' "$R/armA-gen-$STAMP.json" \
    && PASS "A: generation served" || FAIL "A: generation failed"
elif grep -q "CUDA_ERROR_OUT_OF_MEMORY" "$A"; then
  # SKIP, not PASS and not FAIL: another tenant holds the card, so this arm was never run.
  # Recording it as anything else would be the same dishonesty arm C nearly committed.
  SKIP "A: card busy (another tenant holds VRAM) — arm not run"
else
  FAIL "A: server did not come up ($ST)"; tail -20 "$A" | tee -a "$LOG" >&3
fi
kill_server

# ---- ARM B (negative): a NON-step35 model with no drafter must stay QUIET ----
# The no-noise half of the contract. A warning on every headless model the server has ever
# hosted is how a real warning gets ignored.
echo "########## ARM B: non-step35, no drafter -> quiet ##########" | tee -a "$LOG" >&3
B=$R/armB-quiet-$STAMP.log
boot "m=$Q9" "$B"
if [ "$ST" = UP ]; then
  grep -q "no MTP drafter attached" "$B" \
    && FAIL "B: warned on a non-step35 model (noise)" || PASS "B: quiet on non-step35"
elif grep -q "CUDA_ERROR_OUT_OF_MEMORY" "$B"; then
  SKIP "B: card busy (another tenant holds VRAM) — arm not run"
else
  FAIL "B: server did not come up ($ST)"; tail -20 "$B" | tee -a "$LOG" >&3
fi
kill_server

# ---- ARM C: a drafter path that CANNOT load -> REFUSE, with the cause quoted ----
echo "########## ARM C: bad drafter path -> refuse to start ##########" | tee -a "$LOG" >&3
C=$R/armC-refuse-baddraft-$STAMP.log
BAD=/tmp/step-draft-not-a-gguf-$$.gguf
printf 'this is not a GGUF file' > "$BAD"
boot "m=$Q9+$BAD" "$C"
if [ "$ST" = DOWN ]; then
  if not_incidental "$C" C skip; then
    PASS "C: refused to start (did not degrade to plain decode)"
    grep -q "FATAL: worker init failed" "$C" \
      && PASS "C: refusal is FATAL, not a warning" || FAIL "C: no FATAL line"
    grep -q "refusing to start rather than silently serving plain decode" "$C" \
      && PASS "C: refusal names the silent-degradation it is preventing" \
      || FAIL "C: refusal text missing the rationale"
    grep -q "$BAD" "$C" \
      && PASS "C: refusal quotes the offending path" || FAIL "C: path not quoted"
  fi
else
  FAIL "C: server came up with an unloadable drafter ($ST) — SILENT DEGRADATION"
  tail -20 "$C" | tee -a "$LOG" >&3
fi
kill_server
rm -f "$BAD"

# ---- ARM D: a MISSING drafter path -> refuse at PARSE time, BEFORE touching the GPU ----
# This arm is why the drafter path now gets `validate_model_path`'s existence check: before
# the fix, a typo'd path survived parse and only failed after the whole trunk load, so on a
# busy card the operator saw CUDA_ERROR_OUT_OF_MEMORY on the TRUNK and never learned the
# drafter path was wrong. The "before the GPU" assertion is the real content here.
echo "########## ARM D: missing drafter path -> parse-time refusal ##########" | tee -a "$LOG" >&3
D=$R/armD-refuse-missing-$STAMP.log
boot "m=$Q9+/nonexistent/draft.gguf" "$D"
if [ "$ST" = DOWN ]; then
  if not_incidental "$D" D fail; then
    PASS "D: refused to start on a nonexistent drafter path"
    grep -q "drafter path" "$D" && grep -q "does not exist" "$D" \
      && PASS "D: cause is quoted and names the DRAFTER (not the trunk)" \
      || FAIL "D: no quoted drafter cause"
    # PARSE-time, not load-time: the refusal must precede the Engine/model load entirely.
    grep -q "Engine ready" "$D" \
      && FAIL "D: refused only AFTER the GPU load — the whole point was to refuse before it" \
      || PASS "D: refused BEFORE any GPU/model work"
  fi
else
  FAIL "D: server came up with a nonexistent drafter ($ST) — SILENT DEGRADATION"
  tail -20 "$D" | tee -a "$LOG" >&3
fi
kill_server

echo "lock released $(date -Is)" | tee -a "$LOG" >&3
) 9> /tmp/memra-5090.lock

FAILS=$(wc -l < "$FAILF" | tr -d ' ')
rm -f "$FAILF"
echo "=== FAILS=$FAILS" | tee -a "$LOG"
echo "=== log: $LOG"
[ "$FAILS" -eq 0 ]

#!/usr/bin/env bash
# lane/step-draft — the #87 preflight, asserted against a REAL booted server, GPU-FREE.
#
# Why this exists as a separate gate from run-loud-gates.sh: the whole POINT of the preflight is
# that it decides before any CUDA context is created, so it is the one arm that can be asserted
# on a fully contended card — and it was written on a rig whose 5090 was 22.4/24.5 GB occupied by
# another lane. A gate that only runs when the box is idle is a gate that does not run.
#
# It also needs no artifact: the trunk path is only opened AFTER the preflight verdict, so the 9B
# stands in for Step-3.7-Flash with nothing lost. The arms that DO need the real 105 GB artifact
# and the real `arch.is_step35()` bit are in run-box-assert.sh.
#
# Three arms. Arm H is the refusal; arms I are the no-collateral counter-arms, and they are not
# optional — a refusal that fires one term too wide takes the 105 GB SKU offline entirely, since
# PP-2 is the only placement it fits in at all.
#
# Usage: research/step-draft-20260807/run-preflight-gate.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
D9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
R=research/step-draft-20260807/raw
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$R/preflight-gate-$STAMP.log
mkdir -p "$R"

[ -f "$Q9" ] || { echo "SKIP: no model at $Q9"; exit 0; }
[ -x target/release/memra-server ] || cargo build --release -p memra-server || exit 1

FAILS=0
PASS() { echo "  ok: $1" | tee -a "$LOG"; }
FAIL() { echo "  FAIL: $1" | tee -a "$LOG"; FAILS=$((FAILS + 1)); }

{ echo "=== lane/step-draft #87 preflight gate $STAMP"
  echo "=== commit: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
  echo "=== NOTE: no GPU needed — the preflight decides before any CUDA context exists."
  echo "=== rig (for the record, contention and all): $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
} > "$LOG"

# THE #87 regime: 2 stages across 2 distinct devices with the sharded loader and stage streams
# at their DEFAULTS. `MEMRA_PP_SHARD=0` and `MEMRA_PP_STREAMS=0` each bring every weight home to
# the primary, which makes `pp_sharded_cross_device()` false — set either and this gate would
# assert a refusal that legitimately should not fire. (An earlier draft of the box script did
# exactly that; it would have "passed" arm F without ever entering the regime under test.)
PP87="MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1"

# ---- ARM H: drafter attached + spec armed + sharded cross-device PP-2 -> REFUSE, pre-CUDA ----
echo "########## ARM H: the #87 preflight refuses BEFORE any GPU work ##########" | tee -a "$LOG"
H=$R/armH-preflight-refusal-nogpu-$STAMP.log
{ echo "=== arm H: #87 preflight refuses BEFORE any GPU work (no card, no artifact needed)"
  echo "=== env: $PP87 MEMRA_MODELS=step=<trunk>+<draft>"; } > "$H"
# shellcheck disable=SC2086
env $PP87 MEMRA_ADDR=127.0.0.1:8183 MEMRA_MODELS="step=$Q9+$D9" \
  timeout 120 target/release/memra-server >> "$H" 2>&1
echo "EXIT=$?" >> "$H"
grep -q "REFUSING to start" "$H" \
  && PASS "H: refused" || FAIL "H: did NOT refuse the #87 regime"
grep -q "#87" "$H" && PASS "H: cites the quarantine issue" || FAIL "H: no #87 pointer"
grep -q "research/pp2-spec-20260806" "$H" \
  && PASS "H: cites the receipts" || FAIL "H: no receipts pointer"
grep -q "CUDA_ERROR_ILLEGAL_ADDRESS" "$H" \
  && PASS "H: quotes the measured cause" || FAIL "H: cause not quoted"
grep -q "MEMRA_SERVE_SPEC=0" "$H" && PASS "H: names the fix" || FAIL "H: no fix named"
# THE assertion this arm exists for: no engine, no load, no CUDA. `[worker] Engine ready` is the
# first thing the worker prints after `Engine::new`, and `loading model` is the load itself.
grep -q "Engine ready" "$H" \
  && FAIL "H: created a CUDA context before refusing" || PASS "H: no CUDA context created"
grep -q "loading model" "$H" \
  && FAIL "H: started the weight load before refusing (the 20-minute bug)" \
  || PASS "H: refused before the weight load"

# ---- ARMS I: the no-collateral counter-arms — the preflight must NOT fire ----
# One term removed each. Both configs must reach the engine: the standing quarantine flag is how
# PP-2 serves TODAY (872-875 tok/s at c=8) and single-card is where spec is fully live, so a
# preflight that swallowed either would be a worse regression than the bug it guards.
for arm in "quarantine:MEMRA_SERVE_SPEC=0" "doorshut:MEMRA_PP_STAGES=1"; do
  lbl=${arm%%:*}; ev=${arm#*:}
  echo "########## ARM I/$lbl: preflight must NOT fire ($ev) ##########" | tee -a "$LOG"
  I=$R/armI-$lbl-no-refusal-$STAMP.log
  { echo "=== arm I/$lbl: the preflight must NOT fire ($ev)"; } > "$I"
  # shellcheck disable=SC2086
  env $PP87 $ev MEMRA_ADDR=127.0.0.1:8184 MEMRA_MODELS="step=$Q9+$D9" \
    timeout 90 target/release/memra-server >> "$I" 2>&1
  echo "EXIT=$?" >> "$I"
  grep -q "REFUSING to start" "$I" \
    && FAIL "I/$lbl: refused a config that must serve" \
    || PASS "I/$lbl: no spurious refusal"
  # And it must have gotten PAST the preflight, not merely failed to print the refusal. On a
  # contended card the trunk load OOMs — that is fine and still proves the point: reaching a
  # CUDA context at all means the preflight let the config through.
  grep -qE "Engine ready|CUDA_ERROR_OUT_OF_MEMORY" "$I" \
    && PASS "I/$lbl: reached the engine (preflight passed it through)" \
    || FAIL "I/$lbl: never reached the engine — something else stopped it, arm inconclusive"
done

echo "=== FAILS=$FAILS" | tee -a "$LOG"
echo "=== log: $LOG"
[ "$FAILS" -eq 0 ]

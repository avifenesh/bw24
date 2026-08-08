#!/usr/bin/env bash
# step35-prime-batch-gate — exact cross-request prime gate over the real PP-2 placement.
#
# Assertions:
#   1. B=2 uneven prompts beyond the 512-token SWA window produce bit-identical logits,
#      h_seed, full hidden stacks, and teacher-forced decode logits vs serial primes.
#   2. The dedicated step35 batch path ran.
#   3. Its PP stage split ran; an unsplit whole-trunk walk is a vacuous correctness pass.
#
# Registered RED (lane/cx-prime-batch, 2026-08-08): prime_cache_batch currently refuses
# step35, so the gate exits nonzero before any liveness counter advances.
#
# --canary sets MEMRA_STEP35_PRIME_BATCH=0. Once the mechanism lands, this must restore the
# refusal and break the naked gate. While the gate is registered-red, both arms are red by
# construction; the canary becomes load-bearing with the implementation commit.
set -uo pipefail
cd "$(dirname "$0")/.."

CANARY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --canary) CANARY=1; shift ;;
        *) echo "step35-prime-batch-gate: unknown arg $1" >&2; exit 2 ;;
    esac
done

MODEL="${MEMRA_STEP37_GGUF:-}"
if [ -z "$MODEL" ]; then
    for cand in \
        "$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
        "/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
        "/data/models/step37/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
        "/data/models/step37/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"; do
        [ -f "$cand" ] && { MODEL="$cand"; break; }
    done
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || {
    echo "step35-prime-batch-gate: SKIP (no Step-3.7-Flash artifact; set MEMRA_STEP37_GGUF)"
    exit 0
}

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$NGPU" -ge 2 ] || {
    echo "step35-prime-batch-gate: SKIP (needs the two-GPU PP placement, have $NGPU)"
    exit 0
}

BIN=./target/release/prime-batch-gate
[ -x "$BIN" ] || {
    echo "step35-prime-batch-gate: FAIL (no $BIN — build release first)"
    exit 1
}

TS=$(date -u +%Y%m%dT%H%M%SZ)
TAG=$([ "$CANARY" = 1 ] && echo canary || echo naked)
D=research/primebatch-20260808/raw
mkdir -p "$D"
LOG=$D/primebatch35-$TAG-$TS.log
exec > >(tee "$LOG") 2>&1

echo "=== step35-prime-batch-gate tag=$TAG ts=$TS model=$MODEL ==="
RC=1
(
    flock -w 3600 9 || { echo "LOCK TIMEOUT"; exit 75; }
    echo "lock acquired $(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    CANARY_ENV=()
    [ "$CANARY" = 1 ] && CANARY_ENV=(MEMRA_STEP35_PRIME_BATCH=0)
    env "${CANARY_ENV[@]}" \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 \
        "$BIN" "$MODEL" --batch 2 --plen 520 --steps 4 --exact --require-pp-split
    rc=$?
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
) 9>/tmp/memra-gpu.lock
RC=$?

if [ "$CANARY" = 1 ]; then
    if [ "$RC" -ne 0 ]; then
        echo "step35-prime-batch-gate: CANARY OK (rollback broke exact+live as required)"
        exit 0
    fi
    echo "step35-prime-batch-gate: CANARY FAILED (gate passed with the rollback seam on)"
    exit 1
fi

if [ "$RC" -eq 0 ]; then
    echo "step35-prime-batch-gate: PASS (serial identity + live PP-split batch)"
else
    echo "step35-prime-batch-gate: FAIL rc=$RC (refusal, mismatch, or split not live)"
fi
exit "$RC"

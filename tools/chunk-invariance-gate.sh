#!/usr/bin/env bash
# chunk-invariance-gate — asserts that chunked prefill is reduction-order-stable, i.e. that
# the SAME prompt primed at DIFFERENT MEMRA_PRIME_CHUNK values yields byte-identical greedy
# output. Self-gating (`kind=cmd` in tools/fast-gate/models.tsv): exit 0 = PASS.
#
#   tools/chunk-invariance-gate.sh [<model.gguf>] [--chunks 2048,64,32] [--steps N]
#                                 [--expect-invariant|--expect-variant] [--canary]
#
# WHY THIS GATE EXISTS (research/chunk-invariance-20260805/VERDICT.md):
# MEMRA_PRIME_CHUNK is documented as a machine-config/OOM knob, but it also decides the
# prefill's arithmetic — so two rigs with different values produced DIFFERENT greedy text for
# the same prompt (97- and 149-token prompts, zero cache reuse). vLLM hit the same class
# twice (#38561 chunked-prefill splits pinned to a fixed grain, #45683 deterministic MoE
# combine) and both fixes are the same shape: constrain the reduction segmentation to a fixed
# grain, then PIN the property with an asserted test so nobody breaks it silently (#40372).
# This is that asserted test.
#
# DEFAULT MODE is --expect-invariant (flipped 2026-08-05, lane/chunkinv-flip): the grain-free
# fix dropped full_attn_prime_fa_dispatch's base_len==0 f32 special case, so EVERY chunk
# (including chunk 0) attends through the quantized KV cache — one numeric class for every
# row, and chunked prefill is byte-identical across MEMRA_PRIME_CHUNK values with NO door or
# grain env. That byte-identity is now the shipped default contract this gate asserts NAKED
# (no env). --expect-variant remains as the LEGACY-arithmetic assertion: it runs under
# MEMRA_PRIME_F32CHUNK0=1 (the rollback seam that restores the old f32 first-chunk class
# edge) and asserts the pinned divergence still reproduces there.
#
# TEETH: --canary INJECTS A BREAK (it does not merely relabel the expectation) and requires
# the gate's assertion to FAIL, proving the gate can fail. Under --expect-invariant the
# canary sets MEMRA_PRIME_F32CHUNK0=1 — the legacy arithmetic is chunk-VARIANT, so the
# invariant assertion must break. Under --expect-variant the canary unsets it (the fixed
# default is invariant, so the variant assertion must break).
# NOTE (trap, hit twice on this lane): a canary that flips only the EXPECTED label re-runs the
# identical configuration, so it passes exactly when the default gate passes and fails exactly
# when it fails — perfectly correlated, therefore worthless as a teeth check. The canary must
# change the WORLD, not the label. (Earlier vacuous shape: a single --chunks value, which has
# nothing to compare and always reported CHUNK-INVARIANT.)
set -uo pipefail
cd "$(dirname "$0")/.."
PROBE=./target/release/concat-prime-probe
D=research/chunk-invariance-20260805
MODEL=""
CHUNKS=2048,64,32
STEPS=48
EXPECT=invariant
CANARY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --chunks) CHUNKS="$2"; shift 2 ;;
        --steps)  STEPS="$2"; shift 2 ;;
        --expect-invariant) EXPECT=invariant; shift ;;
        --expect-variant)   EXPECT=variant; shift ;;
        --canary) CANARY=1; shift ;;
        -*) echo "chunk-invariance-gate: unknown arg $1" >&2; exit 2 ;;
        *)  MODEL="$1"; shift ;;
    esac
done
# default model = the family the finding was measured on (qwen hybrid NVFP4, GDN linear-attn)
MODEL="${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
[ -f "$MODEL" ] || { echo "chunk-invariance-gate: SKIP (no model at $MODEL)"; exit 0; }
[ -x "$PROBE" ] || { echo "chunk-invariance-gate: FAIL (build concat-prime-probe first)"; exit 1; }

# the two prompt lengths the original finding pinned as divergent (97 and 149 tokens)
PROMPTS="$D/prompt-turn1.txt $D/prompt-turn2.txt"
for p in $PROMPTS; do
    [ -f "$p" ] || { echo "chunk-invariance-gate: FAIL (missing pinned prompt $p)"; exit 1; }
done

LOG=$(mktemp /tmp/chunkinv-gate-XXXXXX.log)
# The assertion under test is always EXPECT. The canary does not change the assertion — it
# changes the WORLD (the MEMRA_PRIME_F32CHUNK0 legacy-arithmetic seam on/off), so a working
# gate must report FAIL on the canary run. LEGACY=on restores the pre-fix f32 first-chunk
# class edge (chunk-variant); LEGACY=off is the shipped grain-free default (invariant).
WANT="$EXPECT"
LEGACY=off
[ "$WANT" = variant ] && LEGACY=on
[ "$CANARY" = 1 ] && { [ "$LEGACY" = on ] && LEGACY=off || LEGACY=on; }
ENVX=()
[ "$LEGACY" = on ] && ENVX=(MEMRA_PRIME_F32CHUNK0=1)

rc_all=0
saw_variant=0
saw_invariant=0
for p in $PROMPTS; do
    # evidence discipline: tee the raw log, parse the LOG (never the pipe)
    env "${ENVX[@]}" "$PROBE" "$MODEL" chunkinv --prompt-a "@$p" \
        --chunks "$CHUNKS" --steps "$STEPS" >> "$LOG" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "chunk-invariance-gate: FAIL (probe exit $rc on $p)"; tail -5 "$LOG"; exit 1; }
done
if grep -q "chunkinv verdict: CHUNK-INVARIANT" "$LOG"; then saw_invariant=1; fi
if grep -q "chunkinv verdict: \*\*\* CHUNK-DEPENDENT" "$LOG"; then saw_variant=1; fi
[ $saw_invariant -eq 0 ] && [ $saw_variant -eq 0 ] && {
    echo "chunk-invariance-gate: FAIL (no verdict line in probe output — probe contract changed)"
    tail -10 "$LOG"; exit 1; }
# ANY diverging prompt makes the run variant: under --expect-invariant every pinned prompt must
# be exact, so one CHUNK-DEPENDENT verdict among N must not be masked by the others.
if [ $saw_variant -eq 1 ]; then GOT=variant; else GOT=invariant; fi
# expect-variant additionally requires that the pinned divergence still shows on BOTH prompts —
# a partial disappearance is a silent behavior change, which is exactly what this gate guards.
if [ "$WANT" = variant ] && [ "$CANARY" = 0 ]; then
    nvar=$(grep -c "chunkinv verdict: \*\*\* CHUNK-DEPENDENT" "$LOG")
    npr=$(set -- $PROMPTS; echo $#)
    [ "$nvar" -lt "$npr" ] && {
        echo "chunk-invariance-gate: FAIL — pinned divergence now shows on only $nvar/$npr prompts"
        echo "  the chunk-order class CHANGED without the door; re-root-cause before touching the gate"
        grep -E "chunkinv verdict" "$LOG" | sed 's/^/    /'; echo "  raw log: $LOG"; exit 1; }
fi

echo "chunk-invariance-gate: assert=$WANT legacy-seam=$LEGACY got=$GOT canary=$CANARY chunks=$CHUNKS model=$(basename "$MODEL")"
grep -E "^ *(2048|64|32|chunkinv verdict)" "$LOG" | sed 's/^/    /'
if [ "$GOT" = "$WANT" ]; then
    if [ "$CANARY" = 1 ]; then
        # the injected break did NOT move the verdict => the assertion is insensitive to the
        # very mechanism it claims to guard, so a real regression would also slip through.
        echo "chunk-invariance-gate: CANARY UNEXPECTEDLY MATCHED — flipping the legacy seam did not"
        echo "  change the verdict, so this assertion cannot detect the mechanism. FIX THE GATE. (log $LOG)"
        rc_all=1
    else
        echo "chunk-invariance-gate: PASS (raw log $LOG)"
    fi
elif [ "$CANARY" = 1 ]; then
    echo "chunk-invariance-gate: PASS (canary broke the assertion as required — gate has teeth; log $LOG)"
else
    echo "chunk-invariance-gate: FAIL — chunk-order behavior CHANGED (wanted $WANT, got $GOT)."
    if [ "$WANT" = variant ]; then
        echo "  The MEMRA_PRIME_F32CHUNK0 legacy seam no longer reproduces the pinned divergence"
        echo "  — the rollback arithmetic changed; re-root-cause before touching the gate."
    else
        echo "  The DEFAULT grain-free prefill no longer delivers byte-identity across chunk"
        echo "  sizes — a chunk-variant class edge came back. Re-root-cause (VERDICT.md protocol)."
    fi
    echo "  raw log: $LOG"
    rc_all=1
fi
exit $rc_all

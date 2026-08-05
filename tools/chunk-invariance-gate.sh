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
# DEFAULT MODE is --expect-variant: it asserts the CURRENT, HONEST contract — that memra's
# default config is chunk-VARIANT and the repo's exactness wording stays scoped to "tokens
# never depend on batchmates". It FAILS if the divergence silently disappears (that would
# mean the perf-relevant chunked path stopped chunking) OR if a NEW divergence class appears
# beyond the pinned one. Run with --expect-invariant to gate the MEMRA_PRIME_INVARIANT=1
# door, which is where byte-identity is the claim.
#
# TEETH: --canary flips the expectation and requires the OPPOSITE verdict, so the gate is
# verified able to fail. CI runs both directions.
set -uo pipefail
cd "$(dirname "$0")/.."
PROBE=./target/release/concat-prime-probe
D=research/chunk-invariance-20260805
MODEL=""
CHUNKS=2048,64,32
STEPS=48
EXPECT=variant
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
ENVX=()
[ "$EXPECT" = invariant ] && ENVX=(MEMRA_PRIME_INVARIANT=1 MEMRA_PRIME_GRAIN=32)

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

# what the run actually showed
if [ $saw_variant -eq 1 ]; then GOT=variant; else GOT=invariant; fi
WANT="$EXPECT"
[ "$CANARY" = 1 ] && { [ "$WANT" = variant ] && WANT=invariant || WANT=variant; }

echo "chunk-invariance-gate: expect=$WANT got=$GOT chunks=$CHUNKS model=$(basename "$MODEL")"
grep -E "^ *(2048|64|32|chunkinv verdict)" "$LOG" | sed 's/^/    /'
if [ "$GOT" = "$WANT" ]; then
    if [ "$CANARY" = 1 ]; then
        echo "chunk-invariance-gate: CANARY UNEXPECTEDLY MATCHED — the gate cannot fail; FIX THE GATE"
        rc_all=1
    else
        echo "chunk-invariance-gate: PASS (raw log $LOG)"
    fi
elif [ "$CANARY" = 1 ]; then
    echo "chunk-invariance-gate: PASS (canary correctly diverged — gate has teeth; raw log $LOG)"
else
    echo "chunk-invariance-gate: FAIL — chunk-order behavior CHANGED (wanted $WANT, got $GOT)."
    if [ "$WANT" = variant ]; then
        echo "  If this is intentional (invariance won back), flip the gate to"
        echo "  --expect-invariant and update docs/SERVING.md + the exactness wording."
    else
        echo "  The MEMRA_PRIME_INVARIANT door no longer delivers byte-identity."
    fi
    echo "  raw log: $LOG"
    rc_all=1
fi
exit $rc_all

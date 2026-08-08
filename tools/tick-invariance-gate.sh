#!/usr/bin/env bash
# tick-invariance-gate — asserts that SERVE-STYLE per-tick prefill segmentation is free of
# arithmetic: the SAME prompt primed across SEVERAL prime_cache CALLS (one per scheduler tick,
# `take <= budget` tokens each — the worker's exact loop including the PRIME_MIN_T tail merge)
# must yield prefill logits BIT-IDENTICAL to one monolithic call, at every per-tick budget.
# Self-gating (`kind=cmd` in tools/fast-gate/models.tsv): exit 0 = PASS.
#
#   tools/tick-invariance-gate.sh [<model.gguf>] [--budgets 0,1024,513,512,256,64] [--steps N]
#                                 [--splits 64,256,512] [--prompts <file[,file]>]
#                                 [--seam ENVVAR] [--label L] [--canary]
#
# WHY THIS GATE EXISTS (research/step35-chunkfix-20260807/PROGRESS.md §9 + this lane,
# research/tick-seg-20260807): the SECOND segmentation axis on the launch SKU. chunkinv35 pins
# the split INSIDE one prime_cache call (MEMRA_PRIME_CHUNK); serve ALSO splits a long prompt
# across several prime_cache calls — one per scheduler tick (worker.rs prefill_tick /
# step_session), budget from LanePolicy: MEMRA_PREFILL_TICK=1024 interactive,
# MEMRA_PREFILL_JUDGE=MEMRA_PREFILL_HARVEST=256 dark — and the dark budgets are additionally
# capped by LIVE SLO HEADROOM, so the segmentation is load-dependent. Each call computed its own
# seq_end = cache.pos + t, so step35's SWA arm predicate was free to differ BETWEEN calls even
# though lane/step35-chunkfix made it fixed WITHIN one. Measured (tickinv probe, PP-2 box,
# raw/tickinv35-20260807T022010Z.log): T=4883 budgets 1024/513 EXACT; 512/256/64 DIFFER,
# maxdiff 1.813e0, greedy text diverging at step 6 — the SAME signature as the original chunk
# defect, i.e. the same mechanism reached through the outer door.
#
# Upstream precedent (vLLM #51113, research/upstream-sweeps.md 08-07): mid-block chunk ends
# published as full blocks silently poisoned their prefix cache; single requests were
# accidentally safe. The two laws this gate pins: (1) position-keyed arithmetic must not depend
# on where tick boundaries fall — only the request's own extent may steer it; (2) unaligned
# STARTS (resume-from-cache at an off-grid position — the prefix-cache LCP split, any LCP in
# [64,512]) are a second hole: the --splits arms prime [0,L) then [L,T) as TWO calls (serve's
# exact LCP-split shape) and assert bit-identity vs monolithic.
#
# REGISTERED RED (this lane's first commit, 2026-08-07): the fix that turns it green is this
# lane's deliverable. The chunkfix lane deliberately did not register it to avoid a known-red
# check rotting unowned; this lane owns the red, so the registration and the fix land in
# adjacent commits on the same branch.
#
# TEETH: --canary INJECTS A BREAK (it does not merely relabel the expectation) and requires the
# gate's assertion to FAIL. The seam is $SEAM (default MEMRA_PRIME_CALLLOCAL — the fix's
# rollback to per-call seq_end, tick-VARIANT by construction). A canary that flips only the
# EXPECTED label is perfectly correlated with the default gate and proves nothing (trap
# documented in chunk-invariance-gate.sh, hit twice on that lane).
set -uo pipefail
cd "$(dirname "$0")/.."
PROBE=./target/release/concat-prime-probe
MODEL=""
BUDGETS=0,1024,513,512,256,64
SPLITS=""
STEPS=24
EXPECT=invariant
CANARY=0
SEAM=MEMRA_PRIME_CALLLOCAL
PROMPTS=research/chunk-invariance-20260805/prompt-pp6257.txt
LABEL=step35-tick
while [ $# -gt 0 ]; do
    case "$1" in
        --budgets) BUDGETS="$2"; shift 2 ;;
        --splits) SPLITS="$2"; shift 2 ;;
        --steps)  STEPS="$2"; shift 2 ;;
        --prompts) PROMPTS="$2"; shift 2 ;;
        --seam)   SEAM="$2"; shift 2 ;;
        --label)  LABEL="$2"; shift 2 ;;
        --expect-invariant) EXPECT=invariant; shift ;;
        --canary) CANARY=1; shift ;;
        -*) echo "tick-invariance-gate: unknown arg $1" >&2; exit 2 ;;
        *)  MODEL="$1"; shift ;;
    esac
done
# Artifact resolution (same shape as chunk-invariance-gate.sh's step35-swa label): the launch
# SKU is box-staged, not on /data on every rig — SKIP cleanly when absent (a missing artifact
# must not read as a pass; fast-gate reads this script's own SKIP word).
if [ -z "$MODEL" ]; then
    for cand in "${MEMRA_STEP37_GGUF:-}" \
        "$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
        /data/ai-ml/hf-models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf; do
        [ -n "$cand" ] && [ -f "$cand" ] && { MODEL="$cand"; break; }
    done
    [ -z "$MODEL" ] && { echo "tick-invariance-gate: SKIP (no Step-3.7-Flash artifact; set MEMRA_STEP37_GGUF)"; exit 0; }
fi
[ -f "$MODEL" ] || { echo "tick-invariance-gate: SKIP (no model at $MODEL)"; exit 0; }
[ -x "$PROBE" ] || { echo "tick-invariance-gate: FAIL (build concat-prime-probe first)"; exit 1; }
PROMPTS="${PROMPTS//,/ }"
for p in $PROMPTS; do
    [ -f "$p" ] || { echo "tick-invariance-gate: FAIL (missing pinned prompt $p)"; exit 1; }
done

LOG=$(mktemp /tmp/tickinv-gate-XXXXXX.log)
# The assertion under test is always EXPECT (invariant). The canary changes the WORLD: $SEAM=1
# restores the per-call seq_end (the pre-fix arithmetic, tick-variant by construction), so a
# working gate must report FAIL on the canary run.
#
# CANARY HISTORY (the THIRD vacuous-canary find on this arch, found independently by
# lane/pp-leverb and the v072 battery on 2026-08-08): after Lever A made windowed FA the
# SWA default, a predicate-only MEMRA_PRIME_CALLLOCAL=1 became bitwise INERT — a tick whose
# call-local seq_end is <= win covers only positions < win, whose views hold no maskable
# key, and on such views windowed==unwindowed FA bit-for-bit (the battery-2 G2c identity).
# FIXED IN THE ENGINE (lane/v072-blockers, 73c65c91): the seam now restores BOTH halves of
# the pre-fix arithmetic — the per-call predicate AND the raw (unaligned) SWA view offset,
# whose tile-grid regrouping is the live segmentation-variant mechanism ON THE SHIPPED FA
# DEFAULT — so this script needs no class-pinning env. Independent floor-class receipts
# (lane/pp-leverb, raw/tickinvc-floor-20260808T*.log): the floor arm's canary bites and its
# naked run PASSes, so the tick-seg fix holds on BOTH numeric classes. THE LAW this find
# re-teaches (H100 lane, rounds 35-36): canaries are calibrated against a kernel class —
# re-sweep them when the class under them moves (this one was calibrated 2026-08-07 02:20Z,
# the FA default landed ~14:00Z the same day).
WANT="$EXPECT"
LEGACY=off
[ "$CANARY" = 1 ] && LEGACY=on
ENVX=()
[ "$LEGACY" = on ] && ENVX=("$SEAM=1")

rc_all=0
saw_variant=0
saw_invariant=0
for p in $PROMPTS; do
    # evidence discipline: tee the raw log, parse the LOG (never the pipe)
    env "${ENVX[@]}" "$PROBE" "$MODEL" tickinv --prompt-a "@$p" \
        --budgets "$BUDGETS" ${SPLITS:+--splits "$SPLITS"} --steps "$STEPS" >> "$LOG" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "tick-invariance-gate: FAIL (probe exit $rc on $p)"; tail -5 "$LOG"; exit 1; }
done
if grep -q "tickinv verdict: TICK-INVARIANT" "$LOG"; then saw_invariant=1; fi
if grep -q "tickinv verdict: \*\*\* TICK-DEPENDENT" "$LOG"; then saw_variant=1; fi
[ $saw_invariant -eq 0 ] && [ $saw_variant -eq 0 ] && {
    echo "tick-invariance-gate: FAIL (no verdict line in probe output — probe contract changed)"
    tail -10 "$LOG"; exit 1; }
# ANY diverging arm makes the run variant: one TICK-DEPENDENT verdict among N prompts/arms
# must not be masked by the others.
if [ $saw_variant -eq 1 ]; then GOT=variant; else GOT=invariant; fi

echo "tick-invariance-gate: label=$LABEL assert=$WANT seam=$SEAM legacy-seam=$LEGACY got=$GOT canary=$CANARY budgets=$BUDGETS splits=${SPLITS:--} model=$(basename "$MODEL")"
# summary rows keyed off the ACTUAL --budgets/--splits values (never hardcode the set).
# Split arms print as `sp<L>` in the probe's table.
VALS="$BUDGETS"
if [ -n "$SPLITS" ]; then
    VALS="$VALS,$(echo "$SPLITS" | tr ',' '\n' | sed 's/^/sp/' | paste -sd, -)"
fi
ROWRE="^ *($(echo "$VALS" | tr ',' '|')) *\||tickinv verdict"
grep -E "$ROWRE" "$LOG" | sed 's/^/    /'
if [ "$GOT" = "$WANT" ]; then
    if [ "$CANARY" = 1 ]; then
        echo "tick-invariance-gate: CANARY UNEXPECTEDLY MATCHED — flipping the $SEAM seam did not"
        echo "  change the verdict, so this assertion cannot detect the mechanism. FIX THE GATE. (log $LOG)"
        rc_all=1
    else
        echo "tick-invariance-gate: PASS (raw log $LOG)"
    fi
elif [ "$CANARY" = 1 ]; then
    echo "tick-invariance-gate: PASS (canary broke the assertion as required — gate has teeth; log $LOG)"
else
    echo "tick-invariance-gate: FAIL — per-tick prefill segmentation steers the arithmetic."
    echo "  The prefill logits move with the tick budget (or the LCP split point), so identical"
    echo "  requests prime differently under load (dark lanes are SLO-capped). Re-root-cause per"
    echo "  research/tick-seg-20260807/PROGRESS.md before touching the gate."
    echo "  raw log: $LOG"
    rc_all=1
fi
exit $rc_all

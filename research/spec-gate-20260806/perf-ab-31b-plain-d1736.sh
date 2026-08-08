#!/usr/bin/env bash
# spec-gate: the LEGAL settle for the one local-ci --perf red at this lane's tip.
#
# THE RED: `tools/local-ci.sh --perf` verdicted `31b-plain-d1736` FAIL at 38.02 tok/s, -3.03%
# against a rolling median of 39.21 built from rows dated 2026-07-30..2026-08-06.
#
# WHY local-ci's OWN verdict cannot settle it (the same three reasons the v0.71.0 red was
# settled on, research/v071-prep-20260806/battery-logs/perf-ab.sh — this is that harness
# re-pointed at this cell):
#   1. The denominator is a CROSS-DAY median. This project's measurement law makes cross-run
#      and cross-day comparisons invalid as evidence, denominator included.
#   2. This particular row is `window_clean:false` by its own admission: the owner's
#      hermes-gateway.service holds a persistent idle CUDA context on this card, so the row
#      was recorded under a co-resident and labelled as such.
#   3. The prior rows for this cell are NOT a stable series to begin with — 39.2x on
#      2026-08-03/04/05, then 35.87 and 35.86 on 2026-08-06 (both `window_clean:true`),
#      then 38.02 here. A median across that spread is a wide band, not a baseline.
#
# WHAT THIS LANE COULD EVEN HAVE DONE TO IT, stated so the measurement is not the only
# argument: the cell is gemma-4-31B **plain** greedy decode through `run-gen`, a
# `memra-engine` binary. This lane's whole diff vs the merge-base is +50 lines of NEW
# `impl SpecSession` methods in `crates/memra-engine/src/spec.rs` (0 existing lines touched)
# plus `crates/memra-server/src/worker.rs`. `memra-engine` does not depend on `memra-server`,
# `run-gen` never constructs a `SpecSession`, and no plain-decode call site changed. So the
# only honest possibilities are a code-layout/inlining accident or machine state.
#
# THE ONLY FORM THAT SEPARATES THEM: same thermal window, interleaved A/B/A/B, N=5 each, the
# GPU held exclusively under one flock for the whole run.
#   A = merge-base 9e228f4c (`origin/restructure/public-split` at branch point)
#   B = this lane's tip
# Cell config byte-identical to perf-cells.json's `31b-plain-d1736`: MEMRA_NGEN=128, the
# 1736-token depth prompt.
#
# VERDICT RULE, fixed before the run: compare A-median to B-median FROM THIS RUN ONLY.
#   |B/A - 1| <= 1.5%  -> the rolling median is the invalid side; no lane code regressed.
#   B < A by > 3%      -> a real regression in this lane, and it blocks.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

BASE=/tmp/perf-ab-mb/target/release/run-gen
CAND=target/release/run-gen
MODEL=${MEMRA_MODELS_DIR:-/data/ai-ml/hf-models}/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf
PROMPT=research/gemma4-bringup/depth-prompt-1736-ids.txt
LOGD="$(dirname "$0")/logs/perf-ab"
N=${N:-5}

[ -x "$BASE" ] || { echo "perf-ab: no baseline binary at $BASE"; exit 2; }
[ -x "$CAND" ] || { echo "perf-ab: no candidate binary at $CAND"; exit 2; }
[ -f "$MODEL" ] || { echo "perf-ab: no model at $MODEL"; exit 2; }
[ -f "$PROMPT" ] || { echo "perf-ab: no prompt at $PROMPT"; exit 2; }
mkdir -p "$LOGD"

echo "== perf-ab: 31b-plain-d1736, N=$N interleaved, one lock hold =="
echo "   A = merge-base 9e228f4c ($BASE)"
echo "   B = lane/spec-gate $(git rev-parse --short HEAD) ($CAND)"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader | sed 's/^/   entry: /'
echo "   other GPU compute apps at entry:"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/     /'
echo

: > "$LOGD/A.toks"; : > "$LOGD/B.toks"

one() { # $1 = arm letter, $2 = binary, $3 = rep
    local out toks
    # shellcheck disable=SC2046
    out=$(MEMRA_NGEN=128 timeout 420 "$2" "$MODEL" $(cat "$PROMPT") 2>&1)
    printf '%s\n' "$out" > "$LOGD/$1-r$3.log"
    toks=$(printf '%s\n' "$out" | grep -oE "= [0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+")
    [ -n "$toks" ] || { echo "  $1 rep$3: NO READING (see $LOGD/$1-r$3.log)"; return 1; }
    echo "$toks" >> "$LOGD/$1.toks"
    printf '  %s rep%s: %s tok/s  (%s)\n' "$1" "$3" "$toks" \
        "$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | tr -d ' ')"
}
export -f one
export MODEL PROMPT LOGD BASE CAND N

# ONE lock hold for the whole interleave: a lock taken per-rep lets a neighbour land between
# arms, which is exactly the hole that made the original rows unusable. The loop body goes to
# `bash -c` through an EXPORTED function, not through a newline-stripped `declare -f` —
# collapsing a function body with `tr` broke it on the first attempt (`syntax error near
# unexpected token 'done'`), and the run still exited 0 because the failure was inside a
# pipeline. Both are fixed: exported function, and PIPESTATUS checked below.
flock -w 7200 /tmp/gpu5090.lock bash -c '
    for r in $(seq 1 "$N"); do
        one A "$BASE" "$r" || exit 1
        one B "$CAND" "$r" || exit 1
    done
' 2>&1 | tee "$LOGD/interleave.txt"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "perf-ab: interleave FAILED (see $LOGD/interleave.txt)"; exit 1; }
[ -s "$LOGD/A.toks" ] && [ -s "$LOGD/B.toks" ] || { echo "perf-ab: no readings recorded"; exit 1; }

med() { sort -g "$1" | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
A=$(med "$LOGD/A.toks"); B=$(med "$LOGD/B.toks")
echo
echo "A (merge-base) reps: $(tr '\n' ' ' < "$LOGD/A.toks")  median $A"
echo "B (lane tip)   reps: $(tr '\n' ' ' < "$LOGD/B.toks")  median $B"
awk -v a="$A" -v b="$B" 'BEGIN{
    d=(b/a-1)*100;
    printf "B vs A: %+.2f%%\n", d;
    if (d < -3.0)      print "VERDICT: REAL REGRESSION in this lane — blocks.";
    else if (d < -1.5) print "VERDICT: INCONCLUSIVE (-1.5%..-3%) — widen N before concluding.";
    else               print "VERDICT: NO LANE REGRESSION — the rolling median was the invalid side.";
}'
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader | sed 's/^/   exit: /'
echo PERF_AB_DONE

#!/usr/bin/env bash
# argmax-margin-gate — the run-gen prefill-vs-decode argmax gate, CALIBRATED.
# Self-gating (`kind=cmd` in tools/fast-gate/models.tsv): exit 0 = PASS.
#
#   tools/argmax-margin-gate.sh [<model.gguf>] [--prompt <file>] [--window N]
#                              [--max-flips N] [--margin-floor F] [--canary]
#                              [--logdir DIR]
#
# Raw logs land in $MEMRA_GATE_LOGDIR (default $TMPDIR/memra-argmax-margin-gate), NOT in the
# tracked research dir — this runs every battery. Use --logdir research/<lane>/ to keep one.
#
# WHY THIS GATE EXISTS (research/q8-argmax-20260806/VERDICT.md)
# ------------------------------------------------------------
# run-gen carries a hard assert: `forward_last` (batched prefill) and the `decode_step`
# tokenwise loop must produce the same last-position argmax, else "cache threading bug"
# (run_gen.rs:896). That assert is right to exist and its wording is a landmine: on the 27B
# Q8_0 + board-2048 pair it has been red on the 188-SM pod since at least v0.69.0, and there
# is no cache threading bug. The two configs are two legitimate arithmetics; the last
# position of that prompt happens to sit on a THREE-WAY near-tie (top-3 inside 0.05 on a
# logit scale whose median top-2 gap is 1.25), so the two configs pick differently. Proven by
# the f32 oracle (MEMRA_FAST=0) flipping the same pair — no fast kernel is involved — and by
# the same {332,485} pair firing on 3 silicon classes and 3 model sizes.
#
# THE COVERAGE BUG THIS GATE FIXES
# --------------------------------
# The assert inspects ONE position: the last. So which prompts "pass" is decided by whether
# their final token happens to land on a near-tie — luck of prompt length, not a property of
# the engine. Measured (research/q8-argmax-20260806/RESULTS.jsonl, arm P3): the 27B NVFP4 arm
# the release battery calls GREEN carries a config flip at position 2042 whose margin (0.0184)
# is SMALLER than the failing arm's (0.0307). Same mechanism, invisible, shipping green.
# Conversely a MATCH at maxdiff 1.165 (arm L4) sits beside a MISMATCH at 0.4659 — so the
# `logit maxdiff` the gate prints is NOT a severity signal and must never be read as one.
#
# WHAT THIS GATE ASSERTS INSTEAD
# ------------------------------
# Over the last N positions of a prompt, under both configs, for every position:
#   flip at position p  =>  margin(p) < config_delta(p)
# i.e. every prefill/decode argmax disagreement must be EXPLAINED by a top-2 margin the
# config spread can actually reach across. A flip at a WIDE margin is the real defect class
# (a genuine cache/threading/kernel bug moves a logit by more than the gap it crosses), and
# that is what fails here — at any position, not just the last. It additionally fails if the
# flip COUNT exceeds --max-flips (default 1): a near-tie coin is isolated by nature, while a
# broken kernel disagrees repeatedly.
#
# This turns an uncalibrated one-position assert into a calibrated all-position one, and it
# is strictly stronger: it catches the pos-2042 class the current gate sleeps through, while
# no longer red-flagging a 0.03-margin coin as a "cache threading bug".
#
# TEETH: --canary INJECTS A BREAK rather than relabelling an expectation (the trap documented
# in tools/chunk-invariance-gate.sh's header, learned the hard way on lane/chunkinv-flip).
# It runs the real probe, then FAULT-INJECTS into a copy of the measured table one row of the
# exact class this gate exists to catch — a WIDE-margin flip (margin 5.0 >> delta 0.1, i.e. a
# disagreement the config spread cannot possibly explain, the signature of a genuine
# cache/threading/kernel bug) — and requires the comparator to reject it. This is a mutation
# test of the comparator on its own parse path, so it fires regardless of whether the model
# under test happens to produce any natural flip.
#   Rejected canary designs, and why (both were tried on this lane):
#   (a) flip only the EXPECTED verdict -> re-runs the identical data, perfectly correlated
#       with the default gate, proves nothing (the chunkinv trap).
#   (b) raise a margin floor above the prompt's median -> only touches rows that ALREADY
#       flipped, so on a clean model (0 flips) there is nothing to reject and the canary
#       reports "no teeth". Correct behavior, useless as a teeth check.
set -uo pipefail
cd "$(dirname "$0")/.."
PROBE=./target/release/argmax-margin-probe

MODEL="${1:-}"
if [ -n "$MODEL" ] && [ "${MODEL#--}" != "$MODEL" ]; then MODEL=""; else shift 2>/dev/null || true; fi
PROMPT=research/e2e/prompts/board-2048.txt
WINDOW=12
MAX_FLIPS=1
MARGIN_FLOOR=""      # optional: reject any flip whose margin exceeds this (off by default)
CANARY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --prompt)       PROMPT="$2"; shift 2 ;;
        --window)       WINDOW="$2"; shift 2 ;;
        --max-flips)    MAX_FLIPS="$2"; shift 2 ;;
        --margin-floor) MARGIN_FLOOR="$2"; shift 2 ;;
        --logdir)       MEMRA_GATE_LOGDIR="$2"; shift 2 ;;
        --canary)       CANARY=1; shift ;;
        *)              echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Default model: the fast-gate's Q8_0 row if present, else the first probe model available.
if [ -z "$MODEL" ]; then
    for cand in \
        /root/models/Qwen3.6-27B-Q8_0.gguf \
        /data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf \
        /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
    do [ -f "$cand" ] && { MODEL="$cand"; break; }; done
fi
# SKIP (not PASS, not FAIL) when the inputs are absent: exiting 0 silently on a missing
# artifact is indistinguishable from a real pass by exit code alone — the hole that reported
# chunkinv/chunkinvc as "PASS (0s)" on the 188-SM pod during the v0.70.0 battery. The
# fast-gate cmd runner greps "^<name>: *SKIP", so emit exactly that shape.
[ -n "$MODEL" ] && [ -f "$MODEL" ] || { echo "argmax-margin-gate: SKIP (no probe model on this rig)"; exit 0; }
[ -f "$PROMPT" ] || { echo "argmax-margin-gate: SKIP (prompt $PROMPT missing)"; exit 0; }
[ -x "$PROBE" ] || { echo "argmax-margin-gate: SKIP (build target/release/argmax-margin-probe to enable)"; exit 0; }

# Logs go to a scratch dir, NOT into the tracked research dir: this gate runs on every
# fast-gate invocation, and writing under research/ made each battery run dirty the worktree
# (and committed the canary's .injected scratch table once already). Point --logdir at the
# lane dir when you want a receipt to keep. Same convention as chunk-invariance-gate.sh.
D="${MEMRA_GATE_LOGDIR:-${TMPDIR:-/tmp}/memra-argmax-margin-gate}"; mkdir -p "$D"
LOG="$D/gate-$(basename "${MODEL%.gguf}")-$(basename "${PROMPT%.txt}")$([ $CANARY = 1 ] && echo .canary).log"

echo "argmax-margin-gate: model=$(basename "$MODEL") prompt=$(basename "$PROMPT") window=$WINDOW canary=$CANARY"
# Raw log FIRST, parse second (never let a pipe swallow the failure text).
"$PROBE" "$MODEL" "$PROMPT" "$WINDOW" > "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    # A VRAM shortage says nothing about argmax margins. On a shared rig another lane can hold
    # the card (seen live: "Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, ...)" with 21.6 GB held
    # by a concurrent job), and failing the exactness battery for that is a flake, not a find.
    # SKIP with the cause QUOTED — an inferred cause is not a cause (CLAUDE.md evidence rules).
    if grep -qaE 'CUDA_ERROR_OUT_OF_MEMORY|out of memory' "$LOG"; then
        echo "argmax-margin-gate: SKIP (GPU out of memory — not an exactness signal)"
        grep -aE 'CUDA_ERROR_OUT_OF_MEMORY|out of memory' "$LOG" | head -2 | sed 's/^/  quoted: /'
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
            | sed 's/^/  concurrent-GPU: /'
        exit 0
    fi
    echo "  FAIL: probe exited $rc"; tail -20 "$LOG"; exit 1
fi

# The canary appends ONE synthetic row of the defect class to the table the comparator parses:
# a flip at margin 5.0 against a config delta of 0.1. No near-tie explanation is available for
# it, so a comparator that actually reads margin-vs-delta MUST reject it.
PARSE="$LOG"
if [ $CANARY = 1 ]; then
    PARSE="$LOG.injected"
    cp "$LOG" "$PARSE"
    printf '%-8s %-13s %-12s %-12s %-12s %-14s %s\n' \
        9999 111111 5.0000 222222 5.0000 0.1000 "NO <-- FLIP" >> "$PARSE"
    echo "  [canary] injected a WIDE-margin flip (pos 9999, margin 5.0000, delta 0.1000)"
    echo "           — a disagreement the config spread cannot explain; must be rejected"
fi

# Parse the per-position table: pos prefill_top1 margin_p decode_top1 margin_d delta agree...
verdict=$(awk -v floor="${MARGIN_FLOOR:-}" -v maxflips="$MAX_FLIPS" '
    /^[0-9]+[ \t]+[0-9]+/ {
        pos=$1; mp=$3; md=$5; delta=$6; flipped=($7=="NO")
        if (flipped) {
            flips++
            m = (md < mp ? md : mp)
            if (delta <= m) { printf "UNEXPLAINED pos=%s margin=%.4f delta=%.4f\n", pos, m, delta; bad++ }
            else if (floor != "" && m+0 > floor+0) {
                printf "OVER-FLOOR pos=%s margin=%.4f > floor=%s\n", pos, m, floor; bad++
            }
            else printf "explained pos=%s margin=%.4f < delta=%.4f\n", pos, m, delta
        }
    }
    END {
        if (flips > maxflips) { printf "TOO-MANY-FLIPS %d > %d\n", flips, maxflips; bad++ }
        printf "SUMMARY flips=%d bad=%d\n", flips+0, bad+0
    }' "$PARSE")
echo "$verdict" | sed 's/^/  /'

bad=$(echo "$verdict" | awk '/^SUMMARY/ {print $3}' | cut -d= -f2)
if [ "${bad:-1}" -eq 0 ]; then
    if [ $CANARY = 1 ]; then
        echo "  CANARY FAIL: the injected wide-margin flip did NOT break the assertion — the"
        echo "               comparator is not reading its own margins; this gate has no teeth."
        exit 1
    fi
    echo "  PASS: every prefill/decode argmax flip is explained by a margin the config spread covers"
    echo "  raw: $LOG"
    exit 0
else
    if [ $CANARY = 1 ]; then
        echo "  CANARY PASS: the injected wide-margin flip broke the assertion as required (teeth)"
        echo "  raw: $LOG"
        exit 0
    fi
    echo "  FAIL: an argmax flip is NOT explained by the config spread (or flip count exceeded)"
    echo "  raw: $LOG"
    exit 1
fi

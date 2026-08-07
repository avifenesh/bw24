#!/usr/bin/env bash
# iso-gap-gate — asserts the STAGGERED-DEPTH serve isolation contract at the engine tick:
# a session's logits are BIT-IDENTICAL whether it decodes alone (B=1, batched body) or
# co-resident with sessions at OTHER depths, INCLUDING across a fa_split_keys ladder-rung
# boundary (the LADDER-RUNG STRADDLE class, task #91). Self-gating (`kind=cmd` in
# tools/fast-gate/models.tsv): exit 0 = PASS, "SKIP" = artifact absent.
#
#   tools/iso-gap-gate.sh [<model.gguf>] [--steps N] [--canary]
#
# WHY THIS GATE EXISTS (research/iso-gap-20260807/PROGRESS.md): the equal-depth serve gate
# (16 prompts, 96 tokens, simultaneous arrival) and the kernel-check seqs-vs-loop pin (depths
# all inside ONE rung) shared a blind spot — no gate ever crossed a rung boundary with a
# co-resident present. lane/iso-gap measured the property and found it HOLDS (5 shapes,
# B=2..8, 3 rungs, 300-step horizon, zero bit diffs) because decode_batch's rung guard
# (batch_layer_ctx: all rows must share one fa_split_keys rung or ALL fall to the per-seq
# eager loop) is per-session-correct. This gate pins that property so nobody breaks it
# silently — the exact vLLM #40372 pattern the chunkinv gates follow: measure, then assert.
#
# THE STRADDLE IS PLACED PER-RIG (--auto): fa_split_keys is SM-count- and n_head_kv-keyed
# (82-SM boundary at t_kv=512, 188-SM at 2048), so pinned depths that straddle on one rig
# straddle NOTHING on another. The probe scans the ladder through the public twin and places
# X 32 tokens below the first boundary — X crosses mid-run, the batch straddles for ~32
# steps (per-seq fallback window), then re-merges (seqs arm re-fires). One run therefore
# exercises: same-rung batched, straddling fallback, the two transitions between them.
#
# TEETH: --canary changes the WORLD, not the label (the chunkinv trap, written wrong twice
# on that lane): it injects one wrong token into the co-resident arm's X feed at step 1
# (probe --canary). The comparator must FAIL on it; the probe exits 0 only if the injected
# break IS caught (CANARY-OK), 2 if it slid through.
set -uo pipefail
cd "$(dirname "$0")/.."
PROBE=./target/release/iso-gap-probe
MODEL=""
STEPS=96
CANARY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --steps)  STEPS="$2"; shift 2 ;;
        --canary) CANARY=1; shift ;;
        -*) echo "iso-gap-gate: unknown arg $1" >&2; exit 2 ;;
        *)  MODEL="$1"; shift ;;
    esac
done
# default model = the family the serve receipt was measured on (qwen hybrid NVFP4)
MODEL="${MODEL:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
[ -f "$MODEL" ] || { echo "iso-gap-gate: SKIP (no model at $MODEL)"; exit 0; }
[ -x "$PROBE" ] || { echo "iso-gap-gate: FAIL (build iso-gap-probe first)"; exit 1; }

LOG=$(mktemp /tmp/iso-gap-gate-XXXXXX.log)
ARGS=(--auto --steps "$STEPS")
[ "$CANARY" = 1 ] && ARGS+=(--canary)
# evidence discipline: tee the raw log, parse the LOG (never the pipe)
"$PROBE" "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1
rc=$?
tail -3 "$LOG"
if [ "$CANARY" = 1 ]; then
    if grep -q 'CANARY-OK' "$LOG"; then
        echo "iso-gap-gate (canary): PASS — injected break caught (log $LOG)"
        exit 0
    fi
    echo "iso-gap-gate (canary): FAIL — injected break NOT caught (log $LOG)"
    exit 1
fi
if [ $rc -eq 0 ] && grep -q 'VERDICT.*PASS' "$LOG"; then
    echo "iso-gap-gate: PASS (log $LOG)"
    exit 0
fi
echo "iso-gap-gate: FAIL rc=$rc (log $LOG)"
exit 1

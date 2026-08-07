#!/usr/bin/env bash
# prime-split-gate — the LEVER B bit-identity gate (lane/pp-leverb, 2026-08-08): the chunked
# prime over an OPEN ppN door must produce BIT-IDENTICAL results split vs unsplit, AND the
# split must be provably LIVE. Self-gating (`kind=cmd` in tools/fast-gate/models.tsv):
# exit 0 = PASS.
#
#   tools/prime-split-gate.sh [<model.gguf>] [--devices 0,1] [--stages 2] [--chunks 4096,513]
#                             [--steps 8] [--prompts <f>] [--canary]
#
# WHY THIS GATE EXISTS (research/pp-leverb-20260807/PROGRESS.md): the prime path has NO pp
# stage split — under MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 every prime chunk walks all 45
# layers on dev0, peer-reading stage-1 trunk weights (22% of the pp4096 wall) while dev1 runs
# ZERO kernels (anatomy receipt: kernels per device = [(0, 2337323, 87.6s)]). Prime keeps NO
# refuse_unsplit_if_remote — its unsplit walk is a 22% amortized tax, not the decode 28x
# cliff, and it is precisely this gate's REFERENCE arm (MEMRA_PRIME_PP=0).
#
# REGISTERED RED (the tickinv35 pattern): the walker does not exist yet. The probe's split
# arm requires pp::PRIME_SPLIT_CHUNKS to ADVANCE (bit-identity of two identical unsplit walks
# is vacuous) — so this gate FAILS today and goes green exactly when the per-stage walker
# lands, with bit-identity + KV-write correctness (teacher-forced decode steps read the KV
# the prime wrote) proven in the same run.
#
# TEETH: --canary passes --force-unsplit to the probe (the "split" arm runs unsplit), so a
# GREEN gate must flip RED under the canary. While the gate is still registered-red the
# canary is trivially red too — it becomes load-bearing at the walker commit, and it is
# registered NOW so the flip cannot be forgotten (the GAP-1/battery-2 vacuous-canary class).
#
# NEEDS 2 GPUs with P2P (the PRO 6000 pair). On a single-GPU rig (the local 5090) it SKIPs:
# a same-device "split" exercises the seam but not the placement this lever exists for; the
# box battery is the authority (CLAUDE.md: CI is compile-only, the battery is the real gate).
set -uo pipefail
cd "$(dirname "$0")/.."
PROBE=./target/release/concat-prime-probe
MODEL=""
DEVICES=0,1
STAGES=2
CHUNKS=4096,513
STEPS=8
PROMPTS=""
CANARY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --stages)  STAGES="$2"; shift 2 ;;
        --chunks)  CHUNKS="$2"; shift 2 ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --prompts) PROMPTS="$2"; shift 2 ;;
        --canary)  CANARY=1; shift ;;
        -*) echo "prime-split-gate: unknown arg $1" >&2; exit 2 ;;
        *)  MODEL="$1"; shift ;;
    esac
done
# Default model = the launch SKU (the placement this lever serves); resolves like chunkinv35.
if [ -z "$MODEL" ]; then
    for cand in "${MEMRA_STEP37_GGUF:-}" \
        "$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
        /data/ai-ml/hf-models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf; do
        [ -n "$cand" ] && [ -f "$cand" ] && { MODEL="$cand"; break; }
    done
    [ -z "$MODEL" ] && { echo "prime-split-gate: SKIP (no Step-3.7-Flash artifact; set MEMRA_STEP37_GGUF)"; exit 0; }
fi
[ -f "$MODEL" ] || { echo "prime-split-gate: SKIP (no model at $MODEL)"; exit 0; }
[ -x "$PROBE" ] || { echo "prime-split-gate: FAIL (build concat-prime-probe first)"; exit 1; }
# Distinct-device count must cover the placement (single-GPU rigs SKIP — see header).
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
NDEV=$(echo "$DEVICES" | tr ',' '\n' | sort -u | wc -l)
MAXDEV=$(echo "$DEVICES" | tr ',' '\n' | sort -n | tail -1)
if [ "$NGPU" -le "$MAXDEV" ] || [ "$NDEV" -lt 2 ]; then
    echo "prime-split-gate: SKIP (needs the multi-GPU placement $DEVICES; $NGPU GPU(s) visible)"
    exit 0
fi
# Prompt must exceed the largest chunk so the chunk loop actually segments.
PROMPTS="${PROMPTS:-research/chunk-invariance-20260805/prompt-pp6257.txt}"
[ -f "$PROMPTS" ] || { echo "prime-split-gate: FAIL (missing pinned prompt $PROMPTS)"; exit 1; }

EXTRA=()
[ "$CANARY" = 1 ] && EXTRA=(--force-unsplit)
LOG=$(mktemp /tmp/prime-split-gate-XXXXXX.log)
# evidence discipline: tee the raw log, parse the LOG (never the pipe)
MEMRA_PP_STAGES=$STAGES MEMRA_PP_DEVICES=$DEVICES \
    "$PROBE" "$MODEL" ppsplit --prompt-a "@$PROMPTS" \
    --chunks "$CHUNKS" --steps "$STEPS" "${EXTRA[@]}" > "$LOG" 2>&1
rc=$?
grep -E "^ppsplit|^  chunk" "$LOG" | sed 's/^/    /'
if [ "$CANARY" = 1 ]; then
    if [ $rc -ne 0 ]; then
        echo "prime-split-gate: PASS (canary broke the assertion as required — gate has teeth; log $LOG)"
        exit 0
    fi
    echo "prime-split-gate: CANARY UNEXPECTEDLY MATCHED — forcing the split arm unsplit did not"
    echo "  flip the verdict, so the liveness check cannot detect the mechanism. FIX THE GATE. (log $LOG)"
    exit 1
fi
if [ $rc -eq 0 ]; then
    echo "prime-split-gate: PASS (split prime bit-identical + live; raw log $LOG)"
    exit 0
fi
echo "prime-split-gate: FAIL rc=$rc — split prime absent, not live, or not bit-identical (log $LOG)"
exit 1

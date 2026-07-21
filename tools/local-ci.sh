#!/usr/bin/env bash
# bw24 local CI — the real gate (GitHub CI is compile-only; the rig is the test machine).
#
#   tools/local-ci.sh                correctness stage only (~3 min)
#   tools/local-ci.sh --perf         correctness + full perf battery (~15 min)
#   tools/local-ci.sh --perf-quick   correctness + gemma-31B cells only (~6 min)
#
# Correctness stage: kernel-check, run-gen argmax gate, spec self-consistency,
# VERIFY-GATE logit maxdiff at depth — the standing exactness battery, one command.
#
# Perf stage: the cell battery from research/tune-data/perf-cells.json. Every spec cell
# records tok/s + ACCEPTANCE + tok/round — the drift class that silently cost the spec
# board 2026-07-13..15 (acceptance 1.000 -> 0.669 across ~40 green-gated commits).
# Rows append to research/tune-data/perf-ci.jsonl; each cell is verdicted against the
# rolling median of its last N rows: FAIL on >3% tok/s drop or >0.05 acceptance drop,
# WARN on >1.5%. A FAIL exits non-zero — treat it like a red test.
#
# Contributor machines: cells whose model file is absent are SKIPPED cleanly; the
# correctness stage runs wherever a GPU + at least one model exists. Set
# BW24_MODELS_DIR to your model root (default /data/ai-ml/hf-models).
#
# Window discipline (recorded per row, enforced where it can be): no other compute
# process on the GPU (co-resident engines spill experts and read 10x low), host load
# sane, power profile noted (pin it with gpu-full-power on|off — profiles pair fairly
# only against themselves).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${BW24_MODELS_DIR:-/data/ai-ml/hf-models}"
MANIFEST=research/tune-data/perf-cells.json
OUT=research/tune-data/perf-ci.jsonl
MODE="${1:---correctness}"

command -v jq >/dev/null || { echo "local-ci: jq required"; exit 2; }
[ -x target/release/kernel-check ] || cargo build --release

# ---- window state ----
# allowed co-residents: embedding servers (tiny, CPU-bound; identified by --embedding in cmdline)
apps=""
while IFS=, read -r pid _name; do
    pid=$(echo "$pid" | tr -d ' '); [ -n "$pid" ] || continue
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--embedding"; then
        apps+="$pid $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)\n"
    fi
done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
apps=$(printf "%b" "$apps")
if [ -n "$apps" ]; then
    echo "local-ci: WARNING — other GPU compute apps present (numbers not window-valid):"
    echo "$apps"
    WINDOW_CLEAN=false
else
    WINDOW_CLEAN=true
fi
LOAD=$(awk '{print $1}' /proc/loadavg)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)

echo "== local-ci: correctness stage =="
out=$(target/release/kernel-check 2>&1 | tail -1)
echo "$out" | grep -q "ALL GREEN" || { echo "kernel-check FAIL"; exit 1; }
echo "kernel-check: GREEN"

G31="$MODELS/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf"
DEPTH=research/gemma4-bringup/depth-prompt-1736-ids.txt
if [ -f "$G31" ]; then
    out=$(BW24_NGEN=8 target/release/run-gen "$G31" 55 2>&1)
    echo "$out" | grep -q "MATCH" || { echo "run-gen argmax FAIL (31B)"; exit 1; }
    echo "run-gen argmax: MATCH (31B)"
    # shellcheck disable=SC2046
    out=$(BW24_VERIFY_GATE=7 target/release/gemma-gate "$G31" $(cat "$DEPTH") 2>&1)
    echo "$out" | grep -q "VERIFY-GATE K=7: PASS" || { echo "VERIFY-GATE FAIL (31B depth)"; exit 1; }
    echo "VERIFY-GATE K=7 depth: PASS (31B)"
    D31="$MODELS/gemma4-31b-tooluse-gguf/gemma-4-31B-it-Q4_0-MTP.gguf"
    if [ -f "$D31" ]; then
        # shellcheck disable=SC2046
        out=$(BW24_SPEC=6 BW24_DRAFT="$D31" BW24_NGEN=64 target/release/gemma-gate "$G31" \
            $(cat research/gemma4-bringup/e4b-chat-watercycle-ids.txt) 2>&1)
        echo "$out" | grep -qE "stream agreement 64/64" || { echo "spec self-consistency FAIL (31B)"; exit 1; }
        echo "spec self-consistency 64/64: PASS (31B)"
    fi
else
    echo "run-gen/VERIFY-GATE/spec: SKIP (no 31B model at $G31)"
fi
echo "correctness stage: GREEN"
[ "$MODE" = "--correctness" ] && exit 0

echo "== local-ci: perf stage ($MODE) =="
GIT_SHA=$(git rev-parse --short HEAD)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FAILS=0; WARNS=0

run_cell() {
    local id="$1" model="$2" mode="$3" prompt="$4" ngen="$5" k="$6" draft="$7" ranks="$8"
    local mp="$MODELS/$model"
    [ -f "$mp" ] || { echo "  $id: SKIP (no model)"; return 0; }
    local pfile; pfile=$(jq -r ".prompts[\"$prompt\"]" $MANIFEST)
    local best_toks="0" accept="" tokround=""
    for _rep in 1 2; do
        local out toks
        if [ "$mode" = "plain" ]; then
            # shellcheck disable=SC2046
            out=$(BW24_NGEN="$ngen" timeout 420 target/release/run-gen "$mp" $(cat "$pfile") 2>&1 || true)
            toks=$(echo "$out" | grep -oE "= [0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+" || echo 0)
        else
            local envs=(BW24_SPEC_ONLY=1 "BW24_SPEC=$k" "BW24_DRAFT=$MODELS/$draft" "BW24_NGEN=$ngen")
            [ -n "$ranks" ] && [ "$ranks" != "null" ] && envs+=("BW24_GEMMA_DRAFT_RANKS=$ranks")
            # shellcheck disable=SC2046
            out=$(env "${envs[@]}" timeout 420 target/release/gemma-gate "$mp" $(cat "$pfile") 2>&1 || true)
            toks=$(echo "$out" | grep -oE "spec: [0-9.]+" | grep -oE "[0-9.]+" || echo 0)
            accept=$(echo "$out" | grep -oE "accept-rate=[0-9.]+" | grep -oE "[0-9.]+" | tail -1 || true)
            tokround=$(echo "$out" | grep -oE "tok/round=[0-9.]+" | grep -oE "[0-9.]+" | tail -1 || true)
        fi
        awk -v a="$toks" -v b="$best_toks" 'BEGIN{exit !(a>b)}' && best_toks="$toks"
    done
    [ "$best_toks" = "0" ] && { echo "  $id: FAIL (no reading)"; FAILS=$((FAILS+1)); return 0; }

    # rolling-median verdict from prior rows of this cell
    local base verdict="OK" note="" rows
    rows=$(grep "\"cell\":\"$id\"" "$OUT" 2>/dev/null || true)
    base=$(printf '%s\n' "$rows" | tail -"$(jq -r .gates.baseline_window $MANIFEST)" \
        | jq -s 'map(.toks) | sort | .[length/2|floor] // 0' 2>/dev/null)
    base=${base:-0}
    if awk -v b="$base" 'BEGIN{exit !(b>0)}'; then
        local drop
        drop=$(awk -v n="$best_toks" -v b="$base" 'BEGIN{printf "%.2f", (b-n)/b*100}')
        if awk -v d="$drop" -v t="$(jq -r .gates.cell_drop_fail_pct $MANIFEST)" 'BEGIN{exit !(d>t)}'; then
            verdict="FAIL"; FAILS=$((FAILS+1)); note="tok/s -$drop% vs median $base"
        elif awk -v d="$drop" -v t="$(jq -r .gates.cell_drop_warn_pct $MANIFEST)" 'BEGIN{exit !(d>t)}'; then
            verdict="WARN"; WARNS=$((WARNS+1)); note="tok/s -$drop% vs median $base"
        fi
        if [ -n "$accept" ]; then
            local abase
            abase=$(printf '%s\n' "$rows" | tail -5 \
                | jq -s 'map(.accept // empty) | sort | .[length/2|floor] // 0' 2>/dev/null)
            abase=${abase:-0}
            if awk -v a="$accept" -v b="$abase" -v t="$(jq -r .gates.accept_drop_fail $MANIFEST)" \
                 'BEGIN{exit !(b>0 && b-a>t)}'; then
                verdict="FAIL"; FAILS=$((FAILS+1)); note="$note; ACCEPTANCE $abase -> $accept"
            fi
        fi
    else
        note="first row (baseline seed)"
    fi
    printf '{"ts":"%s","git":"%s","cell":"%s","toks":%s%s%s,"profile":"%s","load":%s,"window_clean":%s}\n' \
        "$TS" "$GIT_SHA" "$id" "$best_toks" \
        "${accept:+,\"accept\":$accept}" "${tokround:+,\"tok_round\":$tokround}" \
        "$PROFILE" "$LOAD" "$WINDOW_CLEAN" >> "$OUT"
    echo "  $id: $best_toks tok/s${accept:+ accept=$accept} [$verdict]${note:+ — $note}"
}

while read -r cell; do
    id=$(echo "$cell" | jq -r .id)
    if [ "$MODE" = "--perf-quick" ] && [[ "$id" != 31b-* ]]; then continue; fi
    # BW24_CI_CELLS: extended-regex cell-id filter (e.g. "26b-|e4b-") — run a subset
    # without touching the manifest; verdicts/rows behave exactly like a full run.
    if [ -n "${BW24_CI_CELLS:-}" ] && ! echo "$id" | grep -qE "$BW24_CI_CELLS"; then continue; fi
    run_cell "$id" "$(echo "$cell" | jq -r .model)" "$(echo "$cell" | jq -r .mode)" \
             "$(echo "$cell" | jq -r .prompt)" "$(echo "$cell" | jq -r .ngen)" \
             "$(echo "$cell" | jq -r '.k // 0')" "$(echo "$cell" | jq -r '.draft // ""')" \
             "$(echo "$cell" | jq -r '.ranks // ""')"
done < <(jq -c '.cells[]' $MANIFEST)

echo "perf stage: $FAILS fail, $WARNS warn"
[ "$FAILS" -eq 0 ] || exit 1

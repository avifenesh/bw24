#!/usr/bin/env bash
# fast-gate — change-scoped regression gate for the dev loop BETWEEN full-battery points.
#
#   tools/fast-gate/fast-gate.sh [--diff <ref>] [--tier 0|1|2] [--probes a,b] [--smoke]
#   tools/fast-gate/fast-gate.sh --refresh-goldens [--probes a,b] [--force]
#
# THE CONTRACT: the full battery (tools/local-ci.sh — kernel-check ALL GREEN + run-gen argmax
# per affected model + run-spec K=1..8 + serve-smoke) REMAINS the merge/tag gate, unchanged.
# fast-gate accelerates the loop between battery points by running only the gates a diff
# actually needs (docs/TESTING.md):
#
#   tier 0  seconds      workspace compile + kernel-check scoped to the touched sections
#                        (MEMRA_KC_FAST=1 / MEMRA_KC_ONLY=csv seams in kernel_check.rs)
#   tier 1  ~1-2 min     tier 0 + golden-token argmax probes: NGEN-token greedy generation on
#                        ONE model per affected kernel class, token-ids byte-compared against
#                        a PINNED golden (goldens/<id>.tokens). No llama runs, no N=5, no
#                        perf protocol. Spec-touching diffs add ONE single-K run-spec probe.
#   tier 2  full battery exec tools/local-ci.sh (unchanged).
#
# --smoke adds a perf smoke verdict per tier-1 probe from the SAME single run-gen rep:
# WARN >10% / FAIL >25% below the golden-point reference (goldens/<id>.perf). Catastrophic-
# regression tripwire ONLY (kernel fell off a fast path) — NOT publishable numbers; the
# published protocol stays N>=5 interleaved per research/benchmarks.md.
#
# Golden refresh protocol: goldens are (re)generated ONLY at full-battery green points —
# --refresh-goldens refuses a dirty tree without --force and records the git SHA per golden.
#
# GPU work runs under flock /tmp/gpu5090.lock (shared-rig serialization) and each probe's raw
# output is tee'd to a log before parsing (evidence discipline: never parse a pipe).
set -uo pipefail
FG_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$FG_DIR/../.." && pwd)"
cd "$ROOT"

MAP="$FG_DIR/map.tsv"
MODELS_TSV="$FG_DIR/models.tsv"
GOLDENS="$FG_DIR/goldens"
LOCK="${MEMRA_GATE_LOCK:-/tmp/gpu5090.lock}"
LOCK_WAIT="${MEMRA_GATE_LOCK_WAIT:-7200}"
LOGDIR="${MEMRA_GATE_LOGDIR:-/tmp/fast-gate-$(date +%Y%m%d-%H%M%S)}"

DIFF_REF="HEAD"
TIER=1
PROBES_OVERRIDE=""
SMOKE=0
REFRESH=0
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --diff)   DIFF_REF="$2"; shift 2 ;;
        --tier)   TIER="$2"; shift 2 ;;
        --probes) PROBES_OVERRIDE="$2"; shift 2 ;;
        --smoke)  SMOKE=1; shift ;;
        --refresh-goldens) REFRESH=1; shift ;;
        --force)  FORCE=1; shift ;;
        *) echo "fast-gate: unknown arg $1"; exit 2 ;;
    esac
done

mkdir -p "$LOGDIR"
T_START=$(date +%s)
stamp() { echo $(( $(date +%s) - T_START )); }
lockrun() { flock -w "$LOCK_WAIT" "$LOCK" "$@"; }

# ---------- tier 2 = the unchanged full battery ----------
if [ "$TIER" = "2" ]; then
    echo "fast-gate: tier 2 = the full battery (tools/local-ci.sh), unchanged."
    exec tools/local-ci.sh
fi

# ---------- probe registry ----------
probe_field() {  # id, field-no -> value
    awk -F'\t' -v id="$1" -v f="$2" '$0 !~ /^#/ && $1 == id { print $f; exit }' "$MODELS_TSV"
}
all_probe_ids() { awk -F'\t' '$0 !~ /^#/ && NF >= 5 { print $1 }' "$MODELS_TSV"; }

# ---------- diff -> plan (map.tsv is the dispatch-structure encoding) ----------
CHANGED=$( { git diff --name-only "$DIFF_REF" 2>/dev/null;
             git ls-files --others --exclude-standard; } | sort -u )
if [ -z "$CHANGED" ] && [ "$REFRESH" = 0 ]; then
    echo "fast-gate: no changes vs $DIFF_REF — nothing to gate."
    exit 0
fi

KC_SCOPE="none"     # none < synthetic < csv < all
KC_CSV=""
PLAN_PROBES=""
PLAN_SPEC=""
UNMATCHED=""
add_csv() { local x; for x in ${1//,/ }; do case ",$KC_CSV," in *",$x,"*) ;; *) KC_CSV="${KC_CSV:+$KC_CSV,}$x" ;; esac; done; }
add_probe() { local x; for x in ${1//,/ }; do [ "$x" = "-" ] && continue; case ",$PLAN_PROBES," in *",$x,"*) ;; *) PLAN_PROBES="${PLAN_PROBES:+$PLAN_PROBES,}$x" ;; esac; done; }
add_spec() { local x; for x in ${1//,/ }; do [ "$x" = "-" ] && continue; case ",$PLAN_SPEC," in *",$x,"*) ;; *) PLAN_SPEC="${PLAN_SPEC:+$PLAN_SPEC,}$x" ;; esac; done; }

for f in $CHANGED; do
    hit=0
    while IFS=$'\t' read -r rx scope probes spec; do
        [ -z "$rx" ] || [ "${rx:0:1}" = "#" ] || [ "$rx" = "DEFAULT" ] && continue
        if echo "$f" | grep -qE "$rx"; then
            hit=1
            case "$scope" in
                all)       KC_SCOPE="all" ;;
                synthetic) [ "$KC_SCOPE" = "none" ] && KC_SCOPE="synthetic" ;;
                none)      ;;
                *)         [ "$KC_SCOPE" != "all" ] && KC_SCOPE="csv"; add_csv "$scope" ;;
            esac
            add_probe "$probes"; add_spec "$spec"
        fi
    done < "$MAP"
    if [ "$hit" = 0 ]; then
        UNMATCHED="$UNMATCHED $f"
        # DEFAULT row (map.tsv): conservative plan for unmapped paths.
        d_scope=$(awk -F'\t' '$1=="DEFAULT"{print $2; exit}' "$MAP")
        d_probes=$(awk -F'\t' '$1=="DEFAULT"{print $3; exit}' "$MAP")
        d_spec=$(awk -F'\t' '$1=="DEFAULT"{print $4; exit}' "$MAP")
        case "${d_scope:-all}" in
            all) KC_SCOPE="all" ;;
            synthetic) [ "$KC_SCOPE" = "none" ] && KC_SCOPE="synthetic" ;;
            none) ;;
            *) [ "$KC_SCOPE" != "all" ] && KC_SCOPE="csv"; add_csv "$d_scope" ;;
        esac
        add_probe "${d_probes:-g12,q9,q35}"; add_spec "${d_spec:--}"
    fi
done
[ -n "$PROBES_OVERRIDE" ] && { PLAN_PROBES=""; PLAN_SPEC="";
    for p in ${PROBES_OVERRIDE//,/ }; do
        case "$(probe_field "$p" 2)" in
            spec|gspec) add_spec "$p" ;;
            *) add_probe "$p" ;;
        esac
    done; }

echo "== fast-gate plan (diff vs $DIFF_REF) =="
echo "$CHANGED" | sed 's/^/  changed: /'
[ -n "$UNMATCHED" ] && echo "  WARNING: unmatched paths (conservative full plan):$UNMATCHED"
echo "  kernel-check scope: $KC_SCOPE${KC_CSV:+ [$KC_CSV]}"
echo "  tier-1 argmax probes: ${PLAN_PROBES:-none}"
echo "  tier-1 spec probes:   ${PLAN_SPEC:-none}"
echo "$CHANGED" | grep -q "^crates/memra-server/" && \
    echo "  NOTE: memra-server touched — run tools/serve-smoke.sh (serving surface is not fast-gated)"
echo "  logs: $LOGDIR"

# ---------- shared probe runner ----------
# run_probe <id> <mode:check|refresh> -> sets PROBE_VERDICT (PASS/FAIL/SKIP) and PROBE_TOKS
run_probe() {
    local id="$1" mode="$2"
    PROBE_VERDICT="FAIL"; PROBE_TOKS=""
    local kind model prompt ngen extra
    kind=$(probe_field "$id" 2); model=$(probe_field "$id" 3)
    prompt=$(probe_field "$id" 4); ngen=$(probe_field "$id" 5); extra=$(probe_field "$id" 6)
    [ -z "$kind" ] && { echo "  $id: UNKNOWN probe id"; return 1; }
    # kind=cmd: a self-gating check command (host unit tests / GPU oracle gates like
    # sample-check). Gate = exit 0. No golden (the command IS its own reference); refresh
    # skips these. col3 = command, col4 = args (or -), col6 = extra env.
    if [ "$kind" = "cmd" ]; then
        local log="$LOGDIR/probe-$id.log"
        local envs=()
        [ "$extra" != "-" ] && [ -n "$extra" ] && { local kv; for kv in $extra; do envs+=("$kv"); done; }
        # shellcheck disable=SC2206
        local cmdline=($model); [ "$prompt" != "-" ] && cmdline+=($prompt)
        local t0 t1; t0=$(date +%s)
        lockrun env "${envs[@]}" timeout 900 "${cmdline[@]}" > "$log" 2>&1
        local rc=$?; t1=$(date +%s)
        if [ $rc -ne 0 ]; then
            echo "  $id: FAIL (exit $rc, $((t1-t0))s) — tail:"; tail -4 "$log" | sed 's/^/      /'
            return 1
        fi
        echo "  $id: PASS (self-gating check green, $((t1-t0))s)"
        PROBE_VERDICT="PASS"; PROBE_LAST_LOG="$log"
        return 0
    fi
    [ -f "$model" ] || { echo "  $id: SKIP (no model at $model)"; PROBE_VERDICT="SKIP"; return 0; }
    local log="$LOGDIR/probe-$id.log"
    local envs=("MEMRA_NGEN=$ngen" "MEMRA_NMEASURE=0")
    [ "$extra" != "-" ] && [ -n "$extra" ] && { local kv; for kv in $extra; do envs+=("$kv"); done; }
    local bin args=()
    case "$kind" in
        spec)  bin=target/release/run-spec ;;
        gspec) bin=target/release/gemma-gate ;;
        *)     bin=target/release/run-gen ;;
    esac
    if [ "${prompt:0:1}" = "@" ]; then
        envs+=("MEMRA_PROMPT_FILE=$ROOT/${prompt:1}")
    else
        # raw token-ids file (family-pinned ids)
        # shellcheck disable=SC2207
        args=($(cat "$ROOT/$prompt"))
    fi
    local t0 t1; t0=$(date +%s)
    lockrun env "${envs[@]}" timeout 900 "$bin" "$model" "${args[@]}" > "$log" 2>&1
    local rc=$?; t1=$(date +%s)
    if [ $rc -ne 0 ]; then
        echo "  $id: FAIL (exit $rc, $((t1-t0))s) — tail:"; tail -4 "$log" | sed 's/^/      /'
        return 1
    fi
    # gates from the raw log (parse the log, never the pipe)
    if [ "$kind" = "gspec" ]; then
        # gemma-gate spec self-compares vs plain IN the run — "stream agreement n/n" full = PASS.
        local agree
        agree=$(grep -oE "stream agreement [0-9]+/[0-9]+" "$log" | tail -1)
        local a="${agree#stream agreement }"
        if [ -z "$agree" ] || [ "${a%/*}" != "${a#*/}" ]; then
            echo "  $id: FAIL (stream agreement '${agree:-absent}', $((t1-t0))s)"; return 1
        fi
        echo "  $id: PASS ($agree, $((t1-t0))s)"
        PROBE_VERDICT="PASS"; PROBE_LAST_LOG="$log"
        return 0
    elif [ "$kind" = "spec" ]; then
        grep -q "SELF-CONSISTENCY PASS" "$log" || { echo "  $id: FAIL (no SELF-CONSISTENCY PASS, $((t1-t0))s)"; return 1; }
    else
        # " MATCH$" cannot match "MISMATCH" (the char before its MATCH substring is 'S').
        grep -qE "argmax=[0-9]+ +decode argmax=[0-9]+ .* MATCH$" "$log" || { echo "  $id: FAIL (no argmax MATCH, $((t1-t0))s)"; return 1; }
        grep -q "MISMATCH-STRUCTURED" "$log" && { echo "  $id: FAIL (batched-prime MISMATCH-STRUCTURED, $((t1-t0))s)"; return 1; }
    fi
    # ^-anchored: "prompt tokens: [...]" also contains the substring — pin the GENERATED line
    # (run-gen "tokens: [...]" at col 0; run-spec indents its plain-generate line "  tokens:").
    PROBE_TOKS=$(grep -oE "^ *tokens: \[[0-9, ]*\]" "$log" | head -1 | sed 's/^ *//')
    [ -n "$PROBE_TOKS" ] || { echo "  $id: FAIL (no generated-tokens line, $((t1-t0))s)"; return 1; }
    if [ "$mode" = "check" ]; then
        local gfile="$GOLDENS/$id.tokens"
        if [ ! -f "$gfile" ]; then
            echo "  $id: gates green but NO GOLDEN pinned ($((t1-t0))s) — run --refresh-goldens at a battery-green point"
            PROBE_VERDICT="SKIP"; return 0
        fi
        local golden; golden=$(sed -n '2p' "$gfile")
        if [ "$PROBE_TOKS" != "$golden" ]; then
            echo "  $id: FAIL — TOKEN DIVERGENCE vs golden ($(sed -n '1p' "$gfile"), $((t1-t0))s)"
            echo "      golden: ${golden:0:100}..."
            echo "      got:    ${PROBE_TOKS:0:100}..."
            return 1
        fi
        echo "  $id: PASS (gates green + golden token-identical, $((t1-t0))s)"
        # perf smoke (--smoke): single-rep tripwire, wide band, never a published number.
        if [ "$SMOKE" = 1 ] && [ -f "$GOLDENS/$id.perf" ]; then
            local toks ref
            toks=$(grep -oE "= [0-9.]+ tok/s" "$log" | tail -1 | grep -oE "[0-9.]+")
            ref=$(sed -n '2p' "$GOLDENS/$id.perf")
            if [ -n "$toks" ] && [ -n "$ref" ]; then
                local drop delta
                drop=$(awk -v n="$toks" -v r="$ref" 'BEGIN{printf "%.1f",(r-n)/r*100}')
                delta=$(awk -v n="$toks" -v r="$ref" 'BEGIN{printf "%+.1f",(n-r)/r*100}')
                if awk -v d="$drop" 'BEGIN{exit !(d>25)}'; then
                    echo "  $id: PERF-SMOKE FAIL — $toks tok/s vs golden-point $ref ($delta%, catastrophic; single rep, diagnose w/ full protocol)"
                    return 1
                elif awk -v d="$drop" 'BEGIN{exit !(d>10)}'; then
                    echo "  $id: PERF-SMOKE WARN — $toks tok/s vs golden-point $ref ($delta%; single rep, NOT evidence — re-measure per research/benchmarks.md)"
                else
                    echo "  $id: perf-smoke ok ($toks tok/s vs $ref, $delta%; single rep, informational)"
                fi
            fi
        fi
    fi
    PROBE_VERDICT="PASS"
    PROBE_LAST_LOG="$log"
    return 0
}

# ---------- golden refresh (battery-green points ONLY) ----------
if [ "$REFRESH" = 1 ]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        if [ "$FORCE" != 1 ]; then
            echo "fast-gate: refusing --refresh-goldens on a dirty tree (goldens pin battery-green"
            echo "commits, never mid-dev states). Commit/stash first, or --force if you know better."
            exit 2
        fi
        echo "fast-gate: WARNING — refreshing goldens on a DIRTY tree (--force)."
    fi
    mkdir -p "$GOLDENS"
    SHA=$(git rev-parse --short HEAD); TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    ids="${PROBES_OVERRIDE:-$(all_probe_ids | tr '\n' ',' | sed 's/,$//')}"
    echo "== fast-gate: refreshing goldens at $SHA ($TS) =="
    FAILS=0
    for id in ${ids//,/ }; do
        run_probe "$id" refresh || { FAILS=$((FAILS+1)); continue; }
        [ "$PROBE_VERDICT" = "SKIP" ] && continue
        # gspec (in-run stream agreement) and cmd (self-gating check) probes pin no golden.
        case "$(probe_field "$id" 2)" in gspec|cmd) continue ;; esac
        { echo "# golden @ $SHA $TS ngen=$(probe_field "$id" 5) model=$(probe_field "$id" 3)";
          echo "$PROBE_TOKS"; } > "$GOLDENS/$id.tokens"
        toks=$(grep -oE "= [0-9.]+ tok/s" "$PROBE_LAST_LOG" | tail -1 | grep -oE "[0-9.]+")
        [ -n "${toks:-}" ] && { echo "# single-rep tok/s @ $SHA $TS (smoke reference only, NOT evidence)";
                                echo "$toks"; } > "$GOLDENS/$id.perf"
        echo "  $id: golden pinned ($(echo "$PROBE_TOKS" | grep -oE '[0-9]+' | wc -l) ids)"
    done
    echo "goldens refresh: $FAILS fail — remember: refresh is ONLY valid at full-battery green points."
    [ "$FAILS" -eq 0 ] || exit 1
    exit 0
fi

# ---------- tier 0: compile + scoped kernel-check ----------
echo "== fast-gate tier 0 =="
T0A=$(date +%s)
cargo build --release > "$LOGDIR/build.log" 2>&1 || { echo "  BUILD FAIL:"; tail -12 "$LOGDIR/build.log"; exit 1; }
T0B=$(date +%s)
echo "  build: OK ($((T0B-T0A))s)"
case "$KC_SCOPE" in
    none)
        echo "  kernel-check: SKIP (no matched kernel scope)" ;;
    all)
        lockrun target/release/kernel-check > "$LOGDIR/kernel-check.log" 2>&1 \
            || { echo "  kernel-check FAIL:"; tail -6 "$LOGDIR/kernel-check.log"; exit 1; }
        echo "  kernel-check (FULL): GREEN ($(( $(date +%s) - T0B ))s)" ;;
    synthetic)
        lockrun env MEMRA_KC_FAST=1 target/release/kernel-check > "$LOGDIR/kernel-check.log" 2>&1 \
            || { echo "  kernel-check FAIL:"; tail -6 "$LOGDIR/kernel-check.log"; exit 1; }
        echo "  kernel-check (synthetic arms): GREEN ($(( $(date +%s) - T0B ))s)" ;;
    csv)
        lockrun env "MEMRA_KC_ONLY=$KC_CSV" target/release/kernel-check > "$LOGDIR/kernel-check.log" 2>&1 \
            || { echo "  kernel-check FAIL:"; tail -6 "$LOGDIR/kernel-check.log"; exit 1; }
        echo "  kernel-check (synthetic + $KC_CSV): GREEN ($(( $(date +%s) - T0B ))s)" ;;
esac
echo "tier 0: GREEN ($(stamp)s total)"
[ "$TIER" = "0" ] && exit 0

# ---------- tier 1: golden-token probes ----------
echo "== fast-gate tier 1 =="
FAILS=0
for id in ${PLAN_PROBES//,/ } ${PLAN_SPEC//,/ }; do
    run_probe "$id" check || FAILS=$((FAILS+1))
done
[ -z "$PLAN_PROBES$PLAN_SPEC" ] && echo "  no probes in plan (diff touches no probed dispatch class)"
echo "tier 1: $FAILS fail ($(stamp)s total)"
echo
echo "fast-gate is the DEV-LOOP gate only — the full battery (tools/local-ci.sh + perf stage)"
echo "still gates every merge and tag."
[ "$FAILS" -eq 0 ] || exit 1

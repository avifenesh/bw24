#!/usr/bin/env bash
# v0.72 tag-blocker 1: tickinv35 must stay invariant and tickinv35c must break it.
set -uo pipefail

export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "${MEMRA_TREE:-$HOME/memra}" || exit

MODEL=${MEMRA_STEP37_GGUF:-/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
SHARD2=${MODEL/00001-of-00003/00002-of-00003}
SHARD3=${MODEL/00001-of-00003/00003-of-00003}
MTP=${MEMRA_STEP37_MTP:-/data/models/step37/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf}
RAW=${RAW:-/tmp/v072-fix1-73c65c91}
mkdir -p "$RAW"

require_size() {
    local path=$1 expected=$2 actual
    [ -f "$path" ] || { echo "MISSING $path"; return 1; }
    actual=$(stat -c %s "$path")
    [ "$actual" = "$expected" ] || {
        echo "INCOMPLETE $path expected=$expected actual=$actual"
        return 1
    }
    echo "$actual $path"
}

copy_probe_log() {
    local summary=$1 name=$2 probe
    probe=$(grep -oE '/tmp/tickinv-gate-[^ )]+\.log' "$summary" | tail -1)
    [ -n "$probe" ] && [ -f "$probe" ] || {
        echo "missing probe raw-log path in $summary"
        return 1
    }
    cp "$probe" "$RAW/$name-probe-raw.log"
}

TARGS=(
    "$MODEL"
    --label step35-tick
    --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
    --budgets "0,1024,513,512,256,64"
    --splits "64,256,512"
    --seam MEMRA_PRIME_CALLLOCAL
    --steps 24
)

run_gate() {
    local name=$1
    shift
    local summary="$RAW/$name-summary.log"
    timeout 5400 tools/tick-invariance-gate.sh "${TARGS[@]}" "$@" 2>&1 | tee "$summary"
    local rc=${PIPESTATUS[0]}
    copy_probe_log "$summary" "$name" || return $?
    echo "$name rc=$rc"
    return "$rc"
}

TS=$(date -u +%Y%m%dT%H%M%SZ)
DRIVER="$RAW/verify-$TS.log"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    remote_base=$(git rev-parse HEAD)
else
    remote_base=${REMOTE_BASE:-unknown}
fi
{
    echo "=== v0.72 fix1 verification $TS local_commit=73c65c91 remote_base=$remote_base"
    echo "=== artifact byte-size gate"
    require_size "$MODEL" 46483327296 &&
        require_size "$SHARD2" 46999941600 &&
        require_size "$SHARD3" 11510293728 &&
        require_size "$MTP" 3707276416 || exit 66
    ls -lh "$MODEL" "$SHARD2" "$SHARD3" "$MTP"
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git diff --check
    fi
    sha256sum crates/memra-engine/src/hybrid_forward.rs docs/FLAGS.md

    unset MEMRA_PRIME_CALLLOCAL MEMRA_STEP35_SWA_TKV MEMRA_STEP35_SWA_FA MEMRA_NOFA
    export MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1
    (
        flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
        echo "lock acquired $(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm \
            --format=csv,noheader

        echo "########## tickinv35: naked invariant ##########"
        run_gate tickinv35
        rc_naked=$?

        echo "########## tickinv35c: canary must break invariant ##########"
        run_gate tickinv35c --canary
        rc_canary=$?

        nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm \
            --format=csv,noheader
        echo "lock released $(date -u +%FT%TZ)"
        [ "$rc_naked" -eq 0 ] && [ "$rc_canary" -eq 0 ]
    ) 9>/tmp/memra-gpu.lock
    rc=$?
    echo "=== verification rc=$rc"
    exit "$rc"
} 2>&1 | tee "$DRIVER"
exit "${PIPESTATUS[0]}"

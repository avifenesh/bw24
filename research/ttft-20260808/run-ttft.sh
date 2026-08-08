#!/usr/bin/env bash
# Cold-cache TTFT phase anatomy for Step-3.7-Flash over the real PP-2 serve path.
set -uo pipefail

REPO=${REPO:-"$HOME/cx-ttft-memra"}
BIN=${BIN:-"$HOME/cx-ttft-target/release/memra-server"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
DRAFT=${DRAFT:-"$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf"}
PROMPT4K=${PROMPT4K:-"$REPO/research/step-sku-20260807/prompt-pp4096.txt"}
RAW=${RAW:-"/tmp/ttft-20260808"}
LABEL=${LABEL:-baseline}
PORT=${PORT:-18096}
GROUPED=${GROUPED:-default}
PREFILL_TICK=${PREFILL_TICK:-default}
BASE="http://127.0.0.1:$PORT"
RUN_DIR="$RAW/$LABEL"
SERVER_LOG="$RUN_DIR/server.log"
SUMMARY_LOG="$RUN_DIR/summary.log"
SHORT_JSONL="$RUN_DIR/client-short.jsonl"
LONG_JSONL="$RUN_DIR/client-4k.jsonl"
JOINED_JSONL="$RUN_DIR/joined.jsonl"
PHASE_TSV="$RUN_DIR/phase-table.tsv"

mkdir -p "$RUN_DIR"
cd "$REPO" || exit 1

stop_server() {
    if [[ -n ${SERVER_PID:-} ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

thermal() {
    nvidia-smi \
        --query-gpu=index,name,temperature.gpu,clocks.sm,memory.used,utilization.gpu \
        --format=csv,noheader
}

boot_server() {
    local control_env=()
    case "$GROUPED" in
        default) ;;
        0|1) control_env+=("MEMRA_MOE_GROUPED=$GROUPED") ;;
        *)
            echo "GROUPED must be default, 0, or 1 (got $GROUPED)"
            return 2
            ;;
    esac
    case "$PREFILL_TICK" in
        default) ;;
        *[!0-9]*|"")
            echo "PREFILL_TICK must be default or a positive integer (got $PREFILL_TICK)"
            return 2
            ;;
        0)
            echo "PREFILL_TICK must be greater than zero"
            return 2
            ;;
        *) control_env+=("MEMRA_PREFILL_TICK=$PREFILL_TICK") ;;
    esac
    env \
        -u MEMRA_MOE_GROUPED \
        -u MEMRA_PRIME_PIPE \
        -u MEMRA_PRIME_CHUNK \
        -u MEMRA_PREFILL_TICK \
        -u MEMRA_SERVE_BATCH \
        -u MEMRA_PRIME_BATCH_HOLD_MS \
        "${control_env[@]}" \
        MEMRA_MODELS="step35=${MODEL}+${DRAFT}" \
        MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_TTFT_TRACE=1 \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_TAG="$LABEL" \
        "$BIN" >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    for attempt in $(seq 1 180); do
        sleep 5
        if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
            echo "server ready after ~$((attempt * 5))s"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "SERVER DIED"
            tail -100 "$SERVER_LOG"
            return 1
        fi
    done
    echo "server readiness timeout"
    tail -100 "$SERVER_LOG"
    return 1
}

run_probe() {
    local shape=$1
    local requests=$2
    local expected_tokens=$3
    local output=$4
    local prompt_args=()
    local expected_args=()
    if [[ "$shape" == "4k" ]]; then
        prompt_args=(--prompt-file "$PROMPT4K")
    fi
    if [[ -n "$expected_tokens" ]]; then
        expected_args=(--expect-prompt-tokens "$expected_tokens")
    fi
    python3 research/ttft-20260808/probe.py \
        --base "$BASE" \
        --model step35 \
        --shape "$shape" \
        "${prompt_args[@]}" \
        --requests "$requests" \
        --warmup 1 \
        --max-tokens 8 \
        "${expected_args[@]}" \
        --label "$LABEL" \
        --out "$output" \
        --timeout 600
}

exec > >(tee "$SUMMARY_LOG") 2>&1
echo "=== TTFT anatomy label=$LABEL ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit=$(git rev-parse HEAD)"
echo "binary=$BIN sha256=$(sha256sum "$BIN" | awk '{print $1}')"
echo "model=$MODEL bytes=$(stat -c %s "$MODEL")"
echo "draft=$DRAFT bytes=$(stat -c %s "$DRAFT")"
echo "prompt4k=$PROMPT4K bytes=$(stat -c %s "$PROMPT4K") sha256=$(sha256sum "$PROMPT4K" | awk '{print $1}')"
echo "protocol=sequential cold namespaces, 1 warmup then N=8 short and N=5 4k"
echo "serve=grouped=$GROUPED prefill_tick=$PREFILL_TICK PP-2 0,1 spec=off"

(
    flock -w 14400 9 || {
        echo "LOCK TIMEOUT"
        exit 75
    }
    trap stop_server EXIT
    echo "lock acquired $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    thermal
    boot_server || exit 1

    echo "########## short N=8 ##########"
    run_probe short 8 228 "$SHORT_JSONL" || exit 1
    thermal

    echo "########## 4k N=5 ##########"
    run_probe 4k 5 "" "$LONG_JSONL" || exit 1
    thermal

    stop_server
    python3 research/ttft-20260808/analyze.py \
        --server-log "$SERVER_LOG" \
        --client "short=$SHORT_JSONL" \
        --client "4k=$LONG_JSONL" \
        --joined "$JOINED_JSONL" \
        --table "$PHASE_TSV" || exit 1

    trace_lines=$(grep -c '^\[ttft\]' "$SERVER_LOG")
    echo "trace_lines=$trace_lines expected=15"
    if [[ "$trace_lines" -ne 15 ]]; then
        exit 1
    fi
    echo "lock released $(date -u +%Y-%m-%dT%H:%M:%SZ)"
) 9>/tmp/memra-gpu.lock
rc=$?
echo "=== TTFT anatomy rc=$rc"
echo "run_dir=$RUN_DIR"
echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$rc"

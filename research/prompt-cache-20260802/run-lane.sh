#!/bin/bash
# lane/prompt-cache receipts runner (2026-08-02). RTX 5090; EVERY GPU phase under
# flock /tmp/gpu5090.lock (short holds — one server or one gate binary per hold).
# Servers killed by pid, never pkill (co-resident llama-server --embedding survives).
# Workflow-args law: literals only.
#
# phases: audit | gate | load | battery | all
set -u
W=/home/avifenesh/projects/wt-prompt-cache
R=$W/research/prompt-cache-20260802
BIN=$W/target/release
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Q35_DRAFT=/data/ai-ml/hf-models/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf
Q9J=/home/avifenesh/models/qwen3.5-9b-judge-q8_0.gguf
PFP2=$W/research/e2e/prompts/p2-code-medium.txt
PORT=8127
PHASE=${1:-all}

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
exec > >(tee -a "$R/console.log") 2>&1
echo "=== PROMPT-CACHE LANE phase=$PHASE $TS git=$GIT_SHA ==="

busy_procs() {
  local n=0 pid
  while IFS=, read -r pid _; do
    pid=$(echo "$pid" | tr -d ' '); [ -n "$pid" ] || continue
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--embedding" || n=$((n+1))
  done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  echo $n
}
wait_idle() {
  local n=0
  while true; do
    local busy; busy=$(busy_procs)
    [ "$busy" -eq 0 ] && break
    sleep 5; n=$((n+1)); [ $n -gt 240 ] && { echo "wait_idle timeout (busy=$busy)"; break; }
  done
}

# serve_and <server-env...> -- <client cmd...>   (runs inside the flock holder)
serve_and() {
  local envv=()
  while [ "$1" != "--" ]; do envv+=("$1"); shift; done
  shift
  local tag=$1; shift
  wait_idle
  env "${envv[@]}" MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_MODELS="m=$Q35" \
    "$BIN/memra-server" > "$R/server-$tag.log" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 240); do
    sleep 2
    curl -s -m 2 "http://127.0.0.1:$PORT/models" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$pid" 2>/dev/null || break
  done
  if [ "$up" -ne 1 ]; then
    echo "SERVER FAILED $tag (see server-$tag.log)"; kill "$pid" 2>/dev/null; return 1
  fi
  "$@"
  local rc=$?
  curl -s -m 5 "http://127.0.0.1:$PORT/metrics" > "$R/metrics-$tag.json" 2>/dev/null
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 3
  return $rc
}

lock() { flock /tmp/gpu5090.lock "$@"; }

if [ "$PHASE" = audit ] || [ "$PHASE" = all ]; then
  echo "--- phase: audit (as-is receipts) ---"
  # 1a. naked defaults (q35 embeds an MTP head -> spec tier serves greedy chat)
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35
    serve_and MEMRA_PREFIX_CACHE_MB=0 -- audit-asis-naked \
      python3 $R/agentic_load.py audit --base http://127.0.0.1:$PORT --model m \
        --label asis-naked --out $R/audit-asis-naked.jsonl"
  # 1b. bulk tier as-is (spec off, prefix cache off = pre-lane session/KV behavior)
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35
    serve_and MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=0 -- audit-asis-bulk \
      python3 $R/agentic_load.py audit --base http://127.0.0.1:$PORT --model m \
        --label asis-bulk --out $R/audit-asis-bulk.jsonl"
  # 1c. bulk tier with the prefix cache ON (the gap-closed twin of 1b)
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35
    serve_and MEMRA_SERVE_SPEC=0 -- audit-cache-on \
      python3 $R/agentic_load.py audit --base http://127.0.0.1:$PORT --model m \
        --label cache-on --out $R/audit-cache-on.jsonl"
fi

if [ "$PHASE" = gate ] || [ "$PHASE" = all ]; then
  echo "--- phase: gate (cached-vs-fresh exactness) ---"
  # refs: cache-OFF whole-prime fresh outputs (cross-config REPORT + A1 same-config control)
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35
    serve_and MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=0 -- gate-refs \
      python3 $R/cache_exact_gate.py --base http://127.0.0.1:$PORT --model m \
        --collect-refs $R/gate-refs.json"
  # gate: cache-ON (1024MB budget: cells stay resident while live; LRU exercised across cells)
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35
    serve_and MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=1024 -- gate-exact \
      python3 $R/cache_exact_gate.py --base http://127.0.0.1:$PORT --model m \
        --ref $R/gate-refs.json --out $R/gate-exact.jsonl"
  echo "gate rc=$?"
fi

if [ "$PHASE" = load ] || [ "$PHASE" = all ]; then
  echo "--- phase: load (TTFT/throughput, c=8, N=3 interleaved arms) ---"
  for rep in 1 2 3; do
    lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35
      serve_and MEMRA_SERVE_SPEC=0 -- load-on-rep$rep \
        python3 $R/agentic_load.py load --base http://127.0.0.1:$PORT --model m \
          --label cache-on-rep$rep --out $R/load-cache-on.jsonl"
    lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35
      serve_and MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=0 -- load-off-rep$rep \
        python3 $R/agentic_load.py load --base http://127.0.0.1:$PORT --model m \
          --label cache-off-rep$rep --out $R/load-cache-off.jsonl"
  done
fi

if [ "$PHASE" = battery ] || [ "$PHASE" = all ]; then
  echo "--- phase: battery ---"
  wait_idle
  lock timeout 3600 "$BIN/kernel-check" "$Q35" > "$R/battery-kernel-check.log" 2>&1
  rc=$?
  echo "kernel-check rc=$rc FAIL=$(grep -c FAIL "$R/battery-kernel-check.log") OK=$(grep -c ' OK' "$R/battery-kernel-check.log")"

  for tag in q35 q9j; do
    m=$Q35; [ "$tag" = q9j ] && m=$Q9J
    wait_idle
    lock timeout 3600 "$BIN/decode-batch-gate" "$m" > "$R/battery-decode-batch-$tag.log" 2>&1
    echo "decode-batch $tag rc=$? $(grep -cE 'ALL GREEN' "$R/battery-decode-batch-$tag.log") green-lines"
    # strict mode runs via run-strict-equalized.sh — the ACTUAL protocol needs per-model
    # env (MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 + worst-draw MEMRA_GATE_SEED q35=16/q9j=0,
    # --batch 4; gate1-recal-20260802 + validate-h100.sh). The first battery pass ran the
    # stale bare "--mode strict" form and misfired exactly like f16g-default-rearb's first
    # attempt; a second attempt with --batch 4 but no env misfired on the same documented
    # FP-composition gap (*-strict-MISFIRE*.log kept, superseded by the equalized greens).
  done

  # serve c1-vs-c16 byte identity at NAKED defaults (q35 = the spec tier; the prefix cache
  # must leave it untouched). Fresh refs.
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35; W=$W
    serve_and MEMRA_LANE=prompt-cache -- serve-c1c16-naked \
      python3 $W/tools/check-batch-exact.py --base http://127.0.0.1:$PORT --model m \
        --n 16 --max-tokens 96 --label q35-naked-prompt-cache-lane \
        --out $R/greedy-hash-q35-naked.jsonl --ref $R/greedy-refs-q35-naked.json"
  echo "serve-c1c16 naked rc=$?"
  # and on the BULK tier with the prefix cache live (the tier the cache serves)
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35; W=$W
    serve_and MEMRA_SERVE_SPEC=0 -- serve-c1c16-bulk \
      python3 $W/tools/check-batch-exact.py --base http://127.0.0.1:$PORT --model m \
        --n 16 --max-tokens 96 --label q35-bulk-cache-on \
        --out $R/greedy-hash-q35-bulk.jsonl --ref $R/greedy-refs-q35-bulk.json"
  echo "serve-c1c16 bulk rc=$?"

  wait_idle
  env MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=64 MEMRA_PROMPT="$(cat "$PFP2")" \
    flock /tmp/gpu5090.lock timeout 3600 "$BIN/run-spec" "$Q35" > "$R/battery-spec-q35.log" 2>&1
  echo "run-spec q35 rc=$? PASS=$(grep -c 'self-consistency: PASS' "$R/battery-spec-q35.log")"
fi

echo "LANE-DONE phase=$PHASE $(date -u +%FT%TZ)"

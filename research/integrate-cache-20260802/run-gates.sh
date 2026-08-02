#!/bin/bash
# integrate-cache merge gate battery (2026-08-02), RTX 5090. The price of merging
# lane/serve-tools (already in HEAD) x lane/prompt-cache — all five gates on the ONE
# merged binary:
#   hold 1: kernel-check q35 (engine untouched — expected ALL GREEN)
#   hold 2: NAKED q35 (spec tier): serve greedy c1-vs-c16 16/16 + the tools lane's
#           round-trip battery (roundtrip_gate.py: A/A'/B/C/D/E)
#   hold 3: bulk tier, cache OFF: cache-exactness refs (cache_exact_gate --collect-refs)
#   hold 4: bulk tier, cache ON 1024MB: cache-exactness gate (partial==split, full==cold,
#           usage truth incl cached_tokens)
#   hold 5: bulk tier, cache ON default 256MB: THE INTERSECTION gate (tools request
#           repeated 3x -> third hits, call parses identically, usage exact)
# Laws: literals only; every GPU run inside `flock /tmp/gpu5090.lock`; servers killed by
# PID, never pkill (co-resident llama-server survives); raw logs land next to this file.
set -u
W=/home/avifenesh/projects/bw24-integrate-cache
R=$W/research/integrate-cache-20260802
ST=$W/research/serve-tools-20260802
PC=$W/research/prompt-cache-20260802
BIN=$W/target/release
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PORT=8127
PHASE=${1:-all}

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
exec > >(tee -a "$R/console.log") 2>&1
echo "=== INTEGRATE-CACHE GATES phase=$PHASE $TS git=$GIT_SHA (merged tree, pre-commit) ==="

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
    sleep 5; n=$((n+1)); [ $n -gt 720 ] && { echo "wait_idle timeout (busy=$busy)"; break; }
  done
}

# serve_and <server-env...> -- <tag> <client cmd...>  (runs inside the flock holder)
serve_and() {
  local envv=()
  while [ "$1" != "--" ]; do envv+=("$1"); shift; done
  shift
  local tag=$1; shift
  wait_idle
  env "${envv[@]}" MEMRA_ADDR="127.0.0.1:$PORT" \
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

if [ "$PHASE" = kernel ] || [ "$PHASE" = all ]; then
  echo "--- hold 1: kernel-check q35 ---"
  wait_idle
  lock timeout 3600 "$BIN/kernel-check" "$Q35" > "$R/battery-kernel-check.log" 2>&1
  rc=$?
  echo "kernel-check rc=$rc FAIL=$(grep -c FAIL "$R/battery-kernel-check.log") OK=$(grep -c ' OK' "$R/battery-kernel-check.log")"
fi

if [ "$PHASE" = naked ] || [ "$PHASE" = all ]; then
  echo "--- hold 2: naked q35 (spec tier) — c1-vs-c16 + tools round-trip battery ---"
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; W=$W; ST=$ST; Q35=$Q35
    serve_and MEMRA_MODELS=q35=$Q35 -- naked-q35 bash -c '
      set -o pipefail
      python3 $W/tools/check-batch-exact.py --base http://127.0.0.1:$PORT --model q35 \
        --n 16 --max-tokens 96 --label q35-naked-integrate-cache \
        --out $R/greedy-hash-q35-naked.jsonl --ref $R/greedy-refs-q35-naked.json \
        2>&1 | tee $R/greedy-hash-q35-naked.log
      rc1=\$?
      python3 $ST/roundtrip_gate.py --base http://127.0.0.1:$PORT --model q35 \
        --gguf $Q35 --tok-check $BIN/tok-check --out $R --tag q35 \
        2>&1 | tee $R/roundtrip-q35-console.log
      rc2=\$?
      echo \"c1c16 rc=\$rc1 roundtrip rc=\$rc2\"
      [ \$rc1 -eq 0 ] && [ \$rc2 -eq 0 ]'"
  echo "hold2 rc=$?"
fi

if [ "$PHASE" = cachegate ] || [ "$PHASE" = all ]; then
  echo "--- hold 3: bulk cache OFF — exactness refs ---"
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; PC=$PC; Q35=$Q35
    serve_and MEMRA_MODELS=m=$Q35 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=0 -- gate-refs \
      python3 $PC/cache_exact_gate.py --base http://127.0.0.1:$PORT --model m \
        --collect-refs $R/gate-refs.json"
  echo "refs rc=$?"
  echo "--- hold 4: bulk cache ON 1024MB — exactness gate ---"
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; PC=$PC; Q35=$Q35
    serve_and MEMRA_MODELS=m=$Q35 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=1024 -- gate-exact \
      python3 $PC/cache_exact_gate.py --base http://127.0.0.1:$PORT --model m \
        --ref $R/gate-refs.json --out $R/gate-exact.jsonl"
  echo "cache-exact rc=$?"
fi

if [ "$PHASE" = intersection ] || [ "$PHASE" = all ]; then
  echo "--- hold 5: bulk cache ON (default 256MB) — THE INTERSECTION (tools x cache) ---"
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35
    serve_and MEMRA_MODELS=q35=$Q35 MEMRA_SERVE_SPEC=0 -- intersection \
      python3 $R/intersection_gate.py --base http://127.0.0.1:$PORT --model q35 \
        --gguf $Q35 --tok-check $BIN/tok-check --out $R"
  echo "intersection rc=$?"
fi

echo "GATES-DONE phase=$PHASE $(date -u +%FT%TZ)"

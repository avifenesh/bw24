#!/bin/bash
# PC-ISO gate battery (lane/pc-iso, 2026-08-02), RTX 5090. Two holds on the PC-ISO binary:
#   hold 1 (no-salt regression): the integrate-cache INTERSECTION gate, UNMODIFIED
#           (research/integrate-cache-20260802/intersection_gate.py) — no request carries
#           cache_salt, so the whole run rides the default "" namespace; 8/8 gated rows
#           = requirement (c), existing behavior byte-preserved.
#   hold 2 (isolation): salt_gate.py — same-salt hit / cross-salt miss both directions /
#           default-namespace blindness, on the cached_tokens oracle. Budget 1024MB so
#           LRU eviction cannot masquerade as isolation.
# Laws: literals only; every GPU run inside `flock /tmp/gpu5090.lock`; server killed by
# PID, never pkill (co-resident llama-server survives); raw logs land next to this file.
set -u
W=/home/avifenesh/projects/wt-pc-iso
R=$W/research/pc-iso-20260802
IC=$W/research/integrate-cache-20260802
BIN=$W/target/release
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PORT=8129
PHASE=${1:-all}

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
exec > >(tee -a "$R/console.log") 2>&1
echo "=== PC-ISO GATES phase=$PHASE $TS git=$GIT_SHA ==="

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

if [ "$PHASE" = intersection ] || [ "$PHASE" = all ]; then
  echo "--- hold 1: NO-SALT regression — the intersection gate, unmodified (8 gated rows) ---"
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; IC=$IC; Q35=$Q35
    serve_and MEMRA_MODELS=q35=$Q35 MEMRA_SERVE_SPEC=0 -- intersection \
      python3 $IC/intersection_gate.py --base http://127.0.0.1:$PORT --model q35 \
        --gguf $Q35 --tok-check $BIN/tok-check --out $R"
  echo "intersection rc=$?"
fi

if [ "$PHASE" = salt ] || [ "$PHASE" = all ]; then
  echo "--- hold 2: ISOLATION — salt_gate.py (same-salt hit / cross-salt miss / default ns) ---"
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; Q35=$Q35
    serve_and MEMRA_MODELS=q35=$Q35 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=1024 -- salt \
      python3 $R/salt_gate.py --base http://127.0.0.1:$PORT --model q35 --out $R"
  echo "salt rc=$?"
fi

echo "GATES-DONE phase=$PHASE $(date -u +%FT%TZ)"

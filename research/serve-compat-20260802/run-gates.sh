#!/bin/bash
# serve-compat gate battery (lane/serve-compat, 2026-08-03), RTX 5090. Three holds:
#   hold 1 (SDK): sdk_gate.py — the official `openai` Python SDK round-trips completion +
#           stream + tool-call + reasoning + errors against a live server (gap-scan F1
#           acceptance); the disconnect probe's [abort] server-log line asserted here.
#   hold 2 (no-regression): the integrate-cache INTERSECTION gate, UNMODIFIED
#           (research/integrate-cache-20260802/intersection_gate.py) on q35 — tools x
#           prompt-cache behavior byte-preserved under the envelope/reasoning changes.
#           NOTE: post-F13, think prose rides message.reasoning; the gate's identity keys
#           (content + tool_calls + finish) and usage crosschecks are unchanged.
#   hold 3 (SDK model): hold 1 runs on q9 (the MTP spec-serve daily driver) — the fast
#           model; hold 2 keeps the intersection gate's own pinned q35.
# Laws: literals only; every GPU run inside `flock /tmp/gpu5090.lock`; server killed by
# PID, never pkill; raw logs land next to this file.
set -u
W=/home/avifenesh/projects/wt-serve-compat
R=$W/research/serve-compat-20260802
IC=$W/research/integrate-cache-20260802
BIN=$W/target/release
PY=/tmp/openai-gate-venv/bin/python
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PORT=8131
PHASE=${1:-all}

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
exec > >(tee -a "$R/console.log") 2>&1
echo "=== SERVE-COMPAT GATES phase=$PHASE $TS git=$GIT_SHA ==="

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

if [ "$PHASE" = sdk ] || [ "$PHASE" = all ]; then
  echo "--- hold 1: SDK — official openai client round-trip on q9 ---"
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; PY=$PY; Q9=$Q9
    serve_and MEMRA_MODELS=q9=$Q9 -- sdk \
      $PY $R/sdk_gate.py --base http://127.0.0.1:$PORT --model q9 --out $R"
  rc=$?
  echo "sdk rc=$rc"
  # G8 server-side receipt: the disconnect must have produced an [abort] log line.
  if grep -q "\[abort\] client disconnected" "$R/server-sdk.log"; then
    echo '{"gate":"G8-abort-logline","verdict":"PASS"}' >> "$R/sdk-gates.jsonl"
    echo "G8-abort-logline PASS"
  else
    echo '{"gate":"G8-abort-logline","verdict":"FAIL"}' >> "$R/sdk-gates.jsonl"
    echo "G8-abort-logline FAIL (no [abort] line in server-sdk.log)"
  fi
fi

if [ "$PHASE" = intersection ] || [ "$PHASE" = all ]; then
  echo "--- hold 2: NO-REGRESSION — the intersection gate, unmodified (q35) ---"
  lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; BIN=$BIN; IC=$IC; Q35=$Q35
    serve_and MEMRA_MODELS=q35=$Q35 MEMRA_SERVE_SPEC=0 -- intersection \
      python3 $IC/intersection_gate.py --base http://127.0.0.1:$PORT --model q35 \
        --gguf $Q35 --tok-check $BIN/tok-check --out $R"
  echo "intersection rc=$?"
fi

echo "GATES-DONE phase=$PHASE $(date -u +%FT%TZ)"

#!/bin/bash
# Attribution probes for the leg-R cold-vs-full-hit think-text divergence (2026-08-02).
#   probe 1: MERGED binary, bulk, cache OFF  — cold determinism (leg-R tools chat 2x)
#   probe 2: MERGED binary, bulk, cache ON   — raw rendered prompt 3x (cold vs hit)
#   probe 3: CACHE-LANE binary (af72c3db, pre-merge), bulk, cache ON — same raw 3x
# If probe 1 is identical and probe 3 diverges like probe 2, the class PRE-DATES the merge.
set -u
W=/home/avifenesh/projects/bw24-integrate-cache
R=$W/research/integrate-cache-20260802
LANEBIN=/home/avifenesh/projects/wt-prompt-cache/target/release/memra-server
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PORT=8127

exec > >(tee -a "$R/attribution-console.log") 2>&1
echo "=== ATTRIBUTION PROBES $(date -u +%FT%TZ) ==="

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
serve_and() { # <binary> <env...> -- <tag> <client...>
  local bin=$1; shift
  local envv=()
  while [ "$1" != "--" ]; do envv+=("$1"); shift; done
  shift
  local tag=$1; shift
  wait_idle
  env "${envv[@]}" MEMRA_ADDR="127.0.0.1:$PORT" \
    "$bin" > "$R/server-$tag.log" 2>&1 &
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
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 3
  return $rc
}
lock() { flock /tmp/gpu5090.lock "$@"; }

echo "--- probe 1: merged, cache OFF — cold determinism ---"
lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; W=$W; Q35=$Q35
  serve_and $W/target/release/memra-server MEMRA_MODELS=q35=$Q35 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=0 -- attr-cold2 \
    python3 $R/attribution_probe.py --base http://127.0.0.1:$PORT --model q35 \
      --mode cold2 --tag merged-cold2 --out $R"
echo "probe1 rc=$?"

echo "--- probe 2: merged, cache ON — raw rendered prompt 3x ---"
lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; W=$W; Q35=$Q35
  serve_and $W/target/release/memra-server MEMRA_MODELS=q35=$Q35 MEMRA_SERVE_SPEC=0 -- attr-merged-raw3 \
    python3 $R/attribution_probe.py --base http://127.0.0.1:$PORT --model q35 \
      --mode raw3 --tag merged-raw3 --out $R"
echo "probe2 rc=$?"

echo "--- probe 3: cache-lane binary (pre-merge), cache ON — raw rendered prompt 3x ---"
lock bash -c "$(declare -f serve_and busy_procs wait_idle); R=$R; PORT=$PORT; LANEBIN=$LANEBIN; Q35=$Q35
  serve_and $LANEBIN MEMRA_MODELS=q35=$Q35 MEMRA_SERVE_SPEC=0 -- attr-lane-raw3 \
    python3 $R/attribution_probe.py --base http://127.0.0.1:$PORT --model q35 \
      --mode raw3 --tag lane-raw3 --out $R"
echo "probe3 rc=$?"

echo "ATTRIBUTION-DONE $(date -u +%FT%TZ)"

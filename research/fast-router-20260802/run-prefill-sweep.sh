#!/bin/bash
# fast-router: prefill recovery A/B/C on the 5090 (RTX 5090 Laptop 24463 MiB).
# Arms per rep (interleaved, rep loop outside):
#   exact0 = MEMRA_ROUTER_PREFILL_EXACT=0  (pre-fix cuBLASLt router+shexp — the m-DEPENDENT
#            reference the concat lane replaced; q35 board-2048 was ~3501 here)
#   plain  = MEMRA_ROUTER_BATCH=0          (the concat-lane fix as merged: per-(e,tok) w8
#            kernels at prefill m — the ~3151 regression arm)
#   batch  = naked default                 (this lane: register-tiled batch twins,
#            bit-identical to plain per row — the recovery arm)
# Cells: q35 board-2048 (board prompt, the -10% cell) + o35b pp512 (residency-cap 1079.2
# baseline shape, resident-if-fits default). NGEN=128, run-gen argmax MATCH per run,
# busy-proc gate (co-resident llama-server --embedding allowlisted), every GPU run under
# flock /tmp/gpu5090.lock. Workflow-args law: every parameter is a literal here.
# usage: run-prefill-sweep.sh [nreps]
set -u
N=${1:-5}
W=/home/avifenesh/projects/wt-fast-router
R=$W/research/fast-router-20260802
OUT=$R/prefill-sweep.jsonl
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)
gpu-full-power on >/dev/null 2>&1 || true

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
    sleep 5; n=$((n+1)); [ $n -gt 120 ] && { echo "wait_idle timeout (busy=$busy)"; break; }
  done
}
row() { # cell arm metric value rep
  printf '{"ts":"%s","git":"%s","cell":"%s","arm":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1/$2 rep$5] $3 = $4"
}

run_arm() { # cell model promptfile arm rep
  local cell=$1 model=$2 pf=$3 arm=$4 rep=$5 log="$R/$1-$4-rep$5.log"
  local -a env_extra=()
  case "$arm" in
    exact0) env_extra=(MEMRA_ROUTER_PREFILL_EXACT=0) ;;
    plain)  env_extra=(MEMRA_ROUTER_BATCH=0) ;;
    batch)  ;;
  esac
  wait_idle
  env "${env_extra[@]}" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$pf" \
    flock /tmp/gpu5090.lock timeout 900 "$W/target/release/run-gen" "$model" > "$log" 2>&1
  local pp tg gate
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  row "$cell" "$arm" prefill_toks "${pp:-null}" "$rep"
  row "$cell" "$arm" decode_toks "${tg:-null}" "$rep"
  row "$cell" "$arm" argmax_match_lines "${gate:-0}" "$rep"
}

for rep in $(seq 1 "$N"); do
  echo "=== rep $rep/$N ==="
  for arm in exact0 plain batch; do
    run_arm q35-board2048 "$Q35" "$W/research/e2e/prompts/board-2048.txt" "$arm" "$rep"
  done
  for arm in exact0 plain batch; do
    run_arm o35b-pp512 "$O35B" "$W/research/e2e/prompts/pp512.txt" "$arm" "$rep"
  done
done

echo "=== medians ==="
python3 - "$OUT" <<'EOF'
import json, sys, statistics
rows = [json.loads(l) for l in open(sys.argv[1])]
cells = {}
for r in rows:
    if r["metric"] == "prefill_toks" and r["value"] is not None:
        cells.setdefault((r["cell"], r["arm"]), []).append(float(r["value"]))
for (c, a), v in sorted(cells.items()):
    print(f"{c:16s} {a:8s} N={len(v)} median={statistics.median(v):.1f} range=[{min(v):.1f},{max(v):.1f}]")
EOF

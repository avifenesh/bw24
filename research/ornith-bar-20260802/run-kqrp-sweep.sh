#!/bin/bash
# ornith-bar: 5090 KQRP A/B sweep on the Ornith-35B Q4_K trunk (MEMRA_KQRP is Hopper-default,
# unswept on this rig). Interleaved x5 (arm loop inside rep loop), board shape
# (pp512.txt prompt, NGEN=128), run-gen argmax gate per run, A/B token bit-identity check
# (KQRP claims bit-identical — verified per rep via the generated-tokens line).
# No local q27-class gguf exists on this rig (see q27-local-check.log) — Ornith-35B alone.
# usage: run-kqrp-sweep.sh [nreps]
set -u
N=${1:-5}
W=/home/avifenesh/projects/wt-ornith-bar
R=$W/research/ornith-bar-20260802
PF=$W/research/e2e/prompts/pp512.txt
OUT=$R/kqrp-sweep.jsonl
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
row() { # arm metric value rep
  printf '{"ts":"%s","git":"%s","cell":"o35b-kqrp","arm":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1 rep$4] $2 = $3"
}

run_arm() { # arm(off|on) rep
  local arm=$1 rep=$2 log="$R/kqrp-$1-rep$2.log"
  local -a env_extra=()
  [ "$arm" = on ] && env_extra=(MEMRA_KQRP=1)
  env "${env_extra[@]}" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF" \
    flock /tmp/gpu5090.lock timeout 900 "$W/target/release/run-gen" "$O35B" > "$log" 2>&1
  local rc=$?
  local pp tg gate nmir thash
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  nmir=$(grep -oE "split-plane decode mirrors built: [0-9]+" "$log" | grep -oE "[0-9]+" | tail -1)
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  if [ -z "${tg:-}" ]; then row "$arm" ERROR "$rc" "$rep"; tail -5 "$log"; return; fi
  row "$arm" prefill_toks "${pp:-0}" "$rep"
  row "$arm" decode_toks "$tg" "$rep"
  row "$arm" argmax_match "$gate" "$rep"
  row "$arm" mirrors_built "${nmir:-0}" "$rep"
  echo "  [$arm rep$rep] tokens_sha=$thash" | tee -a "$R/kqrp-token-hashes.log"
}

echo "=== O35B KQRP SWEEP x$N $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/kqrp-console.log"
{
  for rep in $(seq 1 "$N"); do
    wait_idle; run_arm off "$rep"
    wait_idle; run_arm on  "$rep"
  done
  echo "KQRP-SWEEP-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/kqrp-console.log"

#!/bin/bash
# residency-cap: Ornith-35B Q4_K_M expert-residency A/B on the 5090 (24463 MiB).
# Arms per rep (interleaved, rep loop outside): slru (naked default = SLRU spill cache),
# resident (MEMRA_MOE_RESIDENT_GB=21 -> clears the 20.9GB projected check), llama
# (llama-bench fork bb090d1f1, -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p 512 -n 128 -r 1).
# Board shape: pp512.txt prompt, NGEN=128, run-gen argmax gate per memra run, per-run
# peak-VRAM sampler (1s), busy-proc gate (co-resident llama-server --embedding allowlisted),
# every GPU run under flock /tmp/gpu5090.lock.
# usage: run-residency-sweep.sh [nreps]
set -u
N=${1:-5}
W=/home/avifenesh/projects/wt-residency-cap
R=$W/research/residency-cap-20260802
PF=$W/research/e2e/prompts/pp512.txt
OUT=$R/residency-sweep.jsonl
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
LLAMA=/home/avifenesh/projects/llama.cpp/build/bin/llama-bench

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
  printf '{"ts":"%s","git":"%s","cell":"o35b-residency","arm":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1 rep$4] $2 = $3"
}

run_memra() { # arm(slru|resident) rep
  local arm=$1 rep=$2 log="$R/$1-rep$2.log" vram="$R/$1-rep$2.vram"
  local -a env_extra=()
  [ "$arm" = resident ] && env_extra=(MEMRA_MOE_RESIDENT_GB=21)
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vram"; sleep 1; done ) &
  local sampler=$!
  env "${env_extra[@]}" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF" \
    flock /tmp/gpu5090.lock timeout 900 "$W/target/release/run-gen" "$O35B" > "$log" 2>&1
  local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  local pp tg gate peak dec thash
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  peak=$(sort -n "$vram" | tail -1)
  dec=$(grep -oE "resident-experts decision: .* -> (RESIDENT|SLRU cache)" "$log" | grep -oE "(RESIDENT|SLRU cache)$" | head -1 | tr ' ' '_')
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  if [ -z "${tg:-}" ]; then row "$arm" ERROR "$rc" "$rep"; tail -5 "$log"; return; fi
  row "$arm" prefill_toks "${pp:-0}" "$rep"
  row "$arm" decode_toks "$tg" "$rep"
  row "$arm" argmax_match "$gate" "$rep"
  row "$arm" peak_vram_mib "${peak:-0}" "$rep"
  echo "  [$arm rep$rep] decision=$dec tokens_sha=$thash" | tee -a "$R/token-hashes.log"
}

run_llama() { # rep
  local rep=$1 log="$R/llama-rep$1.log" vram="$R/llama-rep$1.vram"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vram"; sleep 1; done ) &
  local sampler=$!
  flock /tmp/gpu5090.lock timeout 900 "$LLAMA" -m "$O35B" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 \
    -p 512 -n 128 -r 1 -o json > "$log" 2>&1
  local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  local pp tg peak
  pp=$(python3 -c "import json,sys; d=json.load(open('$log')); print([r['avg_ts'] for r in d if r['n_prompt']>0][0])" 2>/dev/null)
  tg=$(python3 -c "import json,sys; d=json.load(open('$log')); print([r['avg_ts'] for r in d if r['n_gen']>0][0])" 2>/dev/null)
  peak=$(sort -n "$vram" | tail -1)
  if [ -z "${tg:-}" ]; then row llama ERROR "$rc" "$rep"; tail -5 "$log"; return; fi
  row llama prefill_toks "$pp" "$rep"
  row llama decode_toks "$tg" "$rep"
  row llama peak_vram_mib "${peak:-0}" "$rep"
}

echo "=== O35B RESIDENCY SWEEP x$N $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/sweep-console.log"
{
  for rep in $(seq 1 "$N"); do
    wait_idle; run_memra slru "$rep"
    wait_idle; run_memra resident "$rep"
    wait_idle; run_llama "$rep"
  done
  echo "RESIDENCY-SWEEP-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/sweep-console.log"

#!/bin/bash
# kat-anomaly: KAT-Coder IQ4_XS decode anomaly on the 5090 (24463 MiB).
# Arms per rep (interleaved, rep loop OUTSIDE): kat-naked (residency decision + anomaly
# confirm), kat-iqfast (MEMRA_IQ_FAST=1 -> IQ4_XS trunk matvecs ride qmatvec_iq4_XS_dp4a
# instead of the Stage-A f32 oracle path — the op-class experiment), ctrl (Qwen3.6-35B
# UD-IQ4_XS, MEMRA_PRIME_TOKENWISE=1 per the residency-cap branch finding; decode leg is
# prime-mode-independent), llama-kat (llama-bench fork bb090d1f1 on the KAT gguf — the
# vs-llama plain cell for the #42 flip).
# Board shape: pp512.txt prompt, NGEN=128, run-gen argmax gate per memra run, per-run
# peak-VRAM sampler (1s), busy-proc gate (co-resident llama-server --embedding allowlisted),
# every GPU run under flock /tmp/gpu5090.lock.
# usage: run-kat-sweep.sh [nreps]
set -u
N=${1:-5}
W=/home/avifenesh/projects/wt-kat-anomaly
R=$W/research/kat-anomaly-20260802
PF=$W/research/e2e/prompts/pp512.txt
OUT=$R/kat-sweep.jsonl
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
CTRL=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
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
    sleep 5; n=$((n+1)); [ $n -gt 240 ] && { echo "wait_idle timeout (busy=$busy)"; break; }
  done
}
row() { # arm metric value rep
  printf '{"ts":"%s","git":"%s","cell":"kat-anomaly","arm":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1 rep$4] $2 = $3"
}

run_memra() { # arm model rep env...
  local arm=$1 model=$2 rep=$3; shift 3
  local log="$R/$arm-rep$rep.log" vram="$R/$arm-rep$rep.vram"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vram"; sleep 1; done ) &
  local sampler=$!
  env "$@" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF" \
    flock /tmp/gpu5090.lock timeout 900 "$W/target/release/run-gen" "$model" > "$log" 2>&1
  local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  local pp tg gate peak dec thash stop
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  gate=$(grep -c "MATCH" "$log")
  peak=$(sort -n "$vram" | tail -1)
  dec=$(grep -oE "resident-experts decision: .*" "$log" | head -1)
  stop=$(grep -oE "\[stop: [A-Za-z]+\]" "$log" | head -1)
  thash=$(grep -A1 "^generated" "$log" | grep "tokens:" | sha256sum | cut -c1-16)
  if [ -z "${tg:-}" ]; then row "$arm" ERROR "$rc" "$rep"; tail -5 "$log"; return; fi
  row "$arm" prefill_toks "${pp:-0}" "$rep"
  row "$arm" decode_toks "$tg" "$rep"
  row "$arm" argmax_match "$gate" "$rep"
  row "$arm" peak_vram_mib "${peak:-0}" "$rep"
  echo "  [$arm rep$rep] $stop tokens_sha=$thash" | tee -a "$R/token-hashes.log"
  [ -n "$dec" ] && echo "  [$arm rep$rep] $dec" | tee -a "$R/decision-lines.log"
}

run_llama() { # rep
  local rep=$1 log="$R/llama-kat-rep$1.log" vram="$R/llama-kat-rep$1.vram"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vram"; sleep 1; done ) &
  local sampler=$!
  flock /tmp/gpu5090.lock timeout 900 "$LLAMA" -m "$KAT" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 \
    -p 512 -n 128 -r 1 -o json > "$log" 2>&1
  local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  local pp tg peak
  pp=$(python3 -c "import json,sys; d=json.load(open('$log')); print([r['avg_ts'] for r in d if r['n_prompt']>0][0])" 2>/dev/null)
  tg=$(python3 -c "import json,sys; d=json.load(open('$log')); print([r['avg_ts'] for r in d if r['n_gen']>0][0])" 2>/dev/null)
  peak=$(sort -n "$vram" | tail -1)
  if [ -z "${tg:-}" ]; then row llama-kat ERROR "$rc" "$rep"; tail -5 "$log"; return; fi
  row llama-kat prefill_toks "$pp" "$rep"
  row llama-kat decode_toks "$tg" "$rep"
  row llama-kat peak_vram_mib "${peak:-0}" "$rep"
}

echo "=== KAT-ANOMALY SWEEP x$N $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/sweep-console.log"
{
  for rep in $(seq 1 "$N"); do
    wait_idle; run_memra kat-naked  "$KAT"  "$rep" MEMRA_DUMMY=0
    wait_idle; run_memra kat-iqfast "$KAT"  "$rep" MEMRA_IQ_FAST=1
    wait_idle; run_memra ctrl-q35   "$CTRL" "$rep" MEMRA_PRIME_TOKENWISE=1
    wait_idle; run_llama "$rep"
  done
  echo "KAT-SWEEP-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/sweep-console.log"

#!/bin/bash
# depth-decode item 2: mechanism. (a) nsys kernel-time split of the decode window at d512 vs
# d6144 (KAT — the bar-binding model; MEMRA_PROFILE_GEN brackets ONLY the timed generate, so
# the capture excludes load/prefill/gate passes). (b) MEMRA_FA_SPLIT re-sweep at d6144 —
# the sp-ladder's >3072 rung (sp64) was swept 2026-07-08 on older kernels (stale-verdict law).
# (c) MEMRA_FA_V4_MAX=3072 probe (v4 off above the rung — the gemma depth lesson, 5090 recheck).
# All single runs to RANK; winners get x3 interleaved confirms + argmax gates before any code move.
# usage: run-mech.sh [nsys|split|v4]
set -u
W=/home/avifenesh/projects/wt-depth-decode
R=$W/research/depth-decode-20260802
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
NSYS=/usr/local/cuda-13.1/bin/nsys

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
    local busy; busy=$(busy_procs); [ "$busy" -eq 0 ] && break
    sleep 5; n=$((n+1)); [ $n -gt 240 ] && { echo "wait_idle timeout (busy=$busy)"; break; }
  done
}

nsys_point() { # depth
  local d=$1
  wait_idle
  MEMRA_PROFILE_GEN=1 MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$R/depth-$d-kat.txt" \
    flock /tmp/gpu5090.lock timeout 1800 \
    "$NSYS" profile -c cudaProfilerApi --trace=cuda -f true -o "$R/nsys-kat-d$d" \
    "$W/target/release/run-gen" "$KAT" > "$R/nsys-kat-d$d.log" 2>&1
  echo "nsys d$d rc=$?"
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv --output "$R/nsys-kat-d$d" \
    "$R/nsys-kat-d$d.nsys-rep" > /dev/null 2>&1
  echo "stats d$d rc=$?"
}

split_point() { # depth split-or-naked
  local d=$1 sp=$2 log="$R/sweep-fa-split-$2-d$1.log"
  wait_idle
  local -a env=(MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$R/depth-$d-kat.txt")
  [ "$sp" != naked ] && env+=(MEMRA_FA_SPLIT="$sp")
  env "${env[@]}" flock /tmp/gpu5090.lock timeout 1800 \
    "$W/target/release/run-gen" "$KAT" > "$log" 2>&1
  local tg match
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  match=$(grep -c "argmax.*MATCH" "$log")
  echo "fa_split=$sp d$d tg=${tg:-null} match_lines=$match temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" | tee -a "$R/mech-console.log"
  printf '{"cell":"mech-split","depth":%s,"arm":"%s","metric":"tg128_toks","value":%s,"ts":"%s"}\n' \
    "$d" "$sp" "${tg:-null}" "$(date -u +%FT%TZ)" >> "$R/mech.jsonl"
}

v4_point() { # depth v4max-or-naked
  local d=$1 vm=$2 log="$R/sweep-fa-v4max-$2-d$1.log"
  wait_idle
  local -a env=(MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$R/depth-$d-kat.txt")
  [ "$vm" != naked ] && env+=(MEMRA_FA_V4_MAX="$vm")
  env "${env[@]}" flock /tmp/gpu5090.lock timeout 1800 \
    "$W/target/release/run-gen" "$KAT" > "$log" 2>&1
  local tg match
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  match=$(grep -c "argmax.*MATCH" "$log")
  echo "fa_v4max=$vm d$d tg=${tg:-null} match_lines=$match temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" | tee -a "$R/mech-console.log"
  printf '{"cell":"mech-v4","depth":%s,"arm":"%s","metric":"tg128_toks","value":%s,"ts":"%s"}\n' \
    "$d" "$vm" "${tg:-null}" "$(date -u +%FT%TZ)" >> "$R/mech.jsonl"
}

case "${1:-nsys}" in
  nsys)  nsys_point 512; nsys_point 6144 ;;
  split) for sp in naked 8 16 32 96 128; do split_point 6144 "$sp"; done
         # rank order at d6144; naked (=64 via ladder) is the control
         ;;
  v4)    for vm in naked 3072; do v4_point 6144 "$vm"; done ;;
esac

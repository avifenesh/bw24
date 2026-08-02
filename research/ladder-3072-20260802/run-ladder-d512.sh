#!/bin/bash
# lane/ladder-3072 supplemental: d512 cells (sp8/sp32/sp64, kat+q35, N=3 interleaved) —
# places the LOW boundary of the new rung (sp8's original 2026-07-08 win was short-ctx;
# the main sweep floor is d1024). Same protocol as run-ladder-sweep.sh; prompts = the
# depth-decode lane's depth-512-{kat,q35}.txt.
set -u
W=/home/avifenesh/projects/wt-ladder-3072
R=$W/research/ladder-3072-20260802
P=$W/research/depth-decode-20260802
OUT=$R/ladder-sweep.jsonl
declare -A GGUF=(
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
)
N=3
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
wait_ready() {
  local n=0
  while true; do
    local busy ac; busy=$(busy_procs); ac=$(cat /sys/class/power_supply/ADP0/online)
    [ "$busy" -eq 0 ] && [ "$ac" = 1 ] && break
    sleep 5; n=$((n+1)); [ $n -gt 240 ] && { echo "wait_ready timeout"; break; }
  done
}
row() {
  printf '{"ts":"%s","git":"%s","cell":"ladder-3072-sweep","model":"%s","arm":"%s","depth":%s,"metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s,"quarantined":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$6" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" "${7:-false}" >> "$OUT"
  echo "  [$1/sp$2 d$3 rep$6] $4 = $5 q=${7:-false}"
}
memra_point() {
  local m=$1 sp=$2 d=$3 rep=$4 log="$R/mem-sp$2-$1-d$3-rep$4.log"
  wait_ready
  local ac0 ac1
  ac0=$(cat /sys/class/power_supply/ADP0/online)
  MEMRA_FA_SPLIT=$sp MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P/depth-$d-$m.txt" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "${GGUF[$m]}" > "$log" 2>&1
  ac1=$(cat /sys/class/power_supply/ADP0/online)
  local q=false; { [ "$ac0" != 1 ] || [ "$ac1" != 1 ]; } && q=true
  local tg match
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  match=$(grep -c "argmax.*MATCH" "$log")
  row "$m" "$sp" "$d" tg128_toks "${tg:-null}" "$rep" "$q"
  row "$m" "$sp" "$d" argmax_match_lines "${match:-0}" "$rep" "$q"
  grep -q "MISMATCH" "$log" && echo "  !! ARGMAX MISMATCH $m/sp$sp d$d rep$rep"
}
{
  echo "=== LADDER d512 SUPPLEMENT x$N $(date -u +%FT%TZ) git=$GIT_SHA profile=$PROFILE ==="
  for rep in $(seq 1 $N); do
    for m in kat q35; do
      if [ $((rep % 2)) -eq 1 ]; then order="8 32 64"; else order="64 32 8"; fi
      for sp in $order; do memra_point "$m" "$sp" 512 "$rep"; done
    done
  done
  echo "D512-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/sweep-console.log"

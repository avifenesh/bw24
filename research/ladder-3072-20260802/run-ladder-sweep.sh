#!/bin/bash
# lane/ladder-3072: e2e arbitration of the sp8->sp64 ladder rung under the deep kernel.
# At a fixed depth d, a candidate boundary B delivers exactly one split value to the decode
# window (d..d+128): sp8 if d < B else sp64 — so forcing MEMRA_FA_SPLIT per run measures the
# candidate ladders without code churn (the OnceLock seam pins one split per process; a
# 128-token window never mixes arms). sp32 rides along (kernel data shows it competitive in
# the low band). Candidate map (nkv<=4 branch, boundary B in {1024,2048,3072=current,4096}):
#   d1024: B<=1024 -> 64 else 8;  d2048: B<=2048 -> 64 else 8;
#   d3072: B<=3072(current) -> 64 else 8;  d4096: all candidates -> 64.
# Protocol = fa-deep depth-ab: run-gen 128 greedy tokens, gen-only rate, argmax gate inside,
# arms ADJACENT per (model,depth) in one thermal window, N=3 interleaved, llama fresh
# same-session. POWER GUARD: owner cut wall power this session — every rep records ADP0
# pre/post + any transition inside its window is quarantined (power-log.txt is the record).
set -u
W=/home/avifenesh/projects/wt-ladder-3072
R=$W/research/ladder-3072-20260802
P=$W/research/depth-decode-20260802
OUT=$R/ladder-sweep.jsonl
LLAMA=/home/avifenesh/projects/llama.cpp/build/bin/llama-bench
declare -A GGUF=(
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
)
MODELS="kat q35"
DEPTHS="1024 2048 3072 4096"
ARMS="8 32 64"
N=3
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)
gpu-full-power on >/dev/null 2>&1 || true

prompt_file() { # depth model
  local d=$1 m=$2
  if [ -f "$R/depth-$d-$m.txt" ]; then echo "$R/depth-$d-$m.txt"; else echo "$P/depth-$d-$m.txt"; fi
}
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
    sleep 5; n=$((n+1)); [ $n -gt 240 ] && { echo "wait_ready timeout (busy=$busy ac=$ac)"; break; }
  done
}
row() { # model arm depth metric value rep quarantined
  printf '{"ts":"%s","git":"%s","cell":"ladder-3072-sweep","model":"%s","arm":"%s","depth":%s,"metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s,"quarantined":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$6" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" "${7:-false}" >> "$OUT"
  echo "  [$1/sp$2 d$3 rep$6] $4 = $5 q=${7:-false}"
}

memra_point() { # model split depth rep
  local m=$1 sp=$2 d=$3 rep=$4 log="$R/mem-sp$2-$1-d$3-rep$4.log"
  wait_ready
  local ac0 ac1 t0 t1
  ac0=$(cat /sys/class/power_supply/ADP0/online); t0=$(date -u +%FT%TZ)
  MEMRA_FA_SPLIT=$sp MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$(prompt_file $d $m)" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "${GGUF[$m]}" > "$log" 2>&1
  local rc=$?
  ac1=$(cat /sys/class/power_supply/ADP0/online); t1=$(date -u +%FT%TZ)
  local q=false
  { [ "$ac0" != 1 ] || [ "$ac1" != 1 ] || awk -v a="$t0" -v b="$t1" '$1 >= a && $1 <= b && /ADP0 [01]->/' "$R/power-log.txt" 2>/dev/null | grep -q .; } && q=true
  local tg match mism
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  match=$(grep -c "argmax.*MATCH" "$log")
  mism=$(grep -c "MISMATCH" "$log")
  row "$m" "$sp" "$d" tg128_toks "${tg:-null}" "$rep" "$q"
  row "$m" "$sp" "$d" argmax_match_lines "${match:-0}" "$rep" "$q"
  [ "$mism" -gt 0 ] && echo "  !! ARGMAX MISMATCH $m/sp$sp d$d rep$rep (see $log)"
  [ $rc -ne 0 ] && echo "  WARN memra $m/sp$sp d$d rep$rep rc=$rc (see $log)"
  [ "$q" = true ] && echo "  !! QUARANTINED (power transition) $m/sp$sp d$d rep$rep"
}

llama_model() { # model rep
  local m=$1 rep=$2 log="$R/llama-$1-rep$2.log"
  wait_ready
  local ac0 ac1 t0 t1
  ac0=$(cat /sys/class/power_supply/ADP0/online); t0=$(date -u +%FT%TZ)
  flock /tmp/gpu5090.lock timeout 1800 "$LLAMA" -m "${GGUF[$m]}" -ngl 999 -fa 1 \
    -ctk q8_0 -ctv q5_1 -p 0 -n 128 -d 1024,2048,3072,4096 -r 1 -o json > "$log" 2>&1
  local rc=$?
  ac1=$(cat /sys/class/power_supply/ADP0/online); t1=$(date -u +%FT%TZ)
  local q=false
  { [ "$ac0" != 1 ] || [ "$ac1" != 1 ] || awk -v a="$t0" -v b="$t1" '$1 >= a && $1 <= b && /ADP0 [01]->/' "$R/power-log.txt" 2>/dev/null | grep -q .; } && q=true
  python3 - "$log" "$m" "$rep" "$q" <<'PYEOF' >> "$OUT"
import json, sys, re, subprocess, datetime
log, model, rep, q = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
txt = open(log).read()
m = re.search(r'\[\s*\{.*\}\s*\]', txt, re.S)
temp = subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
if m:
    for r in json.loads(m.group(0)):
        if r.get('n_gen', 0) > 0 and r.get('n_prompt', 0) == 0:
            print(json.dumps({"ts":ts,"cell":"ladder-3072-sweep","model":model,"arm":"llama",
                              "depth":r.get('n_depth',0),"metric":"tg128_toks",
                              "value":r['avg_ts'],"rep":int(rep),"temp_c":int(temp),
                              "quarantined": q == "true"}))
PYEOF
  [ $rc -ne 0 ] && echo "  WARN llama $m rep$rep rc=$rc (see $log)"
  [ "$q" = true ] && echo "  !! QUARANTINED llama $m rep$rep"
  tail -4 "$OUT" | sed 's/^/  [llama] /'
}

echo "=== LADDER-3072 SWEEP x$N $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/sweep-console.log"
{
  for rep in $(seq 1 "$N"); do
    for m in $MODELS; do
      for d in $DEPTHS; do
        # alternate arm order per rep (interleave discipline)
        if [ $((rep % 2)) -eq 1 ]; then order="8 32 64"; else order="64 32 8"; fi
        for sp in $order; do memra_point "$m" "$sp" "$d" "$rep"; done
      done
      llama_model "$m" "$rep"
    done
  done
  echo "LADDER-SWEEP-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/sweep-console.log"

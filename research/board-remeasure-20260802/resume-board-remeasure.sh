#!/bin/bash
# RESUME driver for run-board-remeasure.sh after the 2026-08-02 session restart killed the sweep
# mid-q35-rep1 (memra d6257 died mid-gen, llama pair never ran). Interleaving law: a cell whose
# A/B pair was split across sessions is invalid whole — q35 rep1 rows quarantined to
# board-remeasure-rep1-q35-split-invalid.jsonl, partial logs renamed *.split-partial.log.
# Valid completed state: q9 rep1 + q27 rep1 (complete same-session pairs; llama denominator in
# THIS session's regime, so they remain valid rep1 points).
# This driver: q35 rep1 (whole cell), then reps 2..5 for all three models.
# Measurement shape identical to run-board-remeasure.sh (same functions, same OUT).
set -u
W=/home/avifenesh/projects/wt-board-remeasure
R=$W/research/board-remeasure-20260802
OUT=$R/board-remeasure.jsonl
LLAMA=/home/avifenesh/projects/llama.cpp/build/bin/llama-bench
declare -A GGUF=(
  [q9]=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
  [q27]=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
)
declare -A PROMPT=(
  [512]=$W/research/e2e/prompts/pp512.txt
  [6257]=$W/research/e2e/prompts/p3-agentic-long.txt
)
MODELS="q9 q27 q35"
DEPTHS="512 6257"

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
gpu_state() { # concurrent compute apps, comma-joined
  nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null \
    | tr -d '"' | paste -sd';' - | sed 's/, /:/g'
}
row() { # model engine depth metric value rep
  printf '{"ts":"%s","git":"%s","cell":"board-remeasure","model":"%s","engine":"%s","depth":%s,"metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s,"gpu_concurrent":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$6" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" "$(gpu_state)" >> "$OUT"
  echo "  [$1/$2 d$3 rep$6] $4 = $5"
}

memra_point() { # model depth rep
  local m=$1 d=$2 rep=$3 log="$R/mem-$1-d$2-rep$3.log"
  wait_idle
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE="${PROMPT[$d]}" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "${GGUF[$m]}" > "$log" 2>&1
  local rc=$?
  local tg gen match
  gen=$(grep -oE "generated [0-9]+ tokens" "$log" | grep -oE "[0-9]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  match=$(grep -c "MATCH" "$log")
  row "$m" memra "$d" gen_ntok "${gen:-null}" "$rep"
  row "$m" memra "$d" tg128_toks "${tg:-null}" "$rep"
  row "$m" memra "$d" argmax_match_lines "${match:-0}" "$rep"
  [ $rc -ne 0 ] && echo "  WARN memra $m d$d rep$rep rc=$rc (see $log)"
  grep -m1 "resident-experts decision" "$log" | sed 's/^/    /' || true
}

llama_model() { # model rep
  local m=$1 rep=$2 log="$R/llama-$1-rep$2.log"
  wait_idle
  flock /tmp/gpu5090.lock timeout 1800 "$LLAMA" -m "${GGUF[$m]}" -ngl 999 -fa 1 \
    -ctk q8_0 -ctv q5_1 -p 0 -n 128 -d 512,6257 -r 1 -o json > "$log" 2>&1
  local rc=$?
  python3 - "$log" "$m" "$rep" <<'EOF' >> "$OUT"
import json, sys, re, subprocess, datetime
log, model, rep = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(log).read()
m = re.search(r'\[\s*\{.*\}\s*\]', txt, re.S)
temp = subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
apps = subprocess.run(['nvidia-smi','--query-compute-apps=process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip().replace('\n',';').replace(', ',':')
ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
if m:
    for r in json.loads(m.group(0)):
        if r.get('n_gen', 0) > 0 and r.get('n_prompt', 0) == 0:
            print(json.dumps({"ts":ts,"cell":"board-remeasure","model":model,"engine":"llama",
                              "depth":r.get('n_depth',0),"metric":"tg128_toks",
                              "value":r['avg_ts'],"rep":int(rep),"temp_c":int(temp),
                              "build":r.get('build_commit',''),"gpu_concurrent":apps}))
EOF
  [ $rc -ne 0 ] && echo "  WARN llama $m rep$rep rc=$rc (see $log)"
  tail -2 "$OUT" | sed 's/^/  [llama] /'
}

echo "=== BOARD REMEASURE RESUME $TS git=$GIT_SHA profile=$PROFILE (q35 rep1 whole, then reps 2-5 all) ===" | tee -a "$R/sweep-console.log"
{
  # rep1: only the split q35 cell, re-run whole
  for d in $DEPTHS; do memra_point q35 "$d" 1; done
  llama_model q35 1
  # reps 2..5: full round-robin
  for rep in 2 3 4 5; do
    for m in $MODELS; do
      for d in $DEPTHS; do memra_point "$m" "$d" "$rep"; done
      llama_model "$m" "$rep"
    done
  done
  echo "BOARD-REMEASURE-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/sweep-console.log"

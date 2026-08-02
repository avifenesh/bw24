#!/bin/bash
# fa-decode-deep: the depth table OLD-vs-NEW + a fresh vs-llama restatement.
#
# MEASUREMENT SHAPE (the depth-decode lane's protocol, same prompts, same document):
#  - memra arms: run-gen, MEMRA_PROMPT_FILE = the depth lane's per-model prefix of ONE code
#    document cut to exactly {512,2048,4096,6144} tokens by the model's own tokenizer.
#    Cell value = printed gen-only rate over MEMRA_NGEN=128 greedy tokens (prime excluded,
#    host argmax inside the span). ARM old = MEMRA_FA_DEEP=0 (v4 twins); ARM new = naked
#    (deep twins default-on). Arms run ADJACENT per (model, depth) — old then new in the
#    same thermal window — the pairing that makes a ~3% effect readable across reloads.
#  - llama arm: local fork llama-bench -p 0 -n 128 -d 512,2048,4096,6144 -r 1 -ngl 999
#    -fa 1 -ctk q8_0 -ctv q5_1 (the established denominator config), fresh THIS session —
#    cross-day denominators are clock-drift-invalid (the H100 lane law).
#  Every memra run carries the prefill/decode argmax gate. N reps, medians reported,
#  per-rep values in depth-ab.jsonl.
# usage: run-depth-ab.sh [nreps]
set -u
N=${1:-3}
W=/home/avifenesh/projects/wt-fa-decode-deep
R=$W/research/fa-decode-deep-20260802
P=$W/research/depth-decode-20260802
OUT=$R/depth-ab.jsonl
LLAMA=/home/avifenesh/projects/llama.cpp/build/bin/llama-bench
declare -A GGUF=(
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
  [q35]=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
  [o35b]=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
)
MODELS="kat q35 o35b"
DEPTHS="512 2048 4096 6144"

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
row() { # model arm depth metric value rep
  printf '{"ts":"%s","git":"%s","cell":"fa-deep-depth-ab","model":"%s","arm":"%s","depth":%s,"metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$6" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1/$2 d$3 rep$6] $4 = $5"
}

memra_point() { # model arm depth rep
  local m=$1 arm=$2 d=$3 rep=$4 log="$R/mem-$2-$1-d$3-rep$4.log"
  wait_idle
  local env_deep=()
  [ "$arm" = old ] && env_deep=(MEMRA_FA_DEEP=0)
  env "${env_deep[@]}" MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$P/depth-$d-$m.txt" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "${GGUF[$m]}" > "$log" 2>&1
  local rc=$?
  local tg match mism
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  match=$(grep -c "argmax.*MATCH" "$log")
  mism=$(grep -c "MISMATCH" "$log")
  row "$m" "$arm" "$d" tg128_toks "${tg:-null}" "$rep"
  row "$m" "$arm" "$d" argmax_match_lines "${match:-0}" "$rep"
  [ "$mism" -gt 0 ] && echo "  !! ARGMAX MISMATCH $m/$arm d$d rep$rep (see $log)"
  [ $rc -ne 0 ] && echo "  WARN memra $m/$arm d$d rep$rep rc=$rc (see $log)"
}

llama_model() { # model rep
  local m=$1 rep=$2 log="$R/llama-$1-rep$2.log"
  wait_idle
  flock /tmp/gpu5090.lock timeout 1800 "$LLAMA" -m "${GGUF[$m]}" -ngl 999 -fa 1 \
    -ctk q8_0 -ctv q5_1 -p 0 -n 128 -d 512,2048,4096,6144 -r 1 -o json > "$log" 2>&1
  local rc=$?
  python3 - "$log" "$m" "$rep" <<'EOF' >> "$OUT"
import json, sys, re, subprocess, datetime
log, model, rep = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(log).read()
m = re.search(r'\[\s*\{.*\}\s*\]', txt, re.S)
temp = subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
if m:
    for r in json.loads(m.group(0)):
        if r.get('n_gen', 0) > 0 and r.get('n_prompt', 0) == 0:
            print(json.dumps({"ts":ts,"cell":"fa-deep-depth-ab","model":model,"arm":"llama",
                              "depth":r.get('n_depth',0),"metric":"tg128_toks",
                              "value":r['avg_ts'],"rep":int(rep),"temp_c":int(temp)}))
EOF
  [ $rc -ne 0 ] && echo "  WARN llama $m rep$rep rc=$rc (see $log)"
  tail -4 "$OUT" | sed 's/^/  [llama] /'
}

echo "=== FA-DEEP DEPTH A/B x$N $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/ab-console.log"
{
  for rep in $(seq 1 "$N"); do
    for m in $MODELS; do
      for d in $DEPTHS; do
        memra_point "$m" old "$d" "$rep"
        memra_point "$m" new "$d" "$rep"
      done
      llama_model "$m" "$rep"
    done
  done
  echo "DEPTH-AB-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/ab-console.log"

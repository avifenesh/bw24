#!/bin/bash
# depth-decode: 3 models x 4 depth points x 2 engines, interleaved x3 process rounds.
#
# MEASUREMENT SHAPE (stated honestly):
#  - memra arm: naked run-gen (defaults; no flags), MEMRA_PROMPT_FILE = a per-model prefix of
#    ONE code document (p4-16k.txt) cut to exactly {512,2048,4096,6144} tokens by the model's
#    own tokenizer. Decode rate = the printed gen-only rate over MEMRA_NGEN=128 greedy tokens
#    (real continuation, host argmax inside the timed span; prime/prefill excluded). KV depth
#    during the window: D .. D+128. Every run carries the prefill/decode argmax gate.
#  - llama arm: local fork llama-bench, -p 0 -n 128 -d {512,2048,4096,6144} (one invocation,
#    4 tests, -r 1 per process round), -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 (the established
#    denominator KV config). tg128@dD = 128 decode steps timed after an UNTIMED depth-fill of
#    D tokens; llama-bench feeds synthetic tokens (no sampling/detokenize in the timed loop).
#  Both arms therefore time 128 batch-1 decode steps starting from the same KV depth on the
#  same gguf; they differ in token *content* (real greedy continuation vs synthetic) and in
#  host-side sampling cost (memra includes its argmax; llama-bench has no sampler).
# One rep = all arms round-robin (memra 4 depths then llama 4 depths, per model) in the same
# thermal window. Co-resident llama-server --embedding (332 MiB) allowlisted, inside figures.
# usage: run-depth-sweep.sh [nreps]
set -u
N=${1:-3}
W=/home/avifenesh/projects/wt-depth-decode
R=$W/research/depth-decode-20260802
OUT=$R/depth-sweep.jsonl
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
row() { # model engine depth metric value rep
  printf '{"ts":"%s","git":"%s","cell":"depth-sweep","model":"%s","engine":"%s","depth":%s,"metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$6" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1/$2 d$3 rep$6] $4 = $5"
}

memra_point() { # model depth rep
  local m=$1 d=$2 rep=$3 log="$R/mem-$1-d$2-rep$3.log"
  wait_idle
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$R/depth-$d-$m.txt" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "${GGUF[$m]}" > "$log" 2>&1
  local rc=$?
  local ntok tg gen match
  ntok=$(grep -m1 -oE "prompt tokens: \[[0-9, ]*\]" "$log" | tr -cd ',' | wc -c)
  [ -n "$ntok" ] && ntok=$((ntok+1))
  gen=$(grep -oE "generated [0-9]+ tokens" "$log" | grep -oE "[0-9]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  match=$(grep -c "argmax.*MATCH" "$log")
  row "$m" memra "$d" prompt_ntok "${ntok:-null}" "$rep"
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
            print(json.dumps({"ts":ts,"cell":"depth-sweep","model":model,"engine":"llama",
                              "depth":r.get('n_depth',0),"metric":"tg128_toks",
                              "value":r['avg_ts'],"rep":int(rep),"temp_c":int(temp)}))
EOF
  [ $rc -ne 0 ] && echo "  WARN llama $m rep$rep rc=$rc (see $log)"
  tail -4 "$OUT" | sed 's/^/  [llama] /'
}

echo "=== DEPTH SWEEP x$N $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/sweep-console.log"
{
  for rep in $(seq 1 "$N"); do
    for m in $MODELS; do
      for d in $DEPTHS; do memra_point "$m" "$d" "$rep"; done
      llama_model "$m" "$rep"
    done
  done
  echo "DEPTH-SWEEP-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/sweep-console.log"

#!/bin/bash
# kquant-tile-loaders: Ornith-35B vs llama same-session bar re-check post direct tile loaders.
# Same protocol as research/q4k-expert-prefill-20260802/run-barcheck.sh (the 9B-cell e2e rule).
# Interleaved per rep: memra board (pp2048 pp-only + gen512), llama-bench board (-p 512,2048
# -n 128), per-class memra spec-K=2 (run-spec NGEN=256), llama class rates (-p 27,1845,6257 -n 256).
# usage: run-barcheck-o35b.sh [nreps]
set -u
N=${1:-3}
W=/home/avifenesh/projects/bw24-kquant-tile-loaders
R=$W/research/kquant-tile-loaders-20260802
PDIR=$W/research/e2e/prompts
OUT=$R/barcheck-o35b.jsonl
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
DRAFT=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf
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
  printf '{"ts":"%s","git":"%s","cell":"o35b-barcheck-kq","arm":"%s","metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1 rep$4] $2 = $3"
}

memra_board() { # rep
  local rep=$1 log="$R/obar-memra-board-rep$1.log"
  wait_idle
  MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 MEMRA_PROMPT_FILE="$PDIR/board-2048.txt" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$O35B" > "$log" 2>&1
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  row memra pp2048_toks "${med:-null}" "$rep"
  local log2="$R/obar-memra-gen512-rep$1.log"
  wait_idle
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PDIR/pp512.txt" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$O35B" > "$log2" 2>&1
  local pp tg
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log2" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log2" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  row memra pp512_toks "${pp:-null}" "$rep"
  row memra tg128_toks "${tg:-null}" "$rep"
}

llama_board() { # rep
  local rep=$1 log="$R/obar-llama-board-rep$1.log"
  wait_idle
  flock /tmp/gpu5090.lock timeout 1800 "$LLAMA" -m "$O35B" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 \
    -p 512,2048 -n 128 -r 1 -o json > "$log" 2>&1
  python3 - "$log" "$rep" <<'EOF' >> "$OUT"
import json, sys, re, subprocess
log, rep = sys.argv[1], sys.argv[2]
txt = open(log).read()
m = re.search(r'\[\s*\{.*\}\s*\]', txt, re.S)
temp = subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
if m:
    for r in json.loads(m.group(0)):
        if r['n_prompt'] > 0 and r['n_gen'] == 0:
            met = f"pp{r['n_prompt']}_toks"
        elif r['n_gen'] > 0 and r['n_prompt'] == 0:
            met = f"tg{r['n_gen']}_toks"
        else:
            continue
        print(json.dumps({"cell":"o35b-barcheck-kq","arm":"llama","metric":met,"value":r['avg_ts'],"rep":int(rep),"temp_c":int(temp)}))
EOF
  tail -3 "$OUT" | sed 's/^/  [llama] /'
}

memra_class() { # class rep
  local cls=$1 rep=$2 log="$R/obar-memra-$1-rep$2.log"
  wait_idle
  MEMRA_MTP_DRAFT="$DRAFT" MEMRA_SPEC_K=2 MEMRA_NGEN=256 MEMRA_PROMPT="$(cat "$PDIR/$cls.txt")" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-spec" "$O35B" > "$log" 2>&1
  local plain spec prime ntok cons
  ntok=$(grep -oE "text prompt .* -> [0-9]+ tokens" "$log" | grep -oE "[0-9]+ tokens" | grep -oE "[0-9]+")
  plain=$(grep -oE "\[generate\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  spec=$(grep -oE "\[generate_spec K=2\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  prime=$(grep -oE "this run's prime [0-9.]+s" "$log" | grep -oE "[0-9.]+" | tail -1)
  cons=$(grep -c "self-consistency: PASS" "$log")
  row "memra-$cls" prompt_ntok "${ntok:-null}" "$rep"
  row "memra-$cls" plain_decode_toks "${plain:-null}" "$rep"
  row "memra-$cls" spec_k2_decode_toks "${spec:-null}" "$rep"
  row "memra-$cls" prime_s "${prime:-null}" "$rep"
  row "memra-$cls" spec_consistency_pass "${cons:-0}" "$rep"
}

llama_class() { # rep
  local rep=$1 log="$R/obar-llama-class-rep$1.log"
  wait_idle
  flock /tmp/gpu5090.lock timeout 1800 "$LLAMA" -m "$O35B" -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 \
    -p 27,1845,6257 -n 256 -r 1 -o json > "$log" 2>&1
  python3 - "$log" "$rep" <<'EOF' >> "$OUT"
import json, sys, re, subprocess
log, rep = sys.argv[1], sys.argv[2]
txt = open(log).read()
m = re.search(r'\[\s*\{.*\}\s*\]', txt, re.S)
temp = subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
if m:
    for r in json.loads(m.group(0)):
        if r['n_prompt'] > 0 and r['n_gen'] == 0:
            met = f"pp{r['n_prompt']}_toks"
        elif r['n_gen'] > 0 and r['n_prompt'] == 0:
            met = f"tg{r['n_gen']}_toks"
        else:
            continue
        print(json.dumps({"cell":"o35b-barcheck-kq","arm":"llama-class","metric":met,"value":r['avg_ts'],"rep":int(rep),"temp_c":int(temp)}))
EOF
  tail -4 "$OUT" | sed 's/^/  [llama-class] /'
}

echo "=== O35B BAR RE-CHECK (kq-direct) x$N $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/sweep-console.log"
{
  for rep in $(seq 1 "$N"); do
    memra_board "$rep"
    llama_board "$rep"
    for cls in p1-code-short p2-code-medium p3-agentic-long; do
      memra_class "$cls" "$rep"
    done
    llama_class "$rep"
  done
  echo "OBAR-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/sweep-console.log"

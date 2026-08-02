#!/bin/bash
# depth-decode item 4: drafter acceptance at CONTROLLED depth (same document, four prefix
# lengths — depth isolated from content class, unlike the p1/p2/p3 board prompts which vary
# both). run-spec, own-gen trimmed drafters at their serving K=2, NGEN=256, greedy.
# Acceptance under greedy is deterministic per (prompt, K) — rep2 is the determinism check,
# not a variance estimate; tok/s cells are per-rep wall rates.
# usage: run-depth-accept.sh [nreps]
set -u
N=${1:-2}
W=/home/avifenesh/projects/wt-depth-decode
R=$W/research/depth-decode-20260802
OUT=$R/depth-accept.jsonl
declare -A GGUF=(
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
  [o35b]=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
)
declare -A DRAFT=(
  [kat]=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/draft-katcoder-owntrim-nvfp4head-q4blk.gguf
  [o35b]=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf
)
MODELS="kat o35b"
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
row() { # model depth metric value rep
  printf '{"ts":"%s","git":"%s","cell":"depth-accept","model":"%s","depth":%s,"metric":"%s","value":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1 d$2 rep$5] $3 = $4"
}

point() { # model depth rep
  local m=$1 d=$2 rep=$3 log="$R/acc-$1-d$2-rep$3.log"
  wait_idle
  MEMRA_MTP_DRAFT="${DRAFT[$m]}" MEMRA_SPEC_K=2 MEMRA_NGEN=256 \
    MEMRA_PROMPT_FILE="$R/depth-$d-$m.txt" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-spec" "${GGUF[$m]}" > "$log" 2>&1
  local rc=$?
  local acc plain spec cons
  acc=$(grep -oE "acceptance: [0-9]+/[0-9]+ = [0-9.]+%" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  plain=$(grep -oE "\[generate\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  spec=$(grep -oE "\[generate_spec K=2\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  cons=$(grep -c "self-consistency: PASS" "$log")
  row "$m" "$d" acceptance_pct "${acc:-null}" "$rep"
  row "$m" "$d" plain_decode_toks "${plain:-null}" "$rep"
  row "$m" "$d" spec_k2_decode_toks "${spec:-null}" "$rep"
  row "$m" "$d" spec_consistency_pass "${cons:-0}" "$rep"
  [ $rc -ne 0 ] && echo "  WARN $m d$d rep$rep rc=$rc (see $log)"
}

echo "=== DEPTH ACCEPTANCE x$N $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/accept-console.log"
{
  for rep in $(seq 1 "$N"); do
    for m in $MODELS; do
      for d in $DEPTHS; do point "$m" "$d" "$rep"; done
    done
  done
  echo "DEPTH-ACCEPT-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/accept-console.log"

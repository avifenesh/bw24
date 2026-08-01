#!/bin/bash
# Resume of run-board-cells.sh after the harness stopped the first run mid-cell
# (o9b pair1 complete, pair2 llama-arm only — see board-cells.jsonl rows 1-6).
# Identical protocol, two changes: (1) wait_idle spin capped at 90 s — the co-resident
# drafters lane (frspec-owngen) keeps busy=1 continuously, so the idle gate never opens
# and flock /tmp/gpu5090.lock is the real serializer (both arms run in the same hot
# co-loaded regime; pairs stay adjacent); (2) takes cell list as baked-in resume plan.
# Usage: run-board-cells-resume.sh <phase>   phase in {o9b-finish, o35b, kat}
set -u
W=/home/avifenesh/projects/wt-ornith-serve-bench
R=$W/research/ornith-serve-20260801
LB=/home/avifenesh/projects/llama.cpp/build/bin/llama-bench
PF=$W/research/e2e/prompts/pp512.txt
OUT=$R/board-cells.jsonl

O9B=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)

busy_procs() {
  local n=0 pid
  while IFS=, read -r pid _; do
    pid=$(echo "$pid" | tr -d ' '); [ -n "$pid" ] || continue
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--embedding" || n=$((n+1))
  done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
  echo $n
}
wait_idle() { # capped 90s: co-lane never idles; flock serializes
  local n=0
  while true; do
    local busy clk
    busy=$(busy_procs)
    clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "$busy" -eq 0 ] && [ "${clk:-2000}" -lt 1200 ] && break
    sleep 5; n=$((n+1)); [ $n -ge 18 ] && break
  done
}

row() {
  printf '{"ts":"%s","git":"%s","cell":"%s","arm":"%s","metric":"%s","toks":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1/$2 rep$5] $3 = $4 tok/s"
}

llama_arm() {
  local cell=$1 model=$2 rep=$3
  local log="$R/board-$cell-llama-rep$rep.log"
  flock /tmp/gpu5090.lock "$LB" -m "$model" -ngl 999 -p 512 -n 128 -d 0 -r 3 \
    -fa 1 -ctk q8_0 -ctv q5_1 > "$log" 2>&1
  local pp tg
  pp=$(grep -E "pp512" "$log" | grep -oE '[0-9.]+ ±' | grep -oE '^[0-9.]+' | tail -1)
  tg=$(grep -E "tg128" "$log" | grep -oE '[0-9.]+ ±' | grep -oE '^[0-9.]+' | tail -1)
  row "$cell" llama pp512 "${pp:-0}" "$rep"
  row "$cell" llama tg128 "${tg:-0}" "$rep"
}

memra_arm() {
  local cell=$1 model=$2 rep=$3
  local log="$R/board-$cell-memra-rep$rep.log"
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE="$PF" flock /tmp/gpu5090.lock \
    timeout 900 "$W/target/release/run-gen" "$model" > "$log" 2>&1
  local pp tg
  pp=$(grep -oE "prefill [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  tg=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  row "$cell" memra prefill "${pp:-0}" "$rep"
  row "$cell" memra decode "${tg:-0}" "$rep"
}

case "${1:?phase}" in
  o9b-finish)
    echo "== o9b-q8_0 resume: memra rep2, then pair rep3 =="
    wait_idle; memra_arm o9b-q8_0 "$O9B" 2
    wait_idle; llama_arm o9b-q8_0 "$O9B" 3
    wait_idle; memra_arm o9b-q8_0 "$O9B" 3
    ;;
  o35b)
    echo "== o35b-q4km (interleaved x3) =="
    for rep in 1 2 3; do
      wait_idle; llama_arm o35b-q4km "$O35B" $rep
      wait_idle; memra_arm o35b-q4km "$O35B" $rep
    done
    ;;
  kat)
    echo "== kat-iq4xs (interleaved x3) =="
    for rep in 1 2 3; do
      wait_idle; llama_arm kat-iq4xs "$KAT" $rep
      wait_idle; memra_arm kat-iq4xs "$KAT" $rep
    done
    ;;
esac
echo "PHASE-DONE $1 $(date -u +%FT%TZ)"

# ctrl phase appended (attribution): the supported Qwen3.6-35B UD-IQ4_XS control re-anchored
# in THIS hot co-loaded session (cross-day comparisons are clock-drift-invalid, including
# the competitor denominator). Invoked as: run-board-cells-resume.sh ctrl
if [ "${1:-}" = ctrl ]; then
  CTRL=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
  echo "== ctrl-q35 (interleaved x3, attribution) =="
  for rep in 1 2 3; do
    wait_idle; llama_arm ctrl-q35 "$CTRL" $rep
    wait_idle; memra_arm ctrl-q35 "$CTRL" $rep
  done
  echo "PHASE-DONE ctrl $(date -u +%FT%TZ)"
fi

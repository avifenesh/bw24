#!/bin/bash
# ornith-serve-bench: board-protocol vs-llama cells for the 3 onboarded models.
# Reference recipe = the supported q35-plain cell in tools/full-board-bench.sh (same arch
# class): llama-bench at the qwen documented swept-best (-fa 1 -ctk q8_0 -ctv q5_1, -ngl 999,
# -p 512 -n 128 -r 3) vs memra run-gen naked (MEMRA_PROMPT_FILE=research/e2e/prompts/pp512.txt,
# MEMRA_NGEN=128). INTERLEAVED same-session: llama arm then memra arm, x N_PAIRS=3, wait_idle
# between arms (co-resident --embedding llama-server allowlisted). Every GPU run under
# flock /tmp/gpu5090.lock. Raw logs per arm; one JSONL row per reading.
set -u
W=/home/avifenesh/projects/wt-ornith-serve-bench
R=$W/research/ornith-serve-20260801
LB=/home/avifenesh/projects/llama.cpp/build/bin/llama-bench
PF=$W/research/e2e/prompts/pp512.txt
OUT=$R/board-cells.jsonl
N_PAIRS=3

O9B=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$W" rev-parse --short HEAD)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)
gpu-full-power on >/dev/null 2>&1 || true

busy_procs() { # GPU compute apps minus the allowlisted --embedding co-resident
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
    local busy clk
    busy=$(busy_procs)
    clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "$busy" -eq 0 ] && [ "${clk:-2000}" -lt 1200 ] && break
    sleep 5; n=$((n+1)); [ $n -gt 120 ] && { echo "wait_idle timeout (busy=$busy clk=$clk)"; break; }
  done
}

row() { # cell arm metric value rep
  printf '{"ts":"%s","git":"%s","cell":"%s","arm":"%s","metric":"%s","toks":%s,"rep":%s,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$1" "$2" "$3" "$4" "$5" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$1/$2 rep$5] $3 = $4 tok/s"
}

llama_arm() { # cell model rep
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

memra_arm() { # cell model rep
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

cell() { # cell model
  local cellname=$1 model=$2
  echo "== $cellname (interleaved x$N_PAIRS) =="
  for rep in $(seq 1 $N_PAIRS); do
    wait_idle; llama_arm "$cellname" "$model" "$rep"
    wait_idle; memra_arm "$cellname" "$model" "$rep"
  done
}

echo "=== ORNITH BOARD CELLS $TS git=$GIT_SHA profile=$PROFILE ==="
cell o9b-q8_0    "$O9B"
cell o35b-q4km   "$O35B"
cell kat-iq4xs   "$KAT"
echo "BOARD-CELLS-DONE $(date -u +%FT%TZ)"

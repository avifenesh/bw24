#!/usr/bin/env bash
# pro6000wk-validation: power/perf profile (THE Max-Q question)
#
# HOST LIMITATION (receipt: /root/receipts/power-control-attempts.log): nvidia-smi -pl / -lgc /
# -mig all return Insufficient Permissions in this RunPod container — the 300/450/600 W capped
# cells CANNOT be produced here. Each round still attempts the caps (logged), and the battery
# instead measures the WORKLOAD POWER ENVELOPE at the 600 W default: sustained decode draw vs
# sustained prefill draw at 1 Hz. That bounds the Max-Q question from the measurement side:
# any workload whose sustained draw sits under 300 W is cap-insensitive by definition.
set -uo pipefail
cd /root/bw24
export PATH=/usr/local/cuda-13.1/bin:$HOME/.cargo/bin:$PATH
R=/root/receipts
mkdir -p "$R/power"
M9=/root/models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
M27=/root/models/qwen36-27b-q8/Qwen3.6-27B-Q8_0.gguf
PF=research/e2e/prompts/pp512.txt

nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used,power.limit,fan.speed \
  --format=csv -l 1 > "$R/power/gpu-1hz.csv" 2>&1 &
SMIPID=$!
trap 'kill $SMIPID 2>/dev/null' EXIT

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/power/driver.log"; }
gstate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,power.limit --format=csv,noheader; }

try_cap() {  # watts
  local w=$1
  out=$(nvidia-smi -pl "$w" 2>&1 | head -1)
  cur=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits)
  log "cap-attempt ${w}W: '$out' -> effective ${cur}W"
}

decode_cell() {  # tag model ngen  (long window: >=8s of steady decode for 1Hz sampling)
  local tag=$1 model=$2 ngen=$3
  local lg="$R/power/$tag.log"
  log "$tag pre: $(gstate)"
  MEMRA_NGEN=$ngen MEMRA_PROMPT_FILE=$PF timeout 900 target/release/run-gen "$model" > "$lg" 2>&1
  local tps; tps=$(grep -oE "generated [0-9]+ tokens in [0-9.]+s = [0-9.]+ tok/s" "$lg" | tail -1)
  log "$tag post: $(gstate) | $tps"
}

prefill_cell() {  # tag model reps  (long window: reps sized for >=10s of back-to-back prefill)
  local tag=$1 model=$2 reps=$3
  local lg="$R/power/$tag.log"
  log "$tag pre: $(gstate)"
  MEMRA_PP_ONLY=1 MEMRA_PP_REPS=$reps MEMRA_PROMPT_FILE=$PF timeout 900 target/release/run-gen "$model" > "$lg" 2>&1
  local tps; tps=$(grep -oE "[0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s \(pp512[^)]*\)" "$lg" | tail -1)
  log "$tag post: $(gstate) | pp: $tps"
}

# Rounds interleave the nominal power states (cap attempts logged; on this host all rounds
# run at 600 W — interleave still spreads thermal drift).
for round in 1 2 3; do
  for W in 600 450 300; do
    try_cap $W
    decode_cell  "q9-decode-${W}W-r${round}"  "$M9" 1024
    prefill_cell "q9-pp512-${W}W-r${round}"   "$M9" 100
    decode_cell  "q27-decode-${W}W-r${round}" "$M27" 512
    prefill_cell "q27-pp512-${W}W-r${round}"  "$M27" 40
  done
done
try_cap 600
log "POWER BATTERY DONE"

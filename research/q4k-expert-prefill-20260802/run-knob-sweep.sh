#!/bin/bash
# q4k-expert-prefill: f16g2 sk-form knob sweep on Ornith-35B Q4_K_M (RESIDENT), board-2048
# pp-only. Sweep-grade (sequential, one process per arm, median of 5 in-process reps + 1
# warmup) — the interleaved claim run happens separately vs the naked baseline.
# Arms: sk0 (grid-scan rollback), sk32, sk128 (forced forms), cross{32,64,128,256} (hybrid).
# usage: run-knob-sweep.sh
set -u
W=/home/avifenesh/projects/wt-q4k-expert-prefill
R=$W/research/q4k-expert-prefill-20260802
PF2048=$W/research/e2e/prompts/board-2048.txt
OUT=$R/knob-sweep.jsonl
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf

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

run_arm() { # name env...
  local name=$1; shift
  local log="$R/knob-$name.log"
  wait_idle
  env "$@" MEMRA_MOE_F16G=2 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=5 MEMRA_PP_WARMUP=1 \
    MEMRA_PROMPT_FILE="$PF2048" \
    flock /tmp/gpu5090.lock timeout 1800 "$W/target/release/run-gen" "$O35B" > "$log" 2>&1
  local rc=$?
  local med
  med=$(grep -oE "pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s" "$log" | grep -oE "= [0-9.]+" | grep -oE "[0-9.]+")
  printf '{"ts":"%s","git":"%s","cell":"o35b-f16g2-knobs","arm":"%s","metric":"pp2048_toks_med5","value":%s,"n_inproc":5,"profile":"%s","temp_c":%s}\n' \
    "$TS" "$GIT_SHA" "$name" "${med:-null}" "$PROFILE" \
    "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" >> "$OUT"
  echo "  [$name] pp2048 med5 = ${med:-ERROR(rc=$rc)}"
}

echo "=== O35B F16G2 KNOB SWEEP $TS git=$GIT_SHA profile=$PROFILE ===" | tee -a "$R/sweep-console.log"
{
  run_arm sk0      MEMRA_F16G_SK=0
  run_arm sk32     MEMRA_F16G_SK=32
  run_arm sk128    MEMRA_F16G_SK=128
  run_arm cross32  MEMRA_F16G_SK_CROSS=32
  run_arm cross64  MEMRA_F16G_SK_CROSS=64
  run_arm cross128 MEMRA_F16G_SK_CROSS=128
  run_arm cross256 MEMRA_F16G_SK_CROSS=256
  echo "KNOB-SWEEP-DONE $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$R/sweep-console.log"

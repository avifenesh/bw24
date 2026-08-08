#!/usr/bin/env bash
# lane/pp-leverb INCREMENT 3 battery — the prime stage split (walker commit 564fb04d).
# Gates first (the ppsplit gate must flip GREEN and its canary must have teeth), perf second:
#   arm S: naked            — split prime (the default with the door open)
#   arm U: MEMRA_PRIME_PP=0 — unsplit reference (same binary, same cache placement)
# N=5 rep-major interleaved, one flock hold. Lever-A baseline context: ~141 tok/s
# (raw/leverA-gates2 G6; measured there with Cache::new — this battery's U arm is the same
# walk over pp::new_cache stage-owned KV, so U is the apples-to-apples in-hold reference).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/leverb-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
P4096=$HOME/step37/prompt-pp4096.txt
RAW=$HOME/leverb-raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/inc3-battery-$TS.log
PP=(MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1)
thermal() { nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader; }
CIARGS=(--label step35-swa --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
        --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24)
{
echo "=== leverb inc3 battery $TS commit=564fb04d+ (rsync)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal

  echo; echo "########## G4: prime-split-gate — must be GREEN now ##########"
  MEMRA_STEP37_GGUF=$M timeout 5400 tools/prime-split-gate.sh
  echo "G4 exit=$?"

  echo; echo "########## G4c: ppsplitc canary — forced-unsplit must flip it RED (teeth) ##########"
  MEMRA_STEP37_GGUF=$M timeout 5400 tools/prime-split-gate.sh --canary
  echo "G4c exit=$?"

  echo; echo "########## G2: chunkinv35 naked (split prime live — invariance must hold) ##########"
  MEMRA_STEP37_GGUF=$M env "${PP[@]}" timeout 5400 tools/chunk-invariance-gate.sh "${CIARGS[@]}"
  echo "G2 exit=$?"

  echo; echo "########## G2c: chunkinv35 canary (gate must still have teeth over the split) ##########"
  MEMRA_STEP37_GGUF=$M env "${PP[@]}" timeout 5400 tools/chunk-invariance-gate.sh "${CIARGS[@]}" --canary
  echo "G2c exit=$?"

  echo; echo "########## G1: kernel-check model-backed FULL ##########"
  timeout 3600 ./target/release/kernel-check "$M" 2>&1 | tail -40
  echo "G1 exit=$?"

  echo; echo "########## G3: run-gen argmax over PP-2 (split prime on the real gen path) ##########"
  env "${PP[@]}" MEMRA_NGEN=64 timeout 2400 \
    ./target/release/run-gen "$M" --prompt "Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
  echo "G3 exit=$?"

  echo; echo "########## G5: run-spec K=1..8 (split prime seeds spec; acceptance pinned 82.4% K=1) ##########"
  env "${PP[@]}" MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 \
    MEMRA_PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard." \
    timeout 5400 ./target/release/run-spec "$M"
  echo "G5 exit=$?"

  echo; echo "########## G6: ppprime pp4096, split vs unsplit, N=5 rep-major interleaved ##########"
  for rep in 1 2 3 4 5; do
    echo "--- rep $rep arm=S split (naked) ---"; thermal
    env "${PP[@]}" timeout 1800 \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup $([ $rep -eq 1 ] && echo 1 || echo 0)
    echo "--- rep $rep arm=U unsplit (MEMRA_PRIME_PP=0) ---"; thermal
    env "${PP[@]}" MEMRA_PRIME_PP=0 timeout 1800 \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup 0
  done
  echo "G6 done"; thermal

  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== battery rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

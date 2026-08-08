#!/usr/bin/env bash
# lane/pp-leverb INCREMENT 2 battery — the slab-local MoE arm (commit ec6bfad0) on the Step SKU
# over PP-2. THREE perf arms because cx-503b changed the baseline's meaning:
#   arm A: naked            — per-device RESIDENT slabs + the NEW slab-local arm (the default)
#   arm B: MEMRA_MOE_SLAB=0 — slabs uploaded but DEAD = the train-tip state WITHOUT this lane
#                             (cx-503b flips step35 to per-device RESIDENT, but nothing on a
#                             sigmoid-router arch reads dev_exps; the SLRU also sizes itself on
#                             free-VRAM-after-residents, so this arm may be a regression vs C)
#   arm C: MEMRA_MOE_RESIDENT=0 — no slabs, big SLRU = the Lever-A ~141 tok/s baseline state
# Gates first, perf second, one flock hold, rep-major interleaved (A,B,C per rep, N=5).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/leverb-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
P4096=$HOME/step37/prompt-pp4096.txt
RAW=$HOME/leverb-raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/inc2-battery-$TS.log
PP=(MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1)
thermal() { nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader; }
CIARGS=(--label step35-swa --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
        --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24)
{
echo "=== leverb inc2 battery $TS commit=ec6bfad0 (rsync)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal

  echo; echo "########## G0+G3: run-gen argmax over PP-2, naked (slab arm live) + logits dump ##########"
  # stderr carries the [moe] resident-experts decision lines = the G0 receipt (expect PP dev0 +
  # PP dev1, both RESIDENT post-cx-503b — the premise of arms A/B).
  env "${PP[@]}" MEMRA_NGEN=64 MEMRA_PP_LOGITS=/tmp/leverb-logits-slab.bin timeout 2400 \
    ./target/release/run-gen "$M" --prompt "Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
  echo "G3 exit=$?"

  echo; echo "########## G3b: run-gen MEMRA_MOE_SLAB=0 + logits dump, then the slab-vs-SLRU bit cmp ##########"
  env "${PP[@]}" MEMRA_MOE_SLAB=0 MEMRA_NGEN=64 MEMRA_PP_LOGITS=/tmp/leverb-logits-slru.bin timeout 2400 \
    ./target/release/run-gen "$M" --prompt "Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
  echo "G3b exit=$?"
  if cmp /tmp/leverb-logits-slab.bin /tmp/leverb-logits-slru.bin; then
    echo "G3cmp: slab-vs-SLRU prefill logits BIT-IDENTICAL"
  else
    echo "G3cmp: *** slab-vs-SLRU LOGITS DIFFER — the provenance-only claim is FALSE, stop"
  fi

  echo; echo "########## G1: kernel-check model-backed FULL ##########"
  timeout 3600 ./target/release/kernel-check "$M" 2>&1 | tail -60
  echo "G1 exit=$?"

  echo; echo "########## G2: chunkinv35 naked (slab arm live — invariance must hold) ##########"
  MEMRA_STEP37_GGUF=$M env "${PP[@]}" timeout 5400 tools/chunk-invariance-gate.sh "${CIARGS[@]}"
  echo "G2 exit=$?"

  echo; echo "########## G4: ppsplit gate — REGISTERED-RED receipt (walker absent => exit 1) ##########"
  MEMRA_STEP37_GGUF=$M timeout 3600 tools/prime-split-gate.sh || echo "G4 exit=$? (RED as registered)"

  echo; echo "########## G5: run-spec K=1..8 (acceptance must stay pinned: 14/17 = 82.4% K=1) ##########"
  env "${PP[@]}" MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 \
    MEMRA_PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard." \
    timeout 5400 ./target/release/run-spec "$M"
  echo "G5 exit=$?"

  echo; echo "########## G6: ppprime pp4096, 3 arms x N=5 rep-major interleaved ##########"
  for rep in 1 2 3 4 5; do
    echo "--- rep $rep arm=A naked (slabs + slab arm) ---"; thermal
    env "${PP[@]}" timeout 1800 \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup $([ $rep -eq 1 ] && echo 1 || echo 0)
    echo "--- rep $rep arm=B MEMRA_MOE_SLAB=0 (dead slabs = bare train tip) ---"; thermal
    env "${PP[@]}" MEMRA_MOE_SLAB=0 timeout 1800 \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup 0
    echo "--- rep $rep arm=C MEMRA_MOE_RESIDENT=0 (no slabs, big SLRU = Lever-A baseline state) ---"; thermal
    env "${PP[@]}" MEMRA_MOE_RESIDENT=0 timeout 1800 \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup 0
  done
  echo "G6 done"; thermal

  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== battery rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

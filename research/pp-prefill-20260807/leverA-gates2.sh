#!/usr/bin/env bash
# lane/pp-prefill LEVER A battery 2 (post tile-alignment fix 5c523d5e):
#   G2  chunkinv35 naked — must now be INVARIANT (battery 1: CHUNK-DEPENDENT, the FA tile grid)
#   G2c canary (MEMRA_STEP35_SWA_TKV=1) — must still BREAK it
#   G2f floor-arm chunkinv (MEMRA_STEP35_SWA_FA=0) — the alignment must NOT have moved the
#       floor's bits (the fully-masked-key no-op claim, measured not argued)
#   G3  run-gen argmax (FA-arm logits moved under alignment; re-receipt)
#   G5  run-spec K=1..8 (re-receipt; prompt is 19 tok < win so class unchanged — cheap insurance)
#   G6  ppprime pp4096 A/B FA-vs-floor interleaved N=5 (re-receipt on the aligned arm)
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/ppserve-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
P4096=$HOME/step37/prompt-pp4096.txt
RAW=$HOME/ppserve-raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/leverA-gates2-$TS.log
thermal() { nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader; }
CIARGS=(--label step35-swa --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
        --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24)
{
echo "=== leverA battery 2 $TS commit=5c523d5e (rsync)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal

  echo; echo "########## G2: chunkinv35 naked (must be INVARIANT now) ##########"
  MEMRA_STEP37_GGUF=$M MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    tools/chunk-invariance-gate.sh "${CIARGS[@]}"
  echo "G2 exit=$?"

  echo; echo "########## G2c: canary (MEMRA_STEP35_SWA_TKV=1 must BREAK it) ##########"
  MEMRA_STEP37_GGUF=$M MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    tools/chunk-invariance-gate.sh "${CIARGS[@]}" --canary
  echo "G2c exit=$?"

  echo; echo "########## G2f: floor arm chunkinv (alignment must not move the floor) ##########"
  MEMRA_STEP35_SWA_FA=0 MEMRA_STEP37_GGUF=$M MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    tools/chunk-invariance-gate.sh "${CIARGS[@]}"
  echo "G2f exit=$?"

  echo; echo "########## G3: run-gen argmax gate, PP-2 ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_NGEN=64 timeout 2400 \
    ./target/release/run-gen "$M" --prompt "Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
  echo "G3 exit=$?"

  echo; echo "########## G5: run-spec K=1..8 ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 \
    MEMRA_PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard." \
    timeout 5400 ./target/release/run-spec "$M"
  echo "G5 exit=$?"

  echo; echo "########## G6: ppprime pp4096 A/B interleaved N=5 (aligned FA arm) ##########"
  for rep in 1 2 3 4 5; do
    echo "--- rep $rep arm=FA (default, aligned) ---"; thermal
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 1800 \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup $([ $rep -eq 1 ] && echo 1 || echo 0)
    echo "--- rep $rep arm=FLOOR (MEMRA_STEP35_SWA_FA=0) ---"; thermal
    MEMRA_STEP35_SWA_FA=0 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 1800 \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup 0
  done
  echo "G6 done"; thermal

  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== battery2 rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

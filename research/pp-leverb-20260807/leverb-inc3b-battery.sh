#!/usr/bin/env bash
# lane/pp-leverb inc3b — the remaining binding laws + shape sweep:
#   T1/T1c tickinv35 + canary: serve's per-tick prime loop now takes the SPLIT path — the
#          request-level seq_end must survive tick boundaries over the split too.
#   G7     pp512 + pp2048 split-vs-unsplit N=5 interleaved: pp512 is the STOP-bar shape
#          (single 512-tok chunk = 8MB boundary + per-stage overhead; a small-prompt
#          regression = report-not-ship).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/leverb-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
P4096=$HOME/step37/prompt-pp4096.txt
RAW=$HOME/leverb-raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/inc3b-battery-$TS.log
PP=(MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1)
thermal() { nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader; }
TIARGS=(--label step35-tick --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
        --budgets 0,1024,513,512,256,64 --splits 64,256,512 --seam MEMRA_PRIME_CALLLOCAL --steps 24)
# pp512/pp2048 prompts: head of the pp4096 prompt by words (the tokenizer will land near
# the word count; the arms compare the SAME prompt against itself so exact T is irrelevant).
head -c 2800 "$P4096" > /tmp/prompt-pp512.txt
head -c 11200 "$P4096" > /tmp/prompt-pp2048.txt
{
echo "=== leverb inc3b battery $TS"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal

  echo; echo "########## T1: tickinv35 naked over the split ##########"
  MEMRA_STEP37_GGUF=$M env "${PP[@]}" timeout 7200 tools/tick-invariance-gate.sh "${TIARGS[@]}"
  echo "T1 exit=$?"

  echo; echo "########## T1c: tickinv35 canary (teeth over the split) ##########"
  MEMRA_STEP37_GGUF=$M env "${PP[@]}" timeout 7200 tools/tick-invariance-gate.sh "${TIARGS[@]}" --canary
  echo "T1c exit=$?"

  for shape in pp512 pp2048; do
    P=/tmp/prompt-$shape.txt
    echo; echo "########## G7-$shape: split vs unsplit, N=5 rep-major interleaved ##########"
    for rep in 1 2 3 4 5; do
      echo "--- $shape rep $rep arm=S split ---"; thermal
      env "${PP[@]}" timeout 900 \
        ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P" --reps 1 --warmup $([ $rep -eq 1 ] && echo 1 || echo 0)
      echo "--- $shape rep $rep arm=U unsplit ---"; thermal
      env "${PP[@]}" MEMRA_PRIME_PP=0 timeout 900 \
        ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P" --reps 1 --warmup 0
    done
  done
  echo "G7 done"; thermal

  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== battery rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

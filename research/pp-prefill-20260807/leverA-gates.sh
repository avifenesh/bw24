#!/usr/bin/env bash
# lane/pp-prefill LEVER A gate battery: windowed hd128 FA prefill replaces the f32 SWA floor.
# Workspace: ~/ppserve-memra (rsynced from wt-ppserve @ 8b425742). One flock window.
#   G1 kernel-check model-backed FULL (incl. the new 13 windowed-FA assertions)
#   G2 chunkinv35 naked (INVARIANT must hold with the FA arm as default) + canary teeth
#   G3 run-gen argmax gate over PP-2 (prefill-vs-decode agreement — the fix changes prefill class)
#   G4 ppn-gate stages=2 (PP-2 bit-identity vs door-OFF)
#   G5 run-spec K=1..8 self-consistency (drafter attached; prime feeds prompt_h -> spec)
#   G6 perf: ppprime pp4096 A/B interleaved N=5 per arm, one lock hold
#      (arm FA = naked default; arm FLOOR = MEMRA_STEP35_SWA_FA=0 rollback seam)
#   G7 TTFT serve receipt: 228-tok streaming p50 (the capacity T1 cell, FA default)
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/ppserve-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
P4096=$HOME/step37/prompt-pp4096.txt
RAW=$HOME/ppserve-raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/leverA-gates-$TS.log
PORT=8093; BASE=http://127.0.0.1:$PORT

thermal() { nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,memory.used --format=csv,noheader; }

{
echo "=== leverA gates $TS commit=$(git -C ~/ppserve-memra log --oneline -1 2>/dev/null || echo 8b425742-rsync)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"; thermal

  echo; echo "########## G1: kernel-check model-backed FULL ##########"
  timeout 3600 ./target/release/kernel-check "$M" 2>&1 | tail -80
  echo "G1 exit=$?"

  echo; echo "########## G2: chunkinv35 naked (INVARIANT) ##########"
  MEMRA_STEP37_GGUF=$M MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    tools/chunk-invariance-gate.sh --label step35-swa \
    --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
    --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24
  echo "G2 exit=$?"
  echo; echo "########## G2c: chunkinv35 canary (MEMRA_STEP35_SWA_TKV=1 must BREAK it) ##########"
  MEMRA_STEP37_GGUF=$M MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    tools/chunk-invariance-gate.sh --label step35-swa \
    --prompts research/chunk-invariance-20260805/prompt-pp6257.txt \
    --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24 --canary
  echo "G2c exit=$?"

  echo; echo "########## G3: run-gen argmax gate, PP-2, 64 tokens ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_NGEN=64 timeout 2400 \
    ./target/release/run-gen "$M" --prompt "Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
  echo "G3 exit=$?"

  echo; echo "########## G4: ppn-gate stages=2 ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 2400 ./target/release/ppn-gate "$M" 2 8 16
  echo "G4 exit=$?"

  echo; echo "########## G5: run-spec K=1..8 (drafter attached; the step-sku baseline prompt) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 \
    MEMRA_PROMPT="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard." \
    timeout 5400 ./target/release/run-spec "$M"
  echo "G5 exit=$?"

  echo; echo "########## G6: perf ppprime pp4096, A/B interleaved N=5, one hold ##########"
  for rep in 1 2 3 4 5; do
    echo "--- rep $rep arm=FA (default) ---"; thermal
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 1800 \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup $([ $rep -eq 1 ] && echo 1 || echo 0)
    echo "--- rep $rep arm=FLOOR (MEMRA_STEP35_SWA_FA=0) ---"; thermal
    MEMRA_STEP35_SWA_FA=0 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 1800 \
      ./target/release/concat-prime-probe "$M" ppprime --prompt-a "@$P4096" --reps 1 --warmup 0
  done
  echo "G6 done"; thermal

  echo; echo "########## G7: TTFT serve receipt (streaming, c=1 greedy, N=8) ##########"
  env MEMRA_MODELS="step35=${M}+${D}" MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0 \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR=127.0.0.1:$PORT \
      ./target/release/memra-server > "$RAW/leverA-server-$TS.log" 2>&1 &
  SRV=$!
  READY=0
  for i in $(seq 1 120); do
    sleep 5
    curl -sf "$BASE/readyz" >/dev/null 2>&1 && { echo "server ready after ~$((i*5))s"; READY=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -20 "$RAW/leverA-server-$TS.log"; break; }
  done
  if [ "$READY" = 1 ]; then
    python3 tools/load-serve.py --base "$BASE" --model step35 --concurrency 1 --requests 8 \
      --max-tokens 32 --greedy --stream --warmup 1 --label leverA-ttft \
      --per-request "$RAW/leverA-ttft-$TS.jsonl" --timeout 600
    echo "G7 exit=$?"
  fi
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 2

  thermal
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== battery rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

#!/usr/bin/env bash
# lane/tick-seg: exactness battery on the PP-2 pair — same three receipts as the chunkfix lane
# (this fix threads one usize through the same functions; these gates prove the single-call
# paths are byte-identical and the PP-2 stage boundary is undisturbed).
#   E1 kernel-check FULL, model-backed on the step artifact (single-device by construction).
#   E2 run-gen argmax gate over PP-2 (single prime_cache call, queued_after=0 — the
#      byte-identity-by-construction claim, proven on silicon).
#   E3 ppn-gate stages=2 — PP-2 split logits BIT-IDENTICAL to the door-OFF reference.
#   E4 run-spec K=1..8 self-consistency (primes through prime_cache then feeds the MTP draft
#      head; acceptance recorded alongside the PASS line per the f8f4-flip lesson).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/exact-tickseg-$TS.log
P="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
{
echo "=== lane/tick-seg exactness $TS commit=$(cat BOX-COMMIT.txt)"
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv,noheader
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"

  echo; echo "########## E1: kernel-check model-backed (step35 IQ4_XS), FULL ##########"
  timeout 3600 ./target/release/kernel-check "$M"
  echo "kernel-check exit=$?"

  echo; echo "########## E2: run-gen argmax gate, PP-2, 64 tokens ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_NGEN=64 timeout 2400 \
    ./target/release/run-gen "$M" --prompt "$P"
  echo "run-gen exit=$?"

  echo; echo "########## E3: ppn-gate stages=2 (bit-identity vs door-OFF) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 2400 ./target/release/ppn-gate "$M" 2 8 16
  echo "ppn-gate exit=$?"

  echo; echo "########## E4: run-spec K=1..8, n=32, short prompt (baseline-matched) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 \
    MEMRA_PROMPT="$P" timeout 3600 ./target/release/run-spec "$M"
  echo "run-spec exit=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== exact-tickseg rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

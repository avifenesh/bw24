#!/usr/bin/env bash
# lane/step35-chunkfix: run-spec K=1..8 self-consistency, the third CONTRIBUTING gate.
#
# Why this is not redundant with §6's run-gen argmax gate: generate_spec primes through
# prime_cache — THE path this fix changes — and then feeds that KV to the MTP draft head. So a
# prefill numeric change propagates into the drafter's read set and can move ACCEPTANCE while
# self-consistency stays green (the mechanism receipted in research/f8f4-flip-20260806). Both
# numbers are therefore recorded, not just the PASS line.
#
#   S1 short prompt (T=19), K=1..8 — matched to the pre-fix baseline in
#      raw/mtp-draft-20260806T215132Z.log so acceptance is comparable arm-to-arm.
#   S2 LONG prompt (T=4883, prompt-pp6257) at K=3 — the cell PAST the 512 window, where the fix
#      actually changes kernel selection. No pre-fix baseline exists for it, so it also gets a
#      BEFORE arm via the MEMRA_STEP35_SWA_TKV rollback seam, back-to-back in the same lock hold.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
LONG=research/chunk-invariance-20260805/prompt-pp6257.txt
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/spec35-$TS.log
P="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
{
echo "=== lane/step35-chunkfix run-spec K=1..8 $TS commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"

  echo; echo "########## S1: run-spec K=1..8, n=32, short text prompt (baseline-matched) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 \
    MEMRA_PROMPT="$P" timeout 3600 ./target/release/run-spec "$M"
  echo "run-spec short exit=$?"

  echo; echo "########## S2a: run-spec K=3, n=32, LONG prompt T=4883 — AFTER (default) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 MEMRA_SPEC_K=3 \
    MEMRA_PROMPT_FILE="$LONG" timeout 3600 ./target/release/run-spec "$M"
  echo "run-spec long AFTER exit=$?"

  echo; echo "########## S2b: same cell — BEFORE (MEMRA_STEP35_SWA_TKV=1, pre-fix predicate) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 MEMRA_SPEC_K=3 \
    MEMRA_STEP35_SWA_TKV=1 MEMRA_PROMPT_FILE="$LONG" timeout 3600 ./target/release/run-spec "$M"
  echo "run-spec long BEFORE exit=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== spec35 rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

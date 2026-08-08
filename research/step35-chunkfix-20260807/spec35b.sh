#!/usr/bin/env bash
# lane/step35-chunkfix: the spec cell §8's first pass MISSED.
#
# S2a/S2b compared AFTER vs BEFORE at the DEFAULT chunk (4096) — where §2.1's enumeration says the
# arm sequence is identical pre- and post-fix, so identical acceptance there confirms the default is
# untouched but says NOTHING about acceptance under the arm the fix actually changes. This runs the
# same K=3 comparison at MEMRA_PRIME_CHUNK=512, where post-fix rows [0,512) move from dequant-once
# FA to the f32 windowed kernel. Acceptance is downstream of prefill KV (research/f8f4-flip-20260806
# receipted exactly that propagation), so this is the cell that can actually move it.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
LONG=research/chunk-invariance-20260805/prompt-pp6257.txt
RAW=$HOME/step37/raw; mkdir -p "$RAW"
for _ in $(seq 1 720); do pgrep -f "tickinv35.sh|perf35.sh" >/dev/null || break; sleep 30; done
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/spec35b-$TS.log
{
echo "=== lane/step35-chunkfix run-spec at chunk=512 (the changed-arm cell) $TS commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  for arm in AFTER BEFORE; do
    echo; echo "########## K=3, n=32, T=4883, MEMRA_PRIME_CHUNK=512 — $arm ##########"
    ENVX=""; [ "$arm" = BEFORE ] && ENVX="MEMRA_STEP35_SWA_TKV=1"
    env $ENVX MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$D" MEMRA_NGEN=32 \
      MEMRA_SPEC_K=3 MEMRA_PRIME_CHUNK=512 MEMRA_PROMPT_FILE="$LONG" \
      timeout 3600 ./target/release/run-spec "$M"
    echo "$arm exit=$?"
  done
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== spec35b rc=$?"; echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

#!/usr/bin/env bash
# lane/step35-chunkfix: BAR-2 exactness receipts on the 2x RTX PRO 6000 pair, PP-2, step35 IQ4_XS.
#   E1 kernel-check, MODEL-BACKED against the step artifact (full battery, no MEMRA_KC_FAST /
#      MEMRA_KC_ONLY: a merge-gating run must be complete). Single-device by construction
#      (Engine::new(0)) — it oracles kernels against CPU references, not the PP topology, so no
#      MEMRA_PP_STAGES here (and that is why it fits: it mmaps ONE tensor, never the model).
#   E2 run-gen argmax gate on step35 over PP-2 — the prefill-vs-decode agreement receipt.
#   E3 ppn-gate stages=2 — PP-2 split logits BIT-IDENTICAL to the door-OFF reference. This is the
#      one that would catch the fix perturbing the stage-boundary arithmetic.
# One flock window, cards verified back to 0 MiB before release.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/exact35-$TS.log
{
echo "=== lane/step35-chunkfix BAR-2 exactness $TS"
echo "=== commit: $(cat BOX-COMMIT.txt)"
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv,noheader
(
  flock -w 5400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"

  echo; echo "########## E1: kernel-check model-backed (step35 IQ4_XS), FULL ##########"
  timeout 3600 ./target/release/kernel-check "$M"
  echo "kernel-check exit=$?"

  echo; echo "########## E2: run-gen argmax gate, PP-2, 64 tokens ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_NGEN=64 timeout 2400 \
    ./target/release/run-gen "$M" --prompt "Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
  echo "run-gen exit=$?"

  echo; echo "########## E3: ppn-gate stages=2 (bit-identity vs door-OFF) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 2400 ./target/release/ppn-gate "$M" 2 8 16
  echo "ppn-gate exit=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== exact35 rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

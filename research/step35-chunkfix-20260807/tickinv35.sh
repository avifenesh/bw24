#!/usr/bin/env bash
# lane/step35-chunkfix: TICK-BUDGET invariance — the segmentation axis one level ABOVE chunkinv.
#
# WHY THIS EXISTS. chunkinv varies MEMRA_PRIME_CHUNK *inside one* prime_cache call, and the fix
# makes that invariant. But serve splits a long prompt across SEVERAL prime_cache CALLS — one per
# scheduler tick (worker.rs:3555 and :3111), `take` tokens each, budget from LanePolicy:
# MEMRA_PREFILL_TICK=1024 (interactive) / MEMRA_PREFILL_JUDGE=MEMRA_PREFILL_HARVEST=256 (dark).
# Each CALL computes its own seq_end = cache.pos + take, so the arm predicate CAN differ between
# tick budgets even though every call is internally chunk-invariant. Enumeration of the real loop
# says: budget >= 513 is identical to a monolithic prime at every T in [2,40000); budget <= 512
# diverges for every T >= 513. This measures that instead of asserting it.
#
# NOTE ON ORDERING: this rebuilds the probe binary, so it MUST NOT run while perf35/spec35 are
# using ./target/release — a mid-run binary swap would void their interleaved A/B. It waits for
# both to exit before touching the tree.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
P=research/chunk-invariance-20260805
RAW=$HOME/step37/raw; mkdir -p "$RAW"

# 1) wait out the measured runs (binary-stability barrier, not a GPU lock)
for _ in $(seq 1 720); do
  pgrep -f "perf35.sh|spec35.sh" >/dev/null || break
  sleep 30
done
pgrep -f "perf35.sh|spec35.sh" >/dev/null && { echo "ABORT: perf35/spec35 still running after 6h"; exit 75; }

TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/tickinv35-$TS.log
{
echo "=== lane/step35-chunkfix tickinv $TS commit=$(cat BOX-COMMIT.txt)"
echo "=== rebuild (adds the tickinv probe mode; bin-only change, no engine edit)"
cargo build --release --bin concat-prime-probe 2>&1 | tail -5
echo "BUILD_RC=${PIPESTATUS[0]}"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"

  # A: T=4883, PAST the window. ref = 0 (monolithic) vs the real serve budgets + two below-window
  # budgets. Enumeration predicts 1024 EXACT (interactive default is immune) and 256/64 DIFFER
  # (dark-lane default is NOT). A prediction stated before the run, then checked.
  echo; echo "########## A: T=4883 (past win=512), budgets 0,1024,513,512,256,64 ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 3600 \
    ./target/release/concat-prime-probe "$M" tickinv --prompt-a "@$P/prompt-pp6257.txt" \
    --budgets 0,1024,513,512,256,64 --steps 24
  echo "tickinv long exit=$?"

  # B: CONTROL T=402, entirely BELOW the window — every budget must be EXACT (no arm to flip).
  echo; echo "########## B: CONTROL T=402 (below win), budgets 0,1024,256,64 ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 3600 \
    ./target/release/concat-prime-probe "$M" tickinv --prompt-a "@$P/prompt-pp512.txt" \
    --budgets 0,1024,256,64 --steps 24
  echo "tickinv control exit=$?"

  # C: does the OUTER axis interact with the INNER one? Same budgets, MEMRA_PRIME_CHUNK forced
  # small so both segmentations are active at once.
  echo; echo "########## C: T=4883, budgets 0,1024,256 with MEMRA_PRIME_CHUNK=64 ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PRIME_CHUNK=64 timeout 3600 \
    ./target/release/concat-prime-probe "$M" tickinv --prompt-a "@$P/prompt-pp6257.txt" \
    --budgets 0,1024,256 --steps 24
  echo "tickinv nested exit=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== tickinv35 rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

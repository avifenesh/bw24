#!/bin/bash
# Lane 3 (GPU 3): qwen adaptive-K A/B — interleaved pairs, N=3 per arm, NGEN=256,
# same-invocation plain-generate denominators (run-spec prints [generate] and
# [generate_spec] from ONE process). MEMRA_SPEC_STATS=1 on every run: len_hist is the
# mechanism-engagement receipt (fixed-K shows one bucket; adaptive spreads).
#
# q27 (dense, lane-1 best flags MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3):
#   3 prompt classes x {adapt, fixed} x 3. Plus a floor=1 probe on the short class:
#   q27 n_embd >= 3500 gives adapt_floor default 4, which clamps to k_cap=3 below
#   floor_ctx=1024 — the default law is a designed NO-OP at short ctx (gemma semantics);
#   the probe pins floor=1 to test the live law there.
# q35 (MoE, embedded head, its config K=2): board-2048, {adapt, fixed} x 3.
set -u
cd "$HOME/lane3" || exit 1
BW="$HOME/lane3/target/release"
OUT="$HOME/lane3/research/qwen-adaptive-k-20260801"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=3

Q27=/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf
Q35="$HOME/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
# The 35B GGUF carries NO nextn tensors (nextn=0 — captured in gate-q35 first attempt);
# its spec config loads the own-gen trimmed NVFP4 head sidecar via MEMRA_MTP_DRAFT
# (the board's q35-spec cell recipe, tools/full-board-bench.sh:209-211).
Q35DRAFT="$HOME/models/draft-35b-owntrim-nvfp4head-q4blk.gguf"

summ() { # log
  local gen spec acc pass
  gen=$(grep -oE '^\[generate\] .* = [0-9.]+ tok/s' "$1" | grep -oE '[0-9.]+ tok/s' | grep -oE '^[0-9.]+')
  spec=$(grep -oE '= [0-9.]+ tok/s .[0-9.]+x vs generate' "$1" | tail -1 | grep -oE '[0-9.]+' | head -1)
  acc=$(grep -oE 'acceptance: [0-9]+/[0-9]+ = [0-9.]+%' "$1" | tail -1)
  pass=$(grep -cE 'self-consistency: PASS' "$1")
  echo "gen=$gen spec=$spec $acc pass=$pass"
}

run_q27() { # class pf adapt_env label rep
  local log="$OUT/q27-$1-$4-rep$5.log"
  env $3 MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_STATS=1 \
    MEMRA_NGEN=256 MEMRA_PROMPT_FILE="$2" timeout 1200 "$BW/run-spec" "$Q27" >"$log" 2>&1
  echo "q27 $1 $4 rep$5: $(summ "$log")"
}
run_q35() { # adapt_env label rep
  local log="$OUT/q35-board-$2-rep$3.log"
  env $1 MEMRA_MTP_DRAFT="$Q35DRAFT" MEMRA_SPEC_K=2 MEMRA_SPEC_STATS=1 MEMRA_NGEN=256 \
    MEMRA_PROMPT_FILE=research/e2e/prompts/board-2048.txt timeout 1200 "$BW/run-spec" "$Q35" >"$log" 2>&1
  echo "q35 board $2 rep$3: $(summ "$log")"
}

for rep in 1 2 3; do
  for cls_pf in "short:research/e2e/prompts/p1-code-short.txt" \
                "board:research/e2e/prompts/board-2048.txt" \
                "agentic:research/e2e/prompts/p3-agentic-long-v3.txt"; do
    cls="${cls_pf%%:*}"; pf="${cls_pf#*:}"
    run_q27 "$cls" "$pf" "MEMRA_SPEC_ADAPT=1" adapt "$rep"
    run_q27 "$cls" "$pf" "MEMRA_SPEC_ADAPT=0" fixed "$rep"
  done
done
# floor=1 probe, short class (interleaved against fixed for its own denominator pairing)
for rep in 1 2 3; do
  run_q27 short research/e2e/prompts/p1-code-short.txt "MEMRA_SPEC_ADAPT=1 MEMRA_SPEC_ADAPT_FLOOR=1" adaptf1 "$rep"
  run_q27 short research/e2e/prompts/p1-code-short.txt "MEMRA_SPEC_ADAPT=0" fixedb "$rep"
done
for rep in 1 2 3; do
  run_q35 "MEMRA_SPEC_ADAPT=1" adapt "$rep"
  run_q35 "MEMRA_SPEC_ADAPT=0" fixed "$rep"
done
echo "BENCH DONE"

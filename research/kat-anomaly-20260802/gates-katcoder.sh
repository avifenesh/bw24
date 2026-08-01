#!/usr/bin/env bash
# KAT drafter re-verdict on the post-fix binary (IQ4_XS trunk dp4a default) — the #42 flip.
# Same battery as research/ornith-drafters-20260801/gates-drafter.sh (RECIPE.md §5), katcoder
# arm only, run against THIS worktree's binary:
#   phase gate: run-spec K=1..8 self-consistency (p1 prompt, ngen 128)
#   phase acc:  acceptance table K=2..4 x {p1,p2,p3} (ngen 256, board protocol)
#   phase e2e:  3 dedicated reps at serving K=2 per class (spec/plain ratio x3,
#               plain + spec interleaved in-process per invocation)
# usage: gates-katcoder.sh <gate|acc|e2e|all>
set -euo pipefail
PHASE=${1:-all}
WT=/home/avifenesh/projects/wt-kat-anomaly
RD=$WT/research/kat-anomaly-20260802
PDIR=$WT/research/e2e/prompts
BIN=$WT/target/release/run-spec
MODEL=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
DRAFT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/draft-katcoder-owntrim-nvfp4head-q4blk.gguf
KSERVE=2
GD=$RD/gates
mkdir -p "$GD"

runspec() {  # logfile extra-envs...
  local log=$1; shift
  { echo "=== $(date -Is) $* MEMRA_MTP_DRAFT=$DRAFT"
    env "$@" MEMRA_MTP_DRAFT="$DRAFT" \
      flock /tmp/gpu5090.lock timeout 1800 "$BIN" "$MODEL"
    echo "=== rc=$?"
  } 2>&1 | tee -a "$log"
}

if [ "$PHASE" = gate ] || [ "$PHASE" = all ]; then
  runspec "$GD/gate-k1-8.log" MEMRA_NGEN=128 MEMRA_PROMPT="$(cat "$PDIR/p1-code-short.txt")"
fi
if [ "$PHASE" = acc ] || [ "$PHASE" = all ]; then
  for K in 2 3 4; do
    for P in p1-code-short p2-code-medium p3-agentic-long; do
      runspec "$GD/acc-k$K-$P.log" MEMRA_SPEC_K=$K MEMRA_NGEN=256 \
        MEMRA_PROMPT="$(cat "$PDIR/$P.txt")"
    done
  done
fi
if [ "$PHASE" = e2e ] || [ "$PHASE" = all ]; then
  for REP in 1 2 3; do
    for P in p1-code-short p2-code-medium p3-agentic-long; do
      runspec "$GD/e2e-k$KSERVE-$P-rep$REP.log" MEMRA_SPEC_K=$KSERVE MEMRA_NGEN=256 \
        MEMRA_PROMPT="$(cat "$PDIR/$P.txt")"
    done
  done
fi
echo "gates phase '$PHASE' done -> $GD"

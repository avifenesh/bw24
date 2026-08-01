#!/usr/bin/env bash
# Full gate battery for one donor-block drafter (RECIPE.md §5). Each run-spec invocation
# is its own bounded flock hold and interleaves plain generate + spec in-process.
#   phase gate: run-spec K=1..8 self-consistency (p1 prompt, ngen 128)
#   phase acc:  acceptance table K=2..4 x {p1,p2,p3} (ngen 256, board protocol)
#   phase e2e:  3 dedicated reps at the serving K per class (spec/plain ratio x3)
# usage: gates-drafter.sh <ornith9b|ornith35b|katcoder> <gate|acc|e2e|all>
set -euo pipefail
KEY=$1; PHASE=${2:-all}
WT=/home/avifenesh/projects/wt-ornith-drafters
RD=$WT/research/ornith-drafters-20260801
PDIR=$WT/research/e2e/prompts
BIN=$WT/target/release/run-spec
case $KEY in
  ornith9b)
    MODEL=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
    DRAFT=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/draft-ornith9b-owntrim-nvfp4head-q4blk.gguf
    KSERVE=3 ;;
  ornith35b)
    MODEL=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
    DRAFT=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/draft-ornith35b-owntrim-nvfp4head-q4blk.gguf
    KSERVE=2 ;;
  katcoder)
    MODEL=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf
    DRAFT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/draft-katcoder-owntrim-nvfp4head-q4blk.gguf
    KSERVE=2 ;;
  *) echo "unknown model key: $KEY"; exit 2 ;;
esac
GD=$RD/gates/$KEY
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
echo "gates phase '$PHASE' done for $KEY -> $GD"

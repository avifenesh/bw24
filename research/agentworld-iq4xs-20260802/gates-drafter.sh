#!/usr/bin/env bash
# agentworld-iq4xs: drafter gate battery vs the UD-IQ4_XS artifact (baked literals).
# The drafter is artifact-independent at the ranks level (built vs the Q4_K_M target's
# own-gen ranks; same model weights) but its GATES re-run against the new artifact.
#   phase gate: run-spec K=1..8 self-consistency (p1 prompt, ngen 128)
#   phase acc:  acceptance table K=2..4 x {p1,p2,p3} (ngen 256, board protocol)
# Each run-spec invocation is its own bounded flock hold and interleaves plain
# generate + spec in-process. usage: gates-drafter.sh <gate|acc|all>
set -euo pipefail
PHASE=${1:-all}
W=/home/avifenesh/projects/bw24-aw-iq4xs
R=$W/research/agentworld-iq4xs-20260802
PDIR=$W/research/e2e/prompts
BIN=$W/target/release/run-spec
MODEL=/data/ai-ml/hf-models/agentworld-35b-gguf/Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf
DRAFT=/data/ai-ml/hf-models/agentworld-35b-gguf/draft-agentworld-owntrim-nvfp4head-q4blk.gguf
GD=$R/gates/drafter
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
echo "gates phase '$PHASE' done -> $GD"

#!/bin/bash
# Chain after ranks: 5b trim draft -> 5c validate -> 4e run-spec K=1..3 -> 6 serve pilot
# (marker + regression) -> 7 serve base (regression control) -> 4f base-pair xcheck control.
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24

export MEMRA_GGUFPY=/root/llama.cpp/gguf-py
export MEMRA_QUANTIZE=/root/llama.cpp/build/bin/llama-quantize
export MEMRA_CONVERT_PY=/root/llama.cpp/.venv/bin/python3
M=/root/pilot/q9pilot-Q8_0.gguf
DRAFT=/root/pilot/draft-q9pilot-owntrim.gguf

/root/pilot/stage.sh 5b-trim-draft tools/make-trimmed-draft.sh "$M" /root/pilot/ranks-q9pilot.gguf.txt "$DRAFT" 32768 || exit 1

runspec() {
  MEMRA_SPEC_K=$1 MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=64 target/release/run-spec "$M"
}
/root/pilot/stage.sh 4e-runspec-k1 bash -c "MEMRA_SPEC_K=1 MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=64 target/release/run-spec $M" || exit 1
/root/pilot/stage.sh 4e-runspec-k2 bash -c "MEMRA_SPEC_K=2 MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=64 target/release/run-spec $M" || exit 1
/root/pilot/stage.sh 4e-runspec-k3 bash -c "MEMRA_SPEC_K=3 MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=64 target/release/run-spec $M" || exit 1

/root/pilot/stage.sh 6-serve-pilot /root/pilot/serve_stage.sh pilot || exit 1
/root/pilot/stage.sh 7-serve-base /root/pilot/serve_stage.sh base || exit 1

# control: base GGUF vs base HF dir, same form as 4d (is Q8-vs-bf16 divergence
# artifact-independent?)
/root/pilot/stage.sh 4f-base-xcheck bash -c '
  cd /root/bw24
  MEMRA_CHAT=1 MEMRA_NGEN=64 target/release/run-gen /root/pilot/q9base-Q8_0.gguf --prompt "Explain what a hash map is." | tee /root/pilot/xcheck-base-gguf.txt
  MEMRA_CHAT=1 MEMRA_NGEN=64 target/release/run-gen /root/hf-models/qwen35-9b --prompt "Explain what a hash map is." | tee /root/pilot/xcheck-base-hf.txt
  G=$(grep "^tokens:" /root/pilot/xcheck-base-gguf.txt | tail -1)
  H=$(grep "^tokens:" /root/pilot/xcheck-base-hf.txt | tail -1)
  if [ -n "$G" ] && [ "$G" = "$H" ]; then echo "BASE-XCHECK: IDENTICAL"; else echo "BASE-XCHECK: DIVERGED"; fi
'
echo AFTER-RANKS-CHAIN-DONE

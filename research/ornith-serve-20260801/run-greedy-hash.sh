#!/bin/bash
# ornith-serve-bench: serve-level isolation evidence — greedy c=1 vs c=16.
# check-batch-exact.py phase A = 16 distinct greedy prompts SEQUENTIALLY (c=1, B=1 decode),
# phase B = same 16 CONCURRENTLY (batched to the model's chunk cap, i.e. c=16); every
# completion must be byte-identical to its isolated reference (sha256 per prompt).
# Refs are per config (q8rp routes different — bit-identical-by-contract — kernels; each
# arm is compared against ITS OWN isolated c=1 refs).
set -u
W=/home/avifenesh/projects/wt-ornith-serve-bench
R=$W/research/ornith-serve-20260801
PORT=8098
BASE=http://127.0.0.1:$PORT

O9B=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf

run_hash() { # cfg model_path env_kv
  local cfg=$1 mpath=$2 envv=$3
  (
    flock 9
    env $envv MEMRA_MODELS="m=$mpath" MEMRA_ADDR=127.0.0.1:$PORT \
      "$W/target/release/memra-server" > "$R/server-hash-$cfg.log" 2>&1 &
    SRV=$!
    local up=0
    for _ in $(seq 1 300); do curl -s $BASE/health >/dev/null 2>&1 && { up=1; break; }; sleep 1; done
    if [ "$up" != 1 ]; then
      echo "SERVER FAILED hash-$cfg" | tee -a "$R/failures.log"
      kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; exit 1
    fi
    python3 "$W/tools/check-batch-exact.py" --base $BASE --model m --n 16 \
      --max-tokens 96 --label "$cfg" --out "$R/greedy-hash-$cfg.jsonl" \
      --ref "$R/greedy-refs-$cfg.json" 2>&1 | tee "$R/greedy-hash-$cfg.log" | tail -4
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  ) 9>/tmp/gpu5090.lock
}

run_hash o9b-naked  "$O9B"  ""
run_hash o9b-q8rp   "$O9B"  "MEMRA_Q8RP=1"
run_hash o35b-naked "$O35B" ""
run_hash kat-naked  "$KAT"  ""
echo "GREEDY-HASH-DONE $(date -u +%FT%TZ)"

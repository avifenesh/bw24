#!/bin/bash
# Exactness battery with levers ON (the pod claim, re-proven HERE on the 5090):
#   1. run-spec K=1..8 self-consistency, nv-embedded / nv+draft, PMIN=0.3 PMIN0=1 + BURST=128
#   2. greedy stream identity: default vs burst-only vs full stack (cmp of run-ident.sh outputs)
#   3. serve-smoke.sh on the q9 pair (0 failed = gate)
# Each GPU phase holds the flock via its runner.
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)
NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
NVDRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
P512=$TREE/research/e2e/prompts/pp512.txt
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/driver.log"; }

# ---- 1. run-spec K=1..8 with the levers ON (flock per run)
exec 9>/tmp/gpu5090.lock

flock 9
MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_PMIN0=1 MEMRA_SPEC_BURST=128 \
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 \
  "$TREE/target/release/run-spec" "$NV" > "$R/logs/gate-runspec-nv-embedded-lever.log" 2>&1
rc=$?
log "run-spec nv-embedded lever-on rc=$rc PASS=$(grep -c PASS "$R/logs/gate-runspec-nv-embedded-lever.log")"
flock -u 9

flock 9
MEMRA_MTP_DRAFT=$NVDRAFT MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_PMIN0=1 MEMRA_SPEC_BURST=128 \
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 \
  "$TREE/target/release/run-spec" "$NV" > "$R/logs/gate-runspec-nv-draft-lever.log" 2>&1
rc=$?
log "run-spec nv-draft lever-on rc=$rc PASS=$(grep -c PASS "$R/logs/gate-runspec-nv-draft-lever.log")"
flock -u 9
exec 9>&-

# ---- 2. greedy stream identity: default(K3B32) vs the 82-SM winner (K3B128) vs +pmin
./run-ident.sh nv 3 32  0   nv-K3B32
./run-ident.sh nv 3 128 0   nv-K3B128
./run-ident.sh nv 3 128 0.3 nv-K3B128pm
./run-ident.sh q9 3 32  0   q9-K3B32
./run-ident.sh q9 3 128 0.3 q9-K3B128pm
for pair in "nv-K3B32 nv-K3B128" "nv-K3B32 nv-K3B128pm" "q9-K3B32 q9-K3B128pm"; do
  set -- $pair
  if cmp -s "$R/logs/ident-$1.txt" "$R/logs/ident-$2.txt"; then
    log "identity $1 vs $2: BYTE-IDENTICAL"
  else
    log "identity $1 vs $2: MISMATCH"
  fi
done

# ---- 3. serve-smoke (q9 default pair; flock for the whole battery — it boots its own servers)
exec 9>/tmp/gpu5090.lock
flock 9
( cd "$TREE" && timeout 1800 tools/serve-smoke.sh > "$R/logs/gate-serve-smoke.log" 2>&1 )
log "serve-smoke rc=$? failed=$(grep -c FAIL "$R/logs/gate-serve-smoke.log")"
flock -u 9
exec 9>&-

echo GATES_DONE >> "$R/logs/driver.log"
echo GATES_DONE

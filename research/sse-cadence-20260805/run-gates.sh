#!/bin/bash
# Gate battery with the cadence fix in the binary (fix is default-on; the seam only
# reverts emission): run-spec K=1..8 one arm (nv+draft, BURST=128), serve-smoke,
# serve-st-gate one arm. Each GPU phase = its own flock hold.
set -u
cd "$(dirname "$0")"
R=$PWD
TREE=$(cd ../.. && pwd)
NV=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
NVDRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
P512=$TREE/research/e2e/prompts/pp512.txt
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/logs/gates-driver.log"; }

exec 9>/tmp/gpu5090.lock

# ---- 1. run-spec K=1..8 self-consistency, nv+draft, BURST=128
flock 9
MEMRA_MTP_DRAFT=$NVDRAFT MEMRA_SPEC_BURST=128 \
  MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 \
  "$TREE/target/release/run-spec" "$NV" > "$R/logs/gate-runspec-nv-draft.log" 2>&1
rc=$?
log "run-spec nv-draft B128 rc=$rc PASS=$(grep -c PASS "$R/logs/gate-runspec-nv-draft.log")"
flock -u 9

# ---- 2. serve-smoke (boots its own servers)
flock 9
( cd "$TREE" && timeout 1800 tools/serve-smoke.sh > "$R/logs/gate-serve-smoke.log" 2>&1 )
log "serve-smoke rc=$? failed=$(grep -c FAIL "$R/logs/gate-serve-smoke.log")"
flock -u 9

# ---- 3. serve-st-gate one arm (default ckpt)
flock 9
( cd "$TREE" && timeout 1800 tools/serve-st-gate.sh > "$R/logs/gate-serve-st.log" 2>&1 )
log "serve-st-gate rc=$? FAIL=$(grep -c FAIL "$R/logs/gate-serve-st.log")"
flock -u 9
exec 9>&-

log "GATES_DONE"
echo GATES_DONE

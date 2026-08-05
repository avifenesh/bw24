#!/bin/bash
# PHASE-2 gates: exactness anchors on the fresh 70ce5a0f build, both artifacts.
# Also yields the BARE run-spec K=1..8 curves for both arms (item 2's bare denominator).
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24
R=/root/receipts-p2
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
DRAFT=/root/models/draft-owntrim-nvfp4head-q4blk.gguf
P512=research/e2e/prompts/pp512.txt
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/gates-driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }

# cold-start burn (first run after rebuild is an outlier — phase-1 law)
log "cold-burn start: $(gpustate)"
MEMRA_NGEN=16 MEMRA_PROMPT_FILE=$P512 timeout 900 target/release/run-gen $Q8 > "$R/logs/coldburn-q8.log" 2>&1
log "cold-burn q8 rc=$?"

# gate 1: kernel-check (full)
timeout 1800 target/release/kernel-check $Q8 > "$R/logs/gate-kernel-check-q8.log" 2>&1
log "kernel-check q8 rc=$? bad=$(grep -cE 'BAD|FAIL' $R/logs/gate-kernel-check-q8.log)"

# gate 2: run-gen argmax both artifacts
MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 900 target/release/run-gen $Q8 > "$R/logs/gate-rungen-q8.log" 2>&1
log "run-gen q8 rc=$? $(grep -oE '(MATCH|MISMATCH[A-Z-]*)' $R/logs/gate-rungen-q8.log | sort | uniq -c | tr '\n' ' ')"
MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 900 target/release/run-gen $NV > "$R/logs/gate-rungen-nv.log" 2>&1
log "run-gen nv rc=$? $(grep -oE '(MATCH|MISMATCH[A-Z-]*)' $R/logs/gate-rungen-nv.log | sort | uniq -c | tr '\n' ' ')"

# gate 3: run-spec K=1..8 self-consistency — nv EMBEDDED head (the full-head arm)
MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 target/release/run-spec $NV > "$R/logs/gate-runspec-nv-embedded-K1to8.log" 2>&1
log "run-spec nv-embedded rc=$? $(grep -cE 'PASS' $R/logs/gate-runspec-nv-embedded-K1to8.log) PASS lines"

# gate 4: run-spec K=1..8 — nv + owntrim DRAFT (the serve/daily config)
MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 target/release/run-spec $NV > "$R/logs/gate-runspec-nv-draft-K1to8.log" 2>&1
log "run-spec nv-draft rc=$? $(grep -cE 'PASS' $R/logs/gate-runspec-nv-draft-K1to8.log) PASS lines"

# gate 5: run-spec K=1..8 — Q8_0 + external drafter (THE new arm; phase-1 rc=2'd without a draft)
MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 target/release/run-spec $Q8 > "$R/logs/gate-runspec-q8-draft-K1to8.log" 2>&1
log "run-spec q8-draft rc=$? $(grep -cE 'PASS' $R/logs/gate-runspec-q8-draft-K1to8.log) PASS lines"
log "GATES_DONE: $(gpustate)"
echo GATES_DONE

#!/bin/bash
# PHASE-2 item 3: acceptance vs context depth, serve path, 4 arms:
#   nv-trim K=5 | nv-embedded(full head) K=5 | q8-trim K=4 | q8-full(untrimmed donor) K=4
# Server stderr [spec-acc] ctx= lines = the acceptance receipt; client jsonl = e2e receipt.
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24
R=/root/receipts-p2
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
TRIM=/root/mb/drafts/qwen36-27b-nvfp4/draft-owntrim-nvfp4head-q4blk.gguf
FULL=/root/models/draft-full-untrimmed.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/depth-driver.log"; }
wait_health() { for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

run_arm() { # $1 arm-label  $2 models-spec  $3 K
  MEMRA_SPEC_K=$3 MEMRA_MODELS="$2" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$R/logs/depth-$1.server.log" 2>&1 &
  SPID=$!
  wait_health || { log "no-up $1"; kill $SPID 2>/dev/null; return 1; }
  python3 /root/receipts-p2/scripts/p2-depth.py $BASE "$1" "$R/logs/depth-points.jsonl" \
    >> "$R/logs/depth-client-$1.log" 2>&1
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 2
  log "arm $1 done: $(grep -c 'spec-acc' $R/logs/depth-$1.server.log) spec-acc lines | $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader)"
}

# order alternated across the two passes so drift cancels between paired arms
run_arm nv-trim-K5   "q27=$NV+$TRIM" 5
run_arm nv-full-K5   "q27=$NV"       5
run_arm q8-trim-K4   "q27=$Q8+$TRIM" 4
run_arm q8-full-K4   "q27=$Q8+$FULL" 4
run_arm q8-full-K4-b "q27=$Q8+$FULL" 4
run_arm q8-trim-K4-b "q27=$Q8+$TRIM" 4
run_arm nv-full-K5-b "q27=$NV"       5
run_arm nv-trim-K5-b "q27=$NV+$TRIM" 5
log "DEPTH_DONE"
echo DEPTH_DONE

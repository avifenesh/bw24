#!/bin/bash
# Greedy serve-phase runs (comparable to the bare [spec-phase] decomposition; the sampled
# path never marks verify-wait so the earlier sampled rows lump accept into commit-host).
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24
R=/root/receipts-p2
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
DRAFT=/root/mb/drafts/qwen36-27b-nvfp4/draft-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/phase-serve-greedy-driver.log"; }
wait_health() { for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
one() { # $1 art $2 K
  local art=$1 K=$2 M
  if [ "$art" = nv ]; then M=$NV; else M=$Q8; fi
  MEMRA_SPEC_PHASE=1 MEMRA_SPEC_K=$K MEMRA_MODELS="q27=$M+$DRAFT" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$R/logs/phase-serve-greedy-$art-K$K.server.log" 2>&1 &
  SPID=$!
  wait_health || { log "no-up $art"; kill $SPID 2>/dev/null; return 1; }
  python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 6 \
    --max-tokens 256 --greedy --out "$R/logs/phase-serve-greedy-points.jsonl" \
    --label "phaseg-$art-K$K" >> "$R/logs/phase-serve-greedy-load.log" 2>&1
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 2
  log "phaseg $art K=$K: $(grep -oE 'spec-phase.*' $R/logs/phase-serve-greedy-$art-K$K.server.log | tail -3 | tr '\n' '|')"
}
one nv 5
one q8 4
log PHASEG_DONE
echo PHASEG_DONE

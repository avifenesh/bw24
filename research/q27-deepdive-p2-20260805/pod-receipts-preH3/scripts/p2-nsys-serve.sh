#!/bin/bash
# PHASE-2 item 1: nsys one spec-decode serve session per artifact at its serve-optimum K
# (nv K=5, q8 K=4), plus MEMRA_SPEC_PHASE=1 serve runs (no nsys) for wall decomposition
# on the serve path itself.
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
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/nsys-serve-driver.log"; }

wait_health() { for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

profile_one() { # $1 art  $2 K
  local art=$1 K=$2 M
  if [ "$art" = nv ]; then M=$NV; else M=$Q8; fi
  log "nsys $art K=$K starting"
  MEMRA_SPEC_K=$K MEMRA_MODELS="q27=$M+$DRAFT" MEMRA_ADDR=$ADDR \
    nsys profile -o "$R/nsys/serve-spec-$art-K$K" --force-overwrite=true \
    -t cuda,nvtx --cuda-graph-trace=node \
    target/release/memra-server > "$R/logs/nsys-serve-$art.server.log" 2>&1 &
  NPID=$!
  wait_health || { log "server did not come up under nsys ($art)"; kill $NPID; return 1; }
  python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 6 \
    --max-tokens 128 --out "$R/logs/nsys-serve-points.jsonl" --label "nsys-$art-K$K" \
    >> "$R/logs/nsys-serve-load.log" 2>&1
  # graceful stop so nsys finalizes the report
  kill -INT $NPID 2>/dev/null
  wait $NPID 2>/dev/null
  sleep 3
  log "nsys $art K=$K done rc=$?"
}

phase_serve_one() { # $1 art  $2 K
  local art=$1 K=$2 M
  if [ "$art" = nv ]; then M=$NV; else M=$Q8; fi
  MEMRA_SPEC_PHASE=1 MEMRA_SPEC_K=$K MEMRA_MODELS="q27=$M+$DRAFT" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$R/logs/phase-serve-$art-K$K.server.log" 2>&1 &
  SPID=$!
  wait_health || { log "server no-up phase-serve $art"; kill $SPID; return 1; }
  python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 6 \
    --max-tokens 256 --out "$R/logs/phase-serve-points.jsonl" --label "phase-$art-K$K" \
    >> "$R/logs/phase-serve-load.log" 2>&1
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 2
  log "phase-serve $art K=$K: $(grep -oE 'spec-phase.*' $R/logs/phase-serve-$art-K$K.server.log | tail -2 | tr '\n' ' | ')"
}

profile_one nv 5
profile_one q8 4
phase_serve_one nv 5
phase_serve_one q8 4
log "NSYS_SERVE_DONE"
echo NSYS_SERVE_DONE

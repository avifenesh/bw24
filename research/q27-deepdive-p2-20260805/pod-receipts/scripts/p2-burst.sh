#!/bin/bash
# LEVER L1 (profile-driven): MEMRA_SPEC_BURST at serve c=1. The nsys serve profiles show
# ~13% of session GPU time in m=1 trunk passes (qmatvec fused2/mmvq singles on q8; mr2_rp/
# dual_mr2 on nv) = the per-burst pending-flush + init-feed the SAMPLED path pays at every
# burst boundary (sampled cannot pending-carry). Burst 32 -> 128 amortizes 4 boundaries -> 1
# on a 128-token request. Arms interleaved, order alternated per rep, N=3 reps x 2 passes.
# p50 latency recorded — burst size trades stream granularity; the row reports both.
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
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/burst-driver.log"; }
wait_health() { for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

point() { # $1 art  $2 K  $3 burst  $4 rep
  local art=$1 K=$2 B=$3 r=$4 M
  if [ "$art" = nv ]; then M=$NV; else M=$Q8; fi
  MEMRA_SPEC_BURST=$B MEMRA_SPEC_K=$K MEMRA_MODELS="q27=$M+$DRAFT" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$R/logs/burst-$art-B$B-r$r.server.log" 2>&1 &
  SPID=$!
  wait_health || { log "no-up $art B$B"; kill $SPID 2>/dev/null; return 1; }
  for p in 1 2; do
    python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 4 \
      --max-tokens 128 --out "$R/logs/burst-points.jsonl" --label "$art-B$B-r$r-p$p" \
      >> "$R/logs/burst-load.log" 2>&1
    log "burst $art K=$K B=$B r$r p$p: $(tail -1 $R/logs/burst-points.jsonl | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print("agg=%.1f p50=%.3f err=%d" % (d["agg_tok_s"], d["lat_p50_s"], d["n_err"]))' 2>/dev/null || echo parse-fail)"
  done
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 2
}

for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then BURSTS="32 64 128"; ORDER="q8 nv"; else BURSTS="128 64 32"; ORDER="nv q8"; fi
  for art in $ORDER; do
    if [ "$art" = nv ]; then K=5; else K=4; fi
    for B in $BURSTS; do point $art $K $B $r; done
  done
done
log "BURST_DONE"
echo BURST_DONE

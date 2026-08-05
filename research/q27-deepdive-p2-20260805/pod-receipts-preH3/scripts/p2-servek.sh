#!/bin/bash
# PHASE-2 item 2: serve K-policy per artifact. q8 serve K=2..6 (unswept until now),
# nv serve K=3..6 (confirm the dev-lane K=5 verdict). memra-server + load-serve
# (phase-1 law: never size serve claims on decode-batch-bench).
# Structure: 3 reps x ladder (direction alternated per rep) x 2 load passes per point
# -> N=6 passes per (artifact, K), spread across time so thermal drift cancels.
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24
R=/root/receipts-p2
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
DRAFT=/root/models/draft-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/servek-driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }

start_server() { # $1 env  $2 models  $3 logfile
  env $1 MEMRA_MODELS="$2" MEMRA_ADDR=$ADDR target/release/memra-server > "$3" 2>&1 &
  SPID=$!
  for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  log "SERVER-DID-NOT-COME-UP $3"; tail -5 "$3" >> "$R/logs/servek-driver.log"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }

run_point() { # $1 art  $2 K  $3 rep
  local art=$1 K=$2 r=$3 M DR
  if [ "$art" = nv ]; then M=$NV; else M=$Q8; fi
  start_server "MEMRA_SPEC_K=$K" "q27=$M+$DRAFT" "$R/logs/servek-$art-K$K-r$r.server.log" || return 1
  for p in 1 2; do
    python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 4 \
      --max-tokens 128 --out "$R/logs/servek-points.jsonl" --label "$art-K$K-r$r-p$p" \
      >> "$R/logs/servek-load.log" 2>&1
    log "servek $art K=$K r$r p$p: $(tail -1 $R/logs/servek-points.jsonl | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print("agg=%.1f p50=%.3f err=%d" % (d["agg_tok_s"], d["lat_p50_s"], d["n_err"]))' 2>/dev/null || echo parse-fail) | $(gpustate)"
  done
  stop_server
}

for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then LADDER_Q8="2 3 4 5 6"; LADDER_NV="3 4 5 6"; ORDER="q8 nv"; else LADDER_Q8="6 5 4 3 2"; LADDER_NV="6 5 4 3"; ORDER="nv q8"; fi
  for art in $ORDER; do
    if [ "$art" = q8 ]; then LAD=$LADDER_Q8; else LAD=$LADDER_NV; fi
    for K in $LAD; do run_point $art $K $r; done
  done
done
log "SERVEK_DONE: $(gpustate)"
echo SERVEK_DONE

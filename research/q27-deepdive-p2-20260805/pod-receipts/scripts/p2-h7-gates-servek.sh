#!/bin/bash
# 7ac05f54 (H3-fixed) tree: gates first, then serve K re-sweep + burst re-check.
# Same interleave discipline as the pre-H3 sweep (direction alternated per rep).
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24-h7
R=/root/receipts-p2
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
DRAFT=/root/mb/drafts/qwen36-27b-nvfp4/draft-owntrim-nvfp4head-q4blk.gguf
P512=research/e2e/prompts/pp512.txt
ADDR=127.0.0.1:8199
BASE=http://$ADDR
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/h7-driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }

# ---- gates on the new tree
MEMRA_NGEN=64 MEMRA_PROMPT_FILE=$P512 timeout 900 target/release/run-gen $Q8 > "$R/logs/h7-gate-rungen-q8.log" 2>&1
log "h7 run-gen q8 rc=$? $(grep -coE 'MATCH' $R/logs/h7-gate-rungen-q8.log) MATCH"
MEMRA_NGEN=64 MEMRA_PROMPT_FILE=$P512 timeout 900 target/release/run-gen $NV > "$R/logs/h7-gate-rungen-nv.log" 2>&1
log "h7 run-gen nv rc=$? $(grep -coE 'MATCH' $R/logs/h7-gate-rungen-nv.log) MATCH"
MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 target/release/run-spec $NV > "$R/logs/h7-gate-runspec-nv-embedded.log" 2>&1
log "h7 run-spec nv-embedded rc=$? $(grep -c PASS $R/logs/h7-gate-runspec-nv-embedded.log) PASS"
MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 target/release/run-spec $NV > "$R/logs/h7-gate-runspec-nv-draft.log" 2>&1
log "h7 run-spec nv-draft rc=$? $(grep -c PASS $R/logs/h7-gate-runspec-nv-draft.log) PASS"
MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 3600 target/release/run-spec $Q8 > "$R/logs/h7-gate-runspec-q8-draft.log" 2>&1
log "h7 run-spec q8-draft rc=$? $(grep -c PASS $R/logs/h7-gate-runspec-q8-draft.log) PASS"

start_server() { # $1 env  $2 models  $3 logfile
  env $1 MEMRA_MODELS="$2" MEMRA_ADDR=$ADDR target/release/memra-server > "$3" 2>&1 &
  SPID=$!
  for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  log "NO-UP $3"; tail -5 "$3" >> "$R/logs/h7-driver.log"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }

point() { # $1 art $2 K $3 burst $4 tag
  local art=$1 K=$2 B=$3 tag=$4 M
  if [ "$art" = nv ]; then M=$NV; else M=$Q8; fi
  start_server "MEMRA_SPEC_K=$K MEMRA_SPEC_BURST=$B" "q27=$M+$DRAFT" "$R/logs/h7-$tag.server.log" || return 1
  for p in 1 2; do
    python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 4 \
      --max-tokens 128 --out "$R/logs/h7-points.jsonl" --label "$tag-p$p" \
      >> "$R/logs/h7-load.log" 2>&1
    log "$tag p$p: $(tail -1 $R/logs/h7-points.jsonl | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print(f"agg={d[\"agg_tok_s\"]:.1f} p50={d[\"lat_p50_s\"]:.3f} err={d[\"n_err\"]}")' 2>/dev/null) | $(gpustate)"
  done
  stop_server
}

# ---- serve K re-sweep on H3 tree at default burst(32): 3 reps, ladder alternated
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then QK="3 4 5"; NK="4 5 6"; ORD="q8 nv"; else QK="5 4 3"; NK="6 5 4"; ORD="nv q8"; fi
  for art in $ORD; do
    if [ "$art" = q8 ]; then KS=$QK; else KS=$NK; fi
    for K in $KS; do point $art $K 32 "sk-$art-K$K-r$r"; done
  done
done
log "H7_SERVEK_DONE"

# ---- burst re-check on H3 tree: nv K5 B32vs128, q8 K4/K5 B32vs128, 2 reps alternated
for r in 1 2; do
  if [ $((r % 2)) -eq 1 ]; then BS="32 128"; else BS="128 32"; fi
  for B in $BS; do
    point nv 5 $B "bu-nv-K5-B$B-r$r"
    point q8 4 $B "bu-q8-K4-B$B-r$r"
    point q8 5 $B "bu-q8-K5-B$B-r$r"
  done
done
log "H7_BURST_DONE"
echo H7_ALL_DONE

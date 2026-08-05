#!/bin/bash
# BURST lever guards: (1) c=8 A/B B32 vs B128 (batched latency/throughput regression check),
# (2) greedy stream identity B32 vs B128 (exactness), (3) K-interaction at B128 (does the
# serve-K optimum shift when the burst boundary tax is gone?).
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
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/burstguard-driver.log"; }
wait_health() { for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
srv() { # $1 env-extra $2 model-spec $3 log
  env $1 MEMRA_MODELS="$2" MEMRA_ADDR=$ADDR target/release/memra-server > "$3" 2>&1 &
  SPID=$!; wait_health
}
stop() { kill ${SPID:-0} 2>/dev/null; wait ${SPID:-0} 2>/dev/null || true; sleep 2; }

# ---- guard 1: c=8, B32 vs B128, nv K=5, N=3 passes alternated
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then ARMS="32 128"; else ARMS="128 32"; fi
  for B in $ARMS; do
    srv "MEMRA_SPEC_BURST=$B MEMRA_SPEC_K=5" "q27=$NV+$DRAFT" "$R/logs/bg-c8-B$B-r$r.server.log" || { log "no-up c8 B$B"; continue; }
    python3 tools/load-serve.py --base $BASE --model q27 --concurrency 8 --requests 24 \
      --max-tokens 128 --out "$R/logs/bg-c8-points.jsonl" --label "c8-B$B-r$r" \
      >> "$R/logs/bg-c8-load.log" 2>&1
    log "c8 B$B r$r: $(tail -1 $R/logs/bg-c8-points.jsonl | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print("agg=%.1f p50=%.3f p95=%.3f err=%d" % (d["agg_tok_s"], d["lat_p50_s"], d.get("lat_p95_s",-1), d["n_err"]))' 2>/dev/null || echo parse-fail)"
    stop
  done
done

# ---- guard 2: greedy identity B32 vs B128, both artifacts (one fixed prompt, 128 tok)
for art in nv q8; do
  if [ "$art" = nv ]; then M=$NV; K=5; else M=$Q8; K=4; fi
  for B in 32 128; do
    srv "MEMRA_SPEC_BURST=$B MEMRA_SPEC_K=$K" "q27=$M+$DRAFT" "$R/logs/bg-ident-$art-B$B.server.log" || continue
    curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d \
      '{"model":"q27","messages":[{"role":"user","content":"Explain how a CUDA graph reduces kernel launch overhead, in about 200 words."}],"max_tokens":128,"temperature":0.0,"stream":false}' \
      | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["choices"][0]["message"]["content"])' \
      > "$R/logs/bg-ident-$art-B$B.txt" 2>&1
    stop
  done
  if cmp -s "$R/logs/bg-ident-$art-B32.txt" "$R/logs/bg-ident-$art-B128.txt"; then
    log "identity $art: BYTE-IDENTICAL B32 vs B128"
  else
    log "identity $art: MISMATCH B32 vs B128"
  fi
done

# ---- guard 3: K at B128 — nv K=4/5/6, q8 K=3/4/5, N=4 (2 reps x 2 passes, alternated)
for r in 1 2; do
  if [ $((r % 2)) -eq 1 ]; then KS_NV="4 5 6"; KS_Q8="3 4 5"; else KS_NV="6 5 4"; KS_Q8="5 4 3"; fi
  for K in $KS_NV; do
    srv "MEMRA_SPEC_BURST=128 MEMRA_SPEC_K=$K" "q27=$NV+$DRAFT" "$R/logs/bg-k-nv-K$K-r$r.server.log" || continue
    for p in 1 2; do
      python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 4 \
        --max-tokens 128 --out "$R/logs/bg-k-points.jsonl" --label "nvB128-K$K-r$r-p$p" >> "$R/logs/bg-k-load.log" 2>&1
      log "nv B128 K$K r$r p$p: $(tail -1 $R/logs/bg-k-points.jsonl | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print("agg=%.1f" % d["agg_tok_s"])' 2>/dev/null)"
    done
    stop
  done
  for K in $KS_Q8; do
    srv "MEMRA_SPEC_BURST=128 MEMRA_SPEC_K=$K" "q27=$Q8+$DRAFT" "$R/logs/bg-k-q8-K$K-r$r.server.log" || continue
    for p in 1 2; do
      python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 4 \
        --max-tokens 128 --out "$R/logs/bg-k-points.jsonl" --label "q8B128-K$K-r$r-p$p" >> "$R/logs/bg-k-load.log" 2>&1
      log "q8 B128 K$K r$r p$p: $(tail -1 $R/logs/bg-k-points.jsonl | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print("agg=%.1f" % d["agg_tok_s"])' 2>/dev/null)"
    done
    stop
  done
done
log "BURSTGUARD_DONE"
echo BURSTGUARD_DONE

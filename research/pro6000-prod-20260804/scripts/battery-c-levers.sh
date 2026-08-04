#!/usr/bin/env bash
# pro6000-prod: Battery C — tuning levers on the 96GB card.
#   C1 MEMRA_CTX floor sweep (2048 / 8192 / 32768) at c=8, q27 daily NVFP4, spec off:
#      does the 96GB card pay anything for big KV floors, or is headroom free?
#   C2 MEMRA_ST_E4M3 decode A/B on the nvidia ST checkpoint (N=3 interleaved pairs,
#      vast §5 protocol) — J/token direction on the 600W workstation part.
#   C3 spec-serve c-crossover: spec K=3 at c=1/2/4 vs plain c=1/2/4 (where does plain
#      batching overtake the spec lane on this card — laptop crossover was c=2..4).
# 600W fixed. Single-tenant GPU.
set -u
cd /root/bw24
R=/root/receipts/levers
mkdir -p "$R"
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DRAFT=/root/models/draft-owntrim-nvfp4head-q4blk.gguf
ST=/root/models/nvidia-qwen36-27b-nvfp4
ADDR=127.0.0.1:8199
BASE=http://$ADDR

log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader; }

nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv -l 1 > "$R/gpu-1hz.csv" 2>&1 &
SMPID=$!
trap 'kill $SMPID 2>/dev/null' EXIT

start_server() { # $1 extra-env string, $2 models spec, $3 logfile
  env $1 MEMRA_MODELS="$2" MEMRA_ADDR=$ADDR target/release/memra-server > "$3" 2>&1 &
  SPID=$!
  for _ in $(seq 600); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up: $3"; tail -5 "$3"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }

# ---- C1: MEMRA_CTX floor sweep at c=8, N=3 passes per floor, server restart per floor
for ctx in 2048 8192 32768; do
  log "C1 ctx=$ctx: starting server"
  start_server "MEMRA_SERVE_SPEC=0 MEMRA_CTX=$ctx" "q27=$NV" "$R/server-ctx$ctx.log" || continue
  log "C1 ctx=$ctx up | $(gpustate)"
  for pass in 1 2 3; do
    python3 tools/load-serve.py --base $BASE --model q27 --concurrency 8 --requests 64 \
      --max-tokens 128 --out "$R/points-ctx.jsonl" --label "ctx$ctx-p$pass" >> "$R/load-ctx.log" 2>&1
    log "C1 ctx=$ctx p$pass: $(tail -1 "$R/points-ctx.jsonl" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(f"agg={d[\"agg_tok_s\"]:.1f} p50={d[\"lat_p50_s\"]:.2f} err={d[\"n_err\"]} shed={d[\"n_shed\"]}")' 2>/dev/null || echo parse-fail) | vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  done
  stop_server
done
log "C1 done"

# ---- C2: MEMRA_ST_E4M3 decode A/B, N=3 interleaved pairs (run-gen tg128 @ pp512)
if [ -f "$ST/config.json" ]; then
  for r in 1 2 3; do
    log "C2 pair $r arm A (ST default) pre: $(gpustate)"
    MEMRA_NGEN=128 MEMRA_PROMPT_FILE=research/e2e/prompts/pp512.txt timeout 3600 target/release/run-gen "$ST" > "$R/e4m3-A-r$r.log" 2>&1
    log "C2 pair $r arm A post: $(gpustate) | $(grep -oE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$R/e4m3-A-r$r.log" | head -1) | $(grep -oE 'maxdiff=[0-9.e-]+ +(MATCH|MISMATCH)' "$R/e4m3-A-r$r.log" | head -1)"
    log "C2 pair $r arm B (MEMRA_ST_E4M3=1) pre: $(gpustate)"
    MEMRA_ST_E4M3=1 MEMRA_NGEN=128 MEMRA_PROMPT_FILE=research/e2e/prompts/pp512.txt timeout 3600 target/release/run-gen "$ST" > "$R/e4m3-B-r$r.log" 2>&1
    log "C2 pair $r arm B post: $(gpustate) | $(grep -oE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$R/e4m3-B-r$r.log" | head -1) | $(grep -oE 'maxdiff=[0-9.e-]+ +(MATCH|MISMATCH)' "$R/e4m3-B-r$r.log" | head -1)"
  done
else
  log "C2 SKIPPED: ST checkpoint missing at $ST"
fi
log "C2 done"

# ---- C3: spec-vs-plain c-crossover (K=3 spec lane vs plain batch at c=1/2/4), N=3 each
log "C3: starting spec server (K=3, draft attached)"
start_server "MEMRA_SPEC_K=3" "q27=$NV+$DRAFT" "$R/server-c3-spec.log" || true
for pass in 1 2 3; do
  for c in 1 2 4; do
    python3 tools/load-serve.py --base $BASE --model q27 --concurrency $c --requests $((c*6)) \
      --max-tokens 128 --out "$R/points-c3-spec.jsonl" --label "spec-c$c-p$pass" >> "$R/load-c3.log" 2>&1
    log "C3 spec c$c p$pass: $(tail -1 "$R/points-c3-spec.jsonl" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(f"agg={d[\"agg_tok_s\"]:.1f} p50={d[\"lat_p50_s\"]:.2f}")' 2>/dev/null || echo parse-fail)"
  done
done
stop_server
log "C3: starting plain server"
start_server "MEMRA_SERVE_SPEC=0" "q27=$NV" "$R/server-c3-plain.log" || true
for pass in 1 2 3; do
  for c in 1 2 4; do
    python3 tools/load-serve.py --base $BASE --model q27 --concurrency $c --requests $((c*6)) \
      --max-tokens 128 --out "$R/points-c3-plain.jsonl" --label "plain-c$c-p$pass" >> "$R/load-c3.log" 2>&1
    log "C3 plain c$c p$pass: $(tail -1 "$R/points-c3-plain.jsonl" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(f"agg={d[\"agg_tok_s\"]:.1f} p50={d[\"lat_p50_s\"]:.2f}")' 2>/dev/null || echo parse-fail)"
  done
done
stop_server
log "BATTERY-C DONE"
echo "BATTERY-C DONE"

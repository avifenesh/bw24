#!/usr/bin/env bash
# pro6000-dev: (1) board-drift control — prod-session commit 2299ee0f binary vs dev HEAD 623ce27e
#   binary, interleaved N=3 tg128 d512 (same board, same hour -> isolates code from board).
# (2) bonus: spec-serve K=3..7 sweep on nv (serve optimum question, unswept past 5), N=2 per K.
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
cd /root/bw24
R=/root/receipts-dev/ctrl-serve
mkdir -p "$R"
M=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DRAFT=/root/models/draft-owntrim-nvfp4head-q4blk.gguf
P512=research/e2e/prompts/pp512.txt
ADDR=127.0.0.1:8199
BASE=http://$ADDR
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$R/gpu-1hz.csv" 2>&1 &
SMPID=$!
trap 'kill $SMPID 2>/dev/null' EXIT

# ---- cell 1: cross-commit control, interleaved N=3
for r in 1 2 3; do
  for arm in head ctrl; do
    BIN=/root/bw24/target/release/run-gen
    [ "$arm" = ctrl ] && BIN=/root/bw24-ctrl/target/release/run-gen
    log "ctrl-d512 $arm r$r pre: $(gpustate)"
    MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 900 $BIN $M > "$R/ctrl-d512-$arm-r$r.log" 2>&1
    log "ctrl-d512 $arm r$r post rc=$?: $(gpustate) | $(grep -oE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$R/ctrl-d512-$arm-r$r.log" | head -1) | $(grep -oE '(MATCH|MISMATCH)' "$R/ctrl-d512-$arm-r$r.log" | head -1)"
  done
done

# ---- cell 2: spec serve K=3..7 (server restart per K)
start_server() {
  env $1 MEMRA_MODELS="$2" MEMRA_ADDR=$ADDR target/release/memra-server > "$3" 2>&1 &
  SPID=$!
  for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up: $3"; tail -5 "$3"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }
for K in 3 4 5 6 7; do
  log "serve-spec K=$K: starting"
  start_server "MEMRA_SPEC_K=$K" "q27=$M+$DRAFT" "$R/server-spec-k$K.log" || continue
  for r in 1 2; do
    python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 4 \
      --max-tokens 128 --out "$R/points-spec.jsonl" --label "spec-k$K-r$r" >> "$R/load-spec.log" 2>&1
    log "serve-spec K=$K r$r: $(tail -1 "$R/points-spec.jsonl" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(f"agg={d[\"agg_tok_s\"]} p50={d[\"lat_p50_s\"]}")' 2>/dev/null || echo parse-fail)"
  done
  stop_server
done
log "SWEEP_E_DONE"
echo SWEEP_E_DONE

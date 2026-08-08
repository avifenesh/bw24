#!/usr/bin/env bash
# v0.72 battery — DRIVER E: localize the spec+PP-2 serving slowdown (train tip vs lane receipts).
# E1: ENGINE-level run-spec q9 over PP-2 (exact lane gate shape) — if this matches the lane
#     receipt speeds, the regression is in the SERVING layer, not the engine.
# E2: naked-default serve over PP-2 at c=1 — the #89 gate admits spec at c=1, so this prices
#     the real user-facing default on a PP-2 pair.
# E3: same but MEMRA_SERVE_SPEC=0 — the spec-off denominator.
set -uo pipefail
cd ~/v072/memra
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
BIN=target/release
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
RAW=$HOME/v072/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/perfE-$TS.log
ADDR=127.0.0.1:8123
BASE=http://$ADDR
wait_up() { for _ in $(seq 1 "$1"); do curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

serve_arm() { # $1=label $2="ENVK=V ..." $3=concurrency $4=requests
  local label=$1 envs=$2 c=$3 n=$4
  if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then echo "FAIL: port busy before $label"; return 1; fi
  env $envs MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$RAW/perfE-$label-server.log" 2>&1 &
  local PID=$!
  if ! wait_up 180; then echo "FAIL: $label never came up"; tail -5 "$RAW/perfE-$label-server.log"; kill $PID 2>/dev/null; return 1; fi
  grep -m1 "Engine ready" "$RAW/perfE-$label-server.log"
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency "$c" \
    --requests "$n" --max-tokens 96 --greedy --warmup 1 --label "$label" \
    --out "$RAW/perfE-points-$TS.jsonl" 2>&1 | tail -1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
}

{
echo "=== v072 DRIVER E spec+PP-2 localization $TS commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used,clocks.sm --format=csv,noheader

  echo; echo "########## E1: engine run-spec q9 dev10 PP-2 (lane gate shape: DC=0 NGEN=64) ##########"
  MEMRA_PP_DEVICES=1,0 MEMRA_PP_STAGES=2 MEMRA_QWEN_DC=0 MEMRA_NGEN=64 \
    timeout 3600 $BIN/run-spec "$Q9" 55
  echo "E1 exit=$?"

  echo; echo "########## E2: serve naked defaults PP-2 dev10, c=1 x 6 (the #89 gate admits spec here) ##########"
  serve_arm defaults-c1 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0" 1 6

  echo; echo "########## E3: serve spec-off PP-2 dev10, c=1 x 6 (denominator) ##########"
  serve_arm specoff-c1 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SERVE_SPEC=0" 1 6

  echo "--- summary ---"
  python3 - "$RAW/perfE-points-$TS.jsonl" <<'EOF'
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    print(d["label"], "c", d["concurrency"], "ok", d["n_ok"], "err", d["n_err"],
          "agg", round(d["agg_tok_s"],1), "p50", round(d["lat_p50_s"],2))
EOF
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== driverE rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

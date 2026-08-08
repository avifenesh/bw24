#!/usr/bin/env bash
# v0.72 tag-gate RE-BATTERY — R2: blocker-2 post-merge confirmation on box2.
# spec+PP-2 serving must hold the ~112 class at the MERGED TIP (fix 05ddfef2, merged
# d1abd0f3; pre-merge receipts research/v072-fix2-20260808/ read 111.7-112.0 N=3).
# Cells: dev10 naked c=1 (spec admitted by #89 at c<=2) N=3; dev10 SPEC_GATE=0 c=2 N=3;
# dev01 naked c=1 N=3 (the differentiator — head-affinity says FAST, ~111);
# controls: spec-off c=1 (~222 class), door-shut single-card c=4 (~548 class).
# One lock hold. Arms mirror research/v072-fix2-20260808/box2-fix2-verify.sh exactly.
set -uo pipefail
cd ~/memra
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
BIN=target/release
Q9=/data/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
RAW=$HOME/v072rebat/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/R2-serve-$TS.log
PTS=$RAW/R2-points-$TS.jsonl
ADDR=127.0.0.1:8123
BASE_URL=http://$ADDR

wait_up() { for _ in $(seq 1 "$1"); do curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

serve_arm() { # $1=label $2="ENVK=V ..." $3=concurrency $4=requests
  local label=$1 envs=$2 c=$3 n=$4
  if curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then echo "FAIL: port busy before $label"; return 1; fi
  env $envs MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$RAW/R2-$label-server.log" 2>&1 &
  local PID=$!
  if ! wait_up 180; then echo "FAIL: $label never came up"; tail -5 "$RAW/R2-$label-server.log"; kill $PID 2>/dev/null; return 1; fi
  grep -m1 "Engine ready" "$RAW/R2-$label-server.log"
  python3 tools/load-serve.py --base "$BASE_URL" --model q9 --concurrency "$c" \
    --requests "$n" --max-tokens 96 --greedy --warmup 1 --label "$label" \
    --out "$PTS" 2>&1 | tail -1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
}

{
echo "=== v072 REBATTERY R2 $TS commit=$(git rev-parse HEAD)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  for r in 1 2 3; do
    echo; echo "########## R2a-r$r: dev10 naked c=1 x6 — expect ~112 class ##########"
    serve_arm dev10-spec-c1-r$r "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0" 1 6
  done
  for r in 1 2 3; do
    echo; echo "########## R2b-r$r: dev10 SPEC_GATE=0 c=2 x8 — expect ~112 class ##########"
    serve_arm dev10-spec-c2-r$r "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SPEC_GATE=0" 2 8
  done
  for r in 1 2 3; do
    echo; echo "########## R2c-r$r: dev01 naked c=1 x6 — differentiator, expect ~111 ##########"
    serve_arm dev01-spec-c1-r$r "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1" 1 6
  done
  echo; echo "########## R2d: dev10 spec-OFF c=1 x6 — control, expect ~222 unchanged ##########"
  serve_arm dev10-specoff-c1 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SERVE_SPEC=0" 1 6
  echo; echo "########## R2e: door-shut single-card spec smoke c=4 x16 — expect ~548 unchanged ##########"
  serve_arm doorshut-c4 "" 4 16

  echo; echo "--- summary ---"
  python3 - "$PTS" <<'EOF'
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    print(f'{d["label"]:22s} c{d["concurrency"]} ok {d["n_ok"]:3d} err {d["n_err"]} agg {d["agg_tok_s"]:7.1f} p50 {d["lat_p50_s"]:.2f}')
EOF
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== R2 rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

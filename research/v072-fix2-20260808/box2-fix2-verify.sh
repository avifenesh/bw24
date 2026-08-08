#!/usr/bin/env bash
# v072-fix2 — box2 verification driver: spec+PP-2 serving collapse (112.5 -> 17.5).
# Two binaries, one box, one lock hold:
#   BASE = tree as found (a131e8c7 + spot-guard engine-seam checkpoint; worker.rs = the
#          5f27c55c stage-0 primary pin)  -> expect the collapse to REPRODUCE.
#   FIX  = BASE + the worker-primary-follows-HEAD-stage patch -> expect the 112 class back.
# Arms mirror the battery-E / crash-gate shapes (q9 embedded MTP, greedy, max_tokens 96).
set -uo pipefail
cd ~/memra
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
Q9=/data/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
OUT=$HOME/v072fix2; mkdir -p "$OUT"
TS=$(date -u +%Y%m%dT%H%M%SZ)
PTS=$OUT/points-$TS.jsonl
ADDR=127.0.0.1:8123
BASE_URL=http://$ADDR

wait_up() { for _ in $(seq 1 "$1"); do curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

serve_arm() { # $1=binary $2=label $3="ENVK=V ..." $4=concurrency $5=requests
  local bin=$1 label=$2 envs=$3 c=$4 n=$5
  if curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then echo "FAIL: port busy before $label"; return 1; fi
  env $envs MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    "$bin" > "$OUT/$label-server.log" 2>&1 &
  local PID=$!
  if ! wait_up 180; then echo "FAIL: $label never came up"; tail -5 "$OUT/$label-server.log"; kill $PID 2>/dev/null; return 1; fi
  grep -m1 "Engine ready" "$OUT/$label-server.log"
  python3 tools/load-serve.py --base "$BASE_URL" --model q9 --concurrency "$c" \
    --requests "$n" --max-tokens 96 --greedy --warmup 1 --label "$label" \
    --out "$PTS" 2>&1 | tail -1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
}

{
echo "=== v072-fix2 verify $TS  tree=$(git log --oneline -1 | cut -c1-60)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used,clocks.sm --format=csv,noheader

  # ---------- build BASE ----------
  git status --short | head -5
  echo "### building BASE (tree as found)"
  { cargo build --release -p memra-server \
      && cargo build --release -p memra-engine --bin run-spec; } > "$OUT/build-base.log" 2>&1 \
    || { echo "BASE BUILD FAIL"; tail -20 "$OUT/build-base.log"; exit 1; }
  cp target/release/memra-server "$OUT/memra-server-base"
  cp target/release/run-spec "$OUT/run-spec-bin"

  # ---------- build FIX ----------
  echo "### applying fix patch + building FIX"
  git apply "$OUT/fix2-worker.patch" || { echo "PATCH APPLY FAIL"; exit 1; }
  cargo build --release -p memra-server > "$OUT/build-fix.log" 2>&1 \
    || { echo "FIX BUILD FAIL"; tail -20 "$OUT/build-fix.log"; git checkout -- crates/memra-server/src/worker.rs; exit 1; }
  cp target/release/memra-server "$OUT/memra-server-fix"
  git checkout -- crates/memra-server/src/worker.rs
  echo "tree restored: $(git status --short | wc -l) dirty files (expect the pre-existing set only)"

  # ---------- BASE arms: reproduce the collapse ----------
  echo; echo "########## BASE R1: dev10 naked c=1 x6 (spec admitted at c<=2) — expect ~17.5 class ##########"
  serve_arm "$OUT/memra-server-base" base-dev10-spec-c1 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0" 1 6
  echo; echo "########## BASE R2: dev10 SPEC_GATE=0 c=2 x8 (the 112.5 gate shape) ##########"
  serve_arm "$OUT/memra-server-base" base-dev10-spec-c2 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SPEC_GATE=0" 2 8
  echo; echo "########## BASE R3: dev10 spec-OFF c=1 x6 (control ~223) ##########"
  serve_arm "$OUT/memra-server-base" base-dev10-specoff-c1 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SERVE_SPEC=0" 1 6

  # ---------- FIX arms: the return of 112 (N=3 on the headline cells) ----------
  for r in 1 2 3; do
    echo; echo "########## FIX F1r$r: dev10 naked c=1 x6 — expect ~112 class ##########"
    serve_arm "$OUT/memra-server-fix" fix-dev10-spec-c1-r$r "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0" 1 6
  done
  for r in 1 2 3; do
    echo; echo "########## FIX F2r$r: dev10 SPEC_GATE=0 c=2 x8 — expect ~112 class ##########"
    serve_arm "$OUT/memra-server-fix" fix-dev10-spec-c2-r$r "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SPEC_GATE=0" 2 8
  done
  echo; echo "########## FIX F3: dev10 spec-OFF c=1 x6 — expect ~223 unchanged ##########"
  serve_arm "$OUT/memra-server-fix" fix-dev10-specoff-c1 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SERVE_SPEC=0" 1 6
  echo; echo "########## FIX F4: door-shut single-card spec smoke c=4 x16 — expect ~548 unchanged ##########"
  serve_arm "$OUT/memra-server-fix" fix-doorshut-c4 "" 4 16
  echo; echo "########## FIX F5: dev01 naked c=1 x6 — THEORY DIFFERENTIATOR: was always slow, head-affinity says now FAST ##########"
  serve_arm "$OUT/memra-server-fix" fix-dev01-spec-c1 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1" 1 6
  echo; echo "########## FIX F6: #87 quick crash gate — dev10 SPEC_GATE=0 c=4 x50 — expect 50/50 clean ##########"
  serve_arm "$OUT/memra-server-fix" fix-dev10-crash-c4 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SPEC_GATE=0" 4 50
  grep -ciE "illegal|sentinel|panic|CUDA_ERROR" "$OUT/fix-dev10-crash-c4-server.log" | xargs echo "crash-gate fault-line count (want 0):"

  # ---------- engine gate: run-spec K=1..8 over PP-2 (server change can't touch it; bar anyway) ----------
  echo; echo "########## run-spec q9 dev10 PP-2 (lane gate shape DC=0 NGEN=64) ##########"
  MEMRA_PP_DEVICES=1,0 MEMRA_PP_STAGES=2 MEMRA_QWEN_DC=0 MEMRA_NGEN=64 \
    timeout 3600 "$OUT/run-spec-bin" "$Q9" 55 > "$OUT/runspec-dev10.log" 2>&1
  echo "run-spec exit=$?"; grep -E "SELF-CONSISTENCY|K=| PASS|FAIL" "$OUT/runspec-dev10.log" | tail -12

  echo; echo "--- summary ---"
  python3 - "$PTS" <<'EOF'
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    print(f'{d["label"]:26s} c{d["concurrency"]} ok {d["n_ok"]:3d} err {d["n_err"]} agg {d["agg_tok_s"]:7.1f} p50 {d["lat_p50_s"]:.2f}')
EOF
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$OUT/driver-$TS.log" 2>&1
echo "LOG=$OUT/driver-$TS.log"

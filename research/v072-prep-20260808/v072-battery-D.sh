#!/usr/bin/env bash
# v0.72 battery — DRIVER D: placement A/B on the NEW train binary ONLY, to attribute the
# crash-gate throughput anomaly (17.5 agg on this build vs the pp2spec lane's 112 receipts,
# same shape). NO quarantine override anywhere: the train tip has #87 lifted, spec+PP-2
# serves naked. Discriminator: the 5f27c55c merge makes the serving worker FOLLOW the PP
# primary device (boot line "Engine ready (device=1)" on dev10) — if dev01 placement
# restores the lane-receipt class, the regression is that merge's placement interaction.
# Interleaved x2 per arm in ONE lock hold. Also one door-shut c4x16 smoke (548 tok/s class).
set -uo pipefail
cd ~/v072/memra
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
BIN=target/release
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
RAW=$HOME/v072/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/perfD-$TS.log
ADDR=127.0.0.1:8123
BASE=http://$ADDR
wait_up() { for _ in $(seq 1 "$1"); do curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

one_arm() { # $1=label $2="ENVK=V ..." $3=concurrency $4=requests
  local label=$1 envs=$2 c=$3 n=$4
  if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then echo "FAIL: port busy before $label"; return 1; fi
  env $envs MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$RAW/perfD-$label-server.log" 2>&1 &
  local PID=$!
  if ! wait_up 180; then echo "FAIL: $label never came up"; tail -5 "$RAW/perfD-$label-server.log"; kill $PID 2>/dev/null; return 1; fi
  grep -m1 "Engine ready" "$RAW/perfD-$label-server.log"
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency "$c" \
    --requests "$n" --max-tokens 96 --greedy --warmup 1 --label "$label" \
    --out "$RAW/perfD-points-$TS.jsonl" 2>&1 | tail -1
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
}

{
echo "=== v072 DRIVER D placement A/B $TS commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used,clocks.sm,temperature.gpu --format=csv,noheader

  # interleaved: dev10 (the crash-gate repro placement) vs dev01, spec forced on (the gate shape)
  one_arm dev10-spec-r1 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SPEC_GATE=0" 2 8
  one_arm dev01-spec-r1 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_SPEC_GATE=0" 2 8
  one_arm dev10-spec-r2 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SPEC_GATE=0" 2 8
  one_arm dev01-spec-r2 "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_SPEC_GATE=0" 2 8
  # defaults control on the same placement (#89 gate live, spec demotes at c>=2)
  one_arm dev10-defaults "MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0" 4 16
  # door-shut single-card smoke (the 548 tok/s receipt class)
  one_arm doorshut-smoke "" 4 16

  echo "--- summary ---"
  python3 - "$RAW/perfD-points-$TS.jsonl" <<'EOF'
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    print(d["label"], "c", d["concurrency"], "ok", d["n_ok"], "err", d["n_err"],
          "agg", round(d["agg_tok_s"],1), "p50", round(d["lat_p50_s"],2))
EOF
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== driverD rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

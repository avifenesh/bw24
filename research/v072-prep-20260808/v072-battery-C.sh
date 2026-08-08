#!/usr/bin/env bash
# v0.72 pair-box battery — DRIVER C: the #87 crash gate (spec serve over PP-2, c=4 >=100
# requests, 0 crashes — the fences' regression test), ppspec bit-identity, and the
# serve-smoke full battery (incl. gemma4 arm) on the box-built server.
set -uo pipefail
cd ~/v072/memra
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
BIN=target/release
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q9D=/scratch-models/draft-9b-owntrim-nvfp4head-q4blk.gguf
GEMMA=/scratch-models/gemma-4-12b-it-qat-q4_0.gguf
RAW=$HOME/v072/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/serveC-$TS.log
ADDR=127.0.0.1:8123
BASE=http://$ADDR
FAILS=0
wait_up() { for _ in $(seq 1 "$1"); do curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
{
echo "=== v072 battery DRIVER C $TS commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 21600 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## C1: ppspec bit-identity dev01 (fence must not move bytes) ##########"
  MEMRA_PP_DEVICES=0,1 timeout 2400 $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
  echo "=== C1 rc=$?"
  echo; echo "########## C2: ppspec bit-identity dev10 ##########"
  MEMRA_PP_DEVICES=1,0 timeout 2400 $BIN/decode-batch-gate "$Q9" --mode ppspec --stages 2 --steps 16 --ts 2,5,9 --reps 3
  echo "=== C2 rc=$?"

  echo; echo "########## C3: THE #87 CRASH GATE — spec serve over PP-2, c=4 x 100 + c=8 x 104 ##########"
  # Post-lift tree: no MEMRA_PP2SPEC_UNQUARANTINE (flag removed with the quarantine).
  # MEMRA_SPEC_GATE=0 forces always-spec so c=4/c=8 exercise the spec+PP-2 path (the
  # #89 gate would otherwise demote spec at these concurrencies and hide the crash class).
  if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
    echo "FAIL: something already serving $ADDR"
  else
    env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SPEC_GATE=0 \
      MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
      $BIN/memra-server > "$RAW/crash-server-$TS.log" 2>&1 &
    PID=$!
    if ! wait_up 180; then echo "FAIL: crash-gate server never came up"; tail -20 "$RAW/crash-server-$TS.log"; kill $PID 2>/dev/null
    else
      python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
        --requests 8 --max-tokens 96 --greedy --warmup 1 --label v072-c2 \
        --out "$RAW/crash-points-$TS.jsonl" 2>&1
      echo "=== c2 warmup rc=$?"
      python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
        --requests 100 --max-tokens 96 --greedy --warmup 0 --label v072-c4x100 \
        --out "$RAW/crash-points-$TS.jsonl" 2>&1
      echo "=== c4x100 rc=$?"
      python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 8 \
        --requests 104 --max-tokens 96 --greedy --warmup 0 --label v072-c8x104 \
        --out "$RAW/crash-points-$TS.jsonl" 2>&1
      echo "=== c8x104 rc=$?"
      kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
      if grep -qiE "sentinel|illegal|panicked|CUDA_ERROR" "$RAW/crash-server-$TS.log"; then
        echo "CRASH-GATE FAIL: sentinel/illegal/panic lines in server log:"
        grep -inE "sentinel|illegal|panicked|CUDA_ERROR" "$RAW/crash-server-$TS.log" | head -5
      else
        echo "CRASH-GATE server log clean (no sentinel/illegal/panic/CUDA_ERROR)"
      fi
      echo "--- crash points summary ---"
      python3 - "$RAW/crash-points-$TS.jsonl" <<'EOF'
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    print(d["label"], "ok", d["n_ok"], "err", d["n_err"], "agg", round(d.get("agg_tok_s",0),1))
EOF
    fi
  fi

  echo; echo "########## C4: serve-smoke FULL battery (q9+draft, incl gemma4 arm) ##########"
  GEMMA_MODEL="$GEMMA" timeout 7200 tools/serve-smoke.sh "$Q9" "$Q9D"
  echo "=== C4 serve-smoke rc=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== driverC rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

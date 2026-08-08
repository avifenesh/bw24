#!/usr/bin/env bash
# v0.72 tag-gate RE-BATTERY — R3: fast regression smoke layer on box2 at 5ad87a63.
# kernel-check (step35 + q27), run-gen argmax both topologies (step35 PP-2: baseline
# argmax 6776; q27 single-card: baseline argmax 1178 — battery A3/A4 receipts),
# #87 crash gate c=4 x50 (fault-grep must be 0), serve-smoke FULL battery (incl gemma4).
# One lock hold.
set -uo pipefail
cd ~/memra
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
BIN=target/release
STEP=/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
Q27=/data/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q9=/data/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
Q9D=/data/models/draft-9b-owntrim-nvfp4head-q4blk.gguf
GEMMA=/data/models/gemma-4-12b-it-qat-q4_0.gguf
RAW=$HOME/v072rebat/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/R3-smoke-$TS.log
ADDR=127.0.0.1:8123
BASE=http://$ADDR
P="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
wait_up() { for _ in $(seq 1 "$1"); do curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
{
echo "=== v072 REBATTERY R3 $TS commit=$(git rev-parse HEAD)"
(
  flock -w 21600 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## R3a: kernel-check model-backed step35 IQ4_XS ##########"
  timeout 3600 $BIN/kernel-check "$STEP"
  echo "=== R3a rc=$?"

  echo; echo "########## R3b: kernel-check model-backed q27 NVFP4 ##########"
  timeout 3600 $BIN/kernel-check "$Q27"
  echo "=== R3b rc=$?"

  echo; echo "########## R3c: run-gen argmax gate, step35 PP-2, 64 tok (baseline argmax 6776) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_NGEN=64 timeout 2400 \
    $BIN/run-gen "$STEP" --prompt "$P"
  echo "=== R3c rc=$?"

  echo; echo "########## R3d: run-gen argmax gate, q27 single-card, naked (baseline argmax 1178) ##########"
  MEMRA_NGEN=64 timeout 2400 $BIN/run-gen "$Q27" 55
  echo "=== R3d rc=$?"

  echo; echo "########## R3e: #87 crash gate — dev10 SPEC_GATE=0 c=4 x50, fault-grep must be 0 ##########"
  if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
    echo "FAIL: something already serving $ADDR"
  else
    env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SPEC_GATE=0 \
      MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
      $BIN/memra-server > "$RAW/R3-crash-server-$TS.log" 2>&1 &
    PID=$!
    if ! wait_up 180; then echo "FAIL: crash-gate server never came up"; tail -20 "$RAW/R3-crash-server-$TS.log"; kill $PID 2>/dev/null
    else
      python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
        --requests 50 --max-tokens 96 --greedy --warmup 1 --label rebat-c4x50 \
        --out "$RAW/R3-crash-points-$TS.jsonl" 2>&1
      echo "=== c4x50 rc=$?"
      kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
      if grep -qiE "sentinel|illegal|panicked|CUDA_ERROR" "$RAW/R3-crash-server-$TS.log"; then
        echo "CRASH-GATE FAIL: fault lines in server log:"
        grep -inE "sentinel|illegal|panicked|CUDA_ERROR" "$RAW/R3-crash-server-$TS.log" | head -5
      else
        echo "CRASH-GATE server log clean (no sentinel/illegal/panic/CUDA_ERROR)"
      fi
      python3 - "$RAW/R3-crash-points-$TS.jsonl" <<'EOF'
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    print(d["label"], "ok", d["n_ok"], "err", d["n_err"], "agg", round(d.get("agg_tok_s",0),1))
EOF
    fi
  fi

  echo; echo "########## R3f: serve-smoke FULL battery (q9+draft, incl gemma4 arm) ##########"
  GEMMA_MODEL="$GEMMA" timeout 7200 tools/serve-smoke.sh "$Q9" "$Q9D"
  echo "=== R3f serve-smoke rc=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== R3 rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

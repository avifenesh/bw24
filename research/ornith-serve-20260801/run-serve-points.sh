#!/bin/bash
# ornith-serve-bench: serve load points for the 3 onboarded models (2026-08-01).
# Protocol: memra-server single replica, naked defaults first; per config-rep one server
# session under flock /tmp/gpu5090.lock; points c=1/8/16 (requests = 4x concurrency, min 8;
# max_tokens 128) via tools/load-serve.py; N=3 INTERLEAVED passes (pass loop outside the
# config loop, so each model's reps are spread across the session/thermal window).
# Configs (naked-first):
#   o9b-naked  Ornith-1.0-9B Q8_0, no env        -> expect chunk cap 8 (Q8_0 lacks rp4 naked)
#   o9b-q8rp   Ornith-1.0-9B Q8_0, MEMRA_Q8RP=1  -> expect chunk cap 16 (exact-16 tier)
#   o35b-naked Ornith-1.0-35B Q4_K_M, no env     -> MoE FFN disqualifies exact-16 -> cap 8
#   kat-naked  KAT-Coder-V2.5 IQ4_XS, no env     -> MoE FFN disqualifies exact-16 -> cap 8
# Engagement evidence: the server log's "[worker] m: decode chunk cap N" line per session.
set -u
W=/home/avifenesh/projects/wt-ornith-serve-bench
R=$W/research/ornith-serve-20260801
PORT=8098
BASE=http://127.0.0.1:$PORT

O9B=/data/ai-ml/hf-models/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf
O35B=/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf
KAT=/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf

therm() { # label
  echo "$(date -u +%FT%TZ) $1 $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader)" >> "$R/thermal.log"
}

run_cfg_rep() { # cfg model_path env_kv rep
  local cfg=$1 mpath=$2 envv=$3 rep=$4
  (
    flock 9
    env $envv MEMRA_MODELS="m=$mpath" MEMRA_ADDR=127.0.0.1:$PORT \
      "$W/target/release/memra-server" > "$R/server-$cfg-rep$rep.log" 2>&1 &
    SRV=$!
    local up=0
    for _ in $(seq 1 300); do curl -s $BASE/health >/dev/null 2>&1 && { up=1; break; }; sleep 1; done
    if [ "$up" != 1 ]; then
      echo "SERVER FAILED $cfg rep$rep" | tee -a "$R/failures.log"
      kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; exit 1
    fi
    grep "decode chunk cap" "$R/server-$cfg-rep$rep.log" | tee -a "$R/chunk-cap.log" | sed "s/^/[$cfg rep$rep] /"
    for c in 1 8 16; do
      therm "$cfg-c$c-rep$rep pre"
      python3 "$W/tools/load-serve.py" --base $BASE --model m --concurrency $c \
        --max-tokens 128 --label "$cfg-c$c-rep$rep" \
        --out "$R/serve-points.jsonl" --per-request "$R/serve-per-request.jsonl" \
        2>&1 | tail -1
      therm "$cfg-c$c-rep$rep post"
    done
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  ) 9>/tmp/gpu5090.lock
}

for rep in 1 2 3; do
  echo "=== PASS $rep $(date -u +%FT%TZ) ==="
  run_cfg_rep o9b-naked  "$O9B"  ""            $rep
  run_cfg_rep o9b-q8rp   "$O9B"  "MEMRA_Q8RP=1" $rep
  run_cfg_rep o35b-naked "$O35B" ""            $rep
  run_cfg_rep kat-naked  "$KAT"  ""            $rep
done
echo "SERVE-POINTS-DONE $(date -u +%FT%TZ)"

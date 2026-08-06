#!/usr/bin/env bash
# spec-scaling STEP 2 — the MECHANISM RECEIPT: reproduce the flat-spec c-ladder on ONE card and
# name the serialization point with the engine's own per-tick numbers.
#
# THE FINDING BEING REPRODUCED (research/pp2-spec-20260806/RESULTS.md, arm A vs arm C): spec-mode
# serve throughput is FLAT c=1 -> c=8 (346.5 -> 345.2 agg tok/s, N=5, every rep within 1 tok/s)
# while spec-OFF scales 3.9x (223.7 -> 872.9). That was measured on the PRO 6000 pair with the
# door SHUT for arm A, i.e. it is a single-card property. This harness reproduces it on the 5090
# with q9 + its production regime drafter, and adds the ladder rungs the predecessor lane did not
# take (c=2, c=4) so the shape of the curve — flat, not merely equal at the endpoints — is on the
# record rather than inferred from two points.
#
# WHY THE 5090 IS THE RIGHT RIG FOR THIS ONE. The predecessor lane pushed with MEMRA_SKIP_PERF_CI
# because its subject was a two-card stage split that has no 5090 measurement. This lane's subject
# is the opposite: the flat curve is present with the pp door SHUT, on ONE card, so the 5090 is not
# a proxy here — it is a valid rig for the phenomenon, and it is also the default-flip gate rig for
# anything this lane might change in the serving scheduler.
#
# ARMS (rep-major interleave; server restarts per arm per rep; order alternates by rep parity so a
# monotone thermal drift cannot favour one arm):
#   S  spec ON  (naked default — MEMRA_SERVE_SPEC unset, q9 + regime drafter attached)
#   N  spec OFF (MEMRA_SERVE_SPEC=0 — the batched-decode denominator)
# Each arm walks c=1,2,4,8. GREEDY (temperature 0) throughout: the greedy spec arm is the exact
# one, so the ladder is not confounded by sampled acceptance variance.
#
# DIAGNOSTICS captured per arm, because the mechanism claim needs the engine's own numbers and not
# just the aggregate:
#   MEMRA_TICK_TRACE=1  -> per-tick `[tick] act=N int=N priming=N ready=N decode_ms=X` lines. On
#                          the spec arm `ready` is the BATCHED-decode row count, which is the
#                          direct observable for the hypothesis: if spec sessions never enter the
#                          batched path, ready stays 0 at every concurrency while act rises to 8.
#   MEMRA_SPEC_PHASE=1  -> the per-burst `[spec-phase]` decomposition (draft / verify / wait /
#                          rest) printed at every burst end. Per-session round cost under
#                          concurrency vs solo is what separates "sessions serialize" from
#                          "sessions contend".
# Both are diagnostics-only (no kernel change, no extra syncs on the tick path); MEMRA_SPEC_PHASE
# does add per-phase Instant reads inside the round loop, so the PERF numbers come from the arms
# WITHOUT it and the phase arms are a separate, clearly-labelled pass (see run-phase.sh).
#
# Model: q9 NVFP4+MTP + draft-9b-owntrim-nvfp4head-q4blk (the accept-gate q9 cell's production
# drafter — attached via MEMRA_MODELS "+draft", which REPLACES the embedded head at load). The
# embedded head alone would be a bare-head config, and the acceptance sign follows
# (model x drafter x prompt) per the accept-gate law, so the served config gets the served drafter.
#
# GPU window: the caller holds flock /tmp/memra-5090.lock. 24 GB card with the owner's resident
# llama-server on it, so ctx is 4096 and MEMRA_MAX_SESSIONS is left at its default.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PATH=$HOME/.cargo/bin:$PATH

OUT="$(dirname "$0")/logs/cladder"
mkdir -p "$OUT"
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
BIN=target/release
PORT=8317
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
REPS="${REPS:-3}"
CS="${CS:-1 2 4 8}"

[ -f "$Q9" ] || { echo "SKIP: missing $Q9"; exit 0; }
[ -f "$DRAFT" ] || { echo "SKIP: missing $DRAFT"; exit 0; }
[ -x "$BIN/memra-server" ] || { echo "FAIL: build target/release/memra-server first"; exit 1; }

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-pre.csv"

serve_arm() { # $1 = label, $2 = rep, $3.. = extra env words
  local label="$1" rep="$2"; shift 2
  local log="$OUT/r$rep-$label"
  # STALE-LISTENER GUARD (inherited from the predecessor lane, learned the hard way): a foreign
  # responder on our port answers /v1/models and gets measured as the model under test. Hard
  # abort, never a wait — we cannot prove the responder is ours.
  if ss -tln 2>/dev/null | grep -q "[:.]$PORT "; then
    echo "FAIL: $label rep$rep — port $PORT already LISTENing; refusing to measure against it."
    ss -tlnp 2>/dev/null | grep "[:.]$PORT " | sed 's/^/    /'
    return 1
  fi
  env "$@" MEMRA_MODELS="q9=$Q9+$DRAFT" MEMRA_ADDR=$ADDR MEMRA_CTX=4096 \
    MEMRA_SPEC_K=3 MEMRA_TICK_TRACE=1 \
    $BIN/memra-server > "$log-server.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && break
    kill -0 $pid 2>/dev/null || break
    sleep 2
  done
  if ! curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
    echo "FAIL: $label rep$rep server never came up"; tail -20 "$log-server.log"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
  fi
  # Ownership check on top of the pre-flight guard (accept-gate's belt-and-braces).
  if ! ss -tlnp 2>/dev/null | grep "[:.]$PORT " | grep -q "pid=$pid,"; then
    echo "FAIL: $label rep$rep — $PORT answers but is NOT our child (pid $pid)"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
  fi
  for c in $CS; do
    python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency $c \
      --requests $((c * 4)) --max-tokens 128 --greedy --warmup 1 \
      --label "$label-c$c-r$rep" --out "$OUT/points.jsonl" \
      > "$log-c$c.log" 2>&1
  done
  curl -sf "$BASE/metrics" > "$log-metrics.txt" 2>&1 || true
  kill $pid 2>/dev/null
  local left=60
  while [ "$left" -gt 0 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; left=$((left-1)); done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 3
  return 0
}

for r in $(seq 1 "$REPS"); do
  echo "=== rep $r ==="
  if [ $((r % 2)) -eq 1 ]; then ORDER="S N"; else ORDER="N S"; fi
  for a in $ORDER; do
    case $a in
      S) echo "-- rep $r arm S: spec ON (naked default) --"
         serve_arm S-spec $r MEMRA_SPECSCALE_ARM=S ;;
      N) echo "-- rep $r arm N: spec OFF --"
         serve_arm N-nospec $r MEMRA_SERVE_SPEC=0 ;;
    esac
  done
done

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-post.csv"

echo "==== raw load points ===="
cat "$OUT/points.jsonl"
echo CLADDER_DONE

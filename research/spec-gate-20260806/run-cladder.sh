#!/usr/bin/env bash
# spec-gate MEASUREMENT 1 — the three-arm c-ladder: does the GATE track the better of the two
# fixed policies at every concurrency?
#
# THE CLAIM UNDER TEST. `research/spec-scaling-20260806/RESULTS.md` measured spec ON at 1.82x
# (c=1), 1.13x (c=2), 0.65x (c=4), 0.47x (c=8) against spec OFF, i.e. spec is a WIN at low
# concurrency and a LOSS at high, because the spec path is a serial queue and no batched verify
# exists (that fix is REFUTED by a 16-column exact-kernel width ceiling, not merely untried). This
# lane's gate is the policy answer: admit spec only while active <= LOW=2, demote live spec
# sessions once active >= HIGH=4. If the policy is right, the GATED arm tracks ALWAYS-SPEC at
# c=1-2 and NEVER-SPEC at c=4-8 — an upper envelope, not a compromise.
#
# ARMS (rep-major interleave; server restart per arm per rep; order ROTATES by rep so no arm
# always sits at the same point of a monotone thermal drift):
#   G  gated      — the lane default (naked: gate ON, LOW=2, HIGH=4)
#   S  always-spec— MEMRA_SPEC_GATE=0, the pre-lane behavior and the ceiling at low c
#   N  never-spec — MEMRA_SERVE_SPEC=0, the batched denominator and the ceiling at high c
#
# WHY N=5 AND WHY REP-MAJOR. The H100-lane law: cross-run and cross-day comparisons are
# clock-drift-invalid, so every arm is re-measured inside each rep and only same-rep numbers are
# compared. N=5 reps x 4 rungs x 3 arms = 60 load points.
#
# GREEDY throughout (temperature 0): the greedy spec path is the exact one, so the ladder is not
# confounded by sampled acceptance variance, and it is the regime the exactness proof covers.
#
# Model: q9 NVFP4+MTP + draft-9b-owntrim-nvfp4head-q4blk — the accept-gate q9 cell's PRODUCTION
# drafter, attached via MEMRA_MODELS "+draft" so it REPLACES the embedded head at load (a bare
# embedded head is a different acceptance regime, and acceptance follows model x drafter x prompt).
#
# GPU window: the caller holds flock /tmp/memra-5090.lock. 24 GB card shared with the owner's
# resident llama-server, so MEMRA_CTX=4096.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PATH=$HOME/.cargo/bin:$PATH

OUT="$(dirname "$0")/logs/cladder"
mkdir -p "$OUT"
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
BIN=target/release
PORT=8318
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
REPS="${REPS:-5}"
CS="${CS:-1 2 4 8}"

[ -f "$Q9" ] || { echo "SKIP: missing $Q9"; exit 0; }
[ -f "$DRAFT" ] || { echo "SKIP: missing $DRAFT"; exit 0; }
[ -x "$BIN/memra-server" ] || { echo "FAIL: build target/release/memra-server first"; exit 1; }

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-pre.csv"

serve_arm() { # $1 = label, $2 = rep, $3.. = extra env words
  local label="$1" rep="$2"; shift 2
  local log="$OUT/r$rep-$label"
  # STALE-LISTENER GUARD: a foreign responder on our port answers /v1/models and gets measured as
  # the model under test. Hard abort, never a wait — we cannot prove the responder is ours.
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
  # Ownership check on top of the pre-flight guard.
  if ! ss -tlnp 2>/dev/null | grep "[:.]$PORT " | grep -q "pid=$pid,"; then
    echo "FAIL: $label rep$rep — $PORT answers but is NOT our child (pid $pid)"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
  fi
  for c in $CS; do
    # --stream is REQUIRED here, not cosmetic: the brief's metric is TTFT p50/p95, and with
    # `stream: False` a client has no first-token timestamp at all, so a policy that delays the
    # first token but not the last would measure as neutral.
    python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency $c \
      --requests $((c * 4)) --max-tokens 128 --greedy --warmup 1 --stream \
      --label "$label-c$c-r$rep" --out "$OUT/points.jsonl" \
      --per-request "$OUT/per-request.jsonl" \
      > "$log-c$c.log" 2>&1
  done
  curl -sf "$BASE/metrics" > "$log-metrics.txt" 2>&1 || true
  # DEMOTION COUNT per arm: the gate's own observable, and the thrash check's raw material.
  grep -c "spec-gate\] demoted" "$log-server.log" > "$log-demotions.txt" 2>/dev/null || true
  kill $pid 2>/dev/null
  local left=60
  while [ "$left" -gt 0 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; left=$((left-1)); done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 3
  return 0
}

for r in $(seq 1 "$REPS"); do
  echo "=== rep $r ==="
  # rotate the 3-arm order so each arm visits each position across reps
  case $((r % 3)) in
    1) ORDER="G S N" ;;
    2) ORDER="S N G" ;;
    0) ORDER="N G S" ;;
  esac
  for a in $ORDER; do
    case $a in
      G) echo "-- rep $r arm G: GATED (lane default, LOW=2 HIGH=4) --"
         serve_arm G-gated $r MEMRA_SPECGATE_ARM=G ;;
      S) echo "-- rep $r arm S: always-spec (gate off) --"
         serve_arm S-spec $r MEMRA_SPEC_GATE=0 ;;
      N) echo "-- rep $r arm N: never-spec --"
         serve_arm N-nospec $r MEMRA_SERVE_SPEC=0 ;;
    esac
  done
done

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-post.csv"

echo "==== raw load points ===="
cat "$OUT/points.jsonl"
echo CLADDER_DONE

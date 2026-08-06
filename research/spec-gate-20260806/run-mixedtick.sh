#!/usr/bin/env bash
# spec-gate MEASUREMENT 2 — the MIXED TICK: does a serial spec burst starve the batched rows?
#
# THE UNMEASURED INTERACTION (named as a blocker by the predecessor lane,
# research/spec-scaling-20260806/RESULTS.md §6.2, quoted): "Phase (a) runs whole bursts before
# phase (c) is reached, so 2 spec sessions holding ~21 ms of serial burst per tick would inflate
# the batched rows' TTFT and inter-token latency. That interaction has no receipt in this lane and
# a policy shipped without it would be a latency regression dressed as a throughput fix."
#
# This lane's gate CREATES that mixed state deliberately: sessions admitted while the box is quiet
# hold the spec path, later arrivals are admitted batched, and both run in the same tick. So the
# interaction has to be measured or the policy is unshipped.
#
# THE SHAPE. Every arm runs c=6 — above HIGH=4, so the gate is fully engaged — and the arms differ
# only in whether a spec session is present in the tick and whether its burst is bounded:
#
#   B  baseline      MEMRA_SERVE_SPEC=0. No spec anywhere; pure batched c=6. The floor for what
#                    batched TTFT costs at this concurrency with no burst in the tick.
#   M  mixed         gate ON with HIGH raised ABOVE the load (HIGH=99) so demotion NEVER fires and
#                    LOW=2 so only the first arrivals hold spec. This is the WORST-CASE mixed tick:
#                    spec sessions burst serially forever while batched rows wait behind them.
#                    It is deliberately NOT the shipped default — it isolates the interaction the
#                    default exists to avoid.
#   Q  mixed+bounded same as M but MEMRA_SPEC_BURST=4 instead of the default 32. If the mixed-tick
#                    penalty is real, shrinking the burst quantum is the in-scope mitigation the
#                    brief names; this arm says whether it works and what it costs the spec rows.
#   G  gated default the shipped policy (naked). Demotion fires at HIGH=4, so the mixed state is
#                    transient by construction. The number that actually ships.
#
# READ IT AS: B is the batched-TTFT floor, M is the unmitigated cost of a mixed tick, Q is whether
# burst bounding recovers it, G is what a client sees under the default. If G ~ B on TTFT, the
# policy has closed the interaction and burst bounding is unnecessary; if G >> B, Q's result
# decides whether MEMRA_SPEC_BURST becomes part of the shipped policy.
#
# N=5 rep-major, arm order rotates by rep. GREEDY. TTFT requires --stream.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PATH=$HOME/.cargo/bin:$PATH

OUT="$(dirname "$0")/logs/mixedtick"
mkdir -p "$OUT"
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
DRAFT=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf
BIN=target/release
PORT=8318
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
REPS="${REPS:-5}"
C="${C:-6}"

[ -f "$Q9" ] || { echo "SKIP: missing $Q9"; exit 0; }
[ -f "$DRAFT" ] || { echo "SKIP: missing $DRAFT"; exit 0; }
[ -x "$BIN/memra-server" ] || { echo "FAIL: build target/release/memra-server first"; exit 1; }

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-pre.csv"

serve_arm() { # $1 = label, $2 = rep, $3.. = extra env words
  local label="$1" rep="$2"; shift 2
  local log="$OUT/r$rep-$label"
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
  if ! ss -tlnp 2>/dev/null | grep "[:.]$PORT " | grep -q "pid=$pid,"; then
    echo "FAIL: $label rep$rep — $PORT answers but is NOT our child (pid $pid)"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
  fi
  # 512-token generations: long enough that the tick reaches a steady mixed state rather than
  # measuring only the admission transient.
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency "$C" \
    --requests $((C * 3)) --max-tokens 512 --greedy --warmup 1 --stream \
    --label "$label-c$C-r$rep" --out "$OUT/points.jsonl" \
    --per-request "$OUT/per-request.jsonl" \
    > "$log-c$C.log" 2>&1
  curl -sf "$BASE/metrics" > "$log-metrics.txt" 2>&1 || true
  grep -c "spec-gate\] demoted" "$log-server.log" > "$log-demotions.txt" 2>/dev/null || true
  # tick shape: how many ticks carried BOTH a spec session and batched-ready rows — the mixed
  # tick's own observable, straight out of MEMRA_TICK_TRACE.
  awk '/^\[tick\]/ {
         spec=0; ready=0;
         for (i=1;i<=NF;i++) { if ($i ~ /^spec=/) { sub(/^spec=/,"",$i); spec=$i+0 }
                               if ($i ~ /^ready=/) { sub(/^ready=/,"",$i); ready=$i+0 } }
         n++; if (spec>0 && ready>0) mixed++; if (spec>0) sp++; if (ready>0) rd++
       }
       END { printf "ticks=%d spec_present=%d ready_present=%d MIXED=%d\n", n, sp, rd, mixed }' \
    "$log-server.log" > "$log-tickshape.txt" 2>/dev/null || true
  kill $pid 2>/dev/null
  local left=60
  while [ "$left" -gt 0 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; left=$((left-1)); done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 3
  return 0
}

for r in $(seq 1 "$REPS"); do
  echo "=== rep $r ==="
  case $((r % 4)) in
    1) ORDER="B M Q G" ;;
    2) ORDER="M Q G B" ;;
    3) ORDER="Q G B M" ;;
    0) ORDER="G B M Q" ;;
  esac
  for a in $ORDER; do
    case $a in
      B) echo "-- rep $r arm B: baseline, no spec at all --"
         serve_arm B-batched $r MEMRA_SERVE_SPEC=0 ;;
      M) echo "-- rep $r arm M: MIXED, demotion disabled (HIGH=99), burst default 32 --"
         serve_arm M-mixed $r MEMRA_SPEC_GATE_LOW=2 MEMRA_SPEC_GATE_HIGH=99 ;;
      Q) echo "-- rep $r arm Q: MIXED + bounded burst (SPEC_BURST=4) --"
         serve_arm Q-bounded $r MEMRA_SPEC_GATE_LOW=2 MEMRA_SPEC_GATE_HIGH=99 \
                                MEMRA_SPEC_BURST=4 ;;
      G) echo "-- rep $r arm G: shipped default (gate ON, LOW=2 HIGH=4) --"
         serve_arm G-gated $r MEMRA_SPECGATE_ARM=G ;;
    esac
  done
done

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-post.csv"

echo "==== raw load points ===="
cat "$OUT/points.jsonl"
echo "==== tick shapes ===="
for f in "$OUT"/*-tickshape.txt; do [ -f "$f" ] && echo "$(basename "$f"): $(cat "$f")"; done
echo MIXEDTICK_DONE

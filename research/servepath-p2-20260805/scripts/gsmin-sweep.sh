#!/usr/bin/env bash
# H1 CROSSOVER SWEEP — the serve GraphSession promotion key (MEMRA_GS_MIN).
#
# The worker ALREADY has a B=1 graph door (worker.rs phase a0, MEMRA_SERVE_GS). It is gated
# shut at the measured serve config by MEMRA_GS_MIN=384 — an amortization ESTIMATE
# ("~340ms capture = ~330-token break-even at the 1.03ms/tok graph saving"), never a measured
# crossover. Same class as lever 2's 256 key. This sweeps the real crossover.
#
# Protocol: arms interleaved WITHIN each rep, order alternated ACROSS reps, server restarted
# per arm, warmup point discarded (the post-build/boot cold-start law). Harness = memra-server
# + load-serve.py (never decode-batch-bench).
#
# Usage: gsmin-sweep.sh <model.gguf> <nreps> [mt list...]
set -uo pipefail
cd "$(dirname "$0")/../../.."

MODEL=${1:?model.gguf}
REPS=${2:-3}
shift 2 || true
MTS=("${@:-64 128 256 512}")
# shellcheck disable=SC2206
MTS=(${MTS[*]})

OUT=research/servepath-p2-20260805
L=$OUT/logs
P=$OUT/scripts/parse.py
mkdir -p "$L"
ADDR=127.0.0.1:8188
BASE=http://$ADDR
STEM=$(basename "$MODEL" .gguf)
# arm A = door SHUT (the shipped 384 key, effectively closed at mt<384)
# arm B = door OPEN (promote every solo greedy session)
GS_A=100000
GS_B=1
# spec OFF: the plain-decode path is what this key gates (an MTP model would otherwise
# route greedy solo sessions into the spec burst path, which never graph-promotes).
export MEMRA_SERVE_SPEC=0

run_arm() {  # $1=arm tag, $2=GS_MIN, $3=rep
  local tag=$1 gs=$2 rep=$3
  MEMRA_GS_MIN=$gs MEMRA_MODELS="m=$MODEL" MEMRA_ADDR=$ADDR MEMRA_COMPAT=openai \
    target/release/memra-server > "$L/gs-$STEM-$tag-r$rep-server.log" 2>&1 &
  local spid=$!
  for _ in $(seq 180); do curl -sf $BASE/health >/dev/null 2>&1 && break; sleep 2; done
  # warmup point (discarded)
  python3 tools/load-serve.py --base $BASE --model m --concurrency 1 --requests 2 \
    --max-tokens 128 --greedy --warmup 1 --label warm >/dev/null 2>&1
  for mt in "${MTS[@]}"; do
    local lg="$L/gs-$STEM-mt$mt-$tag-r$rep.log"
    { nvidia-smi --query-gpu=temperature.gpu,clocks.mem,power.draw --format=csv,noheader
      echo "arm=$tag MEMRA_GS_MIN=$gs mt=$mt rep=$rep model=$STEM"; } > "$lg"
    python3 tools/load-serve.py --base $BASE --model m --concurrency 1 --requests 4 \
      --max-tokens "$mt" --greedy --warmup 0 --label "gs-$tag-mt$mt-r$rep" \
      --out "$OUT/serve-points.jsonl" >> "$lg" 2>&1
    echo "  mt=$mt $tag r$rep: $(python3 "$P" point < "$lg")"
  done
  curl -s $BASE/metrics >> "$L/gs-$STEM-$tag-r$rep-metrics.json" 2>/dev/null
  kill $spid 2>/dev/null; wait $spid 2>/dev/null || true
  sleep 3
}

for r in $(seq 1 "$REPS"); do
  echo "===== rep $r ====="
  if [ $((r % 2)) -eq 1 ]; then
    run_arm shut $GS_A "$r"; run_arm open $GS_B "$r"
  else
    run_arm open $GS_B "$r"; run_arm shut $GS_A "$r"
  fi
done
echo "===== sweep complete: $OUT/serve-points.jsonl ====="

#!/usr/bin/env bash
# pp2-spec STEP 6 — OWNERSHIP of the dev10 c=8 illegal address.
#
# THE FINDING (step-4 receipt, N=5): arm D (MEMRA_PP_DEVICES=1,0, spec ON) fails EXACTLY 1 of 32
# requests at c=8, in ALL FIVE reps, with the quoted:
#
#   step error: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")
#
# Deterministic 1/32 x 5/5 is a BUG, not a flake. Arm B (devices 0,1, spec ON) is 160/160 clean,
# arm A (door shut, spec ON) is 160/160 clean, and c=1 is clean on every arm — so it needs BOTH
# the reversed placement AND concurrency. The server's stderr carries NO diagnostic line (257
# lines, zero matches for illegal/panic/abort/error), so the cause is NOT yet quotable beyond the
# driver text: per the evidence law this is "fails with the quoted illegal address; mechanism
# unlocalized" until these arms speak.
#
# WHAT DECIDES OWNERSHIP — the receipt has no dev10 + spec-OFF control, so the finding cannot
# currently be attributed:
#   F1  dev10, spec OFF, c=8  — THE OWNERSHIP ARM. Arm C proved dev01+spec-OFF clean, but the
#                              reversed placement was never run WITHOUT spec. If F1 also fails,
#                              the fault is in the PREDECESSOR's batched/decode split under a
#                              reversed placement and this lane merely exposed it. If F1 is clean,
#                              it needs spec, and it is this lane's to fix.
#   F2  dev10, spec ON, MEMRA_SPEC_NOGRAPH=1, c=8 — the draft-graph arm. The captured draft graph
#                              is the one piece of the spec path that bakes launch args and holds
#                              device pointers across replays, and `graph_draft` disables the
#                              context's event tracking for the whole session
#                              (spec.rs:2937-2942 `disable_event_tracking`) — precisely the
#                              ordering machinery a reversed placement leans on. If NOGRAPH=1 is
#                              clean, the capture is implicated and the eager draft is the seam.
#   F3  dev10, spec ON, c=8   — arm D re-measured in the SAME hold, so F1/F2 have an interleaved
#                              denominator and the 1/32 rate is confirmed inside this run rather
#                              than carried across runs (cross-run comparisons are drift-invalid).
#   F4  dev10, spec ON, c=4 and c=2 — the concurrency threshold. 1/32 at c=8 and 0/4 at c=1 leaves
#                              the onset unmeasured; if the failure count tracks concurrency it is
#                              a cross-session interference bug, if it is always exactly 1 it is
#                              more likely a first-touch/warmup path.
#
# N=3 for F1..F3 (rep-major, alternating order). Receipts to ~/receipts/pp2spec/illegal.
# GPU window held by the caller under flock.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2spec/illegal
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
ADDR=127.0.0.1:8099
BASE=http://$ADDR

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-pre.csv"

serve_arm() { # $1 = label, $2 = rep, $3 = concurrency list (space-sep, quoted), $4.. = env words
  local label="$1" rep="$2" cs="$3"; shift 3
  local log="$OUT/r$rep-$label"
  if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
    echo "FAIL: $label rep$rep — something is ALREADY serving $ADDR (stale server?); refusing \
to measure against it. Investigate with: ss -lntp | grep ${ADDR##*:}"
    return 1
  fi
  env "$@" MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$log-server.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
  if ! curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
    echo "FAIL: $label rep$rep server never came up"; tail -20 "$log-server.log"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
  fi
  for c in $cs; do
    python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency $c \
      --requests $((c * 4)) --max-tokens 128 --greedy --warmup 1 \
      --label "$label-c$c-r$rep" --out "$OUT/points.jsonl" \
      > "$log-c$c.log" 2>&1
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 4
  return 0
}

for r in 1 2 3; do
  echo "=== rep $r ==="
  if [ $((r % 2)) -eq 1 ]; then ORDER="F1 F2 F3 F4"; else ORDER="F4 F3 F2 F1"; fi
  for a in $ORDER; do
    case $a in
      F1) echo "-- rep $r arm F1: dev10 spec OFF (OWNERSHIP arm) --"
          serve_arm F1-dev10-nospec $r "8" \
            MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SERVE_SPEC=0 ;;
      F2) echo "-- rep $r arm F2: dev10 spec ON, NOGRAPH=1 (draft-graph arm) --"
          serve_arm F2-dev10-spec-nograph $r "8" \
            MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SPEC_NOGRAPH=1 ;;
      F3) echo "-- rep $r arm F3: dev10 spec ON (arm D, in-hold denominator) --"
          serve_arm F3-dev10-spec $r "8" MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 ;;
      F4) echo "-- rep $r arm F4: dev10 spec ON, concurrency onset (c=2,4) --"
          serve_arm F4-dev10-spec-conc $r "2 4" MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 ;;
    esac
  done
done

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-post.csv"

echo "==== raw load points ===="
cat "$OUT/points.jsonl"
echo ILLEGAL_DONE

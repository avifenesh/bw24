#!/usr/bin/env bash
# pp2-spec STEP 4 — the COST receipt: what spec-over-PP2 costs vs spec on one card.
#
# HARNESS LAW (servepath-p2, 2026-08-05): serving numbers come from `memra-server` +
# `tools/load-serve.py`, NEVER from decode-batch-bench (which overstates batched cost ~35% via a
# host argmax over n_vocab). Spec has no bench binary that serves requests, and the question here
# IS a serving question ("what does the felt path cost"), so the server is the only valid harness.
#
# Arms per rep (rep-major interleave — cross-run comparisons are clock-drift invalid per the H100
# laws, so all arms run back to back inside each rep, and the server restarts per arm):
#   A  door shut, spec ON            — the denominator: spec on ONE card, full speed
#   B  pp2 dev01, spec ON            — THE new configuration
#   C  pp2 dev01, spec OFF           — the predecessor lane's shipped config (what spec buys
#                                      over the split, measured within the same rep)
#   D  pp2 dev10, spec ON            — THE DRAFT-PLACEMENT ARM. The brief asks whether the
#                                      drafter's placement matters. It needs no new flag: the MTP
#                                      head loads through the PLAIN PRIMARY engine (hybrid.rs
#                                      ~1021 — every `load_t(e, ...)`, where trunk layers use
#                                      `layer_engine(e, n_trunk, il)`), and the primary is always
#                                      device 0. So MEMRA_PP_DEVICES=0,1 puts the drafter on
#                                      STAGE 0's card, and 1,0 puts stage 0 on device 1 while the
#                                      drafter stays on device 0 = the LAST stage's card. Same
#                                      weights, same split, drafter on the other side of the
#                                      boundary — which is exactly the A/B, and both orders are
#                                      already bit-identical in the gate battery, so any delta
#                                      here is pure placement cost.
# c=1 AND c=8: c=1 is the felt path (and the B1FAST lesson's blast radius — a solo spec session
# must not fall off a fusion chain when the door opens); c=8 is the capacity path.
#
# Model: q9 (the vehicle — an EMBEDDED MTP head, so the server self-specs with no external
# draft). q9 fits one card, which is exactly what makes arm A a legitimate denominator: the
# question is what the SPLIT costs, and that needs both sides runnable. (A >VRAM model has no
# single-card arm at all — that is the capacity case the predecessor lane already covered.)
#
# Receipts to ~/receipts/pp2spec/perf. GPU window held by the caller under flock.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2spec/perf
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
ADDR=127.0.0.1:8099
BASE=http://$ADDR

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-pre.csv"

# One server per arm per rep. GREEDY (temperature 0) throughout: the spec path's greedy arm is
# the exact one, so the comparison is not confounded by sampled acceptance variance, and greedy
# also makes the token stream reproducible across arms.
serve_arm() { # $1 = label, $2 = rep, $3.. = env words
  local label="$1" rep="$2"; shift 2
  local log="$OUT/r$rep-$label"
  # STALE-LISTENER GUARD (2026-08-06, learned the hard way): a server from an earlier,
  # externally-killed run can still own $ADDR. The new server then fails to bind, /v1/models
  # answers from the OLD process, and every point in this arm is measured against the wrong
  # binary and the wrong placement — a silently wrong receipt, not a visible failure. Worse on
  # this box: such an orphan also inherits the flock fd and holds the GPU lock. Refuse to
  # measure into an occupied port.
  if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
    echo "FAIL: $label rep$rep — something is ALREADY serving $ADDR (stale server?); refusing \
to measure against it. Investigate with: ss -lntp | grep ${ADDR##*:}"
    return 1
  fi
  env "$@" MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$log-server.log" 2>&1 &
  local pid=$!
  # wait for /models (model load on the pair takes tens of seconds)
  for _ in $(seq 1 180); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && break
    sleep 2
  done
  if ! curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
    echo "FAIL: $label rep$rep server never came up"; tail -20 "$log-server.log"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 1
  fi
  # warmup discarded (--warmup 1 is the harness default; stated for the record)
  for c in 1 8; do
    python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency $c \
      --requests $((c * 4)) --max-tokens 128 --greedy --warmup 1 \
      --label "$label-c$c-r$rep" --out "$OUT/points.jsonl" \
      > "$log-c$c.log" 2>&1
  done
  curl -sf "$BASE/metrics" > "$log-metrics.txt" 2>&1 || true
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 4
  return 0
}

for r in 1 2 3 4 5; do
  echo "=== rep $r ==="
  # order alternates across reps so a monotone thermal drift cannot favour one arm
  if [ $((r % 2)) -eq 1 ]; then ORDER="A B C D"; else ORDER="D C B A"; fi
  for a in $ORDER; do
    case $a in
      A) echo "-- rep $r arm A: door shut, spec ON --"
         serve_arm A-doorshut-spec $r MEMRA_DUMMY=1 ;;
      B) echo "-- rep $r arm B: pp2 dev01, spec ON --"
         serve_arm B-pp2-spec $r MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 ;;
      C) echo "-- rep $r arm C: pp2 dev01, spec OFF --"
         serve_arm C-pp2-nospec $r MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_SERVE_SPEC=0 ;;
      D) echo "-- rep $r arm D: pp2 dev10, spec ON (DRAFT PLACEMENT flip) --"
         serve_arm D-pp2-dev10-spec $r MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 ;;
    esac
  done
done

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-post.csv"

echo "==== raw load points ===="
cat "$OUT/points.jsonl"
echo PERF_DONE

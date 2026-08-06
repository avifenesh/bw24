#!/usr/bin/env bash
# pp2-spec STEP 5b — THE DRAFT-PLACEMENT MECHANISM ISOLATION.
#
# The step-4 cost receipt found a 7x asymmetry between the two placement orders with spec ON
# (arm B, MEMRA_PP_DEVICES=0,1: 17.1 agg tok/s; arm D, 1,0: 123.4 — vs 346.5 on one card).
# Bit-identity and acceptance are IDENTICAL in both orders, so this is pure data movement.
#
# THE HYPOTHESIS (a named, arithmetically-checked cause, not an inference):
#   1. The server always builds `Engine::new(0)` (worker.rs:1079) — the serving PRIMARY is
#      ALWAYS device 0, whatever MEMRA_PP_DEVICES says. (decode-batch-gate instead follows
#      MEMRA_PP_DEVICES[0], which is why the gate battery never saw this.)
#   2. `output_norm` + `output` (the lm head) upload through the LAST stage's engine —
#      hybrid.rs:881 `let e_head = crate::pp::layer_engine(e, n_trunk, n_trunk - 1)`. Correct
#      for the trunk: the last stage is what runs the head.
#   3. Every MTP/draft weight uploads through the PLAIN PRIMARY engine (hybrid.rs ~1021,
#      `load_t(e, ...)`), and every draft forward RUNS on `e` (spec.rs:4276/5434 pass `e`).
#   4. q9 ships NO `blk.32.nextn.shared_head.weight` (verified in the gguf header: only
#      eh_proj / enorm / hnorm / shared_head_norm). So the draft head falls back to the
#      TRUNK's head — spec.rs:781, `mtp.shared_head_head.as_ref().unwrap_or(&self.output)`.
#
#   => With devices=0,1 the last stage is dev1, so `output` lives on dev1, while the draft
#      GEMV runs on dev0. Every draft token does a [4096, 248320] NVFP4 matmul whose WEIGHT
#      is remote: ~508 MB of peer reads per draft step. At PCIe-class bandwidth that is
#      ~10 ms/draft-step, and 128 tokens at 59 ms/token (B's measured p50/128) is exactly
#      that order. With devices=1,0 the last stage IS dev0 = the primary, so the head and the
#      draft are co-resident and the peer traffic disappears (D's 8 ms/token).
#      Arm C (spec OFF) is unaffected because the trunk uses the head ON the stage that owns it.
#
# THE DECISIVE ARM: MEMRA_PP_SHARD=0 is the M1 bring-up placement — `layer_engine` returns the
# PRIMARY engine for every layer, so the head comes home to dev0 while the STAGE STREAMS still
# split exactly as before. If the hypothesis holds, dev01+SHARD=0 with spec ON recovers most of
# arm D's speed. If instead it stays at ~17, the cause is the stage split itself and the head
# story is refuted. Either way this arm decides it, with no code change.
#
# Two supporting arms:
#   - E2 dev01 + SPEC_OFF + SHARD=0: the same placement WITHOUT the draft. If E2 ~= arm C, then
#     SHARD=0 is not just "faster because unsharded" and E1's gain is attributable to the draft.
#   - E3 dev10 + SHARD=1 spec ON: arm D re-measured inside this hold, so E1 has an
#     interleaved denominator rather than a cross-run one (cross-run comparisons are
#     clock-drift invalid).
#
# N=3 interleaved, rep-major, order alternating. c=1 only: the asymmetry is a per-draft-step
# cost and c=1 is where it is cleanest (and c=1 had 0 errors on every step-4 arm).
# Receipts to ~/receipts/pp2spec/drafthead. GPU window held by the caller under flock.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2spec/drafthead
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
ADDR=127.0.0.1:8099
BASE=http://$ADDR

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-pre.csv"

serve_arm() { # $1 = label, $2 = rep, $3.. = env words
  local label="$1" rep="$2"; shift 2
  local log="$OUT/r$rep-$label"
  # STALE-LISTENER GUARD (same law as the step-4 script): refuse to measure into an occupied
  # port rather than silently benchmark someone else's binary.
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
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 1 \
    --requests 4 --max-tokens 128 --greedy --warmup 1 \
    --label "$label-c1-r$rep" --out "$OUT/points.jsonl" \
    > "$log-c1.log" 2>&1
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 4
  return 0
}

for r in 1 2 3; do
  echo "=== rep $r ==="
  if [ $((r % 2)) -eq 1 ]; then ORDER="E1 E2 E3"; else ORDER="E3 E2 E1"; fi
  for a in $ORDER; do
    case $a in
      E1) echo "-- rep $r arm E1: dev01 + SHARD=0, spec ON (THE decisive arm) --"
          serve_arm E1-dev01-shard0-spec $r \
            MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0 ;;
      E2) echo "-- rep $r arm E2: dev01 + SHARD=0, spec OFF (draft-free control) --"
          serve_arm E2-dev01-shard0-nospec $r \
            MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0 MEMRA_SERVE_SPEC=0 ;;
      E3) echo "-- rep $r arm E3: dev10 sharded, spec ON (arm D, in-hold denominator) --"
          serve_arm E3-dev10-spec $r MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 ;;
    esac
  done
done

nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
  --format=csv > "$OUT/gpu-post.csv"

echo "==== raw load points ===="
cat "$OUT/points.jsonl"
echo DRAFTHEAD_DONE

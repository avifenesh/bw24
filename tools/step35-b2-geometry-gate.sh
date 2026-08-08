#!/usr/bin/env bash
# step35-b2-geometry-gate — the standing form of research/step-sku-20260807/b2-geometry-ab.sh.
#
# WHAT IT PINS (two assertions, BOTH required):
#   1. TEXT IDENTITY: step35 served at c=2 and c=4 (concurrent identical greedy requests, the
#      DEFAULT batched scheduler) must return responses byte-identical to the c=1 reference.
#      This is the assertion whose pre-fix failure was HTTP-200 GARBAGE ('::::…') — over PP-2
#      (this SKU's only placement) a B>1 tick walked the generic Full arm: global n_head=96
#      over-reading wq on the 12 full-attn layers, 128-dim rope on all 45 layers, no SWA
#      window, no head-wise gate (research/step-sku-20260807/raw/b2ab-pre-20260807T091553Z.log).
#   2. BATCHED EVIDENCE: the server must actually have RUN a B>1 batched step35 walk —
#      (a) the spawn log's `decode chunk cap` for the step35 model must be >= 2, and
#      (b) the engine's one-shot `[step35-batch] first B>1` line must appear.
#      Without (2) the gate is vacuously green under the fail-closed B=1 pin (chunk_cap_for
#      returns 1, every "batched" tick is a B=1 chunk, and identity holds trivially).
#
# REGISTERED RED (lane/step35-batched-decode, 2026-08-08): under the B=1 pin, assertion 2
# fails by construction — `decode chunk cap 1` and no batched-walk line. The batched arm's
# commits turn it green. Same pattern as tickinv35's red registration.
#
# TEETH (--canary): sets MEMRA_STEP35_BATCH=0 — the rollback seam that re-pins step35 decode
# chunks to B=1 (the pre-arm behavior, fail-closed). The canary PASSES only if the naked
# gate's assertion 2 then FAILS, proving the evidence check can detect the pin. The canary
# changes the WORLD, not the label (the chunkinv lesson, written wrong twice there).
#
# Requires: 2 GPUs (the artifact fits only across a PRO 6000 pair), the step35 artifact.
# SKIPs cleanly when either is absent — a missing artifact must not read as a pass
# (fast-gate reads this script's own SKIP word).
#
#   tools/step35-b2-geometry-gate.sh [--canary] [--port N]
set -uo pipefail
cd "$(dirname "$0")/.."

CANARY=0
PORT=8094
while [ $# -gt 0 ]; do
  case "$1" in
    --canary) CANARY=1; shift ;;
    --port) PORT=$2; shift 2 ;;
    *) echo "step35-b2-geometry-gate: unknown arg $1"; exit 2 ;;
  esac
done

# ---- artifact resolution (MEMRA_STEP37_GGUF override; box1 + box2 staged locations) ----
MODEL="${MEMRA_STEP37_GGUF:-}"
if [ -z "$MODEL" ]; then
  for cand in \
    "$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
    "/data/models/step37/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf" \
    "/data/models/step37/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"; do
    [ -f "$cand" ] && { MODEL="$cand"; break; }
  done
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || {
  echo "step35-b2-geometry-gate: SKIP (no Step-3.7-Flash artifact; set MEMRA_STEP37_GGUF)"; exit 0; }
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$NGPU" -ge 2 ] || {
  echo "step35-b2-geometry-gate: SKIP (needs 2 GPUs for the 105GB PP-2 placement, have $NGPU)"; exit 0; }
BIN=./target/release/memra-server
[ -x "$BIN" ] || { echo "step35-b2-geometry-gate: FAIL (no $BIN — build release first)"; exit 1; }

# drafter (optional — trunk-only serve WARNs but works; attach when staged next to the trunk)
DRAFT="$(dirname "$(dirname "$MODEL")")/Step3.7-flash-mtp-Q8_0.gguf"
[ -f "$DRAFT" ] || DRAFT="$(dirname "$MODEL")/Step3.7-flash-mtp-Q8_0.gguf"
MODELS_SPEC="step35=${MODEL}"
[ -f "$DRAFT" ] && MODELS_SPEC="step35=${MODEL}+${DRAFT}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
D=research/step35-batch-20260808/raw
mkdir -p "$D"
TAG=$([ "$CANARY" = 1 ] && echo canary || echo naked)
SLOG=$D/b2geo35-server-$TAG-$TS.log
GLOG=$D/b2geo35-$TAG-$TS.log
BASE=http://127.0.0.1:$PORT

exec > >(tee "$GLOG") 2>&1
echo "=== step35-b2-geometry-gate tag=$TAG ts=$TS model=$MODEL draft=${DRAFT:-none} ==="

RC=1
(
  flock -w 3600 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  # Serve config: batching default ON, PP-2, spec OFF per #87 (spec over PP-2 is
  # quarantined). MEMRA_SERVE_B1FAST=0 pins the c=1 REFERENCE onto the batched walk at
  # B=1 — the same pin decode-batch-gate's gate2 applies, and for the same reason: with
  # b1-fast live, c=1 rides the m=1 FUSION chain (the eager side of the accepted
  # decode-config FP gap) while c>1 rides the batched walk, so a near-tie greedy flip
  # would fail this gate for a class that is NOT batched-geometry breakage. Within-config,
  # byte-identity is the honest bar (per-row bit-identity is gate2's, engine-level).
  # The b1-fast path's own text fidelity is run-gen/serve-smoke jurisdiction.
  # The canary re-pins B=1 chunks.
  CANARY_ENV=()
  [ "$CANARY" = 1 ] && CANARY_ENV=(MEMRA_STEP35_BATCH=0)
  env "${CANARY_ENV[@]}" \
    MEMRA_MODELS="$MODELS_SPEC" MEMRA_SERVE_SPEC=0 MEMRA_SERVE_B1FAST=0 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR=127.0.0.1:$PORT \
    "$BIN" > "$SLOG" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
  for i in $(seq 1 120); do
    sleep 5; curl -sf "$BASE/readyz" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "FAIL: server died during boot"; sed -n '1,50p' "$SLOG"; exit 1; }
  done
  curl -sf "$BASE/readyz" >/dev/null 2>&1 || { echo "FAIL: server never became ready"; exit 1; }

  BODY='{"model":"step35","messages":[{"role":"user","content":"List the first eight prime numbers, comma separated, then explain in two sentences why 1 is not prime."}],"max_tokens":48,"temperature":0.0}'
  ask() { curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
    | python3 -c 'import json,sys
r=json.load(sys.stdin); c=r.get("choices")
if c:
    m=c[0]["message"]; print(json.dumps({"reasoning": m.get("reasoning"), "content": m.get("content")}))
else:
    print("ERROR", json.dumps(r.get("error")))'; }

  echo "--- c=1 reference (greedy) ---"
  ask > /tmp/b2geo35-ref.txt
  cat /tmp/b2geo35-ref.txt

  for C in 2 4; do
    echo "--- c=$C concurrent identical requests ---"
    # wait ONLY for the curl PIDs — a bare `wait` also waits on the server background
    # job, which never exits (found live: the gate hung after a fully-correct c=2 round).
    CURL_PIDS=()
    for i in $(seq 1 $C); do ask > /tmp/b2geo35-c$C-$i.txt & CURL_PIDS+=($!); done
    wait "${CURL_PIDS[@]}"
    cat /tmp/b2geo35-c$C-*.txt
  done

  kill $SRV; wait $SRV 2>/dev/null; trap - EXIT
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"

  # ---- verdicts ----
  FAILS=0
  REF=$(cat /tmp/b2geo35-ref.txt)
  [ -n "$REF" ] && ! grep -q '^ERROR' /tmp/b2geo35-ref.txt || { echo "FAIL: empty/error c=1 reference"; FAILS=$((FAILS+1)); }
  for C in 2 4; do
    for i in $(seq 1 $C); do
      ROW=$(cat /tmp/b2geo35-c$C-$i.txt)
      if [ "$ROW" = "$REF" ]; then echo "c$C[$i] == ref"
      else echo "c$C[$i] != ref"; echo "  ref: $REF"; echo "  got: $ROW"; FAILS=$((FAILS+1)); fi
    done
  done

  # assertion 2a: the spawn-time chunk cap for step35 must admit B>1
  CAP=$(grep -oE 'step35: decode chunk cap [0-9]+' "$SLOG" | grep -oE '[0-9]+$' | head -1)
  if [ -n "$CAP" ] && [ "$CAP" -ge 2 ]; then echo "chunk cap $CAP >= 2 OK"
  else echo "FAIL: step35 decode chunk cap is '${CAP:-unset}' (< 2) — the B=1 pin is live, nothing batched was tested"; FAILS=$((FAILS+1)); fi
  # assertion 2b: a B>1 batched step35 walk actually executed
  if grep -q '\[step35-batch\] first B>1' "$SLOG"; then
    echo "batched-walk evidence OK: $(grep -m1 -oE '\[step35-batch\] first B>1[^"]*' "$SLOG" | head -c 120)"
  else echo "FAIL: no '[step35-batch] first B>1' line in the server log — no B>1 step35 tick ran"; FAILS=$((FAILS+1)); fi

  echo "server log: $SLOG"
  if [ "$FAILS" -eq 0 ]; then echo "VERDICT: PASS (c=2/c=4 byte-identical to c=1, batched ticks proven)"; exit 0
  else echo "VERDICT: FAIL ($FAILS failed assertions)"; exit 1; fi
) 9>/tmp/memra-gpu.lock
RC=$?

if [ "$CANARY" = 1 ]; then
  # The canary re-pinned B=1 chunks; the naked assertions MUST have failed (assertion 2 at
  # minimum). If they passed, the evidence check has no teeth.
  if [ "$RC" -ne 0 ]; then
    echo "step35-b2-geometry-gate: CANARY OK (B=1 re-pin broke the batched-evidence assertion as required)"
    exit 0
  else
    echo "step35-b2-geometry-gate: CANARY FAILED — gate PASSED under MEMRA_STEP35_BATCH=0 (no teeth)"
    exit 1
  fi
fi
exit $RC

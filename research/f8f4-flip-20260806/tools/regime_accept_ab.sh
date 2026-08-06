#!/bin/bash
# REGIME-MATCHED acceptance A/B — the gate-2 question asked again in the config that actually
# serves traffic.
#
# WHY THIS EXISTS. gate 2 (accept_ab.sh -> run-spec) uses the GGUF's EMBEDDED MTP head. The
# owner's daily 27B server (tools/serve-examples/serve-qwen36-27b-memra) does NOT: it attaches
# draft-daily-owntrim-nvfp4head-q4blk.gguf via MEMRA_MODELS "+draft", which REPLACES the
# embedded head at load (worker.rs load_draft). The 2026-07-10 flip battery's headline numbers
# were also regime/ST numbers. So "bare-head acceptance moved -1.9pp" is a real measurement of
# a DIFFERENT drafter than the one in production, and the flip decision is about production.
#
# Vehicle: the server itself. usage.spec.{rounds,drafted,accepted,acceptance_rate} is an
# additive usage field (main.rs usage_json, lane/accept-telemetry) populated per request
# whenever the request actually ran spec rounds — i.e. exact per-request acceptance from the
# real serve loop, no research binary in the path.
#
# Protocol: one server per arm (MEMRA_MMQ_F8F4 is read once into a OnceLock, so it CANNOT be
# toggled inside a live process), same regime draft, same K, temperature 0 (greedy => drafting
# is deterministic => acceptance is a hard number, not a sample), same prompt set, arms run
# back-to-back under the box lock. Serial, ARM order OFF then ON then OFF again (a 3rd pass) so
# a drift in the box shows up as OFF-vs-OFF disagreement.
#
# Usage: TAG=q27|q9 regime_accept_ab.sh
set -u
W=/home/avifenesh/projects/wt-f8f4flip
D=$W/research/f8f4-flip-20260806
OUT=$D/logs
# TAG selects the model+its regime drafter. Both reachable NVFP4 models have one, so this cell
# is 2-model like every other row of the matrix.
TAG=${TAG:-q27}
L=$OUT/regime-accept-$TAG.log
case $TAG in
  q27) DIR=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp
       MODEL=$DIR/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
       DRAFT=$DIR/draft-daily-owntrim-nvfp4head-q4blk.gguf ;;
  q9)  DIR=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf
       MODEL=$DIR/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
       DRAFT=$DIR/draft-9b-owntrim-nvfp4head-q4blk.gguf ;;
  *) echo "regime_accept_ab.sh: unknown TAG $TAG (want q27|q9)" >&2; exit 2 ;;
esac
BIN=$W/target/release/memra-server
PORT=${PORT:-8117}
CTX=${CTX:-16384}
K=${K:-3}
KEY=f8f4lane
# Prompt set: the lane's own p2 code prompt FIRST (same text gate 2 used, so the two
# measurements are directly comparable), plus p1 (short) and p3 v3 (agentic long) for prompt-
# length breadth — acceptance is prompt-dependent and the prefill-KV mechanism scales with
# prompt length, so a one-length reading would be the weakest possible version of this cell.
PROMPTS=("$W/research/e2e/prompts/p2-code-medium.txt"
         "$W/research/e2e/prompts/p1-code-short.txt"
         "$W/research/e2e/prompts/p3-agentic-long-v3.txt")
{ echo "[start] $(date -Is)"; echo "[commit] $(git -C $W rev-parse HEAD)"
  echo "[model] $MODEL"; echo "[draft] $DRAFT (REGIME draft — replaces the embedded MTP head)"
  echo "[bin] $BIN"; echo "[ctx] $CTX"; echo "[spec K] $K"; echo "[port] $PORT"
  echo "[prompts] ${PROMPTS[*]}"
  echo "[proto] one server per arm, temperature 0 (greedy), max_tokens 128, serial OFF/ON/OFF"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | sed 's/^/[gpu] /'
} > "$L"

serve_up(){ # serve_up <arm>
  local arm=$1 aenv=()
  [ "$arm" = ON ] && aenv=(MEMRA_MMQ_F8F4=1)
  env "${aenv[@]}" \
    MEMRA_MODELS="m=$MODEL+$DRAFT" MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_API_KEY=$KEY \
    MEMRA_CTX=$CTX MEMRA_MAX_SESSIONS=1 MEMRA_REUSE_POOL=1 MEMRA_PRIME_CHUNK=2048 \
    MEMRA_SPEC_K=$K \
    "$BIN" > "$OUT/regime-server-$TAG-$arm.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 120); do
    curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 $SPID 2>/dev/null || { echo "[FATAL] server $arm died during boot" >> "$L"; return 1; }
    sleep 2
  done
  echo "[FATAL] server $arm never became healthy" >> "$L"; return 1
}
serve_down(){ kill $SPID 2>/dev/null; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -9 $SPID 2>/dev/null; sleep 3; }

for PASS in 1 2 3; do
  case $PASS in 1) ARM=OFF ;; 2) ARM=ON ;; 3) ARM=OFF2 ;; esac
  RARM=${ARM%2}
  echo "=== PASS $PASS ARM $ARM  $(date -Is)  $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" >> "$L"
  serve_up "$RARM" || { echo "[rc=1] pass $PASS aborted" >> "$L"; continue; }
  for p in "${PROMPTS[@]}"; do
    # raw-prompt /v1/completions = the pi contract this server serves (client renders the
    # template), so no server-side chat template enters the comparison.
    body=$(python3 - "$p" <<'PY'
import json,sys
print(json.dumps({"model":"m","prompt":open(sys.argv[1]).read(),
                  "max_tokens":128,"temperature":0,"stream":False}))
PY
)
    echo "--- prompt $(basename "$p")" >> "$L"
    curl -s -m 600 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
      -d "$body" "http://127.0.0.1:$PORT/v1/completions" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); u=d.get("usage",{}); s=u.get("spec",{}); print("USAGE",json.dumps({k:u.get(k) for k in ("prompt_tokens","completion_tokens","elapsed_s")}),"SPEC",json.dumps(s)); print("TEXT_SHA", __import__("hashlib").sha256(d["choices"][0]["text"].encode()).hexdigest()[:16]); print("TEXT_HEAD", repr(d["choices"][0]["text"][:120]))' \
      >> "$L" 2>&1
    echo "[curl rc=$?]" >> "$L"
  done
  echo "=== metrics ARM $ARM" >> "$L"
  curl -s -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/metrics" >> "$L" 2>&1
  echo "" >> "$L"
  serve_down
  echo "[done] pass $PASS ARM $ARM" >> "$L"
done
echo "wrote $L"
grep -E "^=== PASS|^--- prompt|^USAGE|^TEXT_SHA|FATAL" "$L"

#!/usr/bin/env bash
# pp2spec-crash STEP 9 — final: (a) NAKED boot proof (no MEMRA_PP2SPEC_UNQUARANTINE — the
# env var no longer exists; the unquarantined binary must boot spec+PP-2 and serve), and
# (b) the quarantine-lifted perf point: N=5 interleaved spec-ON vs spec-OFF at c=2 and c=4
# over PP-2 dev10, verifying the concurrency-gated spec scheduler's crossover shape on PP-2.
# NOTE: spec-ON arms run SPEC_GATE=0 (pure spec path measurement); the gate (#89) is the
# shipping policy layered on top — its crossover check is arm S vs arm N at each c.
set -uo pipefail
cd ~/memra
export PATH=$HOME/.cargo/bin:$PATH
OUT=~/receipts/pp2crash/perf
mkdir -p "$OUT"
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
BIN=target/release
ADDR=127.0.0.1:8123
BASE=http://$ADDR

exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || { echo "FAIL: gpu lock timeout"; exit 1; }
echo "gpu lock acquired $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv > "$OUT/gpu-pre.csv"

wait_up() {
  for _ in $(seq 1 "$1"); do
    curl -sf "$BASE/v1/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then
  echo "FAIL: something already serving $ADDR"; exit 1
fi

# ---- (a) NAKED BOOT: no override var; the binary must boot spec+PP-2 and serve spec ----
echo "=== NAKED boot (quarantine-removed binary, no override env) ==="
env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
  $BIN/memra-server > "$OUT/naked-server.log" 2>&1 &
PID=$!
if ! wait_up 180; then
  echo "FAIL: NAKED boot refused or died"; tail -10 "$OUT/naked-server.log"; kill $PID 2>/dev/null; exit 1
fi
python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 4 \
  --requests 16 --max-tokens 96 --greedy --warmup 1 --label naked-c4 \
  --out "$OUT/naked-points.jsonl" > "$OUT/naked-c4.log" 2>&1
kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 4
grep -q "spec-acc" "$OUT/naked-server.log" && echo "NAKED: spec ran" || echo "NAKED: NO spec-acc lines (check spec-gate demotion at c=4 — expected under #89)"
grep -n -i "illegal\|sentinel\|REFUS" "$OUT/naked-server.log" | head -3 || echo "NAKED: clean"

# ---- (b) N=5 interleaved: spec-ON (SPEC_GATE=0) vs spec-OFF at c=2 / c=4, dev10 ----
arm() { # $1=tag $2=serve_spec $3=spec_gate_env
  local tag=$1 spec=$2 gate=$3
  env MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=1,0 MEMRA_SERVE_SPEC=$spec $gate \
    MEMRA_MODELS="q9=$Q9" MEMRA_ADDR=$ADDR \
    $BIN/memra-server > "$OUT/$tag-server.log" 2>&1 &
  local P=$!
  if ! wait_up 180; then echo "FAIL: $tag never came up"; kill $P 2>/dev/null; return 1; fi
  # warm the placement once so rep-0 isn't a cold outlier
  python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency 2 \
    --requests 4 --max-tokens 64 --greedy --warmup 1 --label $tag-warm \
    --out /dev/null > /dev/null 2>&1
  for c in 2 4; do
    python3 tools/load-serve.py --base "$BASE" --model q9 --concurrency $c \
      --requests $((c*8)) --max-tokens 96 --greedy --warmup 0 --label $tag-c$c \
      --out "$OUT/perf-points.jsonl" >> "$OUT/$tag-load.log" 2>&1
  done
  kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 4
}

# INTERLEAVED x5: S (spec ON, gate off = pure spec), N (spec OFF) alternating per rep.
for rep in 1 2 3 4 5; do
  echo "=== perf rep $rep: S then N ==="
  arm "S$rep" 1 MEMRA_SPEC_GATE=0 || exit 1
  arm "N$rep" 0 MEMRA_LANE_NOOP=1 || exit 1
done

nvidia-smi --query-gpu=index,memory.used --format=csv > "$OUT/gpu-post.csv"
echo "==== perf table ===="
python3 - <<'EOF'
import json, statistics as st
rows = {}
for l in open("/home/ubuntu/receipts/pp2crash/perf/perf-points.jsonl"):
    d = json.loads(l)
    lab = d["label"]  # S1-c2 etc.
    arm, c = lab[0], lab.split("-c")[1]
    rows.setdefault((arm, c), []).append(d["agg_tok_s"])
    if d["n_err"]: print("ERRORS in", lab, d["n_err"])
for k in sorted(rows):
    v = rows[k]
    print(f"arm={k[0]} c={k[1]}: median {st.median(v):.1f} tok/s  N={len(v)}  min {min(v):.1f} max {max(v):.1f}")
EOF
echo PP2CRASH_PERF_DONE

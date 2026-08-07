#!/usr/bin/env bash
# probe-bg-stress — deliverable-2 receipt (lane/darklane-training, 2026-08-07):
# serve bursts against a live background job vs a no-job baseline, INTERLEAVED x5
# (A B A B ... — cross-run comparisons without interleaving are clock/thermal-invalid,
# the H100-lane law). Plus a direct yield-latency measurement per rep.
#
# Arms (fresh server boot per rep per arm — MEMRA_BG_JOB is env-at-boot):
#   base : naked server, 2 bursts of c=8 x 16 streaming requests.
#   bg   : same server + a CPU background job (8 spinner workers of the 24-core host —
#          deliberately CAPPED; a desktop-saturating burner is banned on this rig). The
#          job launches in the pre-burst valley, must SIGSTOP-yield when the burst
#          arrives, resume in the inter-burst valley, and yield again for burst 2.
#
# Measured per rep:
#   * burst lat_p50/p95 + ttft p50/p95 (load-serve.py, streaming);
#   * bg arm: yield latency = wall time from firing a single request out of a valley
#     (job RUNNING) to the job's /proc state flipping to 'T' (polled at ~1ms);
#   * bg arm liveness: /metrics bg block must show launches>=1, yields>=2, resumes>=1 —
#     a rep where the job never ran measures nothing.
#
# Verdict inputs (computed by the caller / RESULTS): median p95 across the 10 bg bursts
# vs the 10 base bursts. STOP bar per the lane brief: bg adds >2% p95 => report the gap.
#
# GPU: caller wraps in `flock /tmp/gpu5090.lock`.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
DRAFT="${2:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf}"
REPS="${3:-5}"
[ -f "$MODEL" ] || { echo "probe-bg-stress: SKIP (no model at $MODEL)"; exit 0; }
ADDR=127.0.0.1:8189
BASE=http://$ADDR
OUT=research/darktrain-20260807/raw
mkdir -p "$OUT"
POINTS=$OUT/bgstress-points.jsonl
YIELDS=$OUT/bgstress-yield.jsonl
: > "$POINTS"; : > "$YIELDS"
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

[ -x target/release/memra-server ] || cargo build --release -p memra-server

# The background job: 8 CPU spinners (of 24 cores — capped on purpose) in one group.
BGJOB='for i in 1 2 3 4 5 6 7 8; do while :; do :; done & done; wait'

SPID=0
start_server() { # $1 = arm (base|bg), $2 = rep
  local envjob=()
  [ "$1" = bg ] && envjob=(MEMRA_BG_JOB="$BGJOB")
  env "${envjob[@]}" MEMRA_COMPAT=openai MEMRA_MODELS="smoke=$MODEL+$DRAFT" MEMRA_ADDR=$ADDR \
    target/release/memra-server > "$OUT/bgstress-server-$1-r$2.log" 2>&1 &
  SPID=$!
  for _ in $(seq 120); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server ($1 rep $2) did not come up; tail:"; tail -5 "$OUT/bgstress-server-$1-r$2.log"
  return 1
}
stop_server() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null || true; }

burst() { # $1 = label
  python3 tools/load-serve.py --base $BASE --model smoke --concurrency 8 --requests 16 \
    --max-tokens 96 --stream --warmup 0 --label "$1" --out "$POINTS" >/dev/null
}

bg_field() { curl -sf $BASE/metrics | python3 -c \
  "import json,sys; print(json.load(sys.stdin).get('bg',{}).get('$1',''))"; }

# Yield latency: from a valley (job RUNNING), fire one request and poll the job's /proc
# state at ~1ms until 'T' (SIGSTOPped). Wall time = detection (<=poll 25ms) + delivery.
measure_yield() { # $1 = rep
  local pid t
  pid=$(bg_field job_pid)
  [ -n "$pid" ] && [ "$pid" != "None" ] || { echo "no-job"; return 1; }
  python3 - "$BASE" "$pid" "$1" "$YIELDS" <<'PY'
import json, sys, threading, time, urllib.request
base, pid, rep, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def state():
    try:
        s = open(f"/proc/{pid}/stat").read()
        return s[s.rfind(")") + 1:].split()[0]
    except OSError:
        return "gone"

assert state() in ("R", "S"), f"job not running before probe: {state()}"
body = json.dumps({"model": "smoke", "messages": [{"role": "user", "content": "Say hi."}],
                   "max_tokens": 8, "temperature": 0}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
# the request must fire on a THREAD: urlopen blocks to response completion, and the yield
# happens DURING the request — polling after it returns would over-report the latency.
done = {}
def fire():
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    done["req_wall_s"] = time.monotonic() - t
th = threading.Thread(target=fire)
t0 = time.monotonic()
th.start()
deadline = t0 + 5.0
while state() != "T" and time.monotonic() < deadline:
    time.sleep(0.001)
dt_stop = time.monotonic() - t0
final = state()
th.join()
row = {"rep": int(rep), "yield_wall_s": round(dt_stop, 4),
       "req_wall_s": round(done.get("req_wall_s", -1), 4), "state_after": final}
print(json.dumps(row))
with open(out, "a") as f:
    f.write(json.dumps(row) + "\n")
assert final == "T", "job never reached stopped state"
assert dt_stop < 0.5, f"yield {dt_stop:.3f}s blew the 500ms bound"
PY
}

for rep in $(seq "$REPS"); do
  echo "== rep $rep: base arm =="
  start_server base "$rep" || { FAIL "base server rep $rep"; continue; }
  sleep 3
  burst "base-r$rep-b1" && burst "base-r$rep-b2" \
    && PASS "base rep $rep bursts" || FAIL "base rep $rep burst"
  stop_server

  echo "== rep $rep: bg arm =="
  start_server bg "$rep" || { FAIL "bg server rep $rep"; continue; }
  sleep 4   # valley threshold (2s) + margin: the job must LAUNCH before the burst
  ST=$(bg_field state)
  [ "$ST" = running ] || FAIL "bg rep $rep: job not running pre-burst (state=$ST)"
  if YROW=$(measure_yield "$rep"); then
    PASS "bg rep $rep yield probe: $YROW"
  else
    FAIL "bg rep $rep yield probe"
  fi
  sleep 4   # let it resume so burst 1 exercises yield-under-burst too
  burst "bg-r$rep-b1"
  sleep 4   # inter-burst valley: resume
  burst "bg-r$rep-b2" && PASS "bg rep $rep bursts" || FAIL "bg rep $rep burst"
  # liveness: the job must have yielded for the probe + both bursts and resumed between.
  Y=$(bg_field yields); R=$(bg_field resumes); L=$(bg_field launches)
  echo "  bg counters rep $rep: launches=$L yields=$Y resumes=$R"
  [ "${Y:-0}" -ge 3 ] && [ "${R:-0}" -ge 2 ] && [ "${L:-0}" -ge 1 ] \
    && PASS "bg rep $rep liveness (yields=$Y resumes=$R)" \
    || FAIL "bg rep $rep liveness (launches=$L yields=$Y resumes=$R)"
  stop_server
done

# ---- summary table ----
python3 - "$POINTS" "$YIELDS" <<'PY'
import json, statistics, sys
pts = [json.loads(l) for l in open(sys.argv[1])]
def col(arm, key):
    return [p[key] for p in pts if p["label"].startswith(arm) and p.get(key) is not None]
print("\n== bg-stress summary (per-burst points; N bursts per arm below) ==")
for key in ("lat_p50_s", "lat_p95_s", "ttft_p50_s", "ttft_p95_s", "agg_tok_s"):
    b, g = col("base", key), col("bg", key)
    if not b or not g:
        continue
    mb, mg = statistics.median(b), statistics.median(g)
    delta = (mg - mb) / mb * 100 if mb else float("nan")
    print(f"  {key:12s} base median {mb:8.3f} (n={len(b)})  "
          f"bg median {mg:8.3f} (n={len(g)})  delta {delta:+.2f}%")
ys = [json.loads(l)["yield_wall_s"] for l in open(sys.argv[2])]
if ys:
    print(f"  yield_wall_s  n={len(ys)} median {statistics.median(ys)*1000:.1f}ms "
          f"max {max(ys)*1000:.1f}ms (bound 500ms)")
b, g = col("base", "lat_p95_s"), col("bg", "lat_p95_s")
if b and g:
    delta = (statistics.median(g) - statistics.median(b)) / statistics.median(b) * 100
    print(f"\nSTOP-BAR: bg adds {delta:+.2f}% to median burst p95 (bar: +2%)")
    print("VERDICT: " + ("WITHIN BAR" if delta <= 2.0 else "OVER BAR — report the gap"))
PY

echo "probe-bg-stress: $FAILS failed"
[ $FAILS -eq 0 ]

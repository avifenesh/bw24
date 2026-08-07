#!/usr/bin/env bash
# probe-pp2-valley — the PP-2 pair receipt (lane/darklane-training, 2026-08-07):
# ONE full valley+yield cycle on the 2x RTX PRO 6000 box, deployment shape.
#
# Server: q27 NVFP4 over PP-2 (MEMRA_PP_DEVICES=0,1; MEMRA_SERVE_SPEC=0 — REQUIRED for
# PP-2 serving per FLAGS.md). Background job: CPU spinners (8 of 24) + a small VRAM
# budget arm exercising the min-across-GPUs fit check on the pair.
#
# The cycle asserted:
#   1. valley: serve_idle_seconds accrues; the bg job launches (state=running);
#   2. traffic: one chat request -> the job SIGSTOPs (state flips yielded; /proc 'T'),
#      the request completes normally;
#   3. valley again: the job resumes (state=running, resumes>=1).
#   4. VRAM arm: a budget larger than min-free is REFUSED (state=refused_vram).
#
# Runs on the box under flock /tmp/memra-gpu.lock (caller wraps).
set -uo pipefail
cd "$(dirname "$0")"   # box copy runs from ~/darktrain-memra root; see driver below
BIN=${BIN:-target/release}
MODEL=${MODEL:-/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf}
ADDR=127.0.0.1:8191
BASE=http://$ADDR
OUT=${OUT:-$HOME/receipts/darktrain}
mkdir -p "$OUT"
FAILS=0
PASS() { echo "  ok: $1"; }
FAIL() { echo "  FAIL: $1"; FAILS=$((FAILS+1)); }

BGJOB='for i in 1 2 3 4 5 6 7 8; do while :; do :; done & done; wait'

# ---- arm 1: full valley+yield cycle over PP-2 ----
MEMRA_BG_JOB="$BGJOB" MEMRA_PP_DEVICES=0,1 MEMRA_SERVE_SPEC=0 MEMRA_COMPAT=openai \
  MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=$ADDR \
  $BIN/memra-server > "$OUT/pp2-valley-server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for _ in $(seq 240); do curl -sf $BASE/health >/dev/null 2>&1 && break; sleep 2; done
curl -sf $BASE/health >/dev/null || { echo "PP-2 server did not come up; tail:"; tail -5 "$OUT/pp2-valley-server.log"; exit 1; }

mfield() { curl -sf $BASE/metrics | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d.get('$1') if '$2'=='' else d.get('$1',{}).get('$2',''))" ; }

# 1. valley: idle accrues + job launches.
sleep 4
IDLE=$(mfield serve_idle_seconds "")
ST=$(mfield bg state)
PID=$(mfield bg job_pid)
echo "valley: idle=$IDLE bg.state=$ST pid=$PID"
python3 -c "assert float('$IDLE') > 2, 'idle did not accrue'" \
  && PASS "PP-2 valley: serve_idle_seconds=$IDLE" || FAIL "idle=$IDLE"
[ "$ST" = running ] && PASS "job launched in valley (pid $PID)" || FAIL "job state=$ST"

# 2. traffic: job must yield ('T') while the request runs; request must complete.
python3 - "$BASE" "$PID" "$OUT" <<'PY'
import json, sys, threading, time, urllib.request
base, pid, out = sys.argv[1], sys.argv[2], sys.argv[3]
def state():
    try:
        s = open(f"/proc/{pid}/stat").read()
        return s[s.rfind(")")+1:].split()[0]
    except OSError:
        return "gone"
body = json.dumps({"model":"q27","messages":[{"role":"user","content":"Name three CUDA memory spaces."}],
                   "max_tokens":64,"temperature":0}).encode()
req = urllib.request.Request(base+"/v1/chat/completions", data=body,
                             headers={"Content-Type":"application/json"})
res = {}
def fire():
    t=time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        res["body"]=json.load(r)
    res["wall"]=time.monotonic()-t
th=threading.Thread(target=fire); t0=time.monotonic(); th.start()
while state()!="T" and time.monotonic()-t0<5.0:
    time.sleep(0.001)
dt=time.monotonic()-t0; final=state(); th.join()
row={"yield_wall_s":round(dt,4),"state_during":final,
     "req_wall_s":round(res.get("wall",-1),4),
     "completion_tokens":res.get("body",{}).get("usage",{}).get("completion_tokens",0)}
print("yield probe:", json.dumps(row))
open(out+"/pp2-valley-yield.json","w").write(json.dumps(row))
assert final=="T", "job never stopped under PP-2 traffic"
assert dt<0.5, f"yield {dt:.3f}s blew the 500ms bound"
assert row["completion_tokens"]>0, "request produced no tokens"
PY
[ $? -eq 0 ] && PASS "PP-2 yield cycle (job 'T' under traffic; request completed)" \
             || FAIL "PP-2 yield cycle"

# 3. valley again: resume.
sleep 4
ST2=$(mfield bg state); RES=$(mfield bg resumes); YS=$(mfield bg yields)
echo "post-traffic: bg.state=$ST2 yields=$YS resumes=$RES"
[ "$ST2" = running ] && [ "${RES:-0}" -ge 1 ] \
  && PASS "job resumed in the next valley (yields=$YS resumes=$RES)" \
  || FAIL "resume (state=$ST2 resumes=$RES)"
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null || true

# 4. VRAM arm: budget > min-free on the PAIR must be refused (weights resident on both).
FREEMIN=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
BIGBUDGET=$((FREEMIN + 50000))   # deliberately unfittable
MEMRA_BG_JOB="$BGJOB" MEMRA_BG_VRAM_MB=$BIGBUDGET MEMRA_PP_DEVICES=0,1 MEMRA_SERVE_SPEC=0 \
  MEMRA_COMPAT=openai MEMRA_MODELS="q27=$MODEL" MEMRA_ADDR=$ADDR \
  $BIN/memra-server > "$OUT/pp2-vram-server.log" 2>&1 &
SPID=$!
for _ in $(seq 240); do curl -sf $BASE/health >/dev/null 2>&1 && break; sleep 2; done
sleep 4
ST3=$(mfield bg state)
echo "vram arm: budget=${BIGBUDGET}MB vs min-free=${FREEMIN}MB -> bg.state=$ST3"
[ "$ST3" = refused_vram ] \
  && PASS "unfittable VRAM budget refused on the pair (min-across-GPUs)" \
  || FAIL "vram arm state=$ST3 (want refused_vram)"
grep -q "REFUSED" "$OUT/pp2-vram-server.log" \
  && PASS "refusal logged loudly" || FAIL "no REFUSED line in server log"
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null || true

echo "probe-pp2-valley: $FAILS failed"
[ $FAILS -eq 0 ]

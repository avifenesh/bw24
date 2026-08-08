#!/usr/bin/env bash
# lane/pp-prefill LEVER A final receipt: TTFT on the 4k prompt, FA default vs floor seam.
# The battery-1 TTFT probe (228-tok prompt) sits under the 512 SWA window — its whole
# request rides fa_prefill_view_ws on both arms, so it could not move. The 4k prompt is
# the shape lever A changes (and the shape the 89.5:1 traffic actually sends).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/ppserve-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
D=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
P4096=$HOME/step37/prompt-pp4096.txt
RAW=$HOME/ppserve-raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/leverA-ttft4k-$TS.log
PORT=8094; BASE=http://127.0.0.1:$PORT

ttft_probe() { # ttft_probe <label> <n>
python3 - "$1" "$2" << 'PYEOF'
import json, sys, time, urllib.request
label, n = sys.argv[1], int(sys.argv[2])
prompt = open("/home/ubuntu/step37/prompt-pp4096.txt").read()
base = "http://127.0.0.1:8094"
ttfts = []
for i in range(n):
    body = {"model": "step35", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); ttft = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            try: obj = json.loads(line[6:])
            except json.JSONDecodeError: continue
            ch = obj.get("choices") or []
            # step35 opens <think> unconditionally: the first tokens stream in the
            # delta's REASONING field, not content (the step-sku think-mode trap) —
            # TTFT = first generated token of either kind.
            d = (ch[0].get("delta") or {}) if ch else {}
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                if ttft is None: ttft = time.monotonic() - t0
    if ttft is None:
        print(f"  req {i}: NO delta seen (stream shape changed?) — skipping"); continue
    ttfts.append(ttft)
    print(f"  req {i}: ttft={ttft:.3f}s")
ttfts.sort()
p50 = ttfts[len(ttfts)//2]; p95 = ttfts[max(0, int(len(ttfts)*0.95)-1)]
print(f"{label}: N={n} ttft p50={p50:.3f}s p95={p95:.3f}s min={ttfts[0]:.3f} max={ttfts[-1]:.3f}")
PYEOF
}

boot() { # boot <extra env...>
  env MEMRA_MODELS="step35=${M}+${D}" MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0 \
      MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_ADDR=127.0.0.1:$PORT "$@" \
      ./target/release/memra-server > "$RAW/ttft4k-server-$TS-$#.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do
    sleep 5
    curl -sf "$BASE/readyz" >/dev/null 2>&1 && { echo "ready ~$((i*5))s"; return 0; }
    kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; return 1; }
  done
  return 1
}

{
echo "=== leverA ttft4k $TS (commit 82b216b8 rsync)"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"

  echo; echo "--- arm FA (default), 4k prompt, stream, N=5 + 1 warmup ---"
  boot MEMRA_TAG=fa || exit 1
  ttft_probe warmup-fa 1
  ttft_probe FA 5
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 3

  echo; echo "--- arm FLOOR (MEMRA_STEP35_SWA_FA=0), same shape ---"
  boot MEMRA_STEP35_SWA_FA=0 || exit 1
  ttft_probe warmup-floor 1
  ttft_probe FLOOR 5
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 2

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== ttft4k rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"

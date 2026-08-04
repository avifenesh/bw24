#!/usr/bin/env bash
# pro6000-prod: Battery D — cold-TTFT N=5 per artifact via cache_salt (forces prefix-cache
# namespace miss per rep), plus warm-TTFT N=3 (explicit cache-hit row).
set -u
cd /root/bw24
R=/root/receipts/serve
mkdir -p "$R"
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR

log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/driver.log"; }

start_server() {
  env MEMRA_SERVE_SPEC=0 MEMRA_MODELS="q27=$1" MEMRA_ADDR=$ADDR target/release/memra-server > "$2" 2>&1 &
  SPID=$!
  for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up"; tail -5 "$2"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }

ttft_salted() { # $1 out-prefix $2 rep $3 salt
  python3 - "$1" "$2" "$3" <<'EOF'
import json, sys, time, urllib.request
prefix, rep, salt = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = open("research/e2e/prompts/pp512.txt").read()
body = {"model": "q27", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64, "temperature": 0, "stream": True}
if salt != "none":
    body["cache_salt"] = salt
req = urllib.request.Request("http://127.0.0.1:8199/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.monotonic(); tfirst = None; ntok = 0
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        delta = d["choices"][0].get("delta", {})
        if delta.get("content") or delta.get("reasoning"):
            ntok += 1
            if tfirst is None:
                tfirst = time.monotonic()
tend = time.monotonic()
res = {"rep": rep, "salt": salt, "ttft_s": round(tfirst - t0, 4) if tfirst else None,
       "total_s": round(tend - t0, 4), "stream_tokens": ntok,
       "decode_tokps": round((ntok - 1) / (tend - tfirst), 2) if tfirst and ntok > 1 else None}
print(json.dumps(res))
with open(f"{prefix}-ttft-cold.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
EOF
}

for arm in nv q8; do
  M=$NV; [ "$arm" = q8 ] && M=$Q8
  log "D $arm: starting plain server for cold-TTFT"
  start_server "$M" "$R/server-$arm-ttft.log" || continue
  for r in 1 2 3 4 5; do
    ttft_salted "$R/$arm" "$r" "cold-$arm-$r-$(date +%s%N)" >> "$R/load-$arm-ttft.log" 2>&1
    log "D $arm cold r$r: $(tail -1 "$R/$arm-ttft-cold.jsonl")"
  done
  for r in 1 2 3; do
    ttft_salted "$R/$arm" "warm$r" "warmshared" >> "$R/load-$arm-ttft.log" 2>&1
    log "D $arm warm r$r: $(tail -1 "$R/$arm-ttft-cold.jsonl")"
  done
  stop_server
done
log "BATTERY-D DONE"
echo "BATTERY-D DONE"

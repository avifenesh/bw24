#!/usr/bin/env bash
# pro6000-prod: Battery B — prod serving rows through memra-server.
# Per artifact (nv = daily NVFP4+MTP, q8 = Q8_0 prod):
#   B1 plain serve c=1 decode tok/s (spec off), N=5
#   B2 spec serve c=1 (draft attached) K=2..5 sweep N=2, best-K N=5
#   B3 TTFT + serve-path prefill: streaming chat, ~pp512-length prompt, N=5
#   B4 batched serving c=2/4/8 aggregate (spec off), N=3 passes each
# 600W fixed (nvidia-smi -pl container-blocked). Single-tenant GPU.
set -u
cd /root/bw24
R=/root/receipts/serve
mkdir -p "$R"
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
DRAFT=/root/models/draft-owntrim-nvfp4head-q4blk.gguf
ADDR=127.0.0.1:8199
BASE=http://$ADDR
PP512=research/e2e/prompts/pp512.txt

log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }

nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv -l 1 > "$R/gpu-1hz.csv" 2>&1 &
SMPID=$!
trap 'kill $SMPID 2>/dev/null' EXIT

start_server() { # $1 extra-env string, $2 models spec, $3 logfile
  env $1 MEMRA_MODELS="$2" MEMRA_ADDR=$ADDR target/release/memra-server > "$3" 2>&1 &
  SPID=$!
  for _ in $(seq 300); do curl -sf $BASE/health >/dev/null 2>&1 && return 0; sleep 2; done
  echo "server did not come up: $3"; tail -5 "$3"; return 1
}
stop_server() { kill "${SPID:-0}" 2>/dev/null; wait "${SPID:-0}" 2>/dev/null || true; sleep 2; }

# TTFT probe: streaming chat with the pp512 prompt text; wall to first content delta + total.
ttft_probe() { # $1 outfile-prefix $2 rep
  python3 - "$1" "$2" <<'EOF'
import json, sys, time, urllib.request
prefix, rep = sys.argv[1], sys.argv[2]
prompt = open("research/e2e/prompts/pp512.txt").read()
body = {"model": "q27", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64, "temperature": 0, "stream": True}
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
res = {"rep": rep, "ttft_s": round(tfirst - t0, 4) if tfirst else None,
       "total_s": round(tend - t0, 4), "stream_tokens": ntok,
       "decode_tokps": round((ntok - 1) / (tend - tfirst), 2) if tfirst and ntok > 1 else None}
print(json.dumps(res))
with open(f"{prefix}-ttft.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
EOF
}

for arm in nv q8; do
  M=$NV; [ "$arm" = q8 ] && M=$Q8

  # ---- B1 + B3: plain serve (spec off): c=1 N=5, TTFT N=5, then B4 c=2/4/8
  log "B1 $arm: starting plain server"
  start_server "MEMRA_SERVE_SPEC=0" "q27=$M" "$R/server-$arm-plain.log" || continue
  log "B1 $arm: server up | $(gpustate)"
  for r in 1 2 3 4 5; do
    python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 4 \
      --max-tokens 128 --out "$R/points-$arm-plain.jsonl" --label "c1-$arm-r$r" >> "$R/load-$arm-plain.log" 2>&1
    log "B1 $arm c1 r$r done: $(tail -1 "$R/points-$arm-plain.jsonl" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(f"agg={d[\"agg_tok_s\"]} p50={d[\"lat_p50_s\"]}")' 2>/dev/null || echo parse-fail)"
  done
  for r in 1 2 3 4 5; do
    ttft_probe "$R/$arm" "$r" >> "$R/load-$arm-plain.log" 2>&1
    log "B3 $arm ttft r$r: $(tail -1 "$R/$arm-ttft.jsonl" 2>/dev/null)"
  done
  for pass in 1 2 3; do
    for c in 2 4 8 16 32; do
      python3 tools/load-serve.py --base $BASE --model q27 --concurrency $c --requests $((c*8)) \
        --max-tokens 128 --out "$R/points-$arm-batch.jsonl" --label "c$c-$arm-p$pass" >> "$R/load-$arm-batch.log" 2>&1
      log "B4 $arm c$c p$pass done: $(tail -1 "$R/points-$arm-batch.jsonl" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(f"agg={d[\"agg_tok_s\"]} p50={d[\"lat_p50_s\"]} p95={d[\"lat_p95_s\"]} err={d[\"n_err\"]}")' 2>/dev/null || echo parse-fail)"
    done
  done
  stop_server
  log "B1/B3/B4 $arm: server stopped"

  # ---- B2: spec serve K sweep (server restart per K; draft attached)
  for K in 2 3 4 5; do
    log "B2 $arm K=$K: starting spec server"
    start_server "MEMRA_SPEC_K=$K" "q27=$M+$DRAFT" "$R/server-$arm-spec-k$K.log" || continue
    NREP=2; [ "$K" = 3 ] && NREP=5   # board class K=3 gets N=5
    for r in $(seq $NREP); do
      python3 tools/load-serve.py --base $BASE --model q27 --concurrency 1 --requests 4 \
        --max-tokens 128 --out "$R/points-$arm-spec.jsonl" --label "spec-k$K-$arm-r$r" >> "$R/load-$arm-spec.log" 2>&1
      log "B2 $arm spec K=$K r$r done: $(tail -1 "$R/points-$arm-spec.jsonl" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(f"agg={d[\"agg_tok_s\"]} p50={d[\"lat_p50_s\"]}")' 2>/dev/null || echo parse-fail)"
    done
    stop_server
  done
  log "B2 $arm done"
done

log "BATTERY-B DONE"
echo "BATTERY-B DONE"

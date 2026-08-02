#!/usr/bin/env bash
# Fleet capacity re-sweep — phase-3 paid-window deliverable (lane/fleet-resweep).
# Replica-scaling: fleet sizes 1 (baseline dev4) / 4 (1 per GPU) / 8 (2 per GPU) /
# 12 (3 per GPU) on devices 4-7. Direct per-replica harness at c=8/replica (the
# R3.3 matched-saturation protocol, research/darklane-serving-20260801), c=16/replica
# secondary at f4/f8 (exact-16 tier era). Multi-tenant QoS probe at f8 through the
# admission proxy (cap 8): interactive c=4 alone vs under bulk c=96 (fleet-cap-resweep
# probe shape). N=3, fleet sizes interleaved pass-wise. Params baked as literals.
set -uo pipefail
OUT=/home/ubuntu/receipts/fleet-resweep
BIN=/home/ubuntu/memra/target/release/memra-server
MODEL=/opt/dlami/nvme/models/Qwen3.5-9B-Q8_0.gguf
LS=/home/ubuntu/memra/tools/load-serve.py
FLEET=/home/ubuntu/memra/tools/serve-fleet.sh
DLOG=$OUT/driver.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }

nvidia-smi --query-gpu=timestamp,index,memory.used,temperature.gpu,power.draw \
  --format=csv -l 1 -i 4,5,6,7 > "$OUT/vram-1hz.csv" 2>&1 &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

ports_for(){
  case $1 in
    1)  echo "9085";;
    4)  echo "9085 9086 9087 9088";;
    8)  echo "9085 9086 9087 9088 9089 9090 9091 9092";;
    12) echo "9085 9086 9087 9088 9089 9090 9091 9092 9093 9094 9095 9096";;
  esac
}

fenv(){ # $1=size -> sets gpus/rpg globals
  case $1 in
    1)  FG="4";       FR=1;;
    4)  FG="4 5 6 7"; FR=1;;
    8)  FG="4 5 6 7"; FR=2;;
    12) FG="4 5 6 7"; FR=3;;
  esac
}

fstart(){ local size=$1 pass=$2; fenv "$size"
  GPUS="$FG" REPLICAS_PER_GPU=$FR MODEL=$MODEL BASE_PORT=9085 PROXY_PORT=9080 \
    CAP=8 SERVER_BIN=$BIN LOAD_GRACE=300 FLEET_RUN=$OUT/fleet-f$size-p$pass \
    bash $FLEET start 2>&1 | tee -a "$DLOG"
}

fstop(){ local size=$1 pass=$2; fenv "$size"
  GPUS="$FG" REPLICAS_PER_GPU=$FR MODEL=$MODEL BASE_PORT=9085 PROXY_PORT=9080 \
    CAP=8 SERVER_BIN=$BIN FLEET_RUN=$OUT/fleet-f$size-p$pass \
    bash $FLEET stop 2>&1 | tee -a "$DLOG"
}

gprobe(){ local port=$1 tag=$2 h
  h=$(curl -s -m 90 http://127.0.0.1:$port/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"List the first eight prime numbers, comma-separated, and nothing else.\"}],\"max_tokens\":64,\"temperature\":0,\"seed\":0,\"stream\":false}" \
    | python3 -c "import json,sys,hashlib
try:
    d=json.load(sys.stdin); c=d[\"choices\"][0][\"message\"][\"content\"]
    print(hashlib.sha256(c.encode()).hexdigest()[:16])
except Exception as e:
    print(\"ERR\", type(e).__name__)")
  echo "GREEDY_HASH $tag $port $h" | tee -a "$OUT/greedy-hashes.txt" >> "$DLOG"
}

cell_direct(){ local size=$1 pass=$2 conc=$3 reqs=$4 port pids=()
  log "cell f$size c${conc}pr p$pass (direct, $reqs req/replica)"
  for port in $(ports_for "$size"); do
    python3 $LS --base http://127.0.0.1:$port --concurrency $conc --requests $reqs \
      --model qwen --max-tokens 128 --out $OUT/points.jsonl \
      --per-request $OUT/perreq/f$size-c${conc}pr-p$pass-r$port.jsonl \
      --label f$size-c${conc}pr-p$pass-r$port \
      >> "$OUT/logs/load-f$size-p$pass.log" 2>&1 &
    pids+=($!)
  done
  local p; for p in "${pids[@]}"; do wait "$p"; done
}

qos_cell(){ local pass=$1
  log "QoS f8 p$pass: interactive alone"
  python3 $LS --base http://127.0.0.1:9080 --concurrency 4 --requests 24 --model qwen \
    --max-tokens 128 --out $OUT/points.jsonl \
    --per-request $OUT/perreq/f8-qosint-alone-p$pass.jsonl \
    --label f8-qosint-alone-p$pass >> "$OUT/logs/qos-p$pass.log" 2>&1
  log "QoS f8 p$pass: bulk c96 + interactive c4 contended"
  python3 $LS --base http://127.0.0.1:9080 --concurrency 96 --requests 288 --model qwen \
    --max-tokens 128 --out $OUT/points.jsonl \
    --per-request $OUT/perreq/f8-qosbulk-p$pass.jsonl \
    --label f8-qosbulk-p$pass >> "$OUT/logs/qos-p$pass.log" 2>&1 &
  local BULK=$!
  sleep 5
  python3 $LS --base http://127.0.0.1:9080 --concurrency 4 --requests 24 --model qwen \
    --max-tokens 128 --out $OUT/points.jsonl \
    --per-request $OUT/perreq/f8-qosint-cont-p$pass.jsonl \
    --label f8-qosint-cont-p$pass >> "$OUT/logs/qos-p$pass.log" 2>&1
  wait $BULK
  curl -s -m 5 http://127.0.0.1:9080/metrics > "$OUT/metrics-f8-qos-p$pass.json" 2>&1
}

log "campaign start — binary $BIN, model $MODEL, devices 4-7, ports 9080/9085+"
sha256sum "$BIN" | tee -a "$DLOG" > "$OUT/binary-sha256.txt"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv \
  > "$OUT/gpu-state-pre.txt" 2>&1

for pass in 1 2 3; do
  for size in 1 4 8 12; do
    log "=== pass $pass fleet f$size UP ==="
    if ! fstart "$size" "$pass"; then
      log "FATAL bring-up f$size p$pass — recording and continuing"
      fstop "$size" "$pass"; sleep 8; continue
    fi
    sleep 5
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,5,6,7 \
      > "$OUT/vram-f$size-p$pass-up.txt" 2>&1
    for port in $(ports_for "$size"); do gprobe "$port" "f$size-p$pass"; done
    if [ "$size" -eq 1 ]; then
      for cr in "8 32" "16 64" "32 128"; do
        set -- $cr
        log "cell f1 c$1 p$pass"
        python3 $LS --base http://127.0.0.1:9085 --concurrency $1 --requests $2 \
          --model qwen --max-tokens 128 --out $OUT/points.jsonl \
          --per-request $OUT/perreq/f1-c$1-p$pass.jsonl \
          --label f1-c$1-p$pass >> "$OUT/logs/load-f1-p$pass.log" 2>&1
      done
    else
      cell_direct "$size" "$pass" 8 64
      if [ "$size" -ne 12 ]; then cell_direct "$size" "$pass" 16 64; fi
      if [ "$size" -eq 8 ]; then qos_cell "$pass"; fi
    fi
    log "=== pass $pass fleet f$size DOWN ==="
    fstop "$size" "$pass"
    sleep 8
  done
done

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv \
  > "$OUT/gpu-state-post.txt" 2>&1
log "campaign COMPLETE"

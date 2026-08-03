#!/usr/bin/env bash
# QoS p95 fix-and-verify (lane/qos-p95, 2026-08-02) — the fleet-resweep QoS@8 scenario
# re-run with the engine-side x-lane SLO gate (lane/dl-metering QoS extraction) ported
# onto the v0.67 serve surface.
#
# Four conditions, same window, interleaved pass-wise (the clock-drift law):
#   off8  — no x-lane headers, proxy CAP=8: the exact fleet-resweep QoS@8 mechanism
#           (lane-blind FIFO + per-backend cap; naked traffic through the gate binary
#           takes the identical engine code path — verified, greedy 56b8502cfb8de57a).
#   on8   — bulk sends x-lane: harvest (+ retry-shed), interactive x-lane: interactive,
#           proxy CAP=8: the literal "engine gate through the unchanged proxy" question.
#           Mechanism risk (smoke): at <=8 sessions/replica step p99 ~19ms << 45ms shed
#           threshold — the gate may be inert when the proxy starves it of visibility.
#   off16 — no lanes, CAP=16: cap-16 alone (attribution control; m2-pp8 found cap 16
#           +17.3% aggregate at c=96 on stacked replicas).
#   on16  — lanes on, CAP=16: queueing moves INTO the engine where the lane gate
#           manages it (harvest lane cap 8/replica sheds the 9th; interactive always
#           admits, decode-rows-first, dark prefill only in SLO headroom).
#
# Scenario per cell (the exact fleet-resweep QoS@8 probe shape):
#   fleet f8 = devices 4-7, 2 replicas/GPU (ports 9085-9092), proxy :9080;
#   interactive alone c=4 24 req; bulk c=96 288 req started 5s before interactive
#   contended c=4 24 req. N=3 passes, conditions interleaved within each pass, full
#   teardown/bring-up per cell (12 bring-ups).
# Params baked as literals (workflow-args do not propagate).
set -uo pipefail
OUT=/home/ubuntu/receipts/qos-p95
BIN=/home/ubuntu/memra/target/release/memra-server
MODEL=/opt/dlami/nvme/models/Qwen3.5-9B-Q8_0.gguf
LS=/home/ubuntu/memra/tools/load-serve.py
FLEET=/home/ubuntu/memra/tools/serve-fleet.sh
DLOG=$OUT/driver.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }
mkdir -p "$OUT/perreq" "$OUT/logs"

nvidia-smi --query-gpu=timestamp,index,memory.used,temperature.gpu,power.draw \
  --format=csv -l 1 -i 4,5,6,7 > "$OUT/vram-1hz.csv" 2>&1 &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

PORTS="9085 9086 9087 9088 9089 9090 9091 9092"

fstart(){ local cell=$1 cap=$2
  GPUS="4 5 6 7" REPLICAS_PER_GPU=2 MODEL=$MODEL BASE_PORT=9085 PROXY_PORT=9080 \
    CAP=$cap SERVER_BIN=$BIN LOAD_GRACE=300 FLEET_RUN=$OUT/fleet-$cell \
    bash $FLEET start 2>&1 | tee -a "$DLOG"
}
fstop(){ local cell=$1 cap=$2
  GPUS="4 5 6 7" REPLICAS_PER_GPU=2 MODEL=$MODEL BASE_PORT=9085 PROXY_PORT=9080 \
    CAP=$cap SERVER_BIN=$BIN FLEET_RUN=$OUT/fleet-$cell \
    bash $FLEET stop 2>&1 | tee -a "$DLOG"
}

gprobe(){ local port=$1 tag=$2 lane=$3 h hdr=()
  [ "$lane" != "none" ] && hdr=(-H "x-lane: $lane")
  h=$(curl -s -m 90 http://127.0.0.1:$port/v1/chat/completions \
    -H "Content-Type: application/json" "${hdr[@]}" \
    -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"List the first eight prime numbers, comma-separated, and nothing else.\"}],\"max_tokens\":64,\"temperature\":0,\"seed\":0,\"stream\":false}" \
    | python3 -c "import json,sys,hashlib
try:
    d=json.load(sys.stdin); c=d[\"choices\"][0][\"message\"][\"content\"]
    print(hashlib.sha256(c.encode()).hexdigest()[:16])
except Exception as e:
    print(\"ERR\", type(e).__name__)")
  echo "GREEDY_HASH $tag $port lane=$lane $h" | tee -a "$OUT/greedy-hashes.txt" >> "$DLOG"
}

yield_snap(){ local cell=$1 port
  for port in $PORTS; do
    curl -s -m 5 "http://127.0.0.1:$port/yield/metrics" \
      > "$OUT/yield-$cell-r$port.json" 2>&1 || true
  done
}

qos_cell(){ local cell=$1 mode=$2   # mode=off|on
  local ILANE=() BLANE=() BRETRY=()
  if [ "$mode" = on ]; then
    ILANE=(--lane interactive --tenant tenant-int)
    BLANE=(--lane harvest --tenant tenant-bulk)
    BRETRY=(--retry-shed)
  fi
  log "cell $cell ($mode): interactive alone c=4 24req"
  python3 $LS --base http://127.0.0.1:9080 --concurrency 4 --requests 24 --model qwen \
    --max-tokens 128 "${ILANE[@]}" --out $OUT/points.jsonl \
    --per-request $OUT/perreq/$cell-int-alone.jsonl \
    --label $cell-int-alone >> "$OUT/logs/$cell.log" 2>&1
  log "cell $cell ($mode): bulk c=96 288req + interactive c=4 24req contended"
  python3 $LS --base http://127.0.0.1:9080 --concurrency 96 --requests 288 --model qwen \
    --max-tokens 128 "${BLANE[@]}" "${BRETRY[@]}" --out $OUT/points.jsonl \
    --per-request $OUT/perreq/$cell-bulk.jsonl \
    --label $cell-bulk >> "$OUT/logs/$cell.log" 2>&1 &
  local BULK=$!
  sleep 5
  python3 $LS --base http://127.0.0.1:9080 --concurrency 4 --requests 24 --model qwen \
    --max-tokens 128 "${ILANE[@]}" --out $OUT/points.jsonl \
    --per-request $OUT/perreq/$cell-int-cont.jsonl \
    --label $cell-int-cont >> "$OUT/logs/$cell.log" 2>&1
  wait $BULK
  curl -s -m 5 http://127.0.0.1:9080/metrics > "$OUT/metrics-$cell.json" 2>&1
}

log "campaign start — gate binary $BIN, model $MODEL, devices 4-7, f8 proxy :9080"
sha256sum "$BIN" | tee -a "$DLOG" > "$OUT/binary-sha256.txt"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv \
  > "$OUT/gpu-state-pre.txt" 2>&1

for pass in 1 2 3; do
  for cond in off8 on8 off16 on16; do
    cell="p$pass-$cond"
    case $cond in
      off8)  mode=off; cap=8;;
      on8)   mode=on;  cap=8;;
      off16) mode=off; cap=16;;
      on16)  mode=on;  cap=16;;
    esac
    log "=== pass $pass cond $cond (cap $cap) UP ==="
    if ! fstart "$cell" "$cap"; then
      log "FATAL bring-up $cell — recording and continuing"
      fstop "$cell" "$cap"; sleep 8; continue
    fi
    sleep 5
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,5,6,7 \
      > "$OUT/vram-$cell-up.txt" 2>&1
    if [ "$mode" = on ]; then
      for port in $PORTS; do gprobe "$port" "$cell" interactive; done
      gprobe 9085 "$cell" harvest
    else
      for port in $PORTS; do gprobe "$port" "$cell" none; done
    fi
    qos_cell "$cell" "$mode"
    yield_snap "$cell"
    log "=== pass $pass cond $cond DOWN ==="
    fstop "$cell" "$cap"
    sleep 8
  done
done

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv \
  > "$OUT/gpu-state-post.txt" 2>&1
log "campaign done"

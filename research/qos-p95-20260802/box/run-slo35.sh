#!/usr/bin/env bash
# QoS p95 confirm sweep (lane/qos-p95): the SLO knob as the p50/p95 dial.
# Main campaign found on16 (gate + cap16) restores the tail class (p95 7.15 -> 3.69) but
# interactive p50 rode up (1.69 -> 2.39s): hypothesis — the gate defends STEP p99 (50ms),
# so admitted request latency scales ~ step_slo x gen_len; the SLO env is the dial, and
# if p50 improves at slo25 the mechanism is admitted-load (not the shed-retry storm).
# Conditions: slo50 (= on16 repeat) vs slo25 (MEMRA_SLO_P99_MS=25), N=3 interleaved.
set -uo pipefail
OUT=/home/ubuntu/receipts/qos-p95
BIN=/home/ubuntu/memra/target/release/memra-server
MODEL=/opt/dlami/nvme/models/Qwen3.5-9B-Q8_0.gguf
LS=/home/ubuntu/memra/tools/load-serve.py
FLEET=/home/ubuntu/memra/tools/serve-fleet.sh
DLOG=$OUT/driver-slo35.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }
mkdir -p "$OUT/perreq" "$OUT/logs"

PORTS="9085 9086 9087 9088 9089 9090 9091 9092"

fstart(){ local cell=$1 slo=$2
  MEMRA_SLO_P99_MS=$slo \
  GPUS="4 5 6 7" REPLICAS_PER_GPU=2 MODEL=$MODEL BASE_PORT=9085 PROXY_PORT=9080 \
    CAP=16 SERVER_BIN=$BIN LOAD_GRACE=300 FLEET_RUN=$OUT/fleet-$cell \
    bash $FLEET start 2>&1 | tee -a "$DLOG"
}
fstop(){ local cell=$1
  GPUS="4 5 6 7" REPLICAS_PER_GPU=2 MODEL=$MODEL BASE_PORT=9085 PROXY_PORT=9080 \
    CAP=16 SERVER_BIN=$BIN FLEET_RUN=$OUT/fleet-$cell \
    bash $FLEET stop 2>&1 | tee -a "$DLOG"
}

qos_cell(){ local cell=$1
  log "cell $cell: interactive alone c=4 24req"
  python3 $LS --base http://127.0.0.1:9080 --concurrency 4 --requests 24 --model qwen \
    --max-tokens 128 --lane interactive --tenant tenant-int --out $OUT/points-slo35.jsonl \
    --per-request $OUT/perreq/$cell-int-alone.jsonl \
    --label $cell-int-alone >> "$OUT/logs/$cell.log" 2>&1
  log "cell $cell: bulk c=96 288req + interactive c=4 24req contended"
  python3 $LS --base http://127.0.0.1:9080 --concurrency 96 --requests 288 --model qwen \
    --max-tokens 128 --lane harvest --tenant tenant-bulk --retry-shed \
    --out $OUT/points-slo35.jsonl \
    --per-request $OUT/perreq/$cell-bulk.jsonl \
    --label $cell-bulk >> "$OUT/logs/$cell.log" 2>&1 &
  local BULK=$!
  sleep 5
  python3 $LS --base http://127.0.0.1:9080 --concurrency 4 --requests 24 --model qwen \
    --max-tokens 128 --lane interactive --tenant tenant-int --out $OUT/points-slo35.jsonl \
    --per-request $OUT/perreq/$cell-int-cont.jsonl \
    --label $cell-int-cont >> "$OUT/logs/$cell.log" 2>&1
  wait $BULK
  curl -s -m 5 http://127.0.0.1:9080/metrics > "$OUT/metrics-$cell.json" 2>&1
  local port
  for port in $PORTS; do
    curl -s -m 5 "http://127.0.0.1:$port/yield/metrics" \
      > "$OUT/yield-$cell-r$port.json" 2>&1 || true
  done
}

log "slo confirm sweep start — cap 16, lanes on, slo50 vs slo25, N=3 interleaved"
for pass in 1 2 3; do
  for slo in 35; do
    cell="s$pass-slo$slo"
    log "=== pass $pass slo $slo UP ==="
    if ! fstart "$cell" "$slo"; then
      log "FATAL bring-up $cell — recording and continuing"
      fstop "$cell"; sleep 8; continue
    fi
    sleep 5
    qos_cell "$cell"
    log "=== pass $pass slo $slo DOWN ==="
    fstop "$cell"
    sleep 8
  done
done
log "slo confirm sweep done"

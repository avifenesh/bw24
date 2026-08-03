#!/usr/bin/env bash
# Fleet cap re-sweep 8/12/16 — phase-3 stale-verdict lever (box-aug2-mission §2c).
# Cap 8 was calibrated on the v0.59 core; this binary moved the tick. The IN-WINDOW
# cap-8 pass is the denominator (clock-drift law); cap arms interleave PASS-WISE.
# Params baked as literals (workflow args do not propagate). Receipts are the raw
# JSONL rows (tee/append) — logs ARE the deliverable.
set -uo pipefail
cd ~/memra
OUT=~/receipts/fleet-cap-resweep
mkdir -p "$OUT"
MODEL=/opt/dlami/nvme/models/Qwen3.5-9B-Q8_0.gguf

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv \
  > "$OUT/gpu-state-pre-fleet.txt" 2>&1

for pass in 1 2; do
  for CAP in 8 12 16; do
    echo "=== pass $pass cap $CAP up ==="
    GPUS="5 6 7" REPLICAS_PER_GPU=2 CAP=$CAP MODEL=$MODEL \
      FLEET_RUN=$OUT/fleet-cap$CAP-p$pass tools/serve-fleet.sh start \
      2>&1 | tee -a "$OUT/fleet-driver.log"
    sleep 45   # health settle
    python3 tools/load-serve.py --base http://127.0.0.1:8080 --concurrency 6 \
      --requests 6 --greedy --model qwen --out "$OUT/greedy.jsonl" \
      --label cap$CAP-greedy-p$pass 2>&1 | tee -a "$OUT/fleet-driver.log"
    python3 tools/load-serve.py --base http://127.0.0.1:8080 --concurrency 48 \
      --requests 192 --model qwen --out "$OUT/points.jsonl" \
      --label cap$CAP-c48-p$pass 2>&1 | tee -a "$OUT/fleet-driver.log"
    python3 tools/load-serve.py --base http://127.0.0.1:8080 --concurrency 96 \
      --requests 288 --model qwen --out "$OUT/points.jsonl" \
      --label cap$CAP-c96-p$pass 2>&1 | tee -a "$OUT/fleet-driver.log"
    # multi-tenant QoS probe rides the same fleet: a latency-sensitive low-c tenant
    # UNDER the c=96 bulk tenant — per-request tail lands in qos-tenant.jsonl.
    python3 tools/load-serve.py --base http://127.0.0.1:8080 --concurrency 96 \
      --requests 192 --model qwen --out "$OUT/points.jsonl" \
      --per-request "$OUT/qos-bulk-cap$CAP-p$pass.jsonl" \
      --label cap$CAP-qosbulk-p$pass 2>&1 | tee -a "$OUT/fleet-driver.log" &
    BULK=$!
    sleep 5    # bulk tenant saturates first
    python3 tools/load-serve.py --base http://127.0.0.1:8080 --concurrency 4 \
      --requests 24 --model qwen --out "$OUT/qos-tenant.jsonl" \
      --per-request "$OUT/qos-lat-cap$CAP-p$pass.jsonl" \
      --label cap$CAP-qoslat-p$pass 2>&1 | tee -a "$OUT/fleet-driver.log"
    wait $BULK
    GPUS="5 6 7" FLEET_RUN=$OUT/fleet-cap$CAP-p$pass tools/serve-fleet.sh stop \
      2>&1 | tee -a "$OUT/fleet-driver.log"
    sleep 10
  done
done

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv \
  > "$OUT/gpu-state-post-fleet.txt" 2>&1
echo "==== fleet cap re-sweep complete ===="

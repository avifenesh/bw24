#!/usr/bin/env bash
# serve-fleet: declarative replica-fleet supervisor for darklanes serving.
#
# Brings up REPLICAS_PER_GPU memra-server replicas on each GPU in GPUS, fronts them
# with the admission proxy (tools/serve-proxy.py, per-backend cap), and supervises:
# a health loop restarts any replica or proxy that dies (with a load grace period).
# systemd-free by design (userland box); pidfile discipline under $FLEET_RUN.
#
# Defaults = the measured darklanes sweet spot (research/darklane-serving-20260801/ R3):
# pairs on every serving GPU, admission cap 8/replica (exactness-tier batch width +
# timeslice anti-thrash bound). Ports are gpu-major from BASE_PORT.
#
# Usage:
#   tools/serve-fleet.sh start|stop|status|restart
#
# Config (env-overridable):
#   GPUS="5 6 7"  REPLICAS_PER_GPU=2  MODEL=~/models/Qwen3.5-9B-Q8_0.gguf
#   BASE_PORT=8085  PROXY_PORT=8080  CAP=8  QUEUE_MAX=256  QUEUE_DEADLINE=30
#   SERVER_BIN=~/memra/target/release/memra-server
#   PROXY_PY=<dir-of-this-script>/serve-proxy.py
#   FLEET_RUN=~/darklane-fleet  (pidfiles, logs)
#   HEALTH_INTERVAL=5  LOAD_GRACE=120   (seconds)
set -u

GPUS=${GPUS:-"5 6 7"}
REPLICAS_PER_GPU=${REPLICAS_PER_GPU:-2}
MODEL=${MODEL:-$HOME/models/Qwen3.5-9B-Q8_0.gguf}
BASE_PORT=${BASE_PORT:-8085}
PROXY_PORT=${PROXY_PORT:-8080}
CAP=${CAP:-8}
QUEUE_MAX=${QUEUE_MAX:-256}
QUEUE_DEADLINE=${QUEUE_DEADLINE:-30}
SERVER_BIN=${SERVER_BIN:-$HOME/memra/target/release/memra-server}
PROXY_PY=${PROXY_PY:-$(cd "$(dirname "$0")" && pwd)/serve-proxy.py}
FLEET_RUN=${FLEET_RUN:-$HOME/darklane-fleet}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-5}
LOAD_GRACE=${LOAD_GRACE:-120}

mkdir -p "$FLEET_RUN/logs"

# ---- fleet layout: PORTS[i] on GPU_OF[i], gpu-major from BASE_PORT ----
PORTS=() GPU_OF=()
p=$BASE_PORT
for g in $GPUS; do
  for _ in $(seq "$REPLICAS_PER_GPU"); do
    PORTS+=("$p"); GPU_OF+=("$g"); p=$((p + 1))
  done
done
BACKENDS=""
for port in "${PORTS[@]}"; do BACKENDS+="http://127.0.0.1:$port,"; done
BACKENDS=${BACKENDS%,}

# /health is the RESTART decision, and since lane/serve-hardening it means inference
# liveness (worker heartbeat + fault latches), not "the process is listening" — so a replica
# whose GPU worker panicked or whose card wedged now actually gets restarted here instead of
# sitting green forever. It 503s while weights load, which is what LOAD_GRACE covers, and
# stays 200 during a drain on purpose (a drain is a healthy shutdown; the supervisor must not
# fight it). ROUTING readiness is the proxy's job and asks /readyz.
healthy() { curl -sf -m 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }
pid_alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
pidfile_of() { echo "$FLEET_RUN/replica-$1.pid"; }

launch_replica() { # $1=port $2=gpu
  local port=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu MEMRA_COMPAT=openai \
    MEMRA_MODELS="qwen=$MODEL" MEMRA_ADDR=127.0.0.1:$port \
    nohup "$SERVER_BIN" >> "$FLEET_RUN/logs/replica-$port.log" 2>&1 < /dev/null &
  echo $! > "$(pidfile_of "$port")"
  date +%s > "$FLEET_RUN/replica-$port.launched"
  echo "[fleet] launched replica :$port on GPU $gpu (pid $!)"
}

launch_proxy() {
  nohup python3 "$PROXY_PY" --port "$PROXY_PORT" --backends "$BACKENDS" \
    --cap "$CAP" --queue-max "$QUEUE_MAX" --queue-deadline "$QUEUE_DEADLINE" \
    >> "$FLEET_RUN/logs/proxy.log" 2>&1 < /dev/null &
  echo $! > "$FLEET_RUN/proxy.pid"
  date +%s > "$FLEET_RUN/proxy.launched"
  echo "[fleet] launched proxy :$PROXY_PORT cap=$CAP -> ${#PORTS[@]} backends"
}

kill_port() { # kill whatever owns the port + its pidfile
  local port=$1 pid
  pid=$(cat "$(pidfile_of "$port")" 2>/dev/null || true)
  pid_alive "$pid" && kill "$pid" 2>/dev/null
  pid=$(lsof -t -i ":$port" 2>/dev/null || true)
  [ -n "$pid" ] && kill $pid 2>/dev/null
  rm -f "$(pidfile_of "$port")" "$FLEET_RUN/replica-$port.launched"
}

supervise() { # the health-restart loop (own pidfile; run in background from start)
  echo "[fleet] supervisor up (interval ${HEALTH_INTERVAL}s, grace ${LOAD_GRACE}s)"
  while true; do
    local i now launched
    now=$(date +%s)
    for i in "${!PORTS[@]}"; do
      local port=${PORTS[$i]} gpu=${GPU_OF[$i]}
      if ! healthy "$port"; then
        launched=$(cat "$FLEET_RUN/replica-$port.launched" 2>/dev/null || echo 0)
        if [ $((now - launched)) -gt "$LOAD_GRACE" ]; then
          echo "[fleet] $(date +%H:%M:%S) replica :$port UNHEALTHY past grace — restarting"
          kill_port "$port"
          sleep 2
          launch_replica "$port" "$gpu"
        fi
      fi
    done
    if ! healthy "$PROXY_PORT"; then
      launched=$(cat "$FLEET_RUN/proxy.launched" 2>/dev/null || echo 0)
      if [ $((now - launched)) -gt 15 ]; then
        echo "[fleet] $(date +%H:%M:%S) proxy UNHEALTHY — restarting"
        pid=$(cat "$FLEET_RUN/proxy.pid" 2>/dev/null || true)
        pid_alive "$pid" && kill "$pid" 2>/dev/null
        launch_proxy
      fi
    fi
    sleep "$HEALTH_INTERVAL"
  done
}

cmd_start() {
  [ -f "$MODEL" ] || { echo "[fleet] FATAL: no model at $MODEL"; exit 1; }
  [ -x "$SERVER_BIN" ] || { echo "[fleet] FATAL: no server bin at $SERVER_BIN"; exit 1; }
  if [ -f "$FLEET_RUN/supervisor.pid" ] && pid_alive "$(cat "$FLEET_RUN/supervisor.pid")"; then
    echo "[fleet] already running (supervisor pid $(cat "$FLEET_RUN/supervisor.pid")); use restart"
    exit 1
  fi
  local i
  for i in "${!PORTS[@]}"; do
    kill_port "${PORTS[$i]}"   # clear any strays on our ports
  done
  pid=$(lsof -t -i ":$PROXY_PORT" 2>/dev/null || true); [ -n "$pid" ] && kill $pid
  sleep 2
  for i in "${!PORTS[@]}"; do
    launch_replica "${PORTS[$i]}" "${GPU_OF[$i]}"
  done
  echo "[fleet] waiting for replicas (model load)..."
  local deadline=$(( $(date +%s) + LOAD_GRACE ))
  for i in "${!PORTS[@]}"; do
    while ! healthy "${PORTS[$i]}"; do
      [ "$(date +%s)" -gt "$deadline" ] && { echo "[fleet] FATAL: :${PORTS[$i]} not up"; exit 1; }
      sleep 2
    done
    echo "[fleet] replica :${PORTS[$i]} healthy"
  done
  launch_proxy
  sleep 2
  nohup bash "$0" __supervise >> "$FLEET_RUN/logs/supervisor.log" 2>&1 < /dev/null &
  echo $! > "$FLEET_RUN/supervisor.pid"
  echo "[fleet] up: ${#PORTS[@]} replicas on GPUs [$GPUS], proxy :$PROXY_PORT (cap $CAP)"
}

cmd_stop() {
  local pid
  pid=$(cat "$FLEET_RUN/supervisor.pid" 2>/dev/null || true)
  pid_alive "$pid" && kill "$pid" 2>/dev/null && echo "[fleet] supervisor stopped"
  rm -f "$FLEET_RUN/supervisor.pid"
  pid=$(cat "$FLEET_RUN/proxy.pid" 2>/dev/null || true)
  pid_alive "$pid" && kill "$pid" 2>/dev/null
  pid=$(lsof -t -i ":$PROXY_PORT" 2>/dev/null || true); [ -n "$pid" ] && kill $pid 2>/dev/null
  rm -f "$FLEET_RUN/proxy.pid" "$FLEET_RUN/proxy.launched"
  local i
  for i in "${!PORTS[@]}"; do
    kill_port "${PORTS[$i]}"
  done
  echo "[fleet] stopped"
}

cmd_status() {
  local pid i
  pid=$(cat "$FLEET_RUN/supervisor.pid" 2>/dev/null || true)
  if pid_alive "$pid"; then echo "supervisor: up (pid $pid)"; else echo "supervisor: DOWN"; fi
  if healthy "$PROXY_PORT"; then echo "proxy :$PROXY_PORT: up"; else echo "proxy :$PROXY_PORT: DOWN"; fi
  for i in "${!PORTS[@]}"; do
    if healthy "${PORTS[$i]}"; then
      echo "replica :${PORTS[$i]} (GPU ${GPU_OF[$i]}): up"
    else
      echo "replica :${PORTS[$i]} (GPU ${GPU_OF[$i]}): DOWN"
    fi
  done
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  restart) cmd_stop; sleep 3; cmd_start ;;
  __supervise) supervise ;;   # internal: nohup re-exec target (survives ssh HUP)
  *) echo "usage: $0 start|stop|status|restart"; exit 1 ;;
esac

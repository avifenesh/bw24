#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SERVER_BIN=${SERVER_BIN:-$ROOT/target/release/bw24-server}
OUT_ROOT=${OUT_ROOT:-$ROOT/research/per-expert-quant/results/spill-prefetch}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$OUT_ROOT/$RUN_ID"
WINDOWS=${WINDOWS:-"8 1 4"}
MMAP_ADVICE=${MMAP_ADVICE:-random}
MODEL=${MODEL:-plain_quant}
ADDR=${ADDR:-127.0.0.1:8080}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-1200}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-3600}
DRY_RUN=${DRY_RUN:-0}

usage() {
  cat <<'EOF'
Cold-cache A/B for mmap expert page-prefetch windows.

Required:
  ARTIFACT=/scratch/artifacts/plain-quant
  PROMPT=/path/to/frozen-prompt.txt

Optional:
  MODEL=plain_quant            server model alias
  WINDOWS="8 1 4"              ordered page-prefetch windows
  MMAP_ADVICE=random|normal    whole expert-map kernel advice
  OUT_ROOT=/path/to/results
  RUN_ID=unique-run-id
  ADDR=127.0.0.1:8080
  DRY_RUN=1                    validate and print the run plan without changing state
EOF
}

die() {
  echo "spill-prefetch-ab: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null || die "missing required command: $1"
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
elif (( $# != 0 )); then
  usage >&2
  die "unexpected arguments: $*"
fi

: "${ARTIFACT:?set ARTIFACT to a staged per-expert artifact directory}"
: "${PROMPT:?set PROMPT to a frozen prompt text file}"

for command in curl git iostat nvidia-smi pgrep pidstat python3 realpath sha256sum; do
  require_command "$command"
done
[[ -x "$SERVER_BIN" ]] || die "missing executable server: $SERVER_BIN"
[[ -d "$ARTIFACT" ]] || die "artifact is not a directory: $ARTIFACT"
[[ -f "$ARTIFACT/manifest.json" ]] || die "missing artifact manifest: $ARTIFACT/manifest.json"
[[ -s "$PROMPT" ]] || die "prompt is missing or empty: $PROMPT"
[[ -n "$MODEL" && "$MODEL" != *[=,]* ]] || die "MODEL must be a nonempty alias without '=' or ','"
[[ "$MMAP_ADVICE" == random || "$MMAP_ADVICE" == normal ]] \
  || die "MMAP_ADVICE must be random or normal"
[[ "$HEALTH_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || die "HEALTH_TIMEOUT_S must be a positive integer"
[[ "$REQUEST_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || die "REQUEST_TIMEOUT_S must be a positive integer"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || die "DRY_RUN must be 0 or 1"

read -r -a WINDOW_LIST <<< "$WINDOWS"
(( ${#WINDOW_LIST[@]} > 0 )) || die "WINDOWS must contain at least one integer"
for window in "${WINDOW_LIST[@]}"; do
  [[ "$window" =~ ^[0-9]+$ ]] || die "invalid page-prefetch window: $window"
done

ARTIFACT=$(realpath "$ARTIFACT")
PROMPT=$(realpath "$PROMPT")

python3 - "$ARTIFACT" <<'PY'
import os
import pathlib
import sys

if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
    raise SystemExit("Python/os does not expose POSIX_FADV_DONTNEED")
files = list((pathlib.Path(sys.argv[1]).resolve() / "experts").glob("*.bin"))
if not files:
    raise SystemExit("artifact has no experts/*.bin files")
PY

conflicts=$(
  {
    pgrep -ax bw24-server || true
    pgrep -af '^([^ ]*/)?python(3([.][0-9]+)?)? .*(prepare_mixed_expert_repack|validate_artifact)\.py' || true
    pgrep -ax rsync || true
  } 2>/dev/null
)
if [[ -n "$conflicts" ]]; then
  printf 'spill-prefetch-ab: refusing contaminated run; active server/builder/rsync:\n%s\n' \
    "$conflicts" >&2
  exit 2
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'artifact=%s\nprompt=%s\nmodel=%s\nwindows=%s\nmmap_advice=%s\nrun_dir=%s\n' \
    "$ARTIFACT" "$PROMPT" "$MODEL" "$WINDOWS" "$MMAP_ADVICE" "$RUN_DIR"
  exit 0
fi

[[ ! -e "$RUN_DIR" ]] || die "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

export ROOT ARTIFACT PROMPT MODEL WINDOWS MMAP_ADVICE RUN_ID RUN_DIR SERVER_BIN ADDR
python3 - "$RUN_DIR/metadata.json" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
from datetime import datetime, timezone

def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

root = pathlib.Path(os.environ["ROOT"])
artifact = pathlib.Path(os.environ["ARTIFACT"]).resolve()
prompt = pathlib.Path(os.environ["PROMPT"]).resolve()
manifest = artifact / "manifest.json"
server = pathlib.Path(os.environ["SERVER_BIN"]).resolve()
manifest_data = json.loads(manifest.read_text())
metadata = {
    "addr": os.environ["ADDR"],
    "artifact": str(artifact),
    "artifact_bytes": manifest_data.get("artifact_bytes"),
    "artifact_manifest_sha256": sha256(manifest),
    "bw24_commit": command("git", "-C", str(root), "rev-parse", "HEAD"),
    "bw24_status": command("git", "-C", str(root), "status", "--short"),
    "environment": {
        "BW24_COMPAT": "native",
        "BW24_CTX": "8192",
        "BW24_FAST": "1",
        "BW24_MMVQ": "1",
        "BW24_MOE_CACHE": "1",
        "BW24_MOE_GROUPED": "1",
        "BW24_MOE_MMAP_ADVICE": os.environ["MMAP_ADVICE"],
        "BW24_MOE_PAGE_PREFETCH": "1",
        "BW24_MOE_PREFETCH": "1",
        "BW24_MOE_PREWARM": "1",
        "BW24_MOE_RESIDENT": "1",
        "BW24_MOE_VRAM_FRAC": "0.85",
    },
    "environment_cleared": [
        "BW24_API_KEY",
        "BW24_FULL_PREC",
        "BW24_MOE_GATE",
        "BW24_MOE_RESIDENT_GB",
        "BW24_MOE_SLOTS",
        "BW24_MOE_STATS",
        "BW24_MOE_TRACE",
    ],
    "model": os.environ["MODEL"],
    "nvidia_smi": command(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader",
    ),
    "platform": platform.platform(),
    "prompt": str(prompt),
    "prompt_bytes": prompt.stat().st_size,
    "prompt_sha256": sha256(prompt),
    "run_id": os.environ["RUN_ID"],
    "server_binary": str(server),
    "server_binary_sha256": sha256(server),
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "windows": [int(value) for value in os.environ["WINDOWS"].split()],
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

SERVER_PID=
MONITOR_PIDS=()

stop_monitors() {
  local pid
  for pid in "${MONITOR_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${MONITOR_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  MONITOR_PIDS=()
}

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=
}

cleanup_cell() {
  stop_monitors
  stop_server
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  cleanup_cell
  exit "$status"
}
trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

evict_expert_pages() {
  local output=$1
  python3 - "$ARTIFACT" "$output" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

artifact = pathlib.Path(sys.argv[1]).resolve()
files = sorted((artifact / "experts").glob("*.bin"))
total = 0
for path in files:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        total += path.stat().st_size
    finally:
        os.close(fd)
result = {
    "advice": "POSIX_FADV_DONTNEED",
    "artifact": str(artifact),
    "bytes": total,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "files": len(files),
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(f"evicted_files={len(files)} bytes={total}")
PY
}

snapshot_proc() {
  local output=$1
  mkdir -p "$output"
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$output/timestamp-utc.txt"
  for name in stat status io schedstat; do
    if [[ -r "/proc/$SERVER_PID/$name" ]]; then
      cat "/proc/$SERVER_PID/$name" > "$output/process-$name.txt"
    fi
  done
  cat /proc/meminfo > "$output/meminfo.txt"
  cat /proc/vmstat > "$output/vmstat.txt"
  cat /proc/diskstats > "$output/diskstats.txt"
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader > "$output/nvidia-smi.csv"
}

wait_for_health() {
  local output=$1
  local deadline=$((SECONDS + HEALTH_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 80 "$output/server.log" >&2 || true
      die "server exited before health check passed"
    fi
    if curl -fsS "http://$ADDR/health" > "$output/health.json.tmp" 2>/dev/null \
      && python3 - "$output/health.json.tmp" "$MODEL" <<'PY'
import json
import pathlib
import sys

health = json.loads(pathlib.Path(sys.argv[1]).read_text())
raise SystemExit(0 if sys.argv[2] in health.get("models", []) else 1)
PY
    then
      mv "$output/health.json.tmp" "$output/health.json"
      return 0
    fi
    sleep 1
  done
  tail -n 80 "$output/server.log" >&2 || true
  die "server health timeout after ${HEALTH_TIMEOUT_S}s"
}

start_monitors() {
  local output=$1
  pidstat -rduw -p "$SERVER_PID" 1 > "$output/pidstat.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  iostat -dxm 1 > "$output/iostat.log" 2>&1 &
  MONITOR_PIDS+=("$!")
  nvidia-smi dmon -s pucvmet -d 1 > "$output/gpu-dmon.log" 2>&1 &
  MONITOR_PIDS+=("$!")
}

run_request() {
  local output=$1
  python3 - "$PROMPT" "$MODEL" "http://$ADDR/v1/completions" \
    "$output/request.json" "$output/response.json" "$output/timing.json" \
    "$REQUEST_TIMEOUT_S" <<'PY'
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

prompt_path, model, url, request_path, response_path, timing_path, timeout = sys.argv[1:]
payload = {
    "chat": False,
    "max_tokens": 1,
    "model": model,
    "prompt": pathlib.Path(prompt_path).read_text(),
    "stream": False,
    "temperature": 0.0,
}
encoded = json.dumps(payload, ensure_ascii=False).encode()
pathlib.Path(request_path).write_bytes(encoded + b"\n")
request = urllib.request.Request(
    url,
    data=encoded,
    headers={"Content-Type": "application/json"},
    method="POST",
)
started_utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
status = None
body = b""
error = None
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=int(timeout)) as response:
        status = response.status
        body = response.read()
except urllib.error.HTTPError as exc:
    status = exc.code
    body = exc.read()
    error = repr(exc)
except Exception as exc:
    error = repr(exc)
wall_s = time.perf_counter() - started
pathlib.Path(response_path).write_bytes(body)
timing = {
    "ended_utc": datetime.now(timezone.utc).isoformat(),
    "error": error,
    "http_status": status,
    "response_bytes": len(body),
    "started_utc": started_utc,
    "wall_s": wall_s,
}
pathlib.Path(timing_path).write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
if error is not None or status != 200:
    raise SystemExit(f"request failed: status={status} error={error}")
try:
    json.loads(body)
except Exception as exc:
    raise SystemExit(f"response is not JSON: {exc}") from exc
PY
}

CELL_DIRS=()
cell_index=0
for window in "${WINDOW_LIST[@]}"; do
  cell_index=$((cell_index + 1))
  printf -v cell_name 'cell-%02d-window-%s' "$cell_index" "$window"
  cell_dir="$RUN_DIR/$cell_name"
  mkdir -p "$cell_dir"
  CELL_DIRS+=("$cell_dir")
  printf '%s\n' "$window" > "$cell_dir/window.txt"

  echo "[$cell_name] evicting only $ARTIFACT/experts/*.bin"
  evict_expert_pages "$cell_dir/eviction.json" | tee "$cell_dir/eviction.log"

  echo "[$cell_name] starting fresh server"
  env \
    -u BW24_API_KEY \
    -u BW24_FULL_PREC \
    -u BW24_MOE_GATE \
    -u BW24_MOE_RESIDENT_GB \
    -u BW24_MOE_SLOTS \
    -u BW24_MOE_STATS \
    -u BW24_MOE_TRACE \
    BW24_CTX=8192 \
    BW24_FAST=1 \
    BW24_MMVQ=1 \
    BW24_MOE_CACHE=1 \
    BW24_MOE_GROUPED=1 \
    BW24_MOE_MMAP_ADVICE="$MMAP_ADVICE" \
    BW24_MOE_PREWARM=1 \
    BW24_MOE_PREFETCH=1 \
    BW24_MOE_PAGE_PREFETCH=1 \
    BW24_MOE_PAGE_PREFETCH_WINDOW="$window" \
    BW24_MOE_RESIDENT=1 \
    BW24_MOE_VRAM_FRAC=0.85 \
    BW24_COMPAT=native \
    BW24_MODELS="$MODEL=$ARTIFACT" \
    BW24_ADDR="$ADDR" \
    "$SERVER_BIN" > "$cell_dir/server.log" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$cell_dir/server.pid"
  wait_for_health "$cell_dir"

  snapshot_proc "$cell_dir/proc-before"
  start_monitors "$cell_dir"
  request_status=0
  run_request "$cell_dir" || request_status=$?
  snapshot_proc "$cell_dir/proc-after" || true
  cleanup_cell
  (( request_status == 0 )) || die "$cell_name request failed; see its raw logs"
  echo "[$cell_name] complete"
done

python3 - "$RUN_DIR/summary.json" "${CELL_DIRS[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

def proc_stat(path):
    raw = path.read_text().strip()
    fields = raw.rsplit(") ", 1)[1].split()
    return {
        "minflt": int(fields[7]),
        "majflt": int(fields[9]),
        "utime_ticks": int(fields[11]),
        "stime_ticks": int(fields[12]),
    }

def proc_io(path):
    return {
        key: int(value.strip())
        for key, value in (line.split(":", 1) for line in path.read_text().splitlines())
    }

cells = []
expected = None
identity_match = True
for cell_dir_raw in sys.argv[2:]:
    cell_dir = pathlib.Path(cell_dir_raw)
    response_bytes = (cell_dir / "response.json").read_bytes()
    response = json.loads(response_bytes)
    identity = {"text": response.get("text"), "tokens": response.get("tokens")}
    if not isinstance(identity["text"], str) or not isinstance(identity["tokens"], list):
        raise SystemExit(f"{cell_dir.name}: native response lacks text/tokens")
    if expected is None:
        expected = identity
    elif identity != expected:
        identity_match = False
    before_stat = proc_stat(cell_dir / "proc-before/process-stat.txt")
    after_stat = proc_stat(cell_dir / "proc-after/process-stat.txt")
    before_io = proc_io(cell_dir / "proc-before/process-io.txt")
    after_io = proc_io(cell_dir / "proc-after/process-io.txt")
    cells.append({
        "cell": cell_dir.name,
        "http": json.loads((cell_dir / "timing.json").read_text()),
        "identity": identity,
        "proc_delta": {
            key: after_stat[key] - before_stat[key]
            for key in before_stat
        } | {
            "read_bytes": after_io["read_bytes"] - before_io["read_bytes"],
            "write_bytes": after_io["write_bytes"] - before_io["write_bytes"],
        },
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "window": int((cell_dir / "window.txt").read_text()),
    })
summary = {
    "cells": cells,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "response_identity_match": identity_match,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
if not identity_match:
    raise SystemExit("response token/text identity differs across windows")
PY

echo "$RUN_DIR"

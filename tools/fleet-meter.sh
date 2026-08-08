#!/usr/bin/env bash
# Snapshot memra's cumulative cache metrics into the fleet receipt ledger.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
METRICS_URL="${FLEET_METRICS_URL:-http://127.0.0.1:8002/metrics}"
LEDGER="${FLEET_LEDGER:-${ROOT}/research/fleet-meter/rig5090-fleet.jsonl}"
INTERVAL_MINUTES="${FLEET_INTERVAL_MINUTES:-30}"
TIMEOUT_SECONDS="${FLEET_TIMEOUT_SECONDS:-10}"
MODE=once

usage() {
    cat <<'EOF'
Usage: tools/fleet-meter.sh [--once | --loop] [--interval-minutes N]

Environment:
  FLEET_METRICS_URL       scrape URL (default http://127.0.0.1:8002/metrics)
  FLEET_LEDGER            JSONL output path
  FLEET_INTERVAL_MINUTES  loop interval (default 30)
  FLEET_TIMEOUT_SECONDS   curl deadline (default 10)

The default is one snapshot, suitable for cron and systemd timers.
EOF
}

while (($#)); do
    case "$1" in
        --once)
            MODE=once
            shift
            ;;
        --loop)
            MODE=loop
            shift
            ;;
        --interval-minutes)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            INTERVAL_MINUTES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf '[fleet-meter] unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$INTERVAL_MINUTES" =~ ^[1-9][0-9]*$ ]]; then
    printf '[fleet-meter] interval must be a positive integer: %s\n' \
        "$INTERVAL_MINUTES" >&2
    exit 2
fi
if [[ ! "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    printf '[fleet-meter] timeout must be a positive integer: %s\n' \
        "$TIMEOUT_SECONDS" >&2
    exit 2
fi
command -v curl >/dev/null || {
    printf '[fleet-meter] curl is required\n' >&2
    exit 1
}
command -v flock >/dev/null || {
    printf '[fleet-meter] flock is required\n' >&2
    exit 1
}

umask 027

snapshot_once() {
    local tmp curl_error
    tmp="$(mktemp "${TMPDIR:-/tmp}/memra-fleet-meter.XXXXXX")"
    trap 'rm -f "$tmp"' RETURN

    if ! curl_error="$(
        curl --fail --silent --show-error \
            --connect-timeout 2 --max-time "$TIMEOUT_SECONDS" \
            "$METRICS_URL" -o "$tmp" 2>&1
    )"; then
        printf '[fleet-meter] skip: scrape failed for %s: %s\n' \
            "$METRICS_URL" "$curl_error" >&2
        return 0
    fi

    mkdir -p -- "$(dirname -- "$LEDGER")"
    exec 9>>"$LEDGER"
    if ! flock -w 5 9; then
        printf '[fleet-meter] skip: ledger is busy: %s\n' "$LEDGER" >&2
        exec 9>&-
        return 0
    fi

    python3 - "$tmp" "$LEDGER" <<'PY'
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


metrics_path = Path(sys.argv[1])
ledger_path = Path(sys.argv[2])


def nonnegative_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def normalize_histogram(value):
    if not isinstance(value, dict):
        raise ValueError("lcp_histogram must be an object")
    edges = value.get("edges") or []
    counts = value.get("counts") or []
    if not isinstance(edges, list) or not isinstance(counts, list):
        raise ValueError("lcp_histogram edges/counts must be arrays")
    if len(edges) != len(counts):
        raise ValueError("lcp_histogram edges/counts length mismatch")
    normalized_edges = [
        nonnegative_int(edge, f"lcp_histogram.edges[{i}]")
        for i, edge in enumerate(edges)
    ]
    normalized_counts = [
        nonnegative_int(count, f"lcp_histogram.counts[{i}]")
        for i, count in enumerate(counts)
    ]
    return {"edges": normalized_edges, "counts": normalized_counts}


def normalize_tenants(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("tenants must be an object")
    normalized = {}
    for name, tenant in sorted(value.items()):
        if not isinstance(tenant, dict):
            raise ValueError(f"tenant {name!r} must be an object")
        prompt = nonnegative_int(
            tenant.get("prompt_tokens_in", 0),
            f"tenants[{name!r}].prompt_tokens_in",
        )
        cached = nonnegative_int(
            tenant.get("cached_tokens_in", 0),
            f"tenants[{name!r}].cached_tokens_in",
        )
        if cached > prompt:
            raise ValueError(f"tenant {name!r} cached tokens exceed prompt tokens")
        normalized[str(name)] = {
            "prompt_tokens_in": prompt,
            "cached_tokens_in": cached,
            "cache_hit_token_ratio": cached / prompt if prompt else 0.0,
        }
    return normalized


def last_row(path):
    previous = None
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                previous = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return previous


def regressed(previous, current):
    for key in ("prompt_tokens_in", "cached_tokens_in", "computed_tokens_in"):
        if current[key] < nonnegative_int(previous.get(key), f"previous {key}"):
            return True

    old_hist = previous.get("lcp_histogram") or {}
    new_hist = current["lcp_histogram"]
    if old_hist.get("edges") == new_hist["edges"]:
        old_counts = old_hist.get("counts") or []
        if len(old_counts) == len(new_hist["counts"]):
            if any(new < old for old, new in zip(old_counts, new_hist["counts"])):
                return True

    old_tenants = previous.get("tenants") or {}
    new_tenants = current["tenants"]
    for name, old in old_tenants.items():
        if name not in new_tenants:
            if old.get("prompt_tokens_in", 0) or old.get("cached_tokens_in", 0):
                return True
            continue
        new = new_tenants[name]
        if (
            new["prompt_tokens_in"] < old.get("prompt_tokens_in", 0)
            or new["cached_tokens_in"] < old.get("cached_tokens_in", 0)
        ):
            return True
    return False


with metrics_path.open(encoding="utf-8") as source:
    metrics = json.load(source)
if not isinstance(metrics, dict):
    raise ValueError("metrics scrape must be a JSON object")

prompt = nonnegative_int(metrics.get("prompt_tokens_in"), "prompt_tokens_in")
cached = nonnegative_int(metrics.get("cached_tokens_in"), "cached_tokens_in")
computed = nonnegative_int(
    metrics.get("computed_tokens_in", prompt - cached),
    "computed_tokens_in",
)
if cached > prompt:
    raise ValueError("cached_tokens_in exceeds prompt_tokens_in")
if computed != prompt - cached:
    raise ValueError(
        f"computed_tokens_in mismatch: got {computed}, expected {prompt - cached}"
    )
ratio = cached / prompt if prompt else 0.0
reported_ratio = float(metrics.get("cache_hit_token_ratio", ratio))
if not math.isfinite(reported_ratio) or not math.isclose(
    reported_ratio, ratio, rel_tol=0.0, abs_tol=1e-9
):
    raise ValueError(
        f"cache_hit_token_ratio mismatch: got {reported_ratio}, expected {ratio}"
    )

row = {
    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "prompt_tokens_in": prompt,
    "cached_tokens_in": cached,
    "computed_tokens_in": computed,
    "cache_hit_token_ratio": ratio,
    "lcp_histogram": normalize_histogram(metrics.get("lcp_histogram") or {}),
    "tenants": normalize_tenants(metrics.get("tenants")),
}
previous = last_row(ledger_path)
state_keys = (
    "prompt_tokens_in",
    "cached_tokens_in",
    "computed_tokens_in",
    "cache_hit_token_ratio",
    "lcp_histogram",
    "tenants",
)
if previous is not None and all(previous.get(key) == row[key] for key in state_keys):
    print(
        f"[fleet-meter] unchanged: no row appended "
        f"(prompt={prompt}, cached={cached})"
    )
    raise SystemExit(0)

row["restart"] = previous is not None and regressed(previous, row)
payload = json.dumps(row, separators=(",", ":")) + "\n"
with ledger_path.open("a", encoding="utf-8") as target:
    target.write(payload)
    target.flush()
    os.fsync(target.fileno())
print(
    f"[fleet-meter] appended {row['ts']} prompt={prompt} cached={cached} "
    f"restart={str(row['restart']).lower()} ledger={ledger_path}"
)
PY

    exec 9>&-
}

if [[ "$MODE" == once ]]; then
    snapshot_once
else
    while true; do
        snapshot_once
        sleep "${INTERVAL_MINUTES}m"
    done
fi

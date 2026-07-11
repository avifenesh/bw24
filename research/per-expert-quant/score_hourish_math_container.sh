#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
: "${1:?usage: score_hourish_math_container.sh RUN_DIR}"
[[ -d "$1" ]] || { echo "run directory does not exist: $1" >&2; exit 2; }
(( BASH_VERSINFO[0] >= 4 )) || { echo "Bash 4 or newer is required" >&2; exit 2; }
RUN_DIR=$(cd "$1" && pwd -P)
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }

mapfile -t math < <(find "$RUN_DIR/shards/hendrycks_math500" -name 'samples_hendrycks_math500_*.jsonl' -type f)
[[ ${#math[@]} == 1 ]] || {
  echo "expected exactly one MATH-500 sample file" >&2
  exit 2
}

OUTPUT="$RUN_DIR/math-score.json"
RECEIPT="$RUN_DIR/math-score.receipt.json"
[[ ! -e "$OUTPUT" && ! -e "$RECEIPT" ]] || {
  echo "refusing to overwrite existing math score evidence" >&2
  exit 3
}

tool_sha=$(python3 - "$HERE/score_hourish_math.py" "$HERE/Dockerfile.math-score" \
  "$HERE/math-score-requirements.lock.txt" <<'PY'
import hashlib, sys

outer = hashlib.sha256()
for raw_path in sys.argv[1:]:
    digest = hashlib.sha256()
    with open(raw_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    outer.update(f"{digest.hexdigest()}  {raw_path}\n".encode())
print(outer.hexdigest())
PY
)
image="bw24-hourish-math-score:${tool_sha:0:16}"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --pull --file "$HERE/Dockerfile.math-score" --tag "$image" "$HERE"
fi
image_id=$(docker image inspect --format '{{.Id}}' "$image")

tmp=$(mktemp "$RUN_DIR/.math-score.XXXXXX")
trap 'rm -f "$tmp"' EXIT
docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 32 \
  --memory 1g \
  --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --mount "type=bind,src=$RUN_DIR,dst=/inputs,readonly" \
  "$image" \
  "/inputs/${math[0]#"$RUN_DIR/"}" > "$tmp"

python3 - "$tmp" <<'PY'
import json, sys

report = json.load(open(sys.argv[1]))
if report.get("format") != "bw24-hourish-math-score-v1":
    raise SystemExit("wrong math-score format")
if report.get("total") != 32:
    raise SystemExit(f"expected 32 math samples, got {report.get('total')}")
if report.get("by_task", {}).get("hendrycks_math500", {}).get("total") != 32:
    raise SystemExit("expected thirty-two MATH-500 samples")
PY
mv "$tmp" "$OUTPUT"
trap - EXIT

export RUN_DIR OUTPUT RECEIPT image image_id tool_sha
python3 - <<'PY'
import hashlib, json, os, pathlib

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

receipt = {
    "format": "bw24-hourish-math-score-receipt-v1",
    "run_dir": os.environ["RUN_DIR"],
    "output": os.environ["OUTPUT"],
    "output_sha256": sha256(os.environ["OUTPUT"]),
    "image": os.environ["image"],
    "image_id": os.environ["image_id"],
    "tool_sha256": os.environ["tool_sha"],
    "sandbox": {
        "network": "none",
        "read_only_root": True,
        "capabilities": "all dropped",
        "no_new_privileges": True,
        "pids_limit": 32,
        "memory_bytes": 1024 * 1024 * 1024,
        "cpus": 1,
    },
}
path = pathlib.Path(os.environ["RECEIPT"])
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

echo "math score complete: $OUTPUT"

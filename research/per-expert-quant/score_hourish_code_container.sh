#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
: "${1:?usage: score_hourish_code_container.sh RUN_DIR}"
RUN_DIR=$(realpath "$1")
[[ -d "$RUN_DIR" ]] || { echo "run directory does not exist: $RUN_DIR" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }

mapfile -t human < <(find "$RUN_DIR/shards/humaneval_instruct" -name 'samples_humaneval_instruct_*.jsonl' -type f)
[[ ${#human[@]} == 1 ]] || {
  echo "expected exactly one HumanEval sample file" >&2
  exit 2
}

OUTPUT="$RUN_DIR/code-score.json"
RECEIPT="$RUN_DIR/code-score.receipt.json"
[[ ! -e "$OUTPUT" && ! -e "$RECEIPT" ]] || {
  echo "refusing to overwrite existing code score evidence" >&2
  exit 3
}

tool_sha=$(sha256sum "$HERE/score_hourish_code.py" "$HERE/Dockerfile.code-score" \
  | sha256sum | cut -d' ' -f1)
image="bw24-hourish-code-score:${tool_sha:0:16}"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --pull --file "$HERE/Dockerfile.code-score" --tag "$image" "$HERE"
fi
image_id=$(docker image inspect --format '{{.Id}}' "$image")

tmp=$(mktemp "$RUN_DIR/.code-score.XXXXXX")
trap 'rm -f "$tmp"' EXIT
docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 32 \
  --memory 768m \
  --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --mount "type=bind,src=$RUN_DIR,dst=/inputs,readonly" \
  "$image" \
  "/inputs/${human[0]#"$RUN_DIR/"}" > "$tmp"

python3 - "$tmp" <<'PY'
import json, sys

report = json.load(open(sys.argv[1]))
if report.get("format") != "bw24-hourish-code-score-v1":
    raise SystemExit("wrong code-score format")
if report.get("total") != 14:
    raise SystemExit(f"expected 14 code samples, got {report.get('total')}")
if report.get("by_task", {}).get("humaneval_instruct", {}).get("total") != 14:
    raise SystemExit("expected fourteen HumanEval samples")
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
    "format": "bw24-hourish-code-score-receipt-v1",
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
        "memory_bytes": 768 * 1024 * 1024,
        "cpus": 1,
    },
}
path = pathlib.Path(os.environ["RECEIPT"])
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

echo "code score complete: $OUTPUT"

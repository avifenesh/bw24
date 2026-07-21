#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
SUITE_LOCK=${SUITE_LOCK:-$HERE/suite.lock.json}
: "${1:?usage: score_promoted_math_container.sh ARM_RUN_DIR}"
[[ -d "$1" ]] || { echo "run directory does not exist: $1" >&2; exit 2; }
(( BASH_VERSINFO[0] >= 4 )) || { echo "Bash 4 or newer is required" >&2; exit 2; }
RUN_DIR=$(cd "$1" && pwd -P)
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
[[ -f "$SUITE_LOCK" ]] || { echo "suite lock does not exist: $SUITE_LOCK" >&2; exit 2; }

readarray -t lock_values < <(python3 - "$SUITE_LOCK" <<'PY'
import hashlib, json, sys
path = sys.argv[1]
value = json.load(open(path))
encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
print(value["eval_documents"]["hendrycks_math500"])
print(hashlib.sha256(open(path, "rb").read()).hexdigest())
print(hashlib.sha256(encoded).hexdigest())
PY
)
EXPECTED_COUNT=${lock_values[0]}
SUITE_SHA256=${lock_values[1]}
SUITE_CANONICAL_SHA256=${lock_values[2]}

mapfile -t math < <(find "$RUN_DIR" -name 'samples_hendrycks_math500_*.jsonl' -type f)
[[ ${#math[@]} == 1 ]] || {
  echo "expected exactly one MATH-500 sample file, found ${#math[@]}" >&2
  exit 2
}
MATH_SHARD=$(dirname "${math[0]}")
while [[ "$MATH_SHARD" != "$RUN_DIR" && ! -f "$MATH_SHARD/suite.lock.json" ]]; do
  MATH_SHARD=$(dirname "$MATH_SHARD")
done
[[ -f "$MATH_SHARD/suite.lock.json" ]] || { echo "MATH shard has no copied suite lock" >&2; exit 2; }
python3 - "$SUITE_LOCK" "$MATH_SHARD/suite.lock.json" <<'PY'
import json, sys
a, b = (json.load(open(path)) for path in sys.argv[1:])
if a != b:
    raise SystemExit("copied suite lock differs from scoring lock")
PY

OUTPUT="$RUN_DIR/math-score.json"
RECEIPT="$RUN_DIR/math-score.receipt.json"
[[ ! -e "$OUTPUT" && ! -e "$RECEIPT" ]] || {
  echo "refusing to overwrite existing promoted math score evidence" >&2
  exit 3
}

tool_sha=$(python3 - "$HERE/score_hourish_math.py" "$HERE/Dockerfile.math-score" \
  "$HERE/math-score-requirements.lock.txt" "$HERE/score_promoted_math_container.sh" <<'PY'
import hashlib, sys
outer = hashlib.sha256()
for raw_path in sys.argv[1:]:
    digest = hashlib.sha256(open(raw_path, "rb").read()).hexdigest()
    outer.update(f"{digest}  {raw_path}\n".encode())
print(outer.hexdigest())
PY
)
image="bw24-promoted-math-score:${tool_sha:0:16}"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --pull --file "$HERE/Dockerfile.math-score" --tag "$image" "$HERE"
fi
image_id=$(docker image inspect --format '{{.Id}}' "$image")

tmp=$(mktemp "$RUN_DIR/.math-score.XXXXXX")
trap 'rm -f "$tmp"' EXIT
docker run --rm \
  --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --pids-limit 32 \
  --memory 1g --cpus 1 --cpu-shares 2 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --mount "type=bind,src=$RUN_DIR,dst=/inputs,readonly" \
  "$image" --format bw24-promoted-math-score-v1 \
  "/inputs/${math[0]#"$RUN_DIR/"}" > "$tmp"

python3 - "$tmp" "$EXPECTED_COUNT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
if report.get("format") != "bw24-promoted-math-score-v1":
    raise SystemExit("wrong promoted math-score format")
if report.get("total") != expected:
    raise SystemExit(f"expected {expected} math samples, got {report.get('total')}")
if report.get("by_task", {}).get("hendrycks_math500", {}).get("total") != expected:
    raise SystemExit(f"expected {expected} MATH-500 samples")
PY
mv "$tmp" "$OUTPUT"
trap - EXIT

export RUN_DIR OUTPUT RECEIPT image image_id tool_sha SUITE_SHA256 SUITE_CANONICAL_SHA256 EXPECTED_COUNT
python3 - <<'PY'
import hashlib, json, os, pathlib
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
receipt = {
    "format": "bw24-promoted-math-score-receipt-v1",
    "run_dir": os.environ["RUN_DIR"],
    "output": os.environ["OUTPUT"],
    "output_sha256": sha256(os.environ["OUTPUT"]),
    "image": os.environ["image"], "image_id": os.environ["image_id"],
    "tool_sha256": os.environ["tool_sha"],
    "suite_lock_sha256": os.environ["SUITE_SHA256"],
    "suite_lock_canonical_sha256": os.environ["SUITE_CANONICAL_SHA256"],
    "expected_sample_count": int(os.environ["EXPECTED_COUNT"]),
    "sandbox": {
        "network": "none", "read_only_root": True, "capabilities": "all dropped",
        "no_new_privileges": True, "pids_limit": 32,
        "memory_bytes": 1024 * 1024 * 1024, "cpus": 1, "cpu_shares": 2,
    },
}
path = pathlib.Path(os.environ["RECEIPT"])
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

echo "promoted math score complete: $OUTPUT"

#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/experiments/hy3-110gb}
SOURCE_DIR=${SOURCE_DIR:-/opt/dlami/nvme/models/hy3-source}
VENV=${VENV:-/data/venvs/hy3-110gb}
MODEL_ID=${MODEL_ID:-tencent/Hy3}
REVISION=${REVISION:-716aa7241bd6d95896be4ebfc761162a9c4d49ef}
MAX_WORKERS=${MAX_WORKERS:-16}
EXPECTED_CONFIG_SHA256=${EXPECTED_CONFIG_SHA256:-663036ceca3d8a178cd772739566c262caffdecebaed6c1d76b464d729bb2951}
EXPECTED_INDEX_SHA256=${EXPECTED_INDEX_SHA256:-9594f1a9419e62ca7afca51bb644f38ef19039374f7812449381ccf42f0ef79b}

mkdir -p "$ROOT/logs" "$ROOT/receipts" "$SOURCE_DIR"
exec > >(tee -a "$ROOT/logs/source-download.log") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] downloading $MODEL_ID@$REVISION with $MAX_WORKERS workers"
"$VENV/bin/hf" download "$MODEL_ID" \
  --revision "$REVISION" \
  --local-dir "$SOURCE_DIR" \
  --max-workers "$MAX_WORKERS"

config_sha=$(sha256sum "$SOURCE_DIR/config.json" | awk '{print $1}')
index_sha=$(sha256sum "$SOURCE_DIR/model.safetensors.index.json" | awk '{print $1}')
[[ "$config_sha" == "$EXPECTED_CONFIG_SHA256" ]] || {
  echo "config hash mismatch: $config_sha" >&2
  exit 1
}
[[ "$index_sha" == "$EXPECTED_INDEX_SHA256" ]] || {
  echo "index hash mismatch: $index_sha" >&2
  exit 1
}

"$VENV/bin/python" - "$SOURCE_DIR" "$ROOT/receipts/source-download.json" \
  "$MODEL_ID" "$REVISION" "$config_sha" "$index_sha" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

source_dir, output, model_id, revision, config_sha, index_sha = sys.argv[1:]
root = pathlib.Path(source_dir)
shards = sorted(root.glob("model-*.safetensors"))
payload = {
    "format": "bw24-hy3-pinned-source-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_id": model_id,
    "revision": revision,
    "source_dir": str(root),
    "config_sha256": config_sha,
    "index_sha256": index_sha,
    "shard_count": len(shards),
    "shard_bytes": sum(path.stat().st_size for path in shards),
    "complete": len(shards) == 99,
}
if not payload["complete"]:
    raise SystemExit(f"expected 99 shards, found {len(shards)}")
path = pathlib.Path(output)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pinned source complete"

#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/experiments/hy3-110gb}
BUCKET=${BUCKET:?set BUCKET to the durable S3 bucket}
RUN_ID=${RUN_ID:?set RUN_ID}
PREFIX=${PREFIX:-runs/$RUN_ID}
IMDS=http://169.254.169.254/latest

mkdir -p "$ROOT/logs/interruption"
while true; do
  token=$(curl -fsS -X PUT "$IMDS/api/token" -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
  if notice=$(curl -fsS -H "X-aws-ec2-metadata-token: $token" \
      "$IMDS/meta-data/spot/instance-action" 2>/dev/null); then
    python3 - "$ROOT/logs/interruption/notice.json" "$notice" <<'PY'
import json
import pathlib
import sys
payload = json.loads(sys.argv[2])
payload["observed_by"] = "bw24-hy3-110gb-spot-watch-v1"
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    timeout 90 aws s3 sync "$ROOT" "s3://$BUCKET/$PREFIX" \
      --only-show-errors \
      --exclude 'cache/*' \
      --exclude 'source/*' \
      --exclude 'tmp/*' || true
    aws s3 cp "$ROOT/logs/interruption/notice.json" \
      "s3://$BUCKET/$PREFIX/INTERRUPTION-NOTICE.json" --only-show-errors || true
    exit 0
  fi
  sleep 5
done

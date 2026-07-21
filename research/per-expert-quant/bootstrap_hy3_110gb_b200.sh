#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/experiments/hy3-110gb}
BUCKET=${BUCKET:?set BUCKET}
RUN_ID=${RUN_ID:?set RUN_ID}
REPO=${REPO:-/data/src/bw24-hy3-110gb}
NVME_ROOT=${NVME_ROOT:-/opt/dlami/nvme}

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  awscli build-essential cmake curl git-lfs jq libopenblas-dev ninja-build pkg-config \
  python3-venv rsync

sudo mkdir -p "$NVME_ROOT/bw24-data" "$NVME_ROOT/bw24-scratch"
sudo chown -R "$USER:$USER" "$NVME_ROOT/bw24-data" "$NVME_ROOT/bw24-scratch"
if [[ ! -e /data ]]; then sudo ln -s "$NVME_ROOT/bw24-data" /data; fi
if [[ ! -e /scratch ]]; then sudo ln -s "$NVME_ROOT/bw24-scratch" /scratch; fi
mkdir -p "$ROOT"/{artifacts,calibration,evidence,logs,plans,receipts,tmp} /data/src

git lfs install --skip-repo
python3 -m venv /data/venvs/hy3-110gb
/data/venvs/hy3-110gb/bin/pip install --upgrade pip wheel
/data/venvs/hy3-110gb/bin/pip install \
  boto3 huggingface_hub numpy scipy safetensors torch transformers

python3 - "$ROOT/receipts/bootstrap.json" "$BUCKET" "$RUN_ID" "$REPO" <<'PY'
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

out, bucket, run_id, repo = sys.argv[1:]
def command(*args):
    return subprocess.check_output(args, text=True).strip()
token = command("curl", "-fsS", "-X", "PUT", "http://169.254.169.254/latest/api/token",
                "-H", "X-aws-ec2-metadata-token-ttl-seconds: 60")
def metadata(path):
    return command("curl", "-fsS", "-H", f"X-aws-ec2-metadata-token: {token}",
                   f"http://169.254.169.254/latest/meta-data/{path}")
payload = {
    "format": "bw24-hy3-110gb-bootstrap-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "instance_id": metadata("instance-id"),
    "instance_type": metadata("instance-type"),
    "availability_zone": metadata("placement/availability-zone"),
    "ami_id": metadata("ami-id"),
    "bucket": bucket,
    "run_id": run_id,
    "repo": repo,
    "git_head": command("git", "-C", repo, "rev-parse", "HEAD"),
    "gpu_inventory": command("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                             "--format=csv,noheader").splitlines(),
    "public_eval_data_used_for_construction": False,
    "mtp_enabled": False,
    "speculative_decoding_enabled": False,
    "kv_reuse_enabled": False,
}
path = pathlib.Path(out)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

sudo systemd-run --unit=bw24-hy3-110gb-checkpoint --property=Restart=always \
  --setenv=ROOT="$ROOT" --setenv=BUCKET="$BUCKET" --setenv=RUN_ID="$RUN_ID" \
  "$REPO/research/per-expert-quant/run_hy3_110gb_s3_checkpoint_loop.sh"
sudo systemd-run --unit=bw24-hy3-110gb-spot-watch --property=Restart=on-failure \
  --setenv=ROOT="$ROOT" --setenv=BUCKET="$BUCKET" --setenv=RUN_ID="$RUN_ID" \
  "$REPO/research/per-expert-quant/run_hy3_110gb_spot_interruption_watch.sh"

aws s3 cp "$ROOT/receipts/bootstrap.json" \
  "s3://$BUCKET/runs/$RUN_ID/receipts/bootstrap.json" --only-show-errors
echo "bootstrap complete: $RUN_ID"

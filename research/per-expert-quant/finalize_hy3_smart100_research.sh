#!/usr/bin/env bash
set -euo pipefail

MODE=${1:---dry-run}
[[ "$MODE" == --dry-run || "$MODE" == --execute ]] || {
  echo "usage: $0 [--dry-run|--execute]" >&2
  exit 2
}

REMOTE=${REMOTE:-bw24-research-g6e}
REGION=${REGION:-us-east-2}
INSTANCE_ID=${INSTANCE_ID:-i-09082605f120e88f0}
EXPECTED_ACCOUNT=${EXPECTED_ACCOUNT:-507286591552}
REMOTE_FULL_ROOT=${REMOTE_FULL_ROOT:-/data/results/per-expert-quant/full-agentic-iq3-iq4-q4-v1}
REMOTE_FULL_READY=${REMOTE_FULL_READY:-/data/logs/full-agentic-iq3-iq4-q4-v1/complete}
LOCAL_ROOT=${LOCAL_ROOT:-/home/avifenesh/projects/bw24-research-archive/smart100-final}

EVIDENCE_ROOTS=(
  /data/results/per-expert-quant
  /data/logs
  /data/plans
  /data/calibration/hy3-routing-v1
  /data/calibration/hy3-confidence-v1
  /data/calibration/hy3-100gb-5f02c37
  /data/calibration/hy3-quant-sensitivity-53de6ca
  /data/calibration/hy3-quant-iq3-iq4-q4-99f3dc3
  /data/heal/per-expert-quant-100gb-5f02c37/router/receipts
  /data/heal/per-expert-quant-100gb-5f02c37/joint/receipts
  /data/heal/per-expert-quant-smart100-2605fde/smart100_empirical/receipts
  /data/heal/per-expert-quant-smart100-2605fde/smart100_balanced/receipts
  /data/heal/per-expert-quant-smart100-2605fde/smart100_rescue/receipts
  /data/heal/per-expert-quant-iq3-iq4-q4-99f3dc3/smart100_iq3_iq4_q4_empirical/receipts
)

die() { echo "smart100 finalizer: $*" >&2; exit 1; }
command -v aws >/dev/null || die "aws CLI is required"
command -v rsync >/dev/null || die "rsync is required"
mkdir -p "$LOCAL_ROOT"
exec 9>"$LOCAL_ROOT/finalizer.lock"
flock -n 9 || die "another finalizer owns $LOCAL_ROOT/finalizer.lock"

account=$(aws sts get-caller-identity --query Account --output text)
[[ "$account" == "$EXPECTED_ACCOUNT" ]] || die "AWS account mismatch: $account"
state=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
[[ "$state" == running ]] || die "instance state is $state, expected running"
ssh "$REMOTE" "test -f '$REMOTE_FULL_READY'" \
  || die "complete smart100 agentic evidence is not ready"

run_id=$(ssh "$REMOTE" "cat '$REMOTE_FULL_ROOT/_active-run-id'")
combined="$REMOTE_FULL_ROOT/comparisons/$run_id/combined.json"
finalist=$(ssh "$REMOTE" "python3 - '$combined'" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["format"] == "bw24-full-agentic-comparison-v1"
assert d["baseline"] == "plain_quant" and d["total_tasks"] == 589
print(d["candidate"])
PY
)
case "$finalist" in
  smart100_empirical|smart100_balanced|smart100_rescue)
    remote_artifact="/scratch/bw24-artifacts-smart100-2605fde/$finalist" ;;
  smart100_iq3_iq4_q4_empirical)
    remote_artifact="/scratch/bw24-artifacts-iq3-iq4-q4-99f3dc3/$finalist" ;;
  prune100_joint_heal)
    remote_artifact="/scratch/bw24-artifacts-100gb-5f02c37/$finalist" ;;
  traffic_nvfp4_53_q2_139)
    remote_artifact=/scratch/bw24-artifacts/traffic-nvfp4-53-q2-139 ;;
  *) die "unexpected full-agentic finalist $finalist" ;;
esac

ssh "$REMOTE" 'set -eu
for f in \
  /data/logs/practical-v1/complete \
  /data/logs/hy3-quant-sensitivity-53de6ca/complete \
  /data/logs/smart100-build-2605fde/complete \
  /data/logs/smart100-directional-v1/complete \
  /data/logs/iq3-iq4-q4-extension-99f3dc3/complete \
  /data/logs/iq3-iq4-q4-directional-v1/complete \
  /data/logs/practical-iq3-iq4-q4-v1/complete \
  /data/logs/trusted-full-iq3-iq4-q4-v1/complete \
  /data/logs/full-agentic-iq3-iq4-q4-v1/complete \
  /data/logs/full-agentic-iq3-iq4-q4-v1/chain-complete; do test -f "$f"; done
test -z "$(pgrep -x bw24-server || true)"
test -z "$(pgrep -af "[/]harbor run " || true)"
test -z "$(docker ps -q)"
' || die "remote completion or idle-process gate failed"

for root in "${EVIDENCE_ROOTS[@]}"; do
  ssh "$REMOTE" "test -e '$root'" || die "missing remote evidence root $root"
done
ssh "$REMOTE" "test -f '$remote_artifact/manifest.json'" \
  || die "missing finalist manifest"

quoted_roots=$(printf ' %q' "${EVIDENCE_ROOTS[@]}" "$remote_artifact")
remote_bytes=$(ssh "$REMOTE" "du -sb$quoted_roots | awk '{sum += \$1} END {print sum}'")
[[ "$remote_bytes" =~ ^[1-9][0-9]*$ ]] || die "invalid remote archive size"
available_bytes=$(df -B1 --output=avail "$LOCAL_ROOT" | tail -1 | tr -d ' ')
required_bytes=$((remote_bytes + remote_bytes / 10))
((available_bytes >= required_bytes)) \
  || die "archive needs $required_bytes bytes with headroom; $available_bytes available"

if [[ "$MODE" == --dry-run ]]; then
  echo "smart100 finalizer dry-run: ready finalist=$finalist bytes=$remote_bytes required=$required_bytes"
  exit 0
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
dest="$LOCAL_ROOT/$stamp"
mkdir "$dest" "$dest/evidence-root" "$dest/finalist" "$dest/inventories"

for root in "${EVIDENCE_ROOTS[@]}"; do
  rsync -a --partial --append-verify "$REMOTE:/./${root#/}/" "$dest/evidence-root/"
done
rsync -a --partial --append-verify "$REMOTE:$remote_artifact/" "$dest/finalist/"

verify_tree() {
  local label=$1 remote_root=$2 local_root=$3
  local remote_out="$dest/inventories/$label.remote.sha256"
  local local_out="$dest/inventories/$label.local.sha256"
  ssh "$REMOTE" "cd '$remote_root' && find . -type f -print0 | sort -z | xargs -0 sha256sum" \
    >"$remote_out"
  (cd "$local_root" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
    >"$local_out"
  cmp "$remote_out" "$local_out" || die "$label inventory differs"
}

index=0
for root in "${EVIDENCE_ROOTS[@]}"; do
  verify_tree "evidence-$index" "$root" "$dest/evidence-root$root"
  index=$((index + 1))
done
verify_tree finalist "$remote_artifact" "$dest/finalist"

python3 - "$dest/finalization-receipt.json" "$run_id" "$finalist" "$INSTANCE_ID" \
  "$REGION" "$remote_bytes" "$dest/inventories" <<'PY'
import datetime,hashlib,json,pathlib,sys
out,run_id,finalist,instance,region,remote_bytes,inventory_root=sys.argv[1:]
root=pathlib.Path(inventory_root)
inventories={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.glob("*.remote.sha256"))}
pathlib.Path(out).write_text(json.dumps({
 "format":"bw24-smart100-local-finalization-v2",
 "synced_and_verified_utc":datetime.datetime.now(datetime.UTC).isoformat(),
 "full_agentic_run_id":run_id,"finalist":finalist,"instance_id":instance,"region":region,
 "remote_synced_bytes":int(remote_bytes),"remote_inventory_sha256":inventories,
 "remote_instance_action":"terminate-after-complete-inventory-verification"
},indent=2,sort_keys=True)+"\n")
PY

aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  >"$dest/terminate-response.json"
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID"
final_state=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)
[[ "$final_state" == terminated ]] || die "final instance state is $final_state"
printf '%s\n' "$final_state" >"$dest/instance-final-state.txt"
echo "smart100 research finalized: finalist=$finalist archive=$dest instance=$final_state"

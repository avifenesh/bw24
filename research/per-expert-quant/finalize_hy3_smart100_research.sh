#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-bw24-research-g6e}
REGION=${REGION:-us-east-2}
INSTANCE_ID=${INSTANCE_ID:-i-09082605f120e88f0}
EXPECTED_ACCOUNT=${EXPECTED_ACCOUNT:-507286591552}
REMOTE_FULL_ROOT=${REMOTE_FULL_ROOT:-/data/results/per-expert-quant/full-agentic-smart100-v1}
REMOTE_FULL_READY=${REMOTE_FULL_READY:-/data/logs/full-agentic-smart100-v1/complete}
LOCAL_ROOT=${LOCAL_ROOT:-/home/avifenesh/projects/bw24-research-archive/smart100-final}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-60}

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

while ! ssh "$REMOTE" "test -f '$REMOTE_FULL_READY'"; do sleep "$WAIT_INTERVAL_S"; done
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
  prune100_joint_heal)
    remote_artifact="/scratch/bw24-artifacts-100gb-5f02c37/$finalist" ;;
  traffic_nvfp4_53_q2_139)
    remote_artifact="/scratch/bw24-artifacts/traffic-nvfp4-53-q2-139" ;;
  *) die "unexpected full-agentic finalist $finalist" ;;
esac

# Require every stage and every final report before copying or terminating anything.
ssh "$REMOTE" 'set -eu
for f in \
  /data/logs/practical-v1/complete \
  /data/logs/hy3-quant-sensitivity-53de6ca/complete \
  /data/logs/smart100-build-2605fde/complete \
  /data/logs/smart100-directional-v1/complete \
  /data/logs/practical-smart100-v1/final-complete \
  /data/logs/trusted-full-smart100-v1/complete \
  /data/logs/full-agentic-smart100-v1/complete; do test -f "$f"; done
test -z "$(pgrep -x bw24-server || true)"
test -z "$(pgrep -af "[/]harbor run " || true)"
test -z "$(docker ps -q)"
' || die "remote completion or idle-process gate failed"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
dest="$LOCAL_ROOT/$stamp"
mkdir "$dest" "$dest/evidence-root" "$dest/finalist"
list="$dest/evidence-files.txt"
manifest="$dest/evidence.sha256"
ssh "$REMOTE" 'set -eu
roots=(
 /data/results/per-expert-quant/expanded-v2
 /data/results/per-expert-quant/100gb-heal-v1
 /data/results/per-expert-quant/practical-v1
 /data/results/per-expert-quant/smart100-directional-v1
 /data/results/per-expert-quant/practical-smart100-v1
 /data/results/per-expert-quant/trusted-full-smart100-v1
 /data/results/per-expert-quant/full-agentic-smart100-v1
 /data/logs/100gb-heal-build-5f02c37
 /data/logs/practical-v1
 /data/logs/hy3-quant-sensitivity-53de6ca
 /data/logs/smart100-build-2605fde
 /data/logs/smart100-directional-v1
 /data/logs/practical-smart100-v1
 /data/logs/trusted-full-smart100-v1
 /data/logs/full-agentic-smart100-v1
 /data/plans/per-expert-quant-smart100-2605fde
 /data/calibration/hy3-quant-sensitivity-53de6ca
 /data/heal/per-expert-quant-smart100-2605fde
 /data/artifacts/per-expert-quant-smart100-2605fde
)
for root in "${roots[@]}"; do test -e "$root"; done
find "${roots[@]}" -type f \
  ! -path "*/experts/*" ! -name "*.safetensors" ! -name "*.f32" ! -name "*.bin" \
  -printf "%p\n" | sed "s#^/##" | LC_ALL=C sort -u
' >"$list"
[[ -s "$list" ]] || die "remote evidence list is empty"
rsync -a --partial --files-from="$list" "$REMOTE:/" "$dest/evidence-root/"
(cd "$dest/evidence-root" && xargs -d '\n' sha256sum <"$list") >"$manifest"

# Compare against hashes recomputed independently on the remote paths.
remote_manifest="$dest/evidence.remote.sha256"
ssh "$REMOTE" 'cd /; xargs -d "\n" sha256sum' <"$list" >"$remote_manifest"
cmp "$manifest" "$remote_manifest" || die "evidence inventory differs"
(cd "$dest/evidence-root" && sha256sum -c "$manifest" >/dev/null)

artifact_bytes=$(ssh "$REMOTE" "du -sb '$remote_artifact' | cut -f1")
available=$(df -B1 --output=avail "$LOCAL_ROOT" | tail -1 | tr -d ' ')
(( available > artifact_bytes + 20000000000 )) || die "insufficient local space for finalist"
rsync -a --partial --info=progress2 "$REMOTE:$remote_artifact/" "$dest/finalist/"
ssh "$REMOTE" "cd '$remote_artifact' && find . -type f -print0 | sort -z | xargs -0 sha256sum" \
  >"$dest/finalist.remote.sha256"
(cd "$dest/finalist" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
  >"$dest/finalist.local.sha256"
cmp "$dest/finalist.remote.sha256" "$dest/finalist.local.sha256" \
  || die "finalist artifact inventory differs"

python3 - "$dest/finalization-receipt.json" "$run_id" "$finalist" "$INSTANCE_ID" \
  "$REGION" "$manifest" "$dest/finalist.local.sha256" <<'PY'
import datetime,hashlib,json,pathlib,sys
out,run_id,finalist,instance,region,evidence,artifact=sys.argv[1:]
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pathlib.Path(out).write_text(json.dumps({
 "format":"bw24-smart100-local-finalization-v1","completed_utc":datetime.datetime.now(datetime.UTC).isoformat(),
 "full_agentic_run_id":run_id,"finalist":finalist,"instance_id":instance,"region":region,
 "evidence_manifest_sha256":sha(evidence),"finalist_inventory_sha256":sha(artifact),
 "remote_instance_action":"terminate-after-verified-sync"
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

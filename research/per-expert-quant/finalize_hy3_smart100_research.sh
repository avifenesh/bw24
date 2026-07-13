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
REMOTE_FULL_ROOT=${REMOTE_FULL_ROOT:-/data/results/per-expert-quant/full-agentic-iq3-iq4-q4-pareto-v1}
REMOTE_FULL_READY=${REMOTE_FULL_READY:-/data/logs/full-agentic-iq3-iq4-q4-pareto-v1/complete}
REMOTE_DIRECTIONAL_ROOT=${REMOTE_DIRECTIONAL_ROOT:-/data/results/per-expert-quant/iq3-iq4-q4-pareto-directional-v1}
REMOTE_PRACTICAL_ROOT=${REMOTE_PRACTICAL_ROOT:-/data/results/per-expert-quant/practical-iq3-iq4-q4-pareto-v1}
REMOTE_TRUSTED_ROOT=${REMOTE_TRUSTED_ROOT:-/data/results/per-expert-quant/trusted-full-iq3-iq4-q4-pareto-v1}
REMOTE_FINALIZER_ROOT=${REMOTE_FINALIZER_ROOT:-/data/src/bw24-finalizer-conclusion-v1}
REMOTE_CONCLUSION_ROOT=${REMOTE_CONCLUSION_ROOT:-/data/analysis/per-expert-quant-final-conclusion-v1}
LOCAL_ROOT=${LOCAL_ROOT:-/home/avifenesh/projects/bw24-research-archive/smart100-final}
BASELINE_ALLOCATION_ANALYSIS=${BASELINE_ALLOCATION_ANALYSIS:-/data/analysis/per-expert-quant-smart100-1a97cb3}
IQ4_ALLOCATION_ANALYSIS=${IQ4_ALLOCATION_ANALYSIS:-/data/analysis/per-expert-quant-iq3-iq4-q4-9a1c92c}
CENTERED_ALLOCATION_ANALYSIS=${CENTERED_ALLOCATION_ANALYSIS:-/data/analysis/per-expert-quant-iq3-iq4-q4-centered-a7200c0}
PARETO_ALLOCATION_ANALYSIS=${PARETO_ALLOCATION_ANALYSIS:-/data/analysis/per-expert-quant-iq3-iq4-q4-dominance-6c5c5ea}
PAIR_ALLOCATION_ANALYSIS=${PAIR_ALLOCATION_ANALYSIS:-/data/analysis/per-expert-quant-prune-vs-smart100-9a1c92c}
BASE_EFFECT_ANALYSIS=${BASE_EFFECT_ANALYSIS:-/data/analysis/per-expert-quant-effects-38af56e}
IQ4_EFFECT_ANALYSIS=${IQ4_EFFECT_ANALYSIS:-/data/calibration/hy3-quant-iq3-iq4-q4-pareto-6c5c5ea}
IQ4_EFFECT_EVIDENCE=${IQ4_EFFECT_EVIDENCE:-/data/logs/iq3-iq4-q4-pareto-6c5c5ea/build/evidence.sha256}
PRIVATE_DAMAGE=${PRIVATE_DAMAGE:-$PARETO_ALLOCATION_ANALYSIS/private-damage-three-way.json}
UNCENTERED_PLAN=${UNCENTERED_PLAN:-/data/plans/per-expert-quant-iq3-iq4-q4-99f3dc3/smart100_iq3_iq4_q4_empirical.json}
CENTERED_PLAN=${CENTERED_PLAN:-$CENTERED_ALLOCATION_ANALYSIS/smart100_iq3_iq4_q4_centered.json}
PARETO_PLAN=${PARETO_PLAN:-$PARETO_ALLOCATION_ANALYSIS/smart100_iq3_iq4_q4_pareto.json}

EVIDENCE_ROOTS=(
  /data/results/per-expert-quant
  /data/logs
  /data/plans
  /data/calibration/hy3-routing-v1
  /data/calibration/hy3-confidence-v1
  /data/calibration/hy3-100gb-5f02c37
  /data/calibration/hy3-quant-sensitivity-53de6ca
  /data/calibration/hy3-quant-iq3-iq4-q4-99f3dc3
  "$BASELINE_ALLOCATION_ANALYSIS"
  "$IQ4_ALLOCATION_ANALYSIS"
  "$CENTERED_ALLOCATION_ANALYSIS"
  "$PARETO_ALLOCATION_ANALYSIS"
  "$PAIR_ALLOCATION_ANALYSIS"
  "$BASE_EFFECT_ANALYSIS"
  "$IQ4_EFFECT_ANALYSIS"
  "$REMOTE_CONCLUSION_ROOT"
  /data/heal/per-expert-quant-100gb-5f02c37/router/receipts
  /data/heal/per-expert-quant-100gb-5f02c37/joint/receipts
  /data/heal/per-expert-quant-smart100-2605fde/smart100_empirical/receipts
  /data/heal/per-expert-quant-smart100-2605fde/smart100_balanced/receipts
  /data/heal/per-expert-quant-smart100-2605fde/smart100_rescue/receipts
  /data/heal/per-expert-quant-iq3-iq4-q4-99f3dc3/smart100_iq3_iq4_q4_empirical/receipts
  /data/heal/per-expert-quant-iq3-iq4-q4-pareto-6c5c5ea/smart100_iq3_iq4_q4_pareto/receipts
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
directional_run=$(ssh "$REMOTE" "cat '$REMOTE_DIRECTIONAL_ROOT/_active-run-id'")
practical_run=$(ssh "$REMOTE" "cat '$REMOTE_PRACTICAL_ROOT/_active-run-id'")
trusted_run=$(ssh "$REMOTE" "cat '$REMOTE_TRUSTED_ROOT/_active-run-id'")
analysis_commit=$(ssh "$REMOTE" "git -C '$REMOTE_FINALIZER_ROOT' rev-parse HEAD")
[[ "$analysis_commit" =~ ^[0-9a-f]{40}$ ]] || die "invalid remote finalizer commit"
directional_frontier="$REMOTE_DIRECTIONAL_ROOT/iq3-iq4-q4-frontier-$directional_run.json"
directional_promotion="$REMOTE_DIRECTIONAL_ROOT/iq3-iq4-q4-promotion-$directional_run.json"
practical_promotion="$REMOTE_PRACTICAL_ROOT/practical-promotion-$practical_run.json"
trusted_report="$REMOTE_TRUSTED_ROOT/_runs/$trusted_run/trusted-full-results.json"
combined="$REMOTE_FULL_ROOT/comparisons/$run_id/combined.json"

ssh "$REMOTE" bash -s -- \
  "$REMOTE_FINALIZER_ROOT" "$analysis_commit" "$REMOTE_CONCLUSION_ROOT" \
  "$IQ4_EFFECT_ANALYSIS/seven-format-effects-map.json" "$PRIVATE_DAMAGE" \
  "$directional_frontier" "$directional_promotion" "$practical_promotion" \
  "$trusted_report" "$combined" "$UNCENTERED_PLAN" "$CENTERED_PLAN" "$PARETO_PLAN" <<'SH'
set -euo pipefail
root=$1
commit=$2
out_root=$3
effects=$4
damage=$5
frontier=$6
directional=$7
practical=$8
trusted=$9
full=${10}
uncentered=${11}
centered=${12}
pareto=${13}
tool="$root/tools/summarize_hy3_quant_research.py"
output="$out_root/conclusion.json"
markdown="$out_root/conclusion.md"
receipt="$out_root/receipt.json"
evidence="$out_root/evidence.sha256"
[[ $(git -C "$root" rev-parse HEAD) == "$commit" ]]
[[ -z $(git -C "$root" symbolic-ref -q HEAD || true) ]]
for path in "$tool" "$effects" "$damage" "$frontier" "$directional" "$practical" \
  "$trusted" "$full" "$uncentered" "$centered" "$pareto"; do
  [[ -f "$path" ]]
done
if [[ ! -f "$receipt" ]]; then
  mkdir -p "$out_root"
  python3 "$tool" --effects "$effects" --damage "$damage" --frontier "$frontier" \
    --directional-promotion "$directional" --practical-promotion "$practical" \
    --trusted-report "$trusted" --full-agentic "$full" \
    --plan "uncentered=$uncentered" --plan "centered=$centered" --plan "pareto=$pareto" \
    --analysis-commit "$commit" --output "$output" --markdown "$markdown" \
    --receipt "$receipt"
  sha256sum "$output" "$markdown" "$receipt" "$tool" >"$evidence"
fi
python3 "$tool" --verify-receipt "$receipt"
sha256sum -c "$evidence"
SH

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
  smart100_iq3_iq4_q4_centered)
    remote_artifact="/scratch/bw24-artifacts-iq3-iq4-q4-centered-0f98d7d/$finalist" ;;
  smart100_iq3_iq4_q4_pareto)
    remote_artifact="/scratch/bw24-artifacts-iq3-iq4-q4-pareto-6c5c5ea/$finalist" ;;
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
  /data/logs/iq3-iq4-q4-pareto-6c5c5ea/complete \
  /data/logs/iq3-iq4-q4-pareto-directional-v1/complete \
  /data/logs/practical-iq3-iq4-q4-pareto-v1/complete \
  /data/logs/trusted-full-iq3-iq4-q4-pareto-v1/complete \
  /data/logs/full-agentic-iq3-iq4-q4-pareto-v1/complete \
  /data/logs/full-agentic-iq3-iq4-q4-pareto-v1/chain-complete; do test -f "$f"; done
test -z "$(pgrep -x bw24-server || true)"
test -z "$(pgrep -af "[/]harbor run " || true)"
test -z "$(docker ps -q)"
' || die "remote completion or idle-process gate failed"

ALLOCATION_RECEIPTS=(
  "$BASELINE_ALLOCATION_ANALYSIS/receipt.json"
  "$IQ4_ALLOCATION_ANALYSIS/receipt.json"
  "$CENTERED_ALLOCATION_ANALYSIS/allocation-comparison.receipt.json"
  "$PARETO_ALLOCATION_ANALYSIS/allocation-comparison.receipt.json"
  "$PAIR_ALLOCATION_ANALYSIS/receipt.json"
)
for receipt_path in "${ALLOCATION_RECEIPTS[@]}"; do
  ssh "$REMOTE" "python3 - '$receipt_path'" <<'PY'
import hashlib,json,pathlib,re,sys

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()

path=pathlib.Path(sys.argv[1])
receipt=json.loads(path.read_text())
assert receipt["format"] == "bw24-hy3-allocation-analysis-receipt-v1"
assert re.fullmatch(r"[0-9a-f]{40}", receipt["analysis_commit"])
assert receipt["public_eval_data_used"] is False
assert receipt["inputs"]
for item in [*receipt["inputs"], receipt["output"], receipt["script"]]:
    target=pathlib.Path(item["path"])
    assert target.is_file()
    assert sha256(target) == item["sha256"]
PY
done
ssh "$REMOTE" "test -f '$CENTERED_ALLOCATION_ANALYSIS/complete' && \
  test -s '$CENTERED_ALLOCATION_ANALYSIS/evidence.sha256' && \
  sha256sum -c '$CENTERED_ALLOCATION_ANALYSIS/evidence.sha256'" \
  >/dev/null || die "centered allocation evidence validation failed"
ssh "$REMOTE" "test -f '$PARETO_ALLOCATION_ANALYSIS/complete'" \
  >/dev/null || die "Pareto allocation evidence validation failed"

ssh "$REMOTE" "test -s '$BASE_EFFECT_ANALYSIS/evidence.sha256' && \
  sha256sum -c '$BASE_EFFECT_ANALYSIS/evidence.sha256'" \
  >/dev/null || die "base quant effects evidence validation failed"
ssh "$REMOTE" "test -s '$IQ4_EFFECT_EVIDENCE' && sha256sum -c '$IQ4_EFFECT_EVIDENCE'" \
  >/dev/null || die "seven-format quant effects evidence validation failed"

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
  # Preserve each absolute subtree below evidence-root so the per-root verifier addresses the
  # same path locally (for example, /data/logs -> evidence-root/data/logs).
  rsync -aR --partial --append-verify "$REMOTE:/./${root#/}/" "$dest/evidence-root/"
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

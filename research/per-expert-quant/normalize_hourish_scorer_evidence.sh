#!/usr/bin/env bash
set -euo pipefail

# Re-score immutable expanded-panel generations with the exact scorer images recorded by a
# baseline frontier.  The original run is never edited; the normalized view is a separate tree.

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:?set SOURCE_RUN_DIR}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
ARM=${ARM:?set ARM}
RUN_ID=${RUN_ID:?set RUN_ID}
BASELINE_FRONTIER=${BASELINE_FRONTIER:?set BASELINE_FRONTIER}
BASELINE_ARM=${BASELINE_ARM:-plain_quant}
PANEL_LOCK=${PANEL_LOCK:?set PANEL_LOCK}

die() { echo "normalize hourish scorers: $*" >&2; exit 1; }
sha256() { sha256sum "$1" | cut -d' ' -f1; }

[[ -d "$SOURCE_RUN_DIR" ]] || die "missing source run directory"
[[ -f "$BASELINE_FRONTIER" && -f "$PANEL_LOCK" ]] || die "missing frozen lock input"
[[ "$ARM" =~ ^[A-Za-z0-9._-]+$ && "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "invalid arm or run id"
command -v docker >/dev/null || die "docker is required"

dest="$OUT_ROOT/$ARM/$RUN_ID"
evidence="$dest/scorer-normalization.evidence.sha256"
if [[ -d "$dest" ]]; then
  [[ -f "$evidence" ]] || die "destination exists without complete evidence"
  sha256sum -c "$evidence"
  echo "normalized scorer evidence already complete: $dest"
  exit 0
fi

mkdir -p "$(dirname "$dest")"
stage=$(mktemp -d "$(dirname "$dest")/.${RUN_ID}.normalize.XXXXXX")
trap 'rm -rf "$stage"' EXIT
cp -a "$SOURCE_RUN_DIR/." "$stage/"
rm -f "$stage/code-score.json" "$stage/code-score.receipt.json" \
  "$stage/math-score.json" "$stage/math-score.receipt.json"

lock="$stage/scorer-normalization.lock.json"
python3 - "$BASELINE_FRONTIER" "$BASELINE_ARM" "$PANEL_LOCK" "$lock" <<'PY'
import hashlib, json, pathlib, sys

frontier_path, baseline_arm, panel_path, output = map(pathlib.Path, sys.argv[1:])
frontier = json.loads(frontier_path.read_text())
if frontier.get("format") != "bw24-cross-run-expanded-capability-frontier-v1":
    raise SystemExit("wrong baseline frontier format")
if str(baseline_arm) not in frontier.get("arms", {}):
    raise SystemExit("baseline arm is absent from frontier")
arm = frontier["arms"][str(baseline_arm)]
run_dir = pathlib.Path(arm["run_dir"])
code_receipt = json.loads((run_dir / "code-score.receipt.json").read_text())
math_receipt = json.loads((run_dir / "math-score.receipt.json").read_text())
for name, scorer, receipt in (
    ("code", arm["code_scorer"], code_receipt),
    ("math", arm["math_scorer"], math_receipt),
):
    if any(receipt.get(key) != scorer.get(key) for key in ("tool_sha256", "image_id")):
        raise SystemExit(f"{name} scorer receipt differs from frontier")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "format": "bw24-hourish-scorer-normalization-lock-v1",
    "baseline_arm": str(baseline_arm),
    "baseline_frontier": {"path": str(frontier_path.resolve()), "sha256": sha(frontier_path)},
    "panel_lock": {"path": str(panel_path.resolve()), "sha256": sha(panel_path)},
    "code": {**arm["code_scorer"], "image": code_receipt["image"],
             "receipt_path": str((run_dir / "code-score.receipt.json").resolve()),
             "receipt_sha256": sha(run_dir / "code-score.receipt.json")},
    "math": {**arm["math_scorer"], "image": math_receipt["image"],
             "receipt_path": str((run_dir / "math-score.receipt.json").resolve()),
             "receipt_sha256": sha(run_dir / "math-score.receipt.json")},
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

mapfile -t scorer_fields < <(python3 - "$lock" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for name in ("code", "math"):
    row = d[name]
    print("\t".join((row["image"], row["image_id"], row["tool_sha256"],
                     str(row["expected_sample_count"]), row["panel_lock_sha256"])))
PY
)
IFS=$'\t' read -r code_image code_image_id code_tool code_n code_panel <<<"${scorer_fields[0]}"
IFS=$'\t' read -r math_image math_image_id math_tool math_n math_panel <<<"${scorer_fields[1]}"
[[ "$code_panel" == "$(sha256 "$PANEL_LOCK")" && "$math_panel" == "$code_panel" ]] \
  || die "baseline scorer panel differs from requested panel"
[[ $(docker image inspect --format '{{.Id}}' "$code_image_id") == "$code_image_id" ]] \
  || die "baseline code scorer image is unavailable"
[[ $(docker image inspect --format '{{.Id}}' "$math_image_id") == "$math_image_id" ]] \
  || die "baseline math scorer image is unavailable"

mapfile -t code_samples < <(find "$stage/shards/humaneval_instruct" \
  -type f -name 'samples_humaneval_instruct_*.jsonl')
mapfile -t math_samples < <(find "$stage/shards/hendrycks_math500" \
  -type f -name 'samples_hendrycks_math500_*.jsonl')
[[ ${#code_samples[@]} == 1 && ${#math_samples[@]} == 1 ]] \
  || die "expected exactly one generated sample file per scored task"

docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --pids-limit 32 --memory 768m --cpus 1 \
  --cpu-shares 2 --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --mount "type=bind,src=$stage,dst=/inputs,readonly" "$code_image_id" \
  "/inputs/${code_samples[0]#"$stage/"}" >"$stage/code-score.json"
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges:true --pids-limit 32 --memory 1g --cpus 1 \
  --cpu-shares 2 --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --mount "type=bind,src=$stage,dst=/inputs,readonly" "$math_image_id" \
  "/inputs/${math_samples[0]#"$stage/"}" >"$stage/math-score.json"

SOURCE_RUN_DIR=$(cd "$SOURCE_RUN_DIR" && pwd -P)
DEST_RUN_DIR=$(cd "$(dirname "$dest")" && pwd -P)/$(basename "$dest")
export SOURCE_RUN_DIR DEST_RUN_DIR PANEL_LOCK lock stage code_image code_image_id code_tool code_n
export math_image math_image_id math_tool math_n
python3 - <<'PY'
import hashlib, json, os, pathlib

source = pathlib.Path(os.environ["SOURCE_RUN_DIR"])
stage = pathlib.Path(os.environ["stage"])
destination = pathlib.Path(os.environ["DEST_RUN_DIR"])
panel_sha = hashlib.sha256(pathlib.Path(os.environ["PANEL_LOCK"]).read_bytes()).hexdigest()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def outcomes(report):
    result = {}
    for row in report["samples"]:
        key = ":".join((row["doc_hash"], row["prompt_hash"], row["target_hash"]))
        result[key] = row["passed"]
    return result

normalized = {}
for kind, task, expected in (
    ("code", "humaneval_instruct", int(os.environ["code_n"])),
    ("math", "hendrycks_math500", int(os.environ["math_n"])),
):
    original_path = source / f"{kind}-score.json"
    output_path = stage / f"{kind}-score.json"
    original = json.loads(original_path.read_text())
    output = json.loads(output_path.read_text())
    if output.get("total") != expected or output.get("by_task", {}).get(task, {}).get("total") != expected:
        raise SystemExit(f"normalized {kind} sample count differs from lock")
    if outcomes(original) != outcomes(output):
        raise SystemExit(f"normalized {kind} outcomes differ from original scorer")
    image = os.environ[f"{kind}_image"]
    image_id = os.environ[f"{kind}_image_id"]
    tool = os.environ[f"{kind}_tool"]
    memory = (768 if kind == "code" else 1024) * 1024 * 1024
    receipt = {
        "format": f"bw24-hourish-{kind}-score-receipt-v1",
        "run_dir": str(destination),
        "output": str(destination / f"{kind}-score.json"),
        "output_sha256": sha(output_path),
        "image": image,
        "image_id": image_id,
        "tool_sha256": tool,
        "panel_lock_sha256": panel_sha,
        "expected_sample_count": expected,
        "sandbox": {"network": "none", "read_only_root": True,
                    "capabilities": "all dropped", "no_new_privileges": True,
                    "pids_limit": 32, "memory_bytes": memory, "cpus": 1,
                    "cpu_shares": 2},
    }
    receipt_path = stage / f"{kind}-score.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    normalized[kind] = {
        "original_output": {"path": str(original_path.resolve()), "sha256": sha(original_path)},
        "original_receipt": {"path": str((source / f"{kind}-score.receipt.json").resolve()),
                             "sha256": sha(source / f"{kind}-score.receipt.json")},
        "normalized_output_sha256": sha(output_path),
        "normalized_receipt_sha256": sha(receipt_path),
        "outcomes_equal": True,
        "passed": output["passed"],
        "total": output["total"],
    }

lock_path = pathlib.Path(os.environ["lock"])
normalization = {
    "format": "bw24-hourish-scorer-normalization-receipt-v1",
    "source_run_dir": str(source),
    "destination_run_dir": str(destination),
    "lock_sha256": sha(lock_path),
    "normalized": normalized,
}
(stage / "scorer-normalization.receipt.json").write_text(
    json.dumps(normalization, indent=2, sort_keys=True) + "\n"
)
PY

mv "$stage" "$dest"
trap - EXIT
sha256sum \
  "$SOURCE_RUN_DIR/code-score.json" "$SOURCE_RUN_DIR/code-score.receipt.json" \
  "$SOURCE_RUN_DIR/math-score.json" "$SOURCE_RUN_DIR/math-score.receipt.json" \
  "$dest/code-score.json" "$dest/code-score.receipt.json" \
  "$dest/math-score.json" "$dest/math-score.receipt.json" \
  "$dest/scorer-normalization.lock.json" "$dest/scorer-normalization.receipt.json" \
  "$BASELINE_FRONTIER" "$PANEL_LOCK" >"$evidence"
sha256sum -c "$evidence"
echo "normalized scorer evidence complete: $dest"

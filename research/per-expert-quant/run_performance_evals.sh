#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/scratch/artifacts}
OUT_ROOT=${OUT_ROOT:-$ROOT/research/per-expert-quant/results/performance}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$OUT_ROOT/$RUN_ID"
PREFILL_TOKENS=${PREFILL_TOKENS:-512}
DECODE_TOKENS=${DECODE_TOKENS:-128}
DECODE_REPS=${DECODE_REPS:-3}
ARMS=(plain-quant plain-reap-quant plain-reap-mix-quant mix-quant mix-quant-prune25)

for bin in run-gen decode-bench; do
  test -x "$ROOT/target/release/$bin" || {
    echo "missing $ROOT/target/release/$bin; build the release binaries first" >&2
    exit 2
  }
done
for arm in "${ARMS[@]}"; do
  test -f "$ARTIFACT_ROOT/$arm/manifest.json" || {
    echo "missing $ARTIFACT_ROOT/$arm/manifest.json" >&2
    exit 2
  }
done

mkdir -p "$RUN_DIR"
export ROOT PREFILL_TOKENS DECODE_TOKENS DECODE_REPS
export BW24_FAST=${BW24_FAST:-1}
export BW24_MMVQ=${BW24_MMVQ:-1}
export BW24_MOE_CACHE=${BW24_MOE_CACHE:-1}
export BW24_MOE_GROUPED=${BW24_MOE_GROUPED:-1}
export BW24_MOE_PREWARM=${BW24_MOE_PREWARM:-1}
export BW24_MOE_PREFETCH=${BW24_MOE_PREFETCH:-1}
export BW24_MOE_PAGE_PREFETCH=${BW24_MOE_PAGE_PREFETCH:-1}
export BW24_MOE_PAGE_PREFETCH_WINDOW=${BW24_MOE_PAGE_PREFETCH_WINDOW:-1}

python3 - "$RUN_DIR/metadata.json" "$ARTIFACT_ROOT" <<'PY'
import json, os, pathlib, platform, subprocess, sys

root = pathlib.Path(os.environ["ROOT"])
artifact_root = pathlib.Path(sys.argv[2]).resolve()

def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()

metadata = {
    "artifact_root": str(artifact_root),
    "bw24_commit": command("git", "-C", str(root), "rev-parse", "HEAD"),
    "decode_reps": int(os.environ["DECODE_REPS"]),
    "decode_tokens": int(os.environ["DECODE_TOKENS"]),
    "environment": {
        key: os.environ[key]
        for key in (
            "BW24_FAST", "BW24_MMVQ", "BW24_MOE_CACHE", "BW24_MOE_GROUPED",
            "BW24_MOE_PREWARM", "BW24_MOE_PREFETCH", "BW24_MOE_PAGE_PREFETCH",
            "BW24_MOE_PAGE_PREFETCH_WINDOW",
        )
    },
    "nvidia_smi": command(
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
    ),
    "platform": platform.platform(),
    "prefill_tokens": int(os.environ["PREFILL_TOKENS"]),
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

printf 'arm\tartifact_bytes\tdirectory_bytes\n' > "$RUN_DIR/sizes.tsv"
prompt=()
for ((i = 0; i < PREFILL_TOKENS; i++)); do
  prompt+=("$((100 + (i * 7) % 900))")
done

for arm in "${ARMS[@]}"; do
  artifact="$ARTIFACT_ROOT/$arm"
  artifact_bytes=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_bytes"])' \
    "$artifact/manifest.json")
  directory_bytes=$(du -sb "$artifact" | cut -f1)
  printf '%s\t%s\t%s\n' "$arm" "$artifact_bytes" "$directory_bytes" >> "$RUN_DIR/sizes.tsv"

  BW24_NGEN=1 "$ROOT/target/release/run-gen" "$artifact" "${prompt[@]:0:32}" \
    > "$RUN_DIR/$arm-correctness.log" 2>&1
  grep -q 'verify-prefill .* MATCH' "$RUN_DIR/$arm-correctness.log"

  BW24_PP_ONLY=1 BW24_PP_WARMUP=2 BW24_PP_REPS=3 \
    /usr/bin/time -v "$ROOT/target/release/run-gen" "$artifact" "${prompt[@]}" \
    > "$RUN_DIR/$arm-prefill.log" 2>&1

  for rep in $(seq 1 "$DECODE_REPS"); do
    /usr/bin/time -v "$ROOT/target/release/decode-bench" \
      "$artifact" "$PREFILL_TOKENS" "$DECODE_TOKENS" eager \
      > "$RUN_DIR/$arm-decode-$rep.log" 2>&1
  done
done

echo "$RUN_DIR"

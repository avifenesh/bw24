#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HERE="$ROOT/research/per-expert-quant"
LOCK="$HERE/mlx-reap50-reference.lock.json"
LOCK_SHA256=dd85692e943bd58a348c7cfbc271508160e53178af43cef976fcebd4fde4c131

: "${RUN_ID:?set the shared hourish RUN_ID}"
: "${ARTIFACT:?set ARTIFACT to a directory containing manifest.json for the pinned MLX checkpoint}"
: "${RUNTIME_IDENTITY:?set RUNTIME_IDENTITY to the immutable MLX runtime receipt}"
: "${SERVER_BIN:?set SERVER_BIN to the pinned mlx_lm.server executable}"
: "${SERVER_LOG:?set SERVER_LOG to the active MLX server log}"

OUT_ROOT=${OUT_ROOT:-$HERE/results-hourish-external}
CACHE_DIR=${CACHE_DIR:-$HERE/.cache}
BASE_URL=${BASE_URL:-http://127.0.0.1:8080/v1/completions}
ARM=${ARM:-mlx_reap50_reference}
MODEL=${MODEL:-default_model}

(( BASH_VERSINFO[0] >= 4 )) || {
  echo "Bash 4 or newer is required (install Homebrew bash on macOS)" >&2
  exit 2
}
[[ -f "$ARTIFACT/manifest.json" ]] || {
  echo "ARTIFACT must contain manifest.json" >&2
  exit 2
}
python3 - "$LOCK" "$LOCK_SHA256" "$ARTIFACT/manifest.json" "$RUNTIME_IDENTITY" <<'PY'
import hashlib, json, sys

lock = json.load(open(sys.argv[1]))
if hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != sys.argv[2]:
    raise SystemExit("MLX reference lock hash mismatch")
manifest = json.load(open(sys.argv[3]))
runtime = json.load(open(sys.argv[4]))

if manifest.get("model_repo") != lock["model"]["repo"]:
    raise SystemExit("artifact manifest model_repo differs from the lock")
if manifest.get("model_revision") != lock["model"]["revision"]:
    raise SystemExit("artifact manifest model_revision differs from the lock")
if manifest.get("artifact_bytes") != lock["model"]["repo_storage_bytes"]:
    raise SystemExit("artifact manifest byte count differs from the lock")
if runtime.get("runtime_repo") != lock["runtime"]["repo"]:
    raise SystemExit("runtime receipt repo differs from the lock")
if runtime.get("runtime_revision") != lock["runtime"]["revision"]:
    raise SystemExit("runtime receipt revision differs from the lock")
if runtime.get("draft_model") not in (None, False):
    raise SystemExit("MLX external reference must not use a draft model")
PY

ARM="$ARM" MODEL="$MODEL" ARTIFACT="$ARTIFACT" RUN_ID="$RUN_ID" \
  SERVER_BIN="$SERVER_BIN" SERVER_LOG="$SERVER_LOG" \
  RUNTIME_KIND=external_openai RUNTIME_IDENTITY="$RUNTIME_IDENTITY" \
  BASE_URL="$BASE_URL" OUT_ROOT="$OUT_ROOT" CACHE_DIR="$CACHE_DIR" \
  "$HERE/run_hourish_arm.sh"

RUN_DIR="$OUT_ROOT/$ARM/$RUN_ID"
"$HERE/score_hourish_code_container.sh" "$RUN_DIR"
"$HERE/score_hourish_math_container.sh" "$RUN_DIR"
echo "MLX hourish reference complete: $ARM/$RUN_ID"

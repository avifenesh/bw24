#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data/src/bw24-hy3-110gb}
ROOT=${ROOT:-/data/experiments/hy3-110gb}
CUDA_ARCH=${CUDA_ARCH:-100a}
TARGET_DIR=${TARGET_DIR:-/data/build/bw24-hy3-b200-sm${CUDA_ARCH}-target}
RUSTUP_HOME=${RUSTUP_HOME:-/data/toolchains/rustup}
CARGO_HOME=${CARGO_HOME:-/data/toolchains/cargo}

mkdir -p "$ROOT/logs" "$ROOT/receipts" "$RUSTUP_HOME" "$CARGO_HOME" "$TARGET_DIR"
exec > >(tee -a "$ROOT/logs/bw24-server-build.log") 2>&1
export RUSTUP_HOME CARGO_HOME
export PATH="$CARGO_HOME/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1; then
  installer=/tmp/bw24-rustup-init
  curl -fsS https://sh.rustup.rs -o "$installer"
  sh "$installer" -y --profile minimal --default-toolchain stable --no-modify-path
  rm -f "$installer"
fi
rustup toolchain install stable --profile minimal
export BW24_NVCC=/usr/local/cuda/bin/nvcc
export BW24_CUDA_ARCH="$CUDA_ARCH"
export CARGO_TARGET_DIR="$TARGET_DIR"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-12}
cargo build --manifest-path "$REPO/Cargo.toml" --release --package bw24-server --bin bw24-server

binary="$TARGET_DIR/release/bw24-server"
[[ -x "$binary" ]]
binary_sha=$(sha256sum "$binary" | awk '{print $1}')
python3 - "$ROOT/receipts/bw24-server-build.json" "$binary" "$binary_sha" \
  "$(rustc --version)" "$(cargo --version)" "$(git -C "$REPO" rev-parse HEAD)" \
  "$CUDA_ARCH" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

output, binary, sha, rustc, cargo, git_head, cuda_arch = sys.argv[1:]
pathlib.Path(output).write_text(json.dumps({
    "format": "bw24-hy3-b200-server-build-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "binary": str(pathlib.Path(binary).resolve()),
    "binary_sha256": sha,
    "rustc": rustc,
    "cargo": cargo,
    "git_head_on_snapshot": git_head,
    "cuda_arch": f"sm_{cuda_arch}",
    "nvcc": "/usr/local/cuda/bin/nvcc",
}, indent=2, sort_keys=True) + "\n")
PY
echo "bw24-server complete binary=$binary sha256=$binary_sha"

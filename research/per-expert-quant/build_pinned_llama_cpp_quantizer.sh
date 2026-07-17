#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/experiments/hy3-110gb}
SOURCE_DIR=${SOURCE_DIR:-/data/build/llama-cpp-99f3dc3}
BUILD_DIR=${BUILD_DIR:-$SOURCE_DIR/build-cpu}
REVISION=${REVISION:-99f3dc32296f825fec94f202da1e9fede1e78cf9}
JOBS=${JOBS:-96}

mkdir -p "$ROOT/logs" "$ROOT/receipts" "$(dirname "$SOURCE_DIR")"
exec > >(tee -a "$ROOT/logs/llama-cpp-build.log") 2>&1
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" fetch --depth=1 origin "$REVISION"
git -C "$SOURCE_DIR" checkout --detach "$REVISION"
[[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$REVISION" ]]

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
  -DGGML_CUDA=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_COMMON=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build "$BUILD_DIR" --parallel "$JOBS" --target ggml-base

lib=$(find "$BUILD_DIR" -type f -name 'libggml-base.so*' | sort | tail -1)
[[ -n "$lib" && -f "$lib" ]]
lib_sha=$(sha256sum "$lib" | awk '{print $1}')
python3 - "$ROOT/receipts/llama-cpp-quantizer.json" "$REVISION" "$lib" "$lib_sha" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

output, revision, library, sha = sys.argv[1:]
pathlib.Path(output).write_text(json.dumps({
    "format": "bw24-pinned-llama-cpp-quantizer-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "llama_cpp_commit": revision,
    "library": str(pathlib.Path(library).resolve()),
    "library_sha256": sha,
}, indent=2, sort_keys=True) + "\n")
PY
echo "pinned quantizer library=$lib sha256=$lib_sha"

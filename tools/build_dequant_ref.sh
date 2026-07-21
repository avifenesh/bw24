#!/usr/bin/env bash
# Build the ggml ground-truth dequant oracle against the local llama.cpp checkout.
set -euo pipefail
LL=/home/avifenesh/projects/llama.cpp
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
g++ -O2 -std=c++17 "$HERE/ggml_dequant_ref.cpp" -o "$HERE/ggml_dequant_ref" \
  -I"$LL/include" -I"$LL/ggml/include" \
  -L"$LL/build/bin" -lllama -lggml -lggml-base
echo "built $HERE/ggml_dequant_ref"

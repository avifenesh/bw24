#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 LLAMA_CPP_DIR [OUTPUT_SO]" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
llama_dir=$(realpath -- "$1")
output=${2:-$repo_dir/target/release/libbw24-cpu-experts.so}
include_dir=$llama_dir/ggml/include
library_dir=$llama_dir/build/bin

for required in \
  "$include_dir/ggml.h" \
  "$include_dir/ggml-cpu.h" \
  "$library_dir/libggml-base.so" \
  "$library_dir/libggml-cpu.so"; do
  if [[ ! -e $required ]]; then
    echo "missing llama.cpp CPU build artifact: $required" >&2
    exit 3
  fi
done

mkdir -p -- "$(dirname -- "$output")"
"${CXX:-c++}" -O3 -march=native -fPIC -shared -fopenmp \
  "$repo_dir/tools/bw24_cpu_experts.cpp" \
  -I"$include_dir" \
  -L"$library_dir" \
  -Wl,-rpath,"$library_dir" \
  -lggml-cpu -lggml-base \
  -o "$output"

sha256sum -- "$output"

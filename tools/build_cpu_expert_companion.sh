#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "usage: $0 [OUTPUT_SO]" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
output=${1:-$repo_dir/target/release/libbw24-cpu-experts.so}

mkdir -p -- "$(dirname -- "$output")"
"${CXX:-c++}" -std=c++17 -O3 -march=native -fPIC -shared -fopenmp \
  -Wall -Wextra -Wpedantic -Werror \
  "$repo_dir/tools/bw24_cpu_experts.cpp" \
  -o "$output"

if ldd -- "$output" | grep -Eqi 'llama|ggml'; then
  echo "native CPU expert backend unexpectedly depends on llama/ggml" >&2
  exit 3
fi

sha256sum -- "$output"

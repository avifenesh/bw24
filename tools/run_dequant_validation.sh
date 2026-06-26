#!/usr/bin/env bash
# End-to-end dequant validation harness for the 5 new dtypes (NVFP4, Q5_K, IQ3_S,
# IQ4_XS, Q3_K) against ggml ground truth, on REAL tensors from the daily GGUFs.
#
#   1. build + run tools/ggml_dequant_ref (links libggml) -> /tmp/dq/<name>.{raw,ref}
#      = exact quant bytes + ggml dequantize_row_<type> (via to_float) f32.
#   2. cargo run dequant_oracle_diff           -> diff bw24 CPU dequant vs ggml .ref
#   3. cargo run --bin dtype_gpu_check5        -> diff bw24 GPU (Stage-A + dp4a) vs
#      cpu_linear(bw24_dequant(W),x), which step 2 proved == ggml.
set -euo pipefail

LL=/home/avifenesh/projects/llama.cpp
BW=/home/avifenesh/projects/bw24
G9B=/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
G35B=/home/avifenesh/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
export LD_LIBRARY_PATH="$LL/build/bin:${LD_LIBRARY_PATH:-}"

echo "### 1/3 build ggml oracle ###"
"$BW/tools/build_dequant_ref.sh"

echo "### 1b dump ggml references (real tensors, 64 superblocks each) ###"
mkdir -p /tmp/dq
REF="$BW/tools/ggml_dequant_ref"
"$REF" "$G9B"  blk.0.ffn_gate.weight        256 /tmp/dq/nvfp4   # NVFP4 (64-elem blocks)
"$REF" "$G9B"  blk.0.attn_gate.weight        64 /tmp/dq/q5k     # Q5_K
"$REF" "$G35B" blk.0.ffn_gate_exps.weight    64 /tmp/dq/iq3s    # IQ3_S (expert 0 region)
"$REF" "$G35B" blk.0.ffn_down_exps.weight    64 /tmp/dq/iq4xs   # IQ4_XS
"$REF" "$G35B" blk.40.ffn_gate_exps.weight   64 /tmp/dq/q3k     # Q3_K

echo
echo "### 2/3 CPU dequant vs ggml byte-for-byte ###"
cargo run -q -p bw24-gguf --example dequant_oracle_diff

echo
echo "### 3/3 GPU qmatvec (Stage-A + dp4a) vs ggml-equivalent oracle ###"
cargo run -q -p bw24-engine --bin dtype_gpu_check5

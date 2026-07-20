
## Sweep 2026-07-15T07:30:04Z (since 2026-07-15T00:00:00Z)

### llama.cpp commits (decode-relevant, CUDA)
- (none)

### vllm-project/vllm releases
- (none)

### sgl-project/sglang releases
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

## Sweep 2026-07-20T06:17:01Z (since 2026-07-15T07:30:04Z)

### llama.cpp commits (decode-relevant, CUDA)
- common : auto-download dflash- and eagle3- HF sidecars (#25811)
- convert : fix dflash target tokenizer mismatch during conversion (#25733)
- cuda : CUDA GGML_OP_LIGHTNING_INDEXER implementation (generic vector kernel + wmma kernel) (#25545)
- CUDA: dedup MoE gate/up activation quantization (#25441)
- cuda: extract Q1_0 elements via __byte_perm (#25628)
- cuda : relax tensor contiguity requirements for quantized concat (#25678)
- CUDA: Support CUDA Virtual Devices (#25228)
- CUDA: tighter MMQ src1 buffer size for native fp4 (#25613)
- DeepseekV4: Add fused hyper-connection ops (#25585)
- DeepseekV4: reduce graph splits (#25702)
- Enable CUDA graphs on volta+turing (#25749)
- metal: fuse snake activation (mul, sin, sqr, mul, add) (#25459)
- model: rotate injected K/V cache for DFlash (#25823)

### vllm-project/vllm releases
- (none)

### sgl-project/sglang releases
- (none)

_Review protocol: anything testable gets ported behind a seam + A/B'd per the_
_flags doctrine; parity items get a one-line note; the jsonl is the record._

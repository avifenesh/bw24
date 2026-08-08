# Geometry-table extraction receipt

Source commit: `3e12cc625f1137ab3dfae7344264b5501f04d251`

Candidate tree used for both GPU runs:
`89d1d368af7db1d0fbfc6097026c8447a1437213`

## Extracted contract

`ArchGeometryTable` is a compact class table plus one class id per model
layer. Each `LayerGeometry` declares:

- mixer kind;
- query and KV head counts;
- K and V head dimensions;
- rotary width and base;
- optional sliding window;
- whether the layer consumes RoPE factors;
- no gate, a fused Q gate, or a separate head gate.

Qwen3.5 builds two classes from GGUF or HF config: linear attention and full
attention. Appended MTP layers are explicitly assigned to the full-attention
class rather than inheriting the trunk periodic formula.

Step35 derives its full and SWA classes from the artifact's per-layer arrays.
The table never fabricates an out-of-range row. A standalone drafter resolves
its MTP row from the drafter file's own table, preserving the original router
position and tensor-shape authority.

Legacy architectures still use their existing scalar or per-architecture
paths. `full_attention_geometry_at()` supplies the old scalar values to those
callers, so this commit does not implicitly migrate Gemma4, GLM-DSA, M3, Hy3,
or Qwen3.

## Migrated consumers

- layer classification and mixer loading;
- qwen35 prefill, chunked prime, eager decode, device-counter decode,
  speculative verify, cross-request prefill, and batched decode;
- Step35 prefill, prime, eager decode, batched decode, external drafter
  attach, embedded MTP attach, and Step MTP forward.

Step35 keeps its dedicated kernels and routing semantics. The extraction
centralizes geometry; it does not pretend the two architectures share their
attention implementation.

## Verification

- `cargo test -p memra-gguf --lib`: 78 passed
- `cargo check --workspace --all-targets`: passed
- qwen35 local RTX 5090 run:
  - prefill/decode `MATCH`
  - batched-prime/tokenwise `MATCH`
  - generated token array exactly equals the pre-refactor baseline
- Step35 Box 1 PP2 run:
  - prefill/decode `MATCH`
  - batched-prime/tokenwise `MATCH`
  - generated token array exactly equals the pre-refactor baseline

The mechanical comparisons are in `raw/post-geometry-comparison.log`.

## Raw receipts

- `raw/post-geometry-build-local.log`
- `raw/post-geometry-build-box1.log`
- `raw/post-geometry-memra-gguf-tests.log`
- `raw/post-geometry-workspace-check.log`
- `raw/post-geometry-qwen35-run-gen.log`
- `raw/post-geometry-step35-run-gen-box1.log`
- `raw/post-geometry-comparison.log`

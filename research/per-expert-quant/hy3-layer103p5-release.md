# Hy3 Layer103.5 release

`layer103-late20` is the smallest tested Layer100-preserving complement that improved the matched
115-question directional screen. It restores 262 experts only in layers 60-79 and leaves every
Layer100 expert choice and projection quantization unchanged.

| arm | logical bytes | math | code | history | other | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer100 matched | 99,999,322,624 | 27 | 32 | 3 | 11 | 73/115 |
| Layer103.5 late20 | 103,489,802,752 | 29 | 32 | 4 | 11 | 76/115 |

The paired comparison is 5 wins, 2 losses, and 108 ties. This is directional evidence, not a claim
of statistical significance: the paired exact-sign p-value is 0.453125 and the domain-macro
bootstrap interval crosses zero. Public capability results were not used to construct or heal the
arm; selection used private routing displacement and layer position only.

The published artifact is a `bw24-expert-overlay-v2`, not a Transformers checkpoint. It contains
78,490,288,128 bytes of expert payloads and resolves non-expert tensors from `tencent/Hy3` revision
`716aa7241bd6d95896be4ebfc761162a9c4d49ef`. The model and receipts are published at
[Avifenesh/Hy3-REAP-Layer103p5-bw24](https://huggingface.co/Avifenesh/Hy3-REAP-Layer103p5-bw24).

Use `tools/relocate_hy3_expert_overlay.py` to create a zero-copy runtime view that points the
immutable published overlay at a local copy of the pinned source checkpoint:

```bash
python3 tools/relocate_hy3_expert_overlay.py \
  /path/to/Hy3-REAP-Layer103p5-bw24 \
  /path/to/tencent-Hy3-716aa724 \
  /path/to/hy3-layer103p5-runtime
```

The runtime still needs the source checkpoint's non-expert tensors and the unmanaged layer-80 MTP
experts, but it does not need a second copy of the main 79-layer expert bank. To build a verified
sparse source view, download the source `config.json`, tensor index, every shard containing a
non-expert tensor, and the layer-80 expert shards, then pass `--sparse-source-view`:

```bash
python3 tools/relocate_hy3_expert_overlay.py \
  /path/to/Hy3-REAP-Layer103p5-bw24 \
  /path/to/tencent-Hy3-716aa724 \
  /path/to/hy3-layer103p5-runtime \
  --sparse-source-view /path/to/hy3-layer103p5-sparse-source
```

The command verifies the pinned source fingerprints and that every managed expert projection is
either supplied by the overlay or explicitly pruned. It keeps dense and layer-80 shards real and
uses valid empty safetensors placeholders only for expert-only shards that the overlay supersedes.

## Run on the 24 GB RTX 5090 profile

Build bw24 and its optional CPU-expert companion, then launch the measured local spill profile:

```bash
cargo build --release
git -C /path/to/llama.cpp checkout bb090d1f1dbf3c29df6778fda123aa352329514e
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build \
  -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/llama.cpp/build --target ggml-cpu ggml-base -j
tools/build_cpu_expert_companion.sh /path/to/llama.cpp
tools/run_hy3_local_5090.sh \
  /path/to/hy3-layer103p5-runtime \
  target/release/libbw24-cpu-experts.so \
  /path/to/expert-mirror/inode-alternates.tsv
```

The companion is trusted native code loaded into the bw24 process. Build it from the pinned,
reviewed checkout above (or an intentionally audited replacement).

The mirror map is optional but enables split reads across two byte-identical NVMe copies. Build it
with `tools/build_dual_nvme_expert_view.py` and `tools/build_expert_mirror_map.py`. The launcher
uses the correctness-gated non-fused LRU winner and retains 4 GiB of live RAM headroom; startup
prints the effective host-cache size when the requested 36 GiB does not fit the current desktop
stack.

Receipt anchors:

- source plan: `9606b1b96890b270534237b1143a5f5f25165245d1b5f08f515c25268d1b056c`
- artifact manifest: `08f206aed555752982585a59a7b5096b9cc6e71faf1f84ad5c6dd60476b7509a`
- screen summary: `661e495c02467acf5f28180eb73d562d25fea8724449e69f305831beb435be3d`
- winner receipt: `4c6d4089d5dec428a5575b8c9971521f1f04ed5041e48e02bda67d3dae993ab3`

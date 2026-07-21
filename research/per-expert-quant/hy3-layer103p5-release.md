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
cargo build --release --bins
tools/build_cpu_expert_companion.sh
BW24_CPU_EXPERT_LIB=target/release/libbw24-cpu-experts.so \
  target/release/cpu_native_check
tools/run_hy3_local_5090.sh \
  /path/to/hy3-layer103p5-dual-nvme \
  target/release/libbw24-cpu-experts.so \
  /path/to/expert-mirror/inode-alternates.tsv
```

The companion is bw24-owned native ABI v2 code loaded into the bw24 process. It has no llama.cpp,
ggml, or other inference-runtime dependency; `cpu_native_check` compares every supported packed
row kernel with bw24's independent Rust dequantization oracle before serving. Because `dlopen`
executes ELF constructors before the ABI check, `BW24_CPU_EXPERT_LIB` must point to a trusted build.

The mirror map is optional but enables split reads across two byte-identical NVMe copies. Build it
with `tools/build_dual_nvme_expert_view.py` and `tools/build_expert_mirror_map.py`. The launcher
must receive the generated dual-NVMe view as its model directory when the map is enabled. Native
ABI v2 pins source and alternate generations by device, inode, size, and ctime, so pairing the map
with the persistent source tree fails closed instead of silently reading an unverified mirror. The
launcher
uses the correctness-gated non-fused LRU winner and retains 4 GiB of live RAM headroom; startup
prints the effective host-cache size when the requested 20 GiB does not fit the current desktop
stack.

## Native ABI v2 local validation

The 2026-07-21 RTX 5090 target battery used only bw24's native CPU expert implementation. The
companion SHA-256 was `26303685576126a829933144be6af7dad6a6c19995b0b90421ca196d47c31621`;
`ldd` showed no llama.cpp or ggml library. The packed-row checker passed all 12 supported formats at
widths 256, 1536, and 4096, plus non-finite input rejection and an independently composed nonzero
MoE token. `kernel-check` reported `ALL GREEN`, `run-gen` reported argmax `40129 == 40129` and
`MATCH` after freezing the measured CPU/GPU expert assignment, and `run-spec` produced identical
output for K=1 through K=8.

The default 128-token residency warmup measured a 4.60 tok/s median over three interleaved N=32
post-freeze pairs with bw24's paired AVX-VNNI Q2_K kernel. Its matched pre-change median was
4.37 tok/s with identical output and residency; median CPU compute fell from 3.237 s to 3.025 s and
exposed CPU wait from 6.310 s to 5.956 s. This is +5.3% by arm medians and +6.9% by median paired
delta.
The MTP-capable default plain control measured 3.76 tok/s over N=7 before the K sweep; K=1 through
K=8 were exact but slower for this short prompt. The MTP control is a single observation; the Q2_K
result is N=3 with both pair orders. Every Q2_K arm was cooled to a 55-56 C start on the active
desktop, with
eight CPU workers, a 20 GiB requested/effective host cache, a 4 GiB live-RAM reserve, and the
generation-pinned dual-NVMe map
`861f58c5ad506f0d62242bed5cd79a97313e83a9df4412ddc4930ce1b0159a15`. They are not
Qwen-board measurements. Raw logs and the failed source-tree/map pairing check are retained under
`evidence/local-5090-native-20260721/`; the Q2_K win and exact gate are under
`evidence/local-5090-native-next-20260721/`. The concise receipts are `native-v2-validation.md`
and `q2k-avxvnni-pair-win.md`. The Hy3
MTP head remains full-vocabulary in this receipt: no `BW24_FRSPEC_TRIM` artifact was supplied.

The earlier 2026-07-19 release receipt used a retired external companion and does not certify native
ABI v2. Its raw logs remain historical evidence under `evidence/local-5090-sota-20260719/` but its
throughput is intentionally excluded from current native claims.

Receipt anchors:

- source plan: `9606b1b96890b270534237b1143a5f5f25165245d1b5f08f515c25268d1b056c`
- artifact manifest: `08f206aed555752982585a59a7b5096b9cc6e71faf1f84ad5c6dd60476b7509a`
- screen summary: `661e495c02467acf5f28180eb73d562d25fea8724449e69f305831beb435be3d`
- winner receipt: `4c6d4089d5dec428a5575b8c9971521f1f04ed5041e48e02bda67d3dae993ab3`

# Hy3 spill profile on a 24 GB GPU

Hy3's expert bank exceeds both VRAM and ordinary host-RAM budgets. bw24 freezes a profiled HBM
resident set, keeps a bounded LRU projection cache in normal RAM, reads misses with positioned
direct I/O, and can split each large read across byte-identical copies on two NVMe devices. The
optional CPU-expert companion is also bw24 code: it implements Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K, IQ3_S, IQ4_XS, NVFP4, Q4_0, BF16, and F32 row dots with a bw24 Q8/16 activation format and
AVX2/AVX-VNNI kernels. It does not compile, link, or load llama.cpp, ggml, or another inference
runtime.

## Running

```bash
tools/build_cpu_expert_companion.sh
BW24_CPU_EXPERT_LIB=target/release/libbw24-cpu-experts.so \
  cargo run -p bw24-engine --bin cpu_native_check
tools/run_hy3_local_5090.sh \
  /path/to/hy3-layer103p5-dual-nvme \
  target/release/libbw24-cpu-experts.so \
  /path/to/expert-mirror/inode-alternates.tsv
```

The companion ABI is versioned and fails closed: the engine requires native ABI v2, so a stale
legacy v1 library cannot be loaded accidentally. `cpu_native_check` compares every supported
packed row dot against bw24's independent Rust dequantization oracle. `dlopen` executes library
constructors before the ABI check, so `BW24_CPU_EXPERT_LIB` must always point to a trusted build.

## Dual-NVMe mirror view

The mirror argument is optional. `tools/build_dual_nvme_expert_view.py` and
`tools/build_expert_mirror_map.py` create the verified striped view and alternate-path map. The run
must use that exact dual-NVMe view as `MODEL_DIR` when the map is enabled: ABI v2 pins both sides by
device, inode, size, and ctime and rejects a map paired with the persistent source tree.

## Run profile and tuning state

The run profile requests a 20 GiB CPU cache, retains 4 GiB of live `MemAvailable` headroom, uses
eight P-cores, profiles residency with 128 discarded tokens, and prints the effective cache cap
before warmup; it reduces the cache instead of exhausting RAM when the desktop stack is too large.
In the controlled native v2 Q2_K sweep, the two-pass means for 8 and 12 threads differed by 0.7%,
while the winner reversed by about 8% between the individual passes; eight remains the
lower-contention default while broader mixed-format end-to-end tuning continues. Each point used
10 warmups and 100 timed calls on the active-desktop powersave regime (55 C start); raw log:
`research/per-expert-quant/evidence/local-5090-native-20260721/cpu-native-v2k-q2k-thread-sweep.log`.

Earlier Hy3 throughput measurements used the retired external CPU backend and are not performance
claims for this implementation. Native ABI v2 results are published only with their dependency,
packed-row oracle, exactness, and raw-run evidence. On the final 2026-07-21 target-rig gate, the
default 128-token residency warmup measured 4.48 tok/s over one N=32 post-freeze `run-gen` window;
the MTP-capable default `run-spec` plain control measured 3.76 tok/s over N=7 before its K sweep.
These are single observations, not board-moving medians. `kernel-check` was all green, the
post-freeze serving assignment passed prefill/decode argmax, and K=1 through K=8 were
self-consistent. Raw logs and the exact thermal/memory regime are under
`research/per-expert-quant/evidence/local-5090-native-20260721/`. Sustained 10 tok/s remains the
target.

## Obtaining the published overlay

The published Hy3 Layer103.5 expert overlay, its receipts, and the relocation tool are documented
in [`research/per-expert-quant/hy3-layer103p5-release.md`](../research/per-expert-quant/hy3-layer103p5-release.md).

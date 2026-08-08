# Hy3 spill profile on a 24 GB GPU

Hy3's expert bank exceeds both VRAM and ordinary host-RAM budgets. memra freezes a profiled HBM
resident set, keeps a bounded LRU projection cache in normal RAM, reads misses with positioned
direct I/O, and can split each large read across byte-identical copies on two NVMe devices. The
optional CPU-expert companion is also memra code: it implements Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K, IQ3_S, IQ4_XS, NVFP4, Q4_0, BF16, and F32 row dots with a memra Q8/16 activation format and
AVX2/AVX-VNNI kernels. It does not compile, link, or load llama.cpp, ggml, or another inference
runtime.

## Running

```bash
tools/build_cpu_expert_companion.sh
MEMRA_CPU_EXPERT_LIB=target/release/libmemra-cpu-experts.so \
  cargo run -p memra-engine --bin cpu_native_check
tools/run_hy3_local_5090.sh \
  /path/to/hy3-layer103p5-dual-nvme \
  target/release/libmemra-cpu-experts.so \
  /path/to/expert-mirror/inode-alternates.tsv
```

The companion ABI is versioned and fails closed: the engine requires native ABI v2, so a stale
legacy v1 library cannot be loaded accidentally. `cpu_native_check` compares every supported
packed row dot against memra's independent Rust dequantization oracle. `dlopen` executes library
constructors before the ABI check, so `MEMRA_CPU_EXPERT_LIB` must always point to a trusted build.

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
packed-row oracle, exactness, and raw-run evidence.

Current state (v0.42.0, measured 2026-07-26): the served Layer103.5 candidate decodes at a
5.13 tok/s m=1 median (N=3: 5.13/5.13/5.16, NGEN=32 post-freeze lockstep windows). Day-to-day
regime drift is real — the same artifact and methodology measured 4.29 the previous day — so
cross-arm comparisons are only made same-day. Step budget at last decomposition: io ~39%,
CPU compute ~47%, GPU ~14%. Mixed multi-request concurrency gains +13% from pinned per-executor
core groups (`MEMRA_CPU_EXPERT_EXECUTOR_CPUSETS="0-7;8-15"`, see `docs/FLAGS.md`). Raw triads:
`research/per-expert-quant/evidence/local-5090-plain-arm-20260725/tp-*.log`.

Closed lanes, each with receipts: speculative/MTP decode (verify positions route disjoint
experts — 0.97x at 100% acceptance), prefetch prediction (trained-predictor ceiling: 0.2-2.6%
non-resident precision cross-layer, 21% best-case cross-token persistence;
`research/moe/route_predictability_ceiling.py`), in-call io/compute overlap (concurrent-DMA
interference, three falsifications), and the artifact axis at the served byte budget (a full
15,168-expert bank at the pure Q2_K floor is still ~110% of the served bank bytes). The
plain-arm methodology study (`research/per-expert-quant/local-5090-10toks-plan.md`) showed
importance-fused tier redistribution holds quality at matched bytes (38/56 vs 36/56 paired
screens at 95.7% bytes) — a method receipt, not a serving change. Sustained 10 tok/s remains
the target; remaining lanes are system work on the served artifact.

The sm_90a check (single H100 80GB, 2026-08-01) reproduces the speculative-decode class at
the spill floor with a full K-sweep: run-spec K=1..8 self-consistency PASS x3 (the MTP head
loads, forwards, and drafts exactly on Hopper), but spec is a net slowdown at every K — best
K=1 at 0.84x, and the accepted count is exactly 10 at every K (the nextn=1 head cannot chain
a second draft token). The first-light 23.8%/1.16x point decomposed into a real chat-content
acceptance rate on a cold-cache denominator: warm, the same config's plain oracle runs 3.73
tok/s and spec loses — the 1.16x is refuted, and the 1-GPU floor row is plain greedy decode
(2.49 tok/s median, N=3), spec OFF. A resident multi-GPU regime changes the staging math and
must run its own K=1 sweep before quoting any spec number
(`research/hy3-hopper-20260801/`, `research/hy3-spec-20260802/`).

The K=1 acceptance profile across six realistic serving classes (same box, same build;
`research/hy3-accept-profile-20260802/`) prices the acceptance rate r that a resident
two-GPU (PP-2) spec decision needs: 44-75% on real content (code-gen 75.3%,
code-review/agentic 64.9%, chat/summarize 44-46%) versus the synthetic d1736 story's
8.5% — never calibrate spec decisions on the synthetic. At the spill floor spec is
roughly break-even at medium/long context (agentic 1.21x, summarize 1.07x — upper
bounds, cache-prewarmed; the K-sweep's spec-OFF floor default stands), and with the bank
resident every realistic class clears the honest estimate S_est = 1 + r/2 >= 1.2x, so
the PP-2 spike wires spec K=1 in and measures the verify-batch overhead phi that the
floor regime cannot price. K stays 1: the nextn=1 head accepts zero chained drafts.

**Status of that spike (2026-08-08):** the PP-2 spec path is now correct and crash-gated.
The verify forward takes its own stage split (`MEMRA_SPEC_PP`, default ON),
`decode-batch-gate --mode ppspec` is 7/7 green, and the #87 reverse-publication fix closed
the old fatal placement. The generic PP-2 serving policy nevertheless defaults spec
admission off because q9 and step35 both lose every measured c=1/2/4 throughput cell
(`research/specplace-20260808/`). That result does **not** price Hy3's different K=1
resident-bank economics: its PP-2 spike must explicitly force spec with
`MEMRA_SPEC_GATE=0`, measure phi against a same-window plain arm, and keep the acceptance
profile frozen. Until then S_est remains an estimate on this shape, not a measured PP-2
result. The single-GPU acceptance profile itself is unaffected.

## Obtaining the published overlay

The published Hy3 Layer103.5 expert overlay, its receipts, and the relocation tool are documented
in [`research/per-expert-quant/hy3-layer103p5-release.md`](../research/per-expert-quant/hy3-layer103p5-release.md).

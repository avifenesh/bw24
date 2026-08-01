# memra — from-scratch LLM inference for RTX 5090 (sm_120a) and H100 (sm_90a)

[![ci](https://github.com/avifenesh/memra/actions/workflows/ci.yml/badge.svg)](https://github.com/avifenesh/memra/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust](https://img.shields.io/badge/rust-edition%202024-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8%20%2F%2013.1-76B900.svg)
![arch](https://img.shields.io/badge/arch-sm__120a%20%2B%20sm__90a-black.svg)

![memra vs llama.cpp perf board](docs/perf-card.svg)

From-scratch LLM inference engine in Rust + CUDA — no frameworks, no ggml. One codebase
serves two architectures, auto-detected at build time: **RTX 50-series Blackwell
(sm_120a)**, tuned single-user against llama.cpp, and **H100 Hopper (sm_90a)**, measured
model-by-model against vLLM on the same box. Every kernel is hand-written against
measured hardware limits, and exactness is the contract: speculative and graph-replay
output is gated token-identical to plain decode, and greedy serving is
isolated-identical under concurrent load — speed never changes what the model says,
and batching never changes what a request gets back.

**Use memra when** you serve one model on an RTX 50-series card and want measured,
exactness-gated speed, or you want a single-GPU H100 engine whose wins *and* losses
against vLLM are published per model. **Use something else when** you have another GPU
([llama.cpp](https://github.com/ggml-org/llama.cpp),
[mistral.rs](https://github.com/EricLBuehler/mistral.rs)) or need multi-GPU serving
(vLLM, SGLang).

**Standing (2026-08-02):** eight supported models on the 5090, all fully gated; MTP-spec
cells run 1.06–2.3x llama.cpp (one Gemma near-parity cell, 0.98x, still open), plain
cells sit at the DRAM wall or above. This wave's headline is a real serving defect the
exactness discipline caught and fixed: the batched F32 prefill router GEMM was
m-dependent, so under cross-request prefill batching a MoE request's own expert routing
could change with its co-arrivals. The fix routes prefill through the decode path's
m-invariant router/gate kernels (default on), and a bit-identical batched twin then
recovered 70% of the prefill it cost — the serving contract above is now explicit and
gated, not assumed (receipts: [`research/concat-prime-exact-20260802/`](research/concat-prime-exact-20260802/),
[`research/fast-router-20260802/`](research/fast-router-20260802/)). On the H100, a
full per-model board against vLLM 0.26: **no end-to-end losses** — six wins and a
dead-even 35B cell (1.00-1.81x); decode wins 7 of 7. The last losses fell
in two days: the 35B MoE's expert prefill jumped +53% when the grouped f16 expert lane
got full dequant coverage, the 27B gained a +54% prefill and +16% decode from K-quant
f16 mirrors and split-plane layout v2, and the 26B flipped to a win when a cross-box
arbitration narrowed that f16g default per dispatch class. Multi-user serving measured
on 3xH100: 1,477 tok/s managed fleet (chaos-tested: kill a replica mid-load, the
breaker + supervisor recover in seconds with only in-flight requests lost). Every number is a same-session
interleaved measurement on a real-text prompt with the argmax exactness gate green;
trimmed MTP drafter heads are published ready-to-use at
[huggingface.co/Avifenesh/memra-bench](https://huggingface.co/Avifenesh/memra-bench).

Running memra on your own rig? A [hardware validation
report](.github/ISSUE_TEMPLATE/hardware-validation.md) is the fastest way to help.

## One engine, two architectures

```bash
cargo build --release   # MEMRA_CUDA_ARCH auto-detected from the GPU (120a / 90a / 100a / 89)
```

The build probes the GPU's compute capability and selects the arch; `MEMRA_CUDA_ARCH`
overrides. At startup the engine verifies the binary matches the device and fails early
with a rebuild hint otherwise (`MEMRA_ARCH_CHECK=0` bypasses). Hopper-only promotions
(wgmma/TMA kernels, graph serving defaults) are compile-gated — the naked sm_120a build
is byte-for-byte the tuned 5090 engine.

## The H100 build (sm_90a)

The full per-model board against vLLM 0.26 on 1×H100 80GB, same box, every cell a
same-session interleaved pair (final cells 2026-08-01). One number per arm:
**end-to-end tok/s** — 512 tokens generated on a ~2100-token real-text prompt, single
request, total wall time (N=5 medians, argmax exactness gate green on every published
row). Cross-artifact by design: vLLM serves what H100 users deploy (w8a8 / FP8-dynamic
/ bf16 HF checkpoints — it rejects these GGUFs); memra serves its GGUF artifacts.

| model | memra e2e | vLLM 0.26 e2e (artifact) | ratio |
|---|---:|---:|---:|
| Gemma-4 12B | **146** | 81 (bf16) | **1.81x** |
| Qwen3.6-27B | **96** | 73 (FP8) | **1.31x** |
| Gemma-4 31B | **75** | 64 (FP8-dyn) | **1.18x** |
| Qwen3.5-9B | **204** | 176 (w8a8) | **1.16x** |
| Gemma-4 E4B | **193** | 168 (bf16) | **1.14x** |
| Gemma-4 26B MoE | **196** | 191 (FP8-dyn) | **1.02x** |
| Qwen3.6-35B MoE | **215** | 214 (FP8) | **1.00x** |

Zero losses: six wins and a dead-even 35B cell, on exact math (the bf16-row wins
carry a quant-advantage caveat — those vLLM arms move 4x the weight bytes). Decode
wins every cell (1.05–1.85x; the last two decode losses fell to the shexp fused dot
and a router-default re-arbitration on real prompts). The 35B cell slipped from
218/1.02x when the concat-prime exactness fix landed (m-invariant prefill router +
shexp gates, `MEMRA_ROUTER_PREFILL_EXACT` default ON): expert prefill pays −13%
on Hopper (8428 → 7311 pp2048) so a session's routing no longer depends on its
co-arrivals — decode is untouched, and the dense 27B row is unaffected (no MoE
router; re-cell bit-stable). Receipts `research/router-fix-recells-20260802/`. A
bit-identical batched twin of the exact router kernel has since recovered 70% of that
prefill cost on the 5090 (`research/fast-router-20260802/`; `MEMRA_ROUTER_BATCH=0` is
its perf-only rollback seam) — the H100 row stands at its post-fix re-cell until the
on-box re-cell with the twin lands. The last two e2e losses — both MoE
expert-prefill cells — closed in round 49: the 35B when the grouped f16 expert lane
reached full dequant coverage and became the Hopper default (expert prefill +53%), the
27B via K-quant f16 prefill mirrors (+54% pp2048) plus split-plane decode mirrors
(+16% decode, bit-identical). The 26B cell flipped from dead-even to a win in round 50,
when a cross-box interleaved arbitration caught that same f16g default *regressing* the
gemma expert class −6 to −8% and narrowed it per-model — the board is tuned per
dispatch class, not per flag. Per-cell prefill still trails vLLM on the dense models —
the int8-GEMM dtype edge: a Q8_0-exact int8 GEMM is mechanism-refuted on Hopper
(per-32-block rescale costs 5.4x naive / 17x pipelined; ptxas serializes cross-bank
GMMA register reads), so crossing it means w8a8-class numerics that change model
outputs — an accuracy-bar decision with measured receipts, not an engineering unknown.

Shipped on this build: FA3-class prefill attention (TMA swizzled ring + wgmma, 4.8x the
mma kernel), fused wgmma GDN chunk kernels with varlen twins, f16 prefill mirrors on
the cuBLASLt lane for the Q8_0/Q4_0/Q4_K/Q5_K/Q6_K classes, K-quant split-plane decode
mirrors (bit-identical layout v2, 27B decode +15-16%), the grouped f16 expert lane (one
grouped GEMM over the routed experts — Hopper default for the qwen/silu MoE class, 35B
expert prefill +53%; per-model off for the gemma class after the round-50 regression
arbitration), a
batched serving decode tick (z-batched attention + KV append, device-side sampling,
lean logits: 654-659 tok/s/replica, +25-36%), an MTP speculative serving fast lane
(1.82x plain serving at c=1 on the 27B), cross-request prefill batching, per-session
CUDA-graph decode with kernel-class segment recapture, and the Hopper wgmma toolkit
([`cu/wgmma_common.cuh`](crates/memra-engine/cu/wgmma_common.cuh)) — canonical
core-matrix pairings probed for bf16/tf32/s8: one byte-geometry, three MMA kinds.

- Evidence ledger (every verdict and refutation): [ARCHITECTURE-H100.md](ARCHITECTURE-H100.md)
- Flags + promoted defaults: [docs/FLAGS.md §7](docs/FLAGS.md)
- One-command battery: `tools/validate-h100.sh <model.gguf> [--quick]`

## Model support

| Tier | Models | State |
|---|---|---|
| **Supported** | Qwen3.5-9B, Qwen3.6-27B, Qwen3.6-35B-A3B MoE (NVFP4/IQ4_XS on the 5090; Q8_0 / Q4_K_M MTP-baked / IQ4_XS on the H100 board); Gemma-4 12B, 26B-A4B MoE, 31B, E4B (QAT Q4_0 + MTP drafters); Ornith-1.0-9B (Q8_0 + donor-block own-gen drafter — the first community post-train over the deployment bar: best-vs-best e2e 2.21/1.67/1.47x, [receipts](research/ornith-bar-20260802/)) | Board-published, fully gated, exactness-first |
| **Supported, under tuning** | Hy3 Layer103.5 overlay (VRAM→RAM→dual-NVMe spill) | Correctness-gated end-to-end; [docs/HY3-SPILL.md](docs/HY3-SPILL.md) |
| **In bring-up** | Ornith-1.0-35B, KAT-Coder-V2.5 (with Ornith-9B, the onboarding wave's top-downloaded HF GGUF repos); Qwen-AgentWorld-35B-A3B verified same-stack | Onboarded with zero code change (native `qwen35`/`qwen35moe` arch strings; argmax + chat gates green — [receipts](research/onboard-ornith-20260801/)). 35B: HOLD — the resident-if-fits default inverted its decode leg vs llama.cpp (1.086x, [receipts](research/residency-cap-20260802/)) and its own-gen drafter is adopted (1.38/1.09/1.05x), but the Q4_K expert-prefill gap (0.27x) still fails the e2e bar ([priced](research/ornith-bar-20260802/)). KAT: HOLD — best drafter acceptance of the batch, but a plain-decode anomaly (~104 vs ~170 tok/s on the same arch class) makes drafting net-negative; the anomaly lane is queued. AgentWorld: same-stack by construction (header bytes verified), unbenched |
| **In progress** | MiniMax-M3 REAP50 (safetensors spill) | Loads + generates; router tuning open |

## Quick start

Prebuilt Linux x86_64 binaries (sm_120a) ship with each
[release](https://github.com/avifenesh/memra/releases) — or build from source:

```bash
cargo build --release
./target/release/kernel-check                     # every kernel vs CPU reference
MEMRA_CHAT=1 ./target/release/run-gen /path/to/model.gguf --prompt "Explain KV caches."
MEMRA_SPEC_K=3 ./target/release/run-spec /path/to/qwen36-27b.gguf   # MTP speculative
./target/release/run-gen hf:owner/repo:Q4_K_M --prompt "hi"        # auto-download from HF
./target/release/memra-server                      # OpenAI-compatible /v1
```

`kernel-check` must end with `ALL GREEN`, and `run-gen` prints its argmax gate
(`... MATCH`) before generating — a MISMATCH voids every number after it. Tuned paths
are the defaults; flags exist only for runtime parameters, machine config, and rollback
seams ([docs/FLAGS.md](docs/FLAGS.md)).

## Performance — Qwen (NVFP4 / IQ4_XS), RTX 5090

<!-- PERF-DATE:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
Measured 2026-07-27 on the target rig (RTX 5090 Laptop, N=2+ medians, both engines interleaved in the same thermal window on the same rig, same exact prompts, no flags (tuned paths are defaults); plain/depth rows from the 2026-07-09 validity-gated cold-start rebaseline, spec rows re-paired 2026-07-18, Gemma card rows from the 2026-07-15 best-vs-best re-audit. Full per-run logs: research/tune-data/ (Qwen) and research/gemma4-bringup/ (Gemma) — every win and every loss; Gemma 12B plain row from the 2026-07-24 official N=5 cell stamp (research/gemma4-bringup/g12tg-cellstamp.log)) against llama.cpp built on the same machine, same exact prompts, both engines re-baselined the same day. Boards move with the tuning campaign — `research/tune-data/rig5090.jsonl` is the running record; the README is refreshed with every board-moving merge.
<!-- PERF-DATE:END -->

**Plain decode** (no speculation, tg128 at 512-token context):

<!-- PERF-PLAIN:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
| Model | memra plain | llama.cpp plain | Ratio |
|---|---|---|---|
| Qwen3.5-9B NVFP4 (GGUF) | 135.7 | 126.7 | **1.07x** |
| Qwen3.6-27B NVFP4 (GGUF) | 48.4 | 44.9 | **1.08x** |
| Qwen3.6-35B-A3B MoE (IQ4_XS) | 178.2 | 167.8 | **1.06x** |
<!-- PERF-PLAIN:END -->

Depth is part of the contract: at 6.3k-token context every lead holds (1.02–1.07x).

**Speculative decoding** (MTP head, both engines at their measured best):

<!-- PERF-SPEC:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
| Model | memra spec | llama.cpp spec-best | Ratio |
|---|---|---|---|
| Qwen3.5-9B (K=3 + own-gen trimmed draft) | 281.0 / 211.7 / 187.1 | 122.2 / 121.5 / 117.7 | **2.30x** / **1.74x** / **1.59x** |
| Qwen3.6-27B (K=3 + own-gen trimmed draft) | 116.4 / 101.2 / 86.0 | 91.7 / 93.3 / 81.5 | **1.27x** / **1.08x** / **1.06x** |
| Qwen3.6-35B-A3B (K=2 + own-gen trimmed draft) | 280.6 / 259.6 / 258.0 | 236.5 / 174.6 / 173.5 | **1.19x** / **1.49x** / **1.49x** |
<!-- PERF-SPEC:END -->

The three columns are three prompt classes: short code / medium code (greedy) / long
agentic (temp 0.7, distribution-exact rejection sampling). One asterisk: the 35B
short-code llama bar (236.5) rode an EOS-margin flip and is not a clean win basis. Every
spec row uses one trimmed draft built by the standard regime
([docs/DRAFT-REGIME.md](docs/DRAFT-REGIME.md)); prebuilt drafts live in the
[bench repo](https://huggingface.co/Avifenesh/memra-bench), or build your own:

```bash
./target/release/frspec-owngen model.gguf ranks.gguf 32768        # ranks from the model's OWN generations
tools/make-trimmed-draft.sh model.gguf ranks.gguf.txt draft.gguf  # extract + trim + quantize
```

## Performance — Gemma-4 (QAT Q4_0), RTX 5090

Same protocol, own campaign log
([`research/gemma4-bringup/rig5090-gemma4.jsonl`](research/gemma4-bringup/rig5090-gemma4.jsonl));
cells re-paired best-vs-best with llama's own draft depth swept per cell. Highlights
(full tables and the per-cell archaeology live in the campaign log):

| Cell | memra | llama.cpp | Ratio |
|---|---|---|---|
| 12B MTP spec, 1.7k ctx (K=4 + own-gen trim) | 269.3 | 175.1 | **1.54x** |
| 12B MTP spec, chat (K=4 + own-gen trim) | 240.9 | 172.4 | **1.40x** |
| 31B MTP spec, chat (K=5 + own-gen trim) | 124.3 | 99.0 | **1.26x** |
| 31B MTP spec, 1.7k (K=6 + FR trim) | 97.3 | 83.9 | **1.16x** |
| 26B MTP spec, short ctx (K=4 + own-gen trim) | 328.1 | 298.0 | **1.10x** |
| E4B plain, short | 199.9 | 181.0 | **1.10x** |
| 26B plain, 4.9k ctx | 162.6 | 142.0 | 1.14x |
| plain decode elsewhere | — | — | 1.00–1.07x |

What buys the margins: an FP8 (e4m3) KV cache, occupancy-tuned attention tiles, wide-load
Q4_0 expert dots, FR-Spec drafter trims (150 MB → 18 MB at unchanged acceptance), a
per-model adaptive-K floor (+20% on 12B/31B spec at unchanged exactness), and an in-round
draft-confidence cut. One near-parity cell remains (26B 1.7k spec, 0.98x). Exact prompts
and llama.cpp's swept-best flags: [docs/COMPETITOR-SETUP.md](docs/COMPETITOR-SETUP.md).

## Performance — Ornith-1.0-9B (Q8_0), RTX 5090

Supported model #8, and the first community post-train to clear the deployment bar
(best-vs-best e2e ≥ 1.1x on every prompt class). Each engine at its measured best on the
same GGUF: memra runs the regime drafter at K=3 (donor-block own-gen trim — the published
model ships no MTP head, [docs/DRAFT-REGIME.md](docs/DRAFT-REGIME.md)); llama.cpp's best
is plain — its draftless speculative doors are structurally broken on this arch (M-RoPE
position faults, screened in the receipts). Interleaved N=3 medians, same session
([`research/ornith-bar-20260802/`](research/ornith-bar-20260802/)):

| Prompt class | memra e2e, 256 tok (spec K=3) | llama.cpp e2e (plain best) | Ratio |
|---|---:|---:|---:|
| code short | 1.395 s | 3.084 s | **2.21x** |
| code medium (1.8k ctx) | 2.050 s | 3.429 s | **1.67x** |
| agentic long (6.3k ctx) | 2.984 s | 4.376 s | **1.47x** |

Spec-vs-own-plain: 2.16/1.77/1.70x at 47-61% acceptance
([`research/ornith-drafters-20260801/`](research/ornith-drafters-20260801/)); build the
drafter with the standard two commands in [docs/DRAFT-REGIME.md](docs/DRAFT-REGIME.md)
(donor-block variant — donor pairs and receipts documented there).

## Known gaps

- **5090 prefill** trails llama.cpp (0.59–0.78x), root-caused: llama benches NVFP4
  prefill at W4A4 (FP4 activations), a numeric class memra's exactness gates reject.
  Output quality outranks the prefill column.
- **H100 prefill** still trails vLLM per cell (the e2e board is loss-free because
  decode wins 7 of 7 and dominates the p2048/g512 shape). The MoE expert-prefill gaps
  largely closed in round 49 (grouped f16 expert lane, K-quant f16 mirrors); the dense
  cells sit on the int8-GEMM dtype edge, refused by the accuracy laws (full refutation
  ledger in [ARCHITECTURE-H100.md](ARCHITECTURE-H100.md)).
- Gemma plain margins are thin where both engines sit at the DRAM wall (1.02–1.06x).
- Hy3 native spill serves at a 5.13 tok/s N=3 median, tuning toward 10
  ([docs/HY3-SPILL.md](docs/HY3-SPILL.md)).

## What's inside

- **NVFP4 / Q4_0 / Q8_0 decode** — split-plane repacked matvecs, warp-level dp4a, int8
  tensor-core prefill GEMM, per-shape auto-dispatch.
- **MTP speculative decoding** — embedded draft head, one batched K+1 verify, adaptive
  draft depth + confidence cut; K=1..8 self-consistency gate.
- **Hopper wgmma/TMA kernels** — FA3-class prefill attention, fused GDN chunk family,
  canonical descriptor pairings for bf16/tf32/s8.
- **MoE on 24 GB** — resident-if-fits expert residency (exact GGUF bank accounting +
  measured headroom; +50.4% Ornith-35B decode over the old spill default), expert-major
  CSR batching, frozen SLRU spill cache, bounded host LRU, mirrored positioned reads
  across VRAM→RAM→NVMe.
- **Quantized-KV attention** — fused FlashAttention-class kernels (q8_0/q5_1 or FP8-e4m3
  KV per layer class), split-K, graph-replayable device-length counters.
- **CUDA-graph decode** — one graph replay per token, 4 bytes/token host traffic;
  per-session capture for serving.
- **Serving** — OpenAI-compatible server with batched decode (one z-batched tick
  across sequences, chunked 16-wide on models with a bit-exact 16-batch kernel class:
  +18.8% at c=16 on a 5090 single replica, same-mirror interleaved N=4), cross-request
  prefill batching, KV prefix reuse, speculative
  serving, and a flat `/metrics` endpoint. Greedy serving is isolated-identical under
  concurrent load at defaults — a request's tokens never depend on its co-arrivals
  (m-invariant prefill router/gate kernels, serve-gate c=1-vs-c=16 byte identity).
  Multi-GPU boxes serve as a replica fleet:
  supervisor + admission proxy + load harness, measured at 1,477 tok/s managed on
  3xH100, chaos-tested ([docs/SERVING.md](docs/SERVING.md)).
- **Loaders** — GGUF (memory-mapped) and safetensors (modelopt NVFP4 byte-exact).

## Correctness discipline

Every kernel change passes, in order: `kernel-check` (CPU reference), the `run-gen`
argmax gate, `run-spec` K=1..8 self-consistency — one command: `tools/local-ci.sh`.
FP summation order is part of the contract. Exactness gates are blind to numeric shifts
where decode and verify move together, so the perf CI (`tools/local-ci.sh --perf`)
re-measures every published cell per engine-touching push and tracks speculative
acceptance per cell against a rolling baseline, enforced by the pre-push hook. On
Hopper, `tools/validate-h100.sh` is the equivalent one-command battery.

## Workspace layout

| Crate | What it does |
|---|---|
| `memra-engine` | CUDA kernels (`cu/`), forward passes, speculative decoding, MoE cache, graph decode |
| `memra-gguf` | GGUF parser + tensor loading (memory-mapped) |
| `memra-tokenizer` | BPE tokenizer + chat templates from GGUF metadata |
| `memra-runtime` | CUDA device/stream/memory primitives over cudarc |
| `memra-kv` | KV cache + format policy behind the KvDev device seam |
| `memra-sampling` | Host sampler + device Philox sampling behind one trait |
| `memra-validate` | Gate harness: tolerance policy, deterministic vectors, N-median runner |
| `memra-server` | OpenAI-compatible HTTP server (axum): batched decode, prefill batching, KV reuse |
| `memra-probe` | Standalone hardware microbenches |

## Requirements

- NVIDIA GPU: RTX 50-series (sm_120a, primary target RTX 5090 Laptop), H100 (sm_90a),
  B200 (sm_100a, compile-gated), or Ada (sm_89, portable eval).
- CUDA 13.1 (plus 12.8 for the dual-toolkit build in [ARCHITECTURE.md](ARCHITECTURE.md);
  `MEMRA_NVCC` overrides the nvcc path). Rust edition 2024, cudarc 0.19.
- A model: GGUF or HF safetensors directory.

## Limitations

- Tuned for two exact memory/compute ratios (5090 Laptop, H100 SXM). Other GPUs compile
  via the portable arch but are untuned — use
  [llama.cpp](https://github.com/ggml-org/llama.cpp) or
  [mistral.rs](https://github.com/EricLBuehler/mistral.rs) there.
- One GPU per engine process; no tensor parallelism yet. The multi-GPU build is under
  way: the NVLink/NCCL comms floor is measured (M0) and the two-stage pipeline-parallel
  seam (M1, `MEMRA_PP_STAGES`) is merged bit-identical — per-stage streams/events and
  real peer-to-peer transport — default off, pending the cross-device gate on an
  8-GPU box. Multi-GPU boxes serve today as a replica fleet
  ([docs/SERVING.md](docs/SERVING.md)); batched decode serving on both architectures.
- Moving research codebase; APIs and flags change without notice.

## Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — sm_120a tech stack + feasibility ledger.
- [`ARCHITECTURE-H100.md`](ARCHITECTURE-H100.md) — the H100 build's full evidence ledger.
- [`HANDOVER.md`](HANDOVER.md) — living state-of-work.
- [`docs/FLAGS.md`](docs/FLAGS.md) — the audited flag catalog.
- [`docs/COMPETITOR-SETUP.md`](docs/COMPETITOR-SETUP.md) — competitor engines at their peak.
- [`docs/DRAFT-REGIME.md`](docs/DRAFT-REGIME.md) — the standard drafter pipeline (own-gen ranks, byte-verbatim extraction, trim + quantize).
- [`docs/SERVING.md`](docs/SERVING.md) — multi-user replica-fleet serving runbook.
- [`docs/HY3-SPILL.md`](docs/HY3-SPILL.md) — Hy3 spill runbook.
- [`research/`](research/) — every experiment as JSONL, wins and losses both;
  [`research/benchmarks.md`](research/benchmarks.md) is the measurement protocol.

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Correctness gates run
on real GPUs (`tools/local-ci.sh`); GitHub CI is compile-only.

## License

MIT — see [LICENSE](LICENSE).

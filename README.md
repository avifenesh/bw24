# bw24 — from-scratch LLM inference for sm_120a (RTX 50-series Blackwell)

[![ci](https://github.com/avifenesh/bw24/actions/workflows/ci.yml/badge.svg)](https://github.com/avifenesh/bw24/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust](https://img.shields.io/badge/rust-edition%202024-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8%20%2F%2013.1-76B900.svg)
![arch](https://img.shields.io/badge/arch-sm__120a%20(Blackwell)-black.svg)

![bw24 vs llama.cpp perf board](docs/perf-card.svg)

From-scratch LLM inference engine in Rust + CUDA, built for one machine: an RTX 5090 Laptop (Blackwell sm_120a, 24 GB). No frameworks, no ggml — every kernel written and tuned against measured hardware limits, with llama.cpp as the benchmark to beat on the same rig.

The headline capability is **MTP speculative decoding**: up to 2.3x over llama.cpp's best spec config, leading on every supported Qwen model and prompt class (1.06-2.30x per cell), with trimmed drafter heads published ready-to-use ([huggingface.co/Avifenesh/bw24-bench](https://huggingface.co/Avifenesh/bw24-bench)) — behind a drop-in OpenAI-compatible server. Exactness is the contract: speculative output is gated token-identical to plain decode, so the speedup never changes what the model says.

**Use bw24 when** you serve one model to one user on an RTX 50-series card and want measured, exactness-gated speed. **Use something else when** you have any other GPU ([llama.cpp](https://github.com/ggml-org/llama.cpp), [mistral.rs](https://github.com/EricLBuehler/mistral.rs)) or need multi-GPU / batched serving (vLLM, SGLang).

Running bw24 on your own rig — desktop 50-series, older NVIDIA, anything? A [hardware validation report](.github/ISSUE_TEMPLATE/hardware-validation.md) is the fastest way to help: 50-series reports bless the rest of the family, older-card reports map the compatibility floor.

**Current standing: seven supported models, all fully gated, and no plain cell anywhere below llama.cpp (2026-07-24). Qwen leads on every cell (plain 1.06-1.08x, spec 1.06-2.30x). Gemma leads decisively where llama lacks the capability or the depth (31B spec 1.7k 1.16x, E4B spec ≥1.23x, E4B plain 1.10x) and holds 1.00-1.07x elsewhere under the strictest best-vs-best pairing (12B bring-up closed 2026-07-24; 26B/31B/E4B re-audit 2026-07-15).** Every number below is a same-session, same-prompt, interleaved measurement against llama.cpp's best config; exactness is gated (argmax match + speculative self-consistency) on every kernel change, so speed never buys different outputs.

## Model support

| Tier | Models | State |
|---|---|---|
| **Supported** | Qwen3.5-9B, Qwen3.6-27B, Qwen3.6-35B-A3B MoE (NVFP4/IQ4_XS); Gemma-4 12B dense, 26B-A4B MoE, 31B dense, E4B (QAT Q4_0 + MTP drafters) | Board-published, fully gated, exactness-first; margins per model in the tables below |
| **Supported, under tuning** | Hy3 Layer103.5 overlay (VRAM→RAM→dual-NVMe spill) | Runs end-to-end through bw24-native CPU/GPU serving, correctness-gated on the 5090 target; see [docs/HY3-SPILL.md](docs/HY3-SPILL.md) |
| **In progress** | MiniMax-M3 REAP50 (safetensors, VRAM→RAM→NVMe spill) | Loads + generates; hybrid/sigmoid-router tuning remains open |

## Quick start

Prebuilt Linux x86_64 binaries (sm_120a) ship with each [release](https://github.com/avifenesh/bw24/releases) — or build from source:

```bash
cargo build --release
./target/release/kernel-check                     # every kernel vs CPU reference
BW24_CHAT=1 ./target/release/run-gen /path/to/model.gguf --prompt "Explain KV caches."
BW24_SPEC_K=3 ./target/release/run-spec /path/to/qwen36-27b.gguf   # MTP speculative
./target/release/run-gen hf:owner/repo:Q4_K_M --prompt "hi"        # auto-download from HF (needs `hf` CLI)
./target/release/frspec-owngen model.gguf trim.gguf --validate     # build + validate an FR-Spec draft trim from the model's OWN generations
./target/release/bw24-server                      # OpenAI-compatible /v1
```

Expected output — `kernel-check` ends with:

```
ALL GREEN: kernels match CPU reference.
```

and `run-gen` prints its correctness gate before any generation:

```
verify-prefill argmax=N  decode argmax=N  logit maxdiff=...  MATCH
```

Tuned paths are the defaults — no flags needed. Flags exist only for runtime parameters, machine config, and rollback seams (`docs/FLAGS.md`). A MISMATCH line from the gate voids every number after it.

Serving Hy3 (a ~100 GB expert bank) on this 24 GB card uses a frozen HBM resident set, a bounded host cache, and positioned dual-NVMe reads — runbook, ABI safety notes, and current gate results in [docs/HY3-SPILL.md](docs/HY3-SPILL.md).
The paired native AVX-VNNI Q2_K path raises the local N=32 median from 4.37 to 4.60 tok/s across three interleaved, correctness-identical pairs (+5.3% by arm medians; [receipt](research/per-expert-quant/evidence/local-5090-native-next-20260721/q2k-avxvnni-pair-win.md)).

## Performance — Qwen (NVFP4 / IQ4_XS)

<!-- PERF-DATE:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
Measured 2026-07-24 on the target rig (RTX 5090 Laptop, N=2+ medians, both engines interleaved in the same thermal window on the same rig, same exact prompts, no flags (tuned paths are defaults); plain/depth rows from the 2026-07-09 validity-gated cold-start rebaseline, spec rows re-paired 2026-07-18, Gemma card rows from the 2026-07-15 best-vs-best re-audit. Full per-run logs: research/tune-data/ (Qwen) and research/gemma4-bringup/ (Gemma) — every win and every loss; Gemma 12B plain row from the 2026-07-24 official N=5 cell stamp (research/gemma4-bringup/g12tg-cellstamp.log)) against llama.cpp built on the same machine, same exact prompts, both engines re-baselined the same day. Boards move with the tuning campaign — `research/tune-data/rig5090.jsonl` is the running record; the README is refreshed with every board-moving merge.
<!-- PERF-DATE:END -->

**Plain decode** (no speculation, tg128 at 512-token context):

<!-- PERF-PLAIN:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
| Model | bw24 plain | llama.cpp plain | Ratio |
|---|---|---|---|
| Qwen3.5-9B NVFP4 (GGUF) | 135.7 | 126.7 | **1.07x** |
| Qwen3.6-27B NVFP4 (GGUF) | 48.4 | 44.9 | **1.08x** |
| Qwen3.6-35B-A3B MoE (IQ4_XS) | 178.2 | 167.8 | **1.06x** |
<!-- PERF-PLAIN:END -->

Depth is part of the contract: at 6.3k-token context every lead holds (1.02-1.07x).

**Speculative decoding** (MTP head, both engines at their measured best):

<!-- PERF-SPEC:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
| Model | bw24 spec | llama.cpp spec-best | Ratio |
|---|---|---|---|
| Qwen3.5-9B (K=3 + own-gen trimmed draft) | 281.0 / 211.7 / 187.1 | 122.2 / 121.5 / 117.7 | **2.30x** / **1.74x** / **1.59x** |
| Qwen3.6-27B (K=3 + own-gen trimmed draft) | 116.4 / 101.2 / 86.0 | 91.7 / 93.3 / 81.5 | **1.27x** / **1.08x** / **1.06x** |
| Qwen3.6-35B-A3B (K=2 + own-gen trimmed draft) | 280.6 / 259.6 / 258.0 | 236.5 / 174.6 / 173.5 | **1.19x** / **1.49x** / **1.49x** |
<!-- PERF-SPEC:END -->

The three columns are three prompt classes: short code / medium code (both greedy) / long agentic, sampled at temp 0.7 with distribution-exact rejection sampling. One asterisk the log carries: the 35B short-code llama bar (236.5) rode an EOS-margin flip and is not a clean win basis — the other two 35B cells are. Every spec row uses **one trimmed draft file built by the standard regime** — the model's own-generation FR-Spec ranks, byte-verbatim MTP extraction, NVFP4 head + Q4_K_M block ([`docs/DRAFT-REGIME.md`](docs/DRAFT-REGIME.md)).

**Drafts: use ours or build your own.** Prebuilt per-model drafts (exact pipeline, exact published bytes) live at [huggingface.co/Avifenesh/bw24-bench](https://huggingface.co/Avifenesh/bw24-bench) under `drafts/<model>/`. For any other model, requant, or finetune, build one in two commands (a finetune's distribution moved, so its draft must too):

```bash
./target/release/frspec-owngen model.gguf ranks.gguf 32768        # ranks from the model's OWN generations
tools/make-trimmed-draft.sh model.gguf ranks.gguf.txt draft.gguf  # extract + trim + quantize
```

Exact prompts and configs also in the bench repo; llama.cpp flags in [docs/COMPETITOR-SETUP.md](docs/COMPETITOR-SETUP.md).

## Performance — Gemma-4 (QAT Q4_0)

Same protocol, own campaign log (`research/gemma4-bringup/rig5090-gemma4.jsonl`); all cells re-paired 2026-07-15 under best-vs-best (llama server MTP at its swept-best flags, exact-token-id prompts, serialized same-window arms). Contexts are real prompt depths.

**12B dense** (bring-up completed 2026-07-24 — every plain cell at or above llama; decode closed by a seven-lever launch-class stack, +2.4% in one day, each step argmax- and stream-gated):

| Cell | bw24 | llama.cpp | Ratio |
|---|---|---|---|
| plain decode, 512 ctx | 92.6 | 92.4 | 1.00x |
| prefill 512 | 4204 | 4161 | 1.01x |
| prefill 1736 | 4123 | 3863 | 1.07x |

**26B-A4B MoE:**

| Cell | bw24 | llama.cpp | Ratio |
|---|---|---|---|
| plain, short ctx | 199.7 | 187.6 | 1.06x |
| plain, 1.7k ctx | 183.1 | 173.1 | 1.06x |
| plain, 4.9k ctx | 162.6 | 142.0 | 1.14x (stale fa-off bar — re-pair pending) |
| MTP spec, short ctx (K=6) | 267 | 271 | 0.99x |
| MTP spec, 1.7k ctx (K=6 + FR trim) | 298 | 286 | 1.04x |

**31B dense / E4B:**

| Cell | bw24 | llama.cpp | Ratio |
|---|---|---|---|
| 31B plain, short | 40.8 | 40.2 | 1.02x |
| 31B plain, 1.7k | 38.4 | 37.4 | 1.03x |
| 31B MTP spec, short (K=3) | 98.0 | 92.3 | 1.06x |
| 31B MTP spec, 1.7k (K=6 + FR trim) | 97.3 | 83.9 | **1.16x** |
| E4B MTP spec (K=6 assistant) | 248 | no llama MTP for E4B — vs its plain 181.0 | **≥1.23x** |
| E4B plain, short | 199.9 | 181.0 | **1.10x** |

What buys the margins: an FP8 (e4m3) KV cache (half the bytes of llama's f16 KV at near-zero dequant cost), occupancy-tuned attention tiles, wide-load Q4_0 expert dots, and FR-Spec drafter-head trims (150 MB → 18 MB at unchanged 0.91-0.94 acceptance on the 26B). One structural finding worth knowing: FP8 noise in the *windowed* KV layers guts the MTP drafter's acceptance, so spec serving automatically keeps those layers at q8_0/q5_1 while plain serving keeps FP8 — a config discovery, not a kernel. The same FP8-KV lever is available for Qwen behind `BW24_KV_FP8` (correctness-proven, ~45% smaller KV). Since 2026-07-19 an adaptive serve-time trim lifts the 31B spec cells a further ~2-5% solo (~105 short / ~104 at 1.7k); those runs are not yet llama-re-paired, so the table keeps the paired ratios.

Earlier published Gemma spec margins (1.37-1.54x, and the 31B 167.5/112.1 pair) do not reproduce — even through era binaries — and are retired; the campaign log carries the full archaeology. llama's spec side is `--spec-type draft-mtp`, warm, at its measured best on the same box; E4B has no llama MTP arm at all (the 2026-06-30 freeze binary can't serve its drafter — fixed upstream later), so its spec row is floored against llama's plain. Correctness is structural, not statistical: decode, verify, and graph replay launch the same kernel symbols, so the verify gate reads a bit-exact 0.000e0 logit maxdiff at every depth.

**Reproducing:** exact-token-id prompts and llama.cpp's swept-best flags in [docs/COMPETITOR-SETUP.md](docs/COMPETITOR-SETUP.md); every row's raw run — and every retired number's archaeology — in [`research/gemma4-bringup/rig5090-gemma4.jsonl`](research/gemma4-bringup/rig5090-gemma4.jsonl).

## Known gaps

- **Prefill** trails llama.cpp (0.59-0.78x), root-caused: llama benches NVFP4 prefill at W4A4 (FP4 activations), a numeric class bw24's exactness gates reject — bw24's in-tree W4A4 arm beats llama but forks argmax on long prompts (`docs/FLAGS.md` §5). Output quality outranks the prefill column.
- Gemma plain margins are thin where both engines sit at the DRAM wall (31B 1.02-1.03x, 26B 1.06x; best kernel = 91% of measured wall, e2e 87-89%). Every mechanism class measured — ours plus llama/vLLM/SGLang current releases — is shipped or carries a falsification row in the campaign log. Open spec cells: 31B short 1.06x, 26B 0.99x/1.04x.
- Hy3 native spill is correctness-gated at a 4.60 tok/s N=3 median after the paired native AVX-VNNI Q2_K win and is being tuned toward a sustained 10 tok/s ([docs/HY3-SPILL.md](docs/HY3-SPILL.md)).
- Safetensors runs checkpoints llama.cpp cannot (NVIDIA NVFP4 ST, 121 GB spilled MoEs) but GGUF is the primary delivery format — ST showed seed-sensitive long-context repetition (`research/tune-data/27b-st-vs-gguf-final.md`). The published Hy3 Layer103.5 expert overlay is the scoped exception.

## What's inside

- **NVFP4 / Q4_0 decode** — split-plane repacked matvecs, warp-level dp4a, int8 W4A8 tensor-core prefill GEMM, per-shape auto-dispatch.
- **MTP speculative decoding** — embedded draft head, one batched K+1 verify, zero-sync async rounds, adaptive draft depth; K=1..8 self-consistency gate.
- **MoE on 24 GB** — expert-major CSR batching, decode-once dequant, frozen SLRU expert residency, bounded host LRU, and mirrored positioned reads across VRAM→RAM→NVMe.
- **Quantized-KV attention** — fused prefill/decode FlashAttention-class kernels (q8_0/q5_1 or FP8-e4m3 KV per layer class), split-K, device-length counters for graph replay.
- **CUDA-graph decode** — one graph replay per token, 4 bytes/token host traffic.
- **Hybrid + sigmoid-router architectures** — gated-delta-net mixes (Qwen3.6), MiniMax/DeepSeek-style routing.
- **Safetensors loader** — modelopt NVFP4 repacks byte-exact; FP8/BF16 re-encode at load; disk-tier expert streaming.

## Correctness discipline

Every kernel change passes, in order: `kernel-check` (CPU reference), the `run-gen` argmax gate, `run-spec` K=1..8 self-consistency — one command: `tools/local-ci.sh`. FP summation order is part of the contract — "faster" kernels that reduce in a different order get rejected when they flip argmax at tight margins.

Exactness gates are structurally blind to numeric shifts where decode and verify move *together* — that class silently cost half a spec margin across ~40 green commits in July 2026. The local perf CI (`tools/local-ci.sh --perf`) closes it: every published cell re-measured per engine-touching push, speculative **acceptance and tokens/round tracked per cell** against a rolling baseline (`research/tune-data/perf-ci.jsonl`), enforced by the pre-push hook. Upstream engines are swept weekly for portable decode mechanisms (`tools/upstream-sweep.sh` → `research/upstream-sweeps.md`).

## Workspace layout

| Crate | What it does |
|---|---|
| `bw24-engine` | CUDA kernels (`cu/`), forward passes, speculative decoding, MoE cache, graph decode |
| `bw24-gguf` | GGUF parser + tensor loading (memory-mapped) |
| `bw24-tokenizer` | BPE tokenizer + chat templates from GGUF metadata |
| `bw24-runtime` | CUDA device/stream/memory primitives over cudarc |
| `bw24-server` | OpenAI-compatible HTTP server (axum) |
| `bw24-probe` | Standalone hardware microbenches |

## Requirements

- NVIDIA Blackwell consumer GPU (sm_120a); primary target RTX 5090 Laptop.
- CUDA 13.1, plus 12.8 for the dual-toolkit build documented in [ARCHITECTURE.md](ARCHITECTURE.md) (`BW24_NVCC` overrides the nvcc path). Rust edition 2024, cudarc 0.19.
- A model: GGUF or HF safetensors directory (pass either path). The optional Hy3 CPU-expert companion additionally needs a C++17 compiler with OpenMP.

## Limitations

- Built for sm_120a only; tuning assumes this exact memory/compute ratio. On any other GPU,
  use [llama.cpp](https://github.com/ggml-org/llama.cpp) (broadest hardware coverage) or
  [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (multi-platform Rust) instead —
  a `feat/portable-ada-correctness` branch exists for Ada (sm_89) but is untuned and its
  L40S lane is closed.
- Single GPU, single stream; no tensor parallelism or continuous batching.
- Moving research codebase; APIs and flags change without notice.

## Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — tech stack + sm_120a feasibility ledger.
- [`HANDOVER.md`](HANDOVER.md) — living state-of-work (standings, laws, open lanes).
- [`docs/decisions/`](docs/decisions/) — design decision records.
- [`docs/COMPETITOR-SETUP.md`](docs/COMPETITOR-SETUP.md) — competitor engines at their peak on this box.
- [`docs/HY3-SPILL.md`](docs/HY3-SPILL.md) — Hy3 spill runbook + [overlay release](research/per-expert-quant/hy3-layer103p5-release.md).
- [`research/tune-data/`](research/tune-data/) + [`research/gemma4-bringup/`](research/gemma4-bringup/) — every experiment as JSONL, wins and losses both.
- [`research/benchmarks.md`](research/benchmarks.md) — the A/B measurement protocol.

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

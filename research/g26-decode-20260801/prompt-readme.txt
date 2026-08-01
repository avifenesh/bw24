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
output is gated token-identical to plain decode, so speed never changes what the model
says.

**Use memra when** you serve one model on an RTX 50-series card and want measured,
exactness-gated speed, or you want a single-GPU H100 engine whose wins *and* losses
against vLLM are published per model. **Use something else when** you have another GPU
([llama.cpp](https://github.com/ggml-org/llama.cpp),
[mistral.rs](https://github.com/EricLBuehler/mistral.rs)) or need multi-GPU serving
(vLLM, SGLang).

**Standing (2026-07-31):** seven supported models on the 5090, all fully gated; every
MTP-spec cell is at or above 1.13x llama.cpp (up to 2.2x), plain cells sit at the DRAM
wall or above. On the H100, a full per-model board against vLLM 0.26: decode wins 5 of
6 models (1.2–2.1x), loses one honestly (35B MoE, 0.79x — mechanism priced); prefill
gaps are published per model and shrinking release-by-release (12B 2.1x and 31B 1.6x
prefill landed this week). Every number is a same-session interleaved measurement;
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

The full per-model board against vLLM 0.26 on 1×H100 80GB, same box, same session.
One number per arm: **end-to-end tok/s** — 512 tokens generated on a ~2048-token
prompt, single request, total wall time (N=5 medians). Cross-artifact by design:
vLLM serves what H100 users deploy (w8a8 / FP8-dynamic / bf16 HF checkpoints — it
rejects these GGUFs); memra serves its GGUF artifacts.

| model | memra e2e | vLLM 0.26 e2e (artifact) | ratio |
|---|---:|---:|---:|
| Gemma-4 12B | **148** | 81 (bf16) | **1.83x** |
| Qwen3.5-9B | **212** | 177 (w8a8) | **1.20x** |
| Gemma-4 31B | **77** | 64 (FP8-dyn) | **1.20x** |
| Gemma-4 E4B | **176** | 168 (bf16) | **1.05x** |
| Gemma-4 26B MoE | 146 | 191 (FP8-dyn) | 0.76x |
| Qwen3.6-35B MoE | 157 | 220 (FP8) | 0.71x |

Wins on exact math (the bf16-row wins carry a quant-advantage caveat — those vLLM arms
move 4x the weight bytes). The losses are published, mechanism-priced, and moving:
both losing cells are MoE prefill/expert paths with mapped levers, and the 12B/31B
rows each gained double-digit e2e in one release (Q4_0→fp16 prefill mirrors). The
remaining Qwen prefill deficit inside these numbers is the int8-GEMM dtype edge — a
Q8_0-exact int8 GEMM is mechanism-refuted on Hopper (per-32-block rescale costs 5.4x
naive / 17x pipelined; ptxas serializes cross-bank GMMA register reads), so crossing
it means w8a8-class numerics that change model outputs — an accuracy-bar decision with
measured receipts, not an engineering unknown.

Shipped on this build: FA3-class prefill attention (TMA swizzled ring + wgmma, 4.8x the
mma kernel), fused wgmma GDN chunk kernels with varlen twins, Q4_0/Q8_0→fp16 prefill
mirrors on the cuBLASLt lane, cross-request prefill batching, per-session CUDA-graph
decode, and the Hopper wgmma toolkit
([`cu/wgmma_common.cuh`](crates/memra-engine/cu/wgmma_common.cuh)) — canonical
core-matrix pairings probed for bf16/tf32/s8: one byte-geometry, three MMA kinds.

- Evidence ledger (every verdict and refutation): [ARCHITECTURE-H100.md](ARCHITECTURE-H100.md)
- Flags + promoted defaults: [docs/FLAGS.md §7](docs/FLAGS.md)
- One-command battery: `tools/validate-h100.sh <model.gguf> [--quick]`

## Model support

| Tier | Models | State |
|---|---|---|
| **Supported** | Qwen3.5-9B, Qwen3.6-27B, Qwen3.6-35B-A3B MoE (NVFP4/IQ4_XS); Gemma-4 12B, 26B-A4B MoE, 31B, E4B (QAT Q4_0 + MTP drafters) | Board-published, fully gated, exactness-first |
| **Supported, under tuning** | Hy3 Layer103.5 overlay (VRAM→RAM→dual-NVMe spill) | Correctness-gated end-to-end; [docs/HY3-SPILL.md](docs/HY3-SPILL.md) |
| **In progress** | MiniMax-M3 REAP50 (safetensors spill) | Loads + generates; router tuning open |

## Quick start

Prebuilt Linux x86_64 binaries (sm_120a) ship with each
[release](https://github.com/avifenesh/memra/releases) — or build from source:

```bash
cargo build --release
./target/release/kernel-check                     # every kernel vs CPU reference
MEMRA_CHAT=1 ./target/release/run-gen /path/to/model.gguf --prompt "Explain KV caches."
MEMRA_SPEC_K=3 ./target/release/run-spec /path/to/qwen36-27b.gguf   # MTP speculative
.
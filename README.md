# bw24

From-scratch LLM inference engine in Rust + CUDA, built for one machine: an RTX 5090 Laptop (Blackwell sm_120a, 24 GB, 175 W with dynamic boost). No frameworks, no ggml — every kernel written and tuned against measured hardware limits, with llama.cpp as the benchmark to beat on the same rig.

On its target models it beats llama.cpp where it counts: 9B generation 1.2-1.5x ahead at every prompt size, 27B (with MTP speculative decoding) ahead on short-code and long-agentic prompts and within 5% on the rest — every number measured on-device with a matched same-prompt protocol (see `research/tune-data/`).

## Why this project

- Use this as a reference for real sm_120a (consumer Blackwell) kernel work: NVFP4 block-scale decode, int8 `m16n8k16` tensor-core MMA, dp4a matvec, cp.async pipelining — with the measured win/loss record for each attempt.
- Use this if you want an inference engine whose every optimization is gated on bit-exactness (argmax match + speculative self-consistency) rather than "looks close enough."
- Use this to run Qwen3.5/3.6 dense (9B/27B, NVFP4 hybrid) and Qwen3.6-35B-A3B MoE from GGUF on a 24 GB card, including MTP speculative decoding and MoE expert caching with VRAM/host/disk spill.
- Read `research/tune-data/*.jsonl` if you want a labeled corpus of kernel-tuning experiments (config → measured perf, wins and losses both recorded).

## Requirements

- NVIDIA Blackwell consumer GPU (sm_120a). Primary target: RTX 5090 Laptop. An sm_89 (Ada) branch exists at `arch/sm89-l40s`.
- CUDA toolkit 12.8 (13.1 optional for the cuBLASLt/CUTLASS paths; 13.1 nvcc miscompiles some sm_120 kernels, see `crates/bw24-engine/build.rs`).
- Rust (edition 2024), [cudarc](https://github.com/coreylowman/cudarc) 0.19 with dynamic loading.
- A GGUF model. Tested: Qwen3.5-9B / Qwen3.6-27B (NVFP4 + Q5_K hybrid), Qwen3.6-35B-A3B (IQ4_XS MoE), plus Q4_K/Q5_K/Q6_K/Q8_0 k-quant variants.

## Quick start

```bash
# build
cargo build --release

# verify all kernels against the CPU reference
./target/release/kernel-check

# generate text
BW24_FAST=1 BW24_GEMM=1 BW24_MMVQ=1 BW24_FA_VEC=1 BW24_CHAT=1 \
  ./target/release/run-gen /path/to/model.gguf --prompt "Explain KV caches in one paragraph."

# speculative decoding with the embedded MTP draft head (Qwen3.6)
BW24_FAST=1 BW24_GEMM=1 BW24_MMVQ=1 BW24_FA_VEC=1 BW24_SPEC_K=3 \
  ./target/release/run-spec /path/to/qwen36-27b.gguf

# OpenAI-compatible server
BW24_FAST=1 BW24_GEMM=1 BW24_MMVQ=1 BW24_FA_VEC=1 \
  ./target/release/bw24-server
```

`run-gen` prints a prefill/decode correctness gate (prefill argmax must match decode argmax) before timing anything — if that line says MISMATCH, the numbers after it don't count.

## Workspace layout

| Crate | What it does |
|---|---|
| `bw24-engine` | Core: CUDA kernels (`cu/`), forward passes, speculative decoding, MoE cache, CUDA-graph decode |
| `bw24-gguf` | GGUF parser + tensor loading (memory-mapped) |
| `bw24-tokenizer` | BPE tokenizer + chat templates from GGUF metadata |
| `bw24-runtime` | CUDA device/stream/memory primitives over cudarc |
| `bw24-server` | HTTP server (axum), OpenAI-compatible `/v1` endpoints |
| `bw24-probe` | Standalone hardware microbenches (`probe/*.cu`: bandwidth, tensor-core peaks, layout experiments) |

## What's inside

- **NVFP4 (W4) decode path** — block-scaled FP4 matvec with split-plane repack, warp-level dp4a, and an int8 W4A8 tensor-core GEMM for prefill. Auto-dispatches per matrix shape.
- **MTP speculative decoding** — draft with the model's embedded multi-token-prediction head, verify K+1 tokens in one batched target forward. The whole draft chain runs inside a captured CUDA graph; exactness is enforced by a K=1..8 self-consistency gate (all K must emit identical tokens).
- **MoE on 24 GB** — expert-major CSR batching, decode-once dequant kernels, int8 tensor-core expert GEMM (`BW24_MOE_MMA=1`), and an SLRU expert-residency cache with VRAM → host → disk spill.
- **FlashAttention-style kernels** — fused prefill and decode attention with quantized KV (q8_0 K / q4_0 V default, FP8 optional), register-resident dequant, split-K for long context.
- **CUDA-graph decode** — the full per-token decode is one graph replay; per-step host round-trip is 4 bytes.
- **Hybrid architectures** — full-attention + gated-delta-net (SSM) layer mixes, as in Qwen3.6.

## Correctness discipline

Every kernel change must pass, in order:
1. `kernel-check` — every quant kernel vs a CPU reference.
2. `run-gen` argmax gate — prefill and decode paths must agree on the next token.
3. `run-spec` self-consistency — speculative output at K=1..8 must be token-identical to plain decode.

Floating-point summation order is part of the contract: two mathematically equal kernels that reduce in different orders can flip an argmax at tight logit margins. Several "faster" kernels were rejected for exactly this (`research/tune-data/`).

## Performance

Measured on the target rig (RTX 5090 Laptop, N≥3 medians) against llama.cpp built on the same machine at its best serve config (MTP spec draft, quantized KV, FA, graphs), same exact prompts. Generation tok/s at three real prompt sizes — short code (28 tok), medium code (1.8k), long agentic (6.3k):

| Model | bw24 | llama.cpp | Ratio |
|---|---|---|---|
| Qwen3.5-9B NVFP4 (spec K=3) | 193 / 156 / 149 | 122 / 121 / 117 | **1.59x / 1.29x / 1.28x** |
| Qwen3.6-27B NVFP4 (spec K=3) | 99 / 88 / 76 | 87 / 92 / 75 | **1.14x** / 0.95x / **1.01x** |
| Qwen3.6-35B-A3B MoE (spec K=2 / plain) | 182 / 158 | 170 | **1.08x** / 0.93x |

Also running, no llama.cpp comparison possible (safetensors-only checkpoints):

- **nvidia/Qwen3.6-27B-NVFP4** (official NVIDIA checkpoint: mixed NVFP4 + FP8 linear-attention + model-trained BF16 MTP head, loaded straight from safetensors) — spec K=2 45-48 tok/s on the laptop rig. The vLLM 0.24.0 reference on an RTX PRO 6000 runs the same checkpoint through Marlin weight-only dequant (no native FP4 on sm_120 in vLLM).
- **MiniMax-M3 REAP50 NVFP4** (121 GB, 60 layers, sigmoid routing) — loads and generates correct text on this 24 GB / 60 GB-RAM machine via an NVMe disk-tier expert loader; decode is I/O-bound by design here (routing-locality measurement in `research/tune-data/` shows capacity, not caching policy, is the binding constraint).

Speculative output is bit-exact: a K=1..8 self-consistency gate pins it token-identical to plain greedy decode. Where bw24 is still behind (27B medium-code and prefill, 35B MoE decode), the gap and its current diagnosis are tracked in the tune-data records, not hidden.

## Limitations

- Built for sm_120a. It compiles nowhere else without the `arch/sm89-l40s` branch, and the tuning choices assume this exact memory/compute ratio.
- Model coverage is what's listed above — this is not a general GGUF runner.
- Single GPU, single stream. No tensor parallelism, no continuous batching.
- APIs and env flags change without notice; this is a moving research codebase.

## Docs

- `ARCHITECTURE.md` — tech stack + sm_120a feasibility ledger (what the silicon can and cannot do, measured).
- `research/sm120-empirical-capabilities.md` — microbenched peaks for this GPU.
- `research/benchmarks.md` — the A/B measurement protocol.
- `research/tune-data/` — every tuning experiment as JSONL (the labeled corpus).

## Contributing

Issues and PRs welcome. Any kernel PR must pass the three correctness gates above and include before/after numbers measured with the protocol in `research/benchmarks.md`.

## License

MIT — see [LICENSE](LICENSE).

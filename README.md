# bw24

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust](https://img.shields.io/badge/rust-edition%202024-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8%20%2F%2013.1-76B900.svg)
![arch](https://img.shields.io/badge/arch-sm__120a%20(Blackwell)-black.svg)

![bw24 vs llama.cpp perf board](docs/perf-card.svg)

From-scratch LLM inference engine in Rust + CUDA, built for one machine: an RTX 5090 Laptop (Blackwell sm_120a, 24 GB, 175 W with dynamic boost). No frameworks, no ggml — every kernel written and tuned against measured hardware limits, with llama.cpp as the benchmark to beat on the same rig.

Plain decode runs at or above llama.cpp on the dense models (27B 1.08x, 9B 1.03x) and at 0.99x on the 35B MoE; with MTP speculative decoding it leads 9B 1.31x/1.23x and 27B up to 1.25x at the same raw-prompt protocol — every number measured on-device against llama.cpp's serve-best config on the same machine, N=3 medians, both engines re-baselined the same day (see `research/tune-data/`). It also loads NVIDIA's official safetensors checkpoints directly (mixed NVFP4 + FP8 + BF16 MTP head) and runs a 121 GB MoE on the 24 GB card.

## Why this project

- Use this as a reference for real sm_120a (consumer Blackwell) kernel work — every optimization ships with its measured win/loss record, not just the winners.
- Use this if you want an inference engine gated on bit-exactness (argmax + speculative self-consistency) rather than "looks close enough."
- Use this to run Qwen3.5/3.6 dense and MoE checkpoints on a 24 GB card — from GGUF or straight from HF safetensors, no conversion step — including models far larger than VRAM+RAM.

## Requirements

- NVIDIA Blackwell consumer GPU (sm_120a). Primary target: RTX 5090 Laptop. An sm_89 (Ada) branch exists at `arch/sm89-l40s`.
- CUDA toolkit 12.8 (13.1 optional for the cuBLASLt/CUTLASS paths; 13.1 nvcc miscompiles some sm_120 kernels, see `crates/bw24-engine/build.rs`).
- Rust (edition 2024), [cudarc](https://github.com/coreylowman/cudarc) 0.19 with dynamic loading.
- A model. GGUF tested: Qwen3.5-9B / Qwen3.6-27B (NVFP4 + Q5_K hybrid), Qwen3.6-35B-A3B (IQ4_XS MoE), plus Q4_K/Q5_K/Q6_K/Q8_0 k-quant variants. Safetensors (HF dir) tested: nvidia/Qwen3.6-27B-NVFP4, MiniMax-M3 REAP50 NVFP4 — pass the directory instead of a .gguf path.

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
- **Safetensors loader** — HF checkpoints load directly (no GGUF conversion): modelopt NVFP4 repacks byte-exact into the GGUF block layout, FP8-E4M3 and large-BF16 tensors re-encode to Q8_0/NVFP4 at load, V-head permutations apply on packed bytes, MoE experts stream through a disk-tier repack cache for models far bigger than VRAM+RAM.
- **Sigmoid-router MoE** (MiniMax/DeepSeek-style) — e_score_correction_bias selection, swigluoai activation, gate-optional attention; with the measured law that cross-kernel-family FP-order differences are architectural on discontinuous top-k routing (exactness binds within a config).

## Correctness discipline

Every kernel change must pass, in order:
1. `kernel-check` — every quant kernel vs a CPU reference.
2. `run-gen` argmax gate — prefill and decode paths must agree on the next token.
3. `run-spec` self-consistency — speculative output at K=1..8 must be token-identical to plain decode.

Floating-point summation order is part of the contract: two mathematically equal kernels that reduce in different orders can flip an argmax at tight logit margins. Several "faster" kernels were rejected for exactly this (`research/tune-data/`).

## Performance

<!-- PERF-DATE:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
Measured 2026-07-08 on the target rig (RTX 5090 Laptop, N≥3 medians, power state verified before every session) against llama.cpp built on the same machine, same exact prompts, both engines re-baselined the same day. Boards move with the tuning campaign — `research/tune-data/rig5090.jsonl` is the running record; the README is refreshed with every board-moving merge.
<!-- PERF-DATE:END -->

**Plain decode first** (no speculation, tg128 at 512-token context — the honest floor comparison):

<!-- PERF-PLAIN:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
| Model | bw24 plain | llama.cpp plain | Ratio |
|---|---|---|---|
| Qwen3.6-27B NVFP4 | 47.2 | 43.6 | **1.08x** |
| Qwen3.5-9B NVFP4 | 131.6 | 124.5 | **1.06x** |
| Qwen3.6-35B-A3B MoE | 169.0 | 170.5 | 0.99x |
<!-- PERF-PLAIN:END -->

Depth behavior is part of the comparison: at 6.3k-token context the 35B decodes at 152.8 vs llama.cpp's 159.9 (0.96x) — split-ladder geometry is validated across the depth axis, not just the short-context point.

**Speculative decoding** (MTP head, both engines at their measured best config) as the bonus layer on top:

<!-- PERF-SPEC:START (generated by tools/update-perf-board.py — do not hand-edit; edit research/tune-data/current-board.json instead) -->
| Model | bw24 spec | llama.cpp spec-best | Ratio |
|---|---|---|---|
| Qwen3.5-9B (K=3 + native trim) | 243 / 195 / 162 | 186 / 158 / 155 | **1.31x** / **1.23x** / 1.05x |
| Qwen3.6-27B (K=3 + generic trim) | 108 / 91 / 79.5 | 86.4 / 89.9 / 73.2 | **1.25x** / 1.01x / **1.09x** |
| Qwen3.6-35B-A3B (K=2 + trim + zero-draft) | 197 / 194 / 177 | 215 / 208 / 202 | 0.92x / 0.93x / 0.88x |
<!-- PERF-SPEC:END -->

All rows are the raw-prompt continuation protocol (llama.cpp measured through llama-server at its serve-best speculative config on the same machine, N=3 medians, full power verified). Config is content-class dependent — the chat protocol shifts both the optimal draft depth and the trim choice (chat short-code runs K=7 at 122 tok/s on the 27B); the published HF artifacts document every configuration.

Three speculative mechanisms shipped in the 2026-07-08 push, vendored-and-verified rather than invented: the FR-Spec vocab trims are *vocabulary* artifacts, not model artifacts — a 32k-row d2t list transfers across every model sharing a tokenizer (the gather reads each model's own lm_head bytes), and for a new vocab the `frspec_rank` tool builds one from any local corpus in minutes (the 9B's 248320-token vocab got its own: p1/p2/p3 +22/+17/+14%); zero-draft rounds (`BW24_SPEC_PMIN0`) apply llama.cpp's whole-round confidence gate so unpredictable stretches run at plain-decode cost (pays below ~75% base acceptance, hurts above ~90%); and per-class draft depth (K is a property of the content's per-slot acceptance decay, protocol included).

On the 27B the two engines bind differently: llama.cpp is cost-bound at short prompts (draft overhead caps it even at near-full acceptance) while bw24's cheaper rounds ride high acceptance (1.25x raw, 1.40x chat); at medium/long prompts both sit near the same content-acceptance ceiling.

**Reproducing these numbers:** every artifact the claims depend on is public — trimmed draft-head GGUFs (generic/code75/balanced for the 27B/35B, the 9B-native ranking, all for `BW24_FRSPEC_TRIM`), the exact prompts, and the full configs (env law, per-class K/pmin, llama.cpp build + serve flags) at [huggingface.co/Avifenesh/bw24-bench](https://huggingface.co/Avifenesh/bw24-bench). The llama.cpp side runs its measured-best serve config per model (build flags and per-model serve lines in [docs/COMPETITOR-SETUP.md](docs/COMPETITOR-SETUP.md)); the harness is `research/e2e/run-e2e.sh`.

Safetensors checkpoints (no llama.cpp comparison possible):

- **nvidia/Qwen3.6-27B-NVFP4** — 92.5 tok/s spec on the laptop rig (2.3x the tuned local vLLM reference, which reaches 40.8 plain and cannot fit its MTP draft head on 24 GB; bw24's trimmed draft head byte-gathers rows from the trunk's own lm_head, zero extra VRAM). Same-silicon vLLM comparison on an RTX PRO 6000 96 GB: vLLM MTP reaches 147-184 tok/s there via batched multi-token drafting — the standing gap bw24 is working (bw24 92-97 on that box).
- **MiniMax-M3 REAP50 NVFP4** (121 GB, 60 layers, sigmoid routing) — loads and generates correct text on this 24 GB / 60 GB-RAM machine via an NVMe disk-tier expert loader (~1.5 tok/s, I/O-bound: measured routing locality shows 77% of experts get touched with weak reuse, so capacity — not caching policy — is the binding constraint). On a 96 GB RTX PRO 6000 the same code reaches ~6 tok/s and climbing with an 80 GB expert cache.

Speculative output is bit-exact: a K=1..8 self-consistency gate pins it token-identical to plain greedy decode. Where bw24 is still behind, the gap and its diagnosis are tracked in the tune-data records, not hidden — currently: **prefill** (pp≈2k same-day board: 9B 3799 vs 6287, 27B 1055 vs 2348, 35B 2338 vs 3981 — llama's int8 tensor-core MMQ GEMM vs our dp4a path; an FP8-activation tensor-core prefill is in flight, targeting the compute headroom this silicon has that neither engine uses), the 35B deep-context decode residual (152.8 vs 159.9 at 6.3k), and vLLM's batched MTP on big-VRAM boxes.

## Limitations

- Built for sm_120a. It compiles nowhere else without the `arch/sm89-l40s` branch, and the tuning choices assume this exact memory/compute ratio.
- Model coverage is what's listed above — this is not a general GGUF runner.
- Single GPU, single stream. No tensor parallelism, no continuous batching.
- APIs and env flags change without notice; this is a moving research codebase.

## Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — tech stack + sm_120a feasibility ledger (what the silicon can and cannot do, measured).
- [`HANDOVER.md`](HANDOVER.md) — the living state-of-work doc (current standings, laws, open lanes); internal but readable.
- [`docs/decisions/`](docs/decisions/) — design decision records: internal weight format, quant/GEMM policy, safetensors import, hybrid-architecture plan.
- [`docs/COMPETITOR-SETUP.md`](docs/COMPETITOR-SETUP.md) — how each competitor engine is built and tuned to its peak on this box (the "beat them at their best" contract).
- [`research/tune-data/current-board.json`](research/tune-data/current-board.json) — the numbers behind the Performance section and `docs/perf-card.svg`, both regenerated by [`tools/update-perf-board.py`](tools/update-perf-board.py) — edit the JSON, never the generated regions directly.
- [`research/sm120-empirical-capabilities.md`](research/sm120-empirical-capabilities.md) — microbenched silicon peaks for this GPU.
- [`research/benchmarks.md`](research/benchmarks.md) — the A/B measurement protocol.
- [`research/tune-data/`](research/tune-data/) — every tuning experiment as JSONL: config → measured result, wins and losses both. ~215 records and counting; treat it as a labeled corpus of what sm_120a actually rewards.

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

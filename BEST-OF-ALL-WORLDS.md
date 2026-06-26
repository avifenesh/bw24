# bw24 — Best-of-all-worlds map + our edges

Philosophy (user 2026-06-26): rewrite everything that can be done better. Take the best idea for each
component from each engine, port it to our format (mostly reading+writing, not inventing algorithms),
and fill the gaps where ALL of them are bad. Target = ONLY RTX 5090 Laptop sm_120. No multi-GPU
genericity. We build new because we're tired of imperfect options + months-long PR review waits.
This is worth a month+ of work.

## What we take from each (port to our format, tuned for sm_120/5090-laptop)

| Component | Source (best impl) | What we take |
|---|---|---|
| Quantization + GGUF quant kernels | **llama.cpp** | k-quant/i-quant block formats, MMQ/MMVQ int8 dp4a, dequant-in-dot |
| Weight quant fast path | **Marlin/vLLM** + **ExLlamaV3** | W4A16 marlin, EXL3 trellis ideas (quality/size) |
| KV cache structure | **SGLang RadixAttention** | radix-tree prefix sharing, page/token reuse |
| Prefix/disk KV tiering | **LMCache** | offload KV to disk, reuse across requests, tiered KV store |
| Expert spill/offload | **KTransformers** | GPU attention/dense + CPU/spill experts, but DONE RIGHT (see edge) |
| Continuous batching/scheduler | **vLLM** | iteration-level batching, chunked prefill, block pool |
| Zero-overhead scheduler / overlap | **SGLang** | event_loop_overlap, run step N+1 while processing N |
| Attention | **FA-2 + FA-3/FA-4 improvements** (hand) | online softmax, async pipeline, warp-spec adapted to sm_120 mma.sync |
| Spec decode | **MTP (NextN) + EAGLE3.1** | built-in draft head + EAGLE drafts (daily serve uses draft-mtp) |

## OUR EDGES (where ALL of them are bad — we fix)

### EDGE 1 — Selective expert staging (the headline, user's own idea)
PROBLEM: vLLM keeps ALL experts resident on GPU (OOMs big MoE). llama.cpp `--n-cpu-moe` keeps experts
on CPU and on offload re-handles experts coarsely. Both effectively **touch/override all experts per
token** instead of staging only the few actually routed. User waited months for a PR review on exactly
this and it never landed.
OUR DESIGN: per token, the router picks 8 of 256 experts. **Stage/transfer ONLY those 8 requested
experts** (the miss set), not all. Keep a GPU-resident **hot-expert cache** (fixed N slots, SLRU +
second-miss admission). On a routed expert:
- HIT (expert in a resident slot) → compute in place, zero transfer.
- MISS → DMA-copy ONLY that expert's weight (gate/up/down slice) into a free/evicted slot, then compute.
Hot ~15-20% of experts stay resident → steady-state per-token PCIe ≈ 0. Prefetch next-layer routed
experts during current-layer compute (overlap on a copy stream). This is the difference from
vLLM/llama.cpp: **override only the requested expert, never all.**
Tiers: GPU slot cache ↔ pinned host RAM ↔ mmap'd NVMe (LMCache-style disk for cold experts).

### EDGE 2 — Whole-system native Rust runtime
No python per-token dispatch (vLLM/SGLang tax), no GC. Single process, fearless overlap of
compute/copy/spill. This is the single-stream decode + cold-start win.

### EDGE 3 — Hybrid-arch KV advantage exploited fully
qwen35 daily models: only 8/32 (or 16/64) layers grow KV; 24/48 carry fixed recurrent state. We size
the dual cache to this asymmetry → far longer context in 24GB than a uniform-KV engine.

### EDGE 4 — sm_120-exact kernels
Everything tuned to ONE chip: 82 SMs, 847 GB/s, 99KB smem, block-FP4 762 TFLOP, no wgmma/tcgen05.
No arch-dispatch overhead, no datacenter-kernel fallbacks.

## NON-GOALS
- Multi-GPU / tensor-parallel / pipeline-parallel (single 5090 laptop only).
- FA-5 or new algorithms (port FA-3/FA-4 IMPROVEMENTS, don't invent).
- Supporting every GPU arch.

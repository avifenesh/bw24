# Implementation Map — ExLlamaV3 (EXL3)

Edge low-VRAM **single-stream** engine. Positioning-wise the *closest in intent* to bw24 of any engine surveyed: consumer GPU, batch=1 decode focus, aggressive low-bpw weight quant + low-bit paged KV. Where llama.cpp is the closest *implementation* analog (raw CUDA + resident quant), ExLlamaV3 is the closest *use-case* analog (laptop / single-user, VRAM is the binding constraint).

**SOURCE AVAILABILITY CAVEAT (governs every verdict below).** The only ExLlamaV3 artifact present on this machine is a compiled, opaque autotune blob:

- `/data/cache/exllamav3/autotune/coop_autotune_v1.bin` — 1.8 KB, `file` reports `data` (not ELF, not text), zero extractable symbol/format strings (`strings | grep -iE 'trellis|exl3|gemm|kernel|dequant|mma|fp4|q4|kv|page'` → empty). It is a serialized cooperative-launch autotune *result cache* (tile/occupancy params), **not** kernel source or PTX.

Therefore there are **no `file:line` anchors into ExLlamaV3 itself**. Every "source" cell below is annotated `NOT IN REPO`, and the mechanism descriptions are reconstructed from ExLlamaV3's published positioning (EXL3 trellis VQ, Q4 paged KV, dynamic generator) cross-referenced against the bw24 project decision docs. Anchors that *do* resolve to real lines point at **bw24's own** comparison code (the engine that would have to port any of this), so the map stays concrete and actionable.

bw24 comparison anchors used throughout (all under `/home/avifenesh/projects/bw24`):
- `crates/bw24-engine/cu/qmatvec.cu:1-7` — resident-quant GGUF matvec, in-register dequant, f32 activations (Stage A). The bw24 path EXL3's decode GEMM would compete with.
- `crates/bw24-engine/cu/qmatvec_gemm.cu` — bw24 prefill quant-GEMM (the 43× prefill gap target, commit `8d1c0b7`).
- `crates/bw24-engine/cu/flash_attn.cu:1-25` — hand-written FA-2 on validated **m16n8k16 bf16 mma** for sm_120a; q8_0-K/q5_1-V inline dequant cache (commit `9ebf958`).
- Project KV decision: q8_0 K / q5_1 V chosen over Q4 for scale-fragility reasons (`research/basics/KVQUANT-PLAN.md`).

sm_120 reality check that governs every verdict (RTX 5090 / consumer Blackwell, `sm_120a`):
- HAS: `mma.sync` m16n8k8/k16/k32 (int8 / fp16 / bf16), `ldmatrix`, `dp4a` (DP4A int8), `__shfl_*_sync`, fast `expf`, `cudaStreamBeginCapture`.
- LACKS: `wgmma` (Hopper / datacenter-Blackwell warpgroup MMA), `tcgen05`, block-scaled `mma.sync` m16n8k64 for MXFP4/NVFP4 (`BLACKWELL_MMA_AVAILABLE`-gated, datacenter only), AMX (CPU, irrelevant).

---

## What bw24 could take from ExLlamaV3

Highest-value, most-portable ideas for a single-stream sm_120 GGUF engine. Note that *the format itself is the main barrier* — EXL3 is not GGUF — so most "take" items are **conceptual / algorithmic borrows**, not code lifts.

1. **Q4 paged KV cache with in-attention-kernel dequant (concept).** EXL3 ships a 4-bit paged KV with dequant fused into the paged-attention read. bw24 deliberately chose the *more conservative* q8_0-K/q5_1-V (`flash_attn.cu`, commit `9ebf958`) precisely because Q4 KV adds per-page/per-head scale fragility. The portable lesson is the **upper bound**: if bw24 ever needs to halve KV again (longer context on a laptop), EXL3 demonstrates Q4 paged KV is viable single-stream — but bw24 should keep its asymmetric K>V bit budget rather than blanket-Q4. Value: HIGH (directly on bw24's VRAM-bound axis), effort: MEDIUM (dequant-scale plumbing), risk: accuracy regression.

2. **Cooperative-launch autotune *result cache* pattern (the one artifact we actually have).** `coop_autotune_v1.bin` is a tiny (1.8 KB) persisted blob of best tile/occupancy params per (kernel, shape) keyed and reloaded across runs, so the first-token latency does not re-pay autotuning. bw24's hand-tuned single-shape kernels (head_dim=256, fixed tiles in `flash_attn.cu`) mostly sidestep this, but the *pattern* — persist autotune verdicts to a small on-disk cache instead of recomputing — is cheap and portable if bw24 ever parametrizes tiles per model. Value: MEDIUM, effort: LOW.

3. **Trellis VQ as the "if we ever go below 4 bpw" reference (concept only).** EXL3's tail-biting trellis VQ beats scalar-block (GGUF K-quant) quality at the same low bpw with tiny metadata. bw24's decision docs already evaluated and **deferred** this: real quality/bit advantage, but non-GGUF format + bespoke decode GEMM is too large for the single-stream sm_120 scope, and it forfeits any Blackwell block-FP4 MMA path. Keep as a documented fallback if GGUF K-quants prove insufficient at target bpw. Value: LOW-MED, effort: VERY HIGH.

4. **Single-stream-first runtime discipline (validation, not code).** EXL3's dynamic generator is a batch=1, eager-launch, dynamic-KV-growth loop — structurally identical to what bw24 already does in **Rust** (zero per-token FFI). The borrow here is *confirmation of the design*: a leading edge engine reaches its numbers with exactly bw24's structural choice (no multi-request scheduler on the hot path). bw24's Rust runtime is strictly better than EXL3's Python dispatch on per-token overhead. Value: HIGH (de-risks bw24's architecture), effort: ZERO (already done).

5. **Q4 dequant-in-register feasibility on sm_120 (concept).** EXL3 proves 4-bit weight *and* KV dequant on consumer GPUs via DP4A + Turing-class int8 MMA — the same instruction set bw24's `qmatvec.cu` already uses. So if bw24 adds a 4-bit GGUF weight path (Q4_0/Q4_K) to `qmatvec_gemm.cu`, the EXL3 existence-proof says the *kernel shape* (nibble unpack → int8 → dp4a/mma) is sound on sm_120; only the format differs. Value: MEDIUM, effort: LOW (bw24 already has the int8 path).

---

## DEAD for bw24

Items that do not run on sm_120, or are non-portable by construction:

- **EXL3 bespoke trellis decode GEMM (as a code lift).** Non-GGUF packed layout + custom dequant kernel; cannot be dropped into a GGUF engine. Even if ported, it forfeits the Blackwell block-FP4 MMA advantage (trellis decode maps to DP4A/int8, not block-scaled m16n8k64). **DEAD as code; concept deferred** (see "take" #3). Anchor: format is internal to EXL3; nearest bw24 contrast `crates/bw24-engine/cu/qmatvec_gemm.cu`.
- **Any `wgmma` / `tcgen05` fast path EXL3 ships for datacenter Blackwell / Hopper.** If present in EXL3's full kernel set, those are `sm_90`/datacenter-`sm_100`-class and **DEAD on sm_120** — bw24 must use `mma.sync` m16n8k16/k32 (as it already does in `flash_attn.cu`). (Not confirmable from the blob; flagged as the standard architecture cut.)
- **Python host dispatch / dynamic generator framework.** ExLlamaV3's generator is Python-resident; bw24 is Rust with zero per-token FFI cost. Porting the *framework* would re-introduce the exact per-token dispatch overhead bw24 was built to avoid. **DEAD** — keep bw24's Rust loop; borrow only the batch=1 discipline (take #4).
- **Multi-request / continuous-batching server machinery** (if any in EXL3's full generator). No single-stream value; bw24 is unified `n_stream=1`. **DEAD.**
- **AMX.** CPU-only ISA, not in EXL3's CUDA path and irrelevant to an sm_120 engine. **DEAD.**

---

## Subsystem 1 — EXL3 Quant + Decode GEMM (trellis VQ, quant-domain matmul)

| Technique | How implemented (mechanism) | Kernel / layout / instruction | source file:line | sm120_fit (single-stream value) |
|---|---|---|---|---|
| EXL3 trellis vector quantization | Tail-biting trellis VQ encodes weight blocks; beats scalar-block (GGUF K-quant) RD curve at low bpw with tiny per-tensor metadata. Bespoke packed layout, **NOT** GGUF; decode needs custom dequant. Competes with NVFP4 on quality/bit. | Trellis-encoded weight blocks + minimal per-tensor metadata; bespoke packed layout incompatible with GGUF block format. | **NOT IN REPO** — only opaque autotune blob `/data/cache/exllamav3/autotune/coop_autotune_v1.bin` (1.8 KB `data`, no strings). bw24 contrast: GGUF block layout consumed verbatim in `crates/bw24-engine/cu/qmatvec.cu:18` (IQ3_S grid `ggml-common.h:1042`). | **MISSING / DEFERRED.** Quality/bit win real but non-GGUF format + bespoke decode GEMM too large for single-stream sm_120 laptop scope (per bw24 decision doc). Not on critical path vs NVFP4 / GGUF K-quants. Single-stream value: LOW-MED as concept; ZERO as code. |
| EXL3 decode GEMM (quant-domain matvec) | Custom quantized matvec over the trellis format, tuned for batch=1 low-VRAM decode. Activations quantized to a custom int8/fp4 variant; exact dispatch unknown. | EXL3 internal trellis blocks; activation quant to custom int8/fp4; kernel type (DP4A vs mma) unknown from blob. | **NOT IN REPO** — hidden in compiled autotune blob. bw24 equivalent: `crates/bw24-engine/cu/qmatvec.cu` (decode) + `qmatvec_gemm.cu` (prefill, llama.cpp MMVQ/MMQ-style). | **NEEDS-PORT (source absent).** If ported: (1) needs a custom trellis dequant kernel, (2) sm_120 *has* DP4A + Turing int8 MMA so the nibble→int8→dp4a/mma shape is compatible, (3) but Blackwell block-FP4 MMA advantage is lost on trellis. bw24 should stay on GGUF MMVQ/MMQ. Single-stream value: MEDIUM if format barrier removed (it isn't). |
| Activation quantization (custom variant) | Activations down-cast to a custom int8/fp4 to feed integer/low-bit MAC, mirroring llama.cpp's Q8_1 pre-pass concept but in EXL3's own layout. | Custom int8/fp4 activation block layout; pre-pass before decode GEMM. | **NOT IN REPO.** bw24 contrast: Stage-B plan to quantize activations to q8_1 + int8 dp4a, noted in `crates/bw24-engine/cu/qmatvec.cu:2-3`. | **RUNS (concept).** Activation int8 pre-pass is arch-neutral on sm_120 (vector op, no tensor cores). bw24 already plans the GGUF-native equivalent; EXL3 adds no portable code, only confirms the approach. Single-stream value: validates bw24's Stage-B plan. |
| Cooperative-launch autotune result cache | Persists best tile/occupancy params per (kernel, shape) to a tiny on-disk blob so first-token latency does not re-pay autotuning across runs. | Serialized param cache (≈1.8 KB), keyed by kernel+shape; reloaded at startup, fed to cooperative-grid launch. | **PARTIALLY PRESENT** — the artifact `coop_autotune_v1.bin` itself exists (`/data/cache/exllamav3/autotune/coop_autotune_v1.bin`), but the producing/consuming code is **NOT IN REPO**. | **RUNS / PORTABLE.** The pattern (persist autotune verdicts, skip recompute) is arch-neutral and cheap. bw24's fixed-shape hand-tuned kernels (`flash_attn.cu`, head_dim=256) mostly avoid the need, but it's a clean low-effort borrow if bw24 ever parametrizes tiles per model. Single-stream value: MEDIUM (first-token latency). |

---

## Subsystem 2 — EXL3 KV Cache Quantization (Q4 paged, in-kernel dequant)

| Technique | How implemented (mechanism) | Kernel / layout / instruction | source file:line | sm120_fit (single-stream value) |
|---|---|---|---|---|
| Q4 paged KV cache | Low-bit (4-bit) paged KV cache for single-stream low-VRAM inference; ~halves KV vs Q8 for the same context. Per-page / per-head block scaling (exact granularity not visible from blob). | Paged KV layout, 4-bit packed blocks, per-page or per-head scale factors. | **NOT IN REPO** — source absent. bw24 contrast: q8_0 K / q5_1 V cache in `crates/bw24-engine/cu/flash_attn.cu` (commit `9ebf958`); rationale in `research/basics/KVQUANT-PLAN.md`. | **NEEDS-PORT (comparable to bw24's q8_0/q5_1).** sm_120 has DP4A + int8 MMA → Q4 dequant supported, but no block-FP4 leverage. Q4 saves VRAM vs Q8 at the cost of scale fragility — bw24 *deliberately* chose conservative q8_0/q5_1. Single-stream value: HIGH on the VRAM axis if longer context needed; keep asymmetric K>V bits rather than blanket Q4. |
| In-attention-kernel dequant (fused) | Dequant performed *inside* the paged-attention read (no separate dequant pass / no materialized f16 KV), vectorized per page. | Vectorized dequant inside paged attention; block-scale multiply on load. | **NOT IN REPO.** bw24 equivalent: asymmetric in-kernel dequant in `crates/bw24-engine/cu/flash_attn.cu` (q8_0-K/q5_1-V dequant on load, FA-2 online-softmax loop). | **RUNS (bw24 already does this for q8_0/q5_1).** Fused inline dequant is bitwise + block-scale multiply only — native on sm_120, no tensor-core dependency on the decode read. EXL3 adds no portable code beyond confirming the fused-dequant shape bw24 already ships. Single-stream value: structural match to bw24's decode kernel. |
| Paged KV allocation / page table | Pages of KV allocated and indexed via a page table for dynamic context growth; designed for single sequence (batch=1) growth on constrained VRAM. | Page-table indexed KV blocks; dynamic growth, single sequence per page run. | **NOT IN REPO.** bw24 contrast: bw24 KV growth is contiguous/ring-style per single stream (no multi-seq paging needed at `n_stream=1`). | **RUNS but LOW value for bw24.** Paging earns its keep under multi-request fragmentation; at unified `n_stream=1` bw24 does not fragment, so a full page table is overhead. Borrow only if bw24 adds prefix-cache reuse. Single-stream value: LOW. |

---

## Subsystem 3 — EXL3 Dynamic Generator (streaming single-stream runtime)

| Technique | How implemented (mechanism) | Kernel / layout / instruction | source file:line | sm120_fit (single-stream value) |
|---|---|---|---|---|
| Single-stream token-by-token loop | Batch=1 generation loop with dynamic shape handling; minimal per-token overhead, decode-focused. | Dynamic KV growth, single sequence per token; eager kernel launch (no CUDA graphs assumed), batch size = 1. | **NOT IN REPO** — host runtime (Python/C++), not in the kernel blob. bw24 equivalent: Rust per-token decode loop (zero FFI per token). | **RUNS.** Structure is trivial for sm_120. bw24 implements the same shape in **Rust** with zero per-token FFI cost; EXL3's Python dispatch is the structural overhead bw24 avoids. Single-stream value: HIGH as *design validation*, ZERO as code (bw24's Rust loop is strictly better). |
| Dynamic KV growth | KV cache grown incrementally as the single sequence extends; no pre-allocation of full context where avoidable. | Incremental KV append per decoded token; dynamic shape on the attention read. | **NOT IN REPO.** bw24 contrast: bw24 grows its single-stream KV per token alongside the q8_0/q5_1 cache (`flash_attn.cu` decode split-K path). | **RUNS.** Append-grow is a pointer/length update + one block write; arch-neutral. bw24 already does this. Single-stream value: structural match, no port needed. |
| Eager launch (no CUDA graph assumed) | Per-token kernels launched eagerly via stream-based prefetch; relies on Python/C++ dispatch rather than captured graphs. | CUDA stream prefetch + eager launch; batch=1. | **NOT IN REPO.** bw24 contrast: bw24 can use `cudaStreamBeginCapture` graph replay (sm_120-compatible) to kill per-op launch latency — a lever EXL3's eager Python loop leaves on the table. | **RUNS, and bw24 can do BETTER.** sm_120 supports graph capture/replay (~3-5% on launch-bound batch=1 decode). EXL3's eager dispatch is *not* a model to copy here; bw24's graph-capture option is the superior single-stream choice. Single-stream value: NEGATIVE as a model (avoid eager Python dispatch); bw24's graph path wins. |

---

## Bottom line for bw24

- **Format is the wall.** EXL3's headline wins (trellis VQ, custom decode GEMM) are locked behind a non-GGUF format and a bespoke decode kernel that is *not in this repo* and would forfeit Blackwell FP4 on sm_120. Correctly **deferred** by the bw24 decision docs — do not port.
- **The genuinely portable borrow is Q4 paged KV as an upper bound** on bw24's VRAM-vs-context tradeoff (`flash_attn.cu`), to be reached for only if longer laptop context forces it — and even then with bw24's asymmetric K>V bit budget, not blanket Q4.
- **EXL3's strongest contribution is architectural confirmation:** a leading edge engine hits its numbers with batch=1, dynamic-KV, single-stream — exactly bw24's design — and bw24's Rust runtime + optional CUDA-graph replay is structurally *ahead* of EXL3's Python eager loop on per-token overhead.
- **One concrete cheap lift exists:** the persisted autotune-result-cache pattern (the lone artifact present), worth adopting only if/when bw24 parametrizes kernel tiles per model.

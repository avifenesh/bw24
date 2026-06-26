# KTransformers — Implementation Map (for a single-stream sm_120 GGUF engine)

**Engine:** kvcache-ai/ktransformers — CPU/GPU **heterogeneous** MoE inference. The headline
lever is **routed-expert offload to CPU**: experts live in host RAM and the FFN GEMM is computed
**on the CPU** (AMX or AVX-512 int8 tiles), while attention / shared-expert / router / dense
layers stay GPU-resident. Placement (which expert on which device) is a **load-time static YAML
decision**, not per-token.

**bw24 host reality (verified this session via `/proc/cpuinfo`):** `avx2` + `avx_vnni` present,
**NO `avx512*`, NO `amx*`**. So KTransformers' fastest CPU-expert path (**AMX int8 tile GEMM**)
is **DEAD on this box**. The AVX2+VNNI fallback *runs* but is the slow path. bw24's own design
already diverges: it does **per-token GPU staging** (H2D the K selected experts, compute the GEMM
on GPU) instead of CPU compute — see `hybrid_forward.rs:233-282`. This map records what is still
worth lifting (the *placement / cache / scheduling* ideas), not the CPU-GEMM kernels.

GPU target context: RTX 5090 Laptop, **GB203 sm_120 (CC 12.0)**, 24463 MiB, ~896 GB/s GDDR7,
PCIe **Gen5 x8 ≈ 31 GB/s** ceiling, host CPU Core Ultra 9 275HX (`ARCHITECTURE.md:3`). Single
stream, batch≈1 decode is the regime that matters.

---

## What bw24 could take from ktransformers

Ranked by portability × single-stream value for an sm_120 GGUF engine. None of the CPU GEMM
kernels make this list — they are AMX-dead or VNNI-slow. The value is in the **offload
*architecture***, which bw24 has already started (`hybrid_forward.rs`) and can finish.

1. **Per-layer GPU "expert mask" → fast route-and-dispatch (HIGH).** KTransformers'
   `generate_gpu_experts_masks` produces a `[n_expert]` boolean per layer answering "is this
   expert resident?" in O(1). bw24 currently re-stages *every* selected expert *every* token
   (`hybrid_forward.rs:261/265/274` — no residency check). A per-layer residency bitmask is the
   exact data structure the **MISSING SLRU cache** (ARCHITECTURE gap D1, `ARCHITECTURE.md:92`)
   needs: `if resident(layer, ex) { use slot } else { stage + admit }`. **Highest-value port** —
   it is the seam that turns per-token PCIe from ~850 KB → ~0 after warmup.

2. **Static "hot-GPU / cold-CPU" placement as a *prior* for SLRU admission (MEDIUM-HIGH).**
   KTransformers' `frequency` / `front-loading` strategies (early/frequent experts → GPU) are a
   cheap **cache-warming prior**: pre-load the predicted-hot experts into the resident slots at
   model init so the first tokens don't all miss. bw24's SLRU is more adaptive per-token, but a
   static warm-start removes the cold-start cliff. Combine: static prior for *admission*, SLRU for
   *eviction*.

3. **Residency-keyed dispatch split (resident → GPU GEMM, miss → stage path) (HIGH).** The
   KTransformers control-flow `if expert_id in gpu_mask: gpu_gemm() else: cpu_dispatch()` maps
   cleanly onto bw24 as `if resident: qmatvec_view(slot) else: stage_expert()+qmatvec_view()`
   (`lib.rs:168` + `lib.rs:179`). Same branch, but the "else" stays on GPU (stage) instead of
   going to CPU. This is the single code change that converts Stage-1 (always-stage) into Stage-2
   (cache-aware).

4. **Contiguous pinned-host expert buffer, indexed by expert ID (MEDIUM).** KTransformers keeps
   CPU experts in contiguous pinned RAM indexed by `(layer, expert)`. bw24 already mirrors this:
   `HostExps.bytes` is one contiguous `Vec<u8>`, `expert_bytes(e)` slices
   `e*expert_stride..` (`model.rs:239-241`). The remaining port is **pinning** it
   (`cudaHostAlloc`) so the H2D in `stage_expert` (`lib.rs:171`) hits the ~31 GB/s Gen5 burst
   instead of pageable-memory speeds. Low effort, direct latency win on the staging path.

5. **VNNI int8 expert GEMM as a *measured* cold-expert fallback only (LOW, conditional).** Keep
   the AVX2+VNNI path on the table strictly for the **fetch-vs-recompute crossover** (ARCHITECTURE
   open-question #4, `ARCHITECTURE.md:199`): for a *very* cold expert, recomputing on CPU may beat
   an x8-PCIe fetch if the expert is large enough. Measure per model; default stays GPU-staging.
   Not a pillar.

6. **Placement strategy *names* as a config surface (LOW).** Exposing `uniform | frequency |
   front-loading | random` as a tunable lets bw24 A/B the warm-start prior cheaply without code
   churn. Pure config ergonomics.

---

## DEAD for bw24

| What | Why dead | Evidence |
|---|---|---|
| **AMX int8 tile expert GEMM** (`amx_*_moe.hpp`, kt-kernel onnx/amx backend) | Host has **no `amx*`** flags (verified `/proc/cpuinfo`: `avx2`, `avx_vnni` only). The 16×64×4 int8 AMX tile fabric does not exist on Core Ultra 9 275HX. | `ARCHITECTURE.md:92` ("KTransformers' headline AMX lever is gone"); `/proc/cpuinfo` |
| **AVX-512 int8 expert GEMM** (512-bit `vpdpbusd` path) | No `avx512*` flags on host. Only the narrower AVX2 256-bit VNNI variant is available. | `/proc/cpuinfo` (no `avx512f/avx512vnni`) |
| **CPU-compute as the *primary* expert path** | At batch=1 the CPU GEMM is memory-latency serialized + cache-thrashing (~5-10 ms/expert est.); bw24 measures GPU-staging (~27 µs H2D hidden under ~500 µs GPU compute) as the win. CPU compute is a fallback, not the design. | `hybrid_forward.rs:258-282`; `ARCHITECTURE.md:93` |
| **Multi-GPU / EP (expert-parallel) wrappers** (`kt_ep_wrapper.py`, SGLang EP layer) | bw24 is single-GPU, single-stream. Expert-parallel sharding across devices is multi-device server machinery with no batch=1 value. | `kt_ep_wrapper.py` (KT repo) |
| **Server / multi-request scheduling around the offload** | bw24 is single-stream; the throughput-oriented continuous-batching layer KT inherits from SGLang adds no batch=1 value. | (KT third_party/sglang) |
| **Static placement *as the only adaptivity*** | KT decides device once at init; bw24 re-derives top-K every token (`hybrid_forward.rs:234`), strictly more adaptive at low concurrency. Static-only placement is *not* portable as-is — use it only as a warm-start prior (idea #2). | `hybrid_forward.rs:233-282` |

> Note: KTransformers does **not** rely on wgmma/tcgen05 megakernels for the offload path (its GPU
> side is conventional GEMM + the CPU does the heavy MoE lifting). So the usual "wgmma/tcgen05-only
> = DEAD on sm_120" exclusion is less relevant here than for vLLM/SGLang; the dead weight here is
> **AMX/AVX-512 CPU kernels** and **multi-device/multi-request** machinery.

---

## Subsystem: CPU-Expert Offload (the whole engine, essentially)

| Technique | How implemented (mechanism) | Kernel / layout / instruction | source file:line | sm120_fit (RUNS/NEEDS-PORT/DEAD + single-stream value) |
|---|---|---|---|---|
| **CPUInfer AMX int8 expert GEMM** | Routed-expert FFN matmul runs **on CPU** via AMX int8 tiles. Experts live in CPU RAM; top-K router picks experts/token; CPU-side GEMM computes activation, returns result to GPU. INT8 weights, dynamic-q8 activations. | AMX **16×64×4 int8** accumulate tiles; experts staged into L2/L3 via buffer copy; micro-kernel `amx_*` from kt-kernel onnx/mlir backend | `kt-kernel onnx_kernels/amx-kernels/.../amx_*_moe.hpp` (AMX path); `kt-kernel python/experts_base.py` (placement) | **DEAD.** Host has no `amx*` (verified `/proc/cpuinfo`: `avx2`,`avx_vnni` only). Fastest KT path cannot run. Zero single-stream value on this box. |
| **AVX2 + AVX_VNNI int8 expert GEMM (fallback)** | INT8 GEMM on AVX2 base + `vpdpbusd` (8×int8→int32 dot-product, 256-bit). Outer loop over output rows, inner unroll-4/8 over input, accumulate int32, denorm to f32. Less parallel than AMX → wall-clock slower but correct. | `vpdpbusd ymm,ymm,mem` (8×int8→int32); INT8 weight `[out_f,in_f]`; q8_1 activation (int8 qs + per-32-block f32 scale); tile 64-128 out-rows/thread | `kt-kernel onnx_kernels/.../rawint4_avxvnni-moe.hpp` (VNNI micro-kernel) + int8 quantizer in expert-prep pipeline | **RUNS (feasible, slow).** Host has `avx2`+`avx_vnni` (verified). Loses to GPU-staging at batch=1: CPU int8 ~10-20 GB/s memory-bound + serialized vs PCIe x8 ~31 GB/s burst hidden under GPU compute. Keep as **measured cold-expert fallback only** (open-q #4, `ARCHITECTURE.md:199`). |
| **YAML static module placement (hot-GPU / cold-CPU)** | Load-time regex rules (`--kt-expert-placement-strategy uniform\|frequency\|front-loading\|random`) assign each module/expert to a device **once at init**. `generate_gpu_experts_masks` emits a per-layer `[n_expert]` bool: resident-on-GPU (0) vs CPU (1). | Per-layer GPU mask = `[n_expert]` u8/bitmask; route-time `if ex & gpu_mask == 0 → GPU GEMM else CPU dispatch`; CPU experts contiguous in pinned host RAM indexed by expert ID | `kt-kernel python/experts_base.py` (`generate_gpu_experts_masks`, strategy select); `third_party/sglang/.../moe/kt_ep_wrapper.py` (placement wrapper); `doc/en/kt-kernel/experts-sched-Tutorial.md` (design) | **RUNS but config-time (NEEDS-PORT to use well).** Static assignment works on any GPU. NOT ideal alone for batch=1: bw24 re-derives top-K every token (`hybrid_forward.rs:234`), strictly more adaptive. **Port the mask data structure** as SLRU residency bitmask + use the `frequency`/`front-loading` strategy as a **warm-start prior** for cache admission (`ARCHITECTURE.md:92` D1 SLRU). |
| **Residency-keyed dispatch split** | Control flow: resident expert → run GPU GEMM directly; non-resident → CPU dispatch (KT) / stage-then-GPU (bw24). The branch is per-expert, decided by the per-layer mask. | `if resident(layer,ex): gemm(slot) else: dispatch()` | `kt-kernel python/experts_base.py` (mask consume); KT moe wrapper dispatch | **NEEDS-PORT (high value).** Maps to bw24 `if resident: qmatvec_view(slot) else: stage_expert()+qmatvec_view()` (`lib.rs:179` + `lib.rs:168`). The exact change converting bw24 Stage-1 (always-stage) → Stage-2 (cache-aware). |
| **Contiguous pinned-host expert buffer** | KT keeps CPU experts in one contiguous pinned RAM region indexed by `(layer, expert)`; DMA source for any device-bound expert. | Contiguous host buffer, expert stride = total/n_expert, slice by expert ID | `kt-kernel python/experts_base.py` (buffer alloc) | **NEEDS-PORT (low effort).** bw24 already mirrors layout: `HostExps.bytes` contiguous, `expert_bytes(e)=&bytes[e*stride..]` (`model.rs:239-241`). Remaining port = `cudaHostAlloc` **pin** it so `stage_expert` H2D (`lib.rs:171`) hits Gen5 x8 burst. |

### bw24 counterpart (already implemented — divergence baseline, RUNS)

| Technique | How implemented (mechanism) | Kernel / layout | source file:line | sm120_fit |
|---|---|---|---|---|
| **EDGE-1 per-token routed-expert GPU staging** | Per token: softmax over 256 logits → stable DESC top-8 → renorm; for each selected expert `stage_expert()` async-H2D copies the ~256KB-1MB quant block into a GPU scratch buffer, then `qmatvec_view()` runs the resident-quant dequant-GEMM on GPU, `axpy_into()` accumulates `w[j]*y`. Router + shared expert stay GPU-resident. **No cross-token cache yet** (gap D1). | Host: `HostExps` contiguous quant bytes `[in_f,out_f,n_expert]`, `expert_stride=total/n_expert` (gate/up 860160, down 1114112). GPU: 3 scratch buffers sized for ONE expert, reused per token. Kernel `qmatvec_f32` (qmatvec.cu), dequant-on-fly → f32. | `hybrid_forward.rs:233-282` (moe loop); `lib.rs:163-173` (`stage_expert`); `lib.rs:179-189` (`qmatvec_view`); `model.rs:196-243` (`HostExps`) | **RUNS on sm_120.** Avoids CPU-compute bottleneck (KT's weakness on this box). Per-token H2D ≈ 8×~106KB ≈ 850KB / 31GB/s ≈ 27µs, hidden under ~500µs GPU compute. Fits 30B-A3B in 24GB where vLLM/SGLang OOM. **Next win = SLRU cache (D1)** → per-token PCIe ≈ 0 after warmup; this is where the KT mask/placement ideas (above) get lifted. |

---

## Bottom line

KTransformers' transferable IP for bw24 is **not its CPU kernels** (AMX-dead, VNNI-slow on this
host) and **not its placement *policy*** (static, less adaptive than bw24's per-token re-pick).
It is the **offload plumbing**: the per-layer expert-residency mask, the resident-vs-miss dispatch
split, the contiguous-by-expert-ID host buffer, and the idea of a cheap static *warm-start prior*
for cache admission. Those are precisely the pieces bw24 needs to close ARCHITECTURE gap **D1**
(the SLRU GPU expert-slot cache, `ARCHITECTURE.md:92`) and convert its current always-stage
Stage-1 loop into a cache-aware Stage-2 loop — turning per-token PCIe from ~850KB into ~0 after
warmup.

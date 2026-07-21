# SOTA Sweep — 2026-07-13 (targeted at the open goal fronts)

Five parallel sweeps (llama.cpp source+PRs; vLLM+SGLang releases/PRs; TRT-LLM+FlashInfer+NVIDIA;
spec-decode papers 2026-05→07, 71 arxiv hits triaged; decode-kernel papers/blogs). Judged against
the 2026-07-13 cell map: 31B spec depth 0.763x (THE gap), plain cells 0.97–1.10x, all other spec
cells cleared. Prior ledger: research/SOTA-ADOPTION-2026-07.md (July 4). Items already adopted or
wall-ledgered are excluded; closed-negative probes are cross-checked (PDL −24% MoE probe, waves
grid-cap, whole-round graph −10%).

## Verdict in one line

The field converged on bw24's own diagnoses (batch-1 decode is overhead-limited past ~85% wall;
long-context spec is verify-bound), and llama.cpp's depth-round advantage is STRUCTURAL — less
work per round, not faster kernels. Three arcs fall out, ranked.

---

## ARC 1 — 31B spec depth 0.763x (highest value; composite of 6 findings)

### 1a. K-sweep at depth + fixed-cap policy [ZERO CODE — measure first]
llama's whole depth policy is `n_draft = 3, p_min = 0` (server default), accept 0.90, round =
3 tiny drafts + one t=4 verify. Our K=7 + adaptive(accepted+1) was tuned on SHORT ctx; at depth
the marginal accept of deep draft positions decays (KVShot 2604.26412: hidden-state signal decays;
our KV-attending heads decay slower but still decay) while verify rows cost more. **Sweep 31B
depth K=2..5 + BW24_SPEC_ADAPT_FLOOR/cap variants before any code.** The same sweep may move 26B
depth further above its bar.

### 1b. llama round anatomy — the three structural mechanisms (HIGH PRIORITY, from source)
1. **Drafter feeds all draft tokens at the SAME position, no drafter KV, no catch-up decode**
   (gemma4-assistant driver: `common/speculative.cpp:1388-1449,1587-91`). We already share the
   target KV (no drafter cache ✓) but audit: any per-round drafter re-embed/catch-up we still do.
2. **Zero-update CUDA graph replay for EVERY decode in the round** (`ggml-cuda.cu:2398-2506`):
   per-shape graph cache + uid short-circuit; KV growth is DATA-driven not shape-driven — KV
   writes via `set_rows` with device index tensors, `n_kv` padded to 256-boundaries so shapes are
   stable for 256-token windows. No param patching at all.
3. **`flash_attn_mask_to_KV_max`** (`fattn-common.cuh:664`): tiny pre-kernel scans the mask and
   writes the true per-tile KV bound; FA loops to the device-side bound, so 256-padded shapes
   cost nothing at depth. This is what reconciles stable shapes with no padding waste.
   → bw24 port: pad the fa iteration extent to a 256 rung + device-side bound (we already have
   device len counters + the one-partition law — the missing piece is BUCKETED SHAPE STABILITY
   so verify/draft graphs replay without exec-updates for 256 tokens at a stretch). Pays on
   plain decode too.

### 1c. Sparse verify attention (Dustin 2606.24957 + SpecPV 2512.02337 + SSV 2605.19893)
Consensus 2026 line: at depth, verify = K+1 rows over deep KV dominates; fix = verify rows attend
a SELECTED KV subset. Selection signal = the DRAFTER's attention over target KV — which our MTP
heads compute for free. SpecPV adds the exactness knob: periodic full-KV verify resyncs drift.
SSV adds the kernel form: the K+1 rows share one KV-block selection, stream the union once.
Dustin: 9.17x e2e at 32k (batch 16, lossy). For bw24: no retraining, but greedy bit-exactness
becomes a cadence knob — gate behind a flag, arbitrate with stream-agreement + quality evals.
Real candidate only if 1a+1b leave a gap; at 1.7k the KV bytes are small vs weights, so this
pays mainly at 8k+ ctx.

### 1d. Depth-aware K cost model (DSpark 2607.05147 + Bastion 2605.29727 + greedy-acceptance
theory 2606.30265)
`k* = argmax_K E[accepted(K, margin)] / T(K, L)` with T profiled per (K, depth-bucket) — the
principled upgrade over accepted+1. DSpark's own gap: their cost model ignores context length —
depth-dependent theta is exactly our regime. Greedy-margin certificates (2606.30265) replace the
flat p-min. Position-level acceptance telemetry (vLLM #32374) is the cheap first step: log accept
histogram BY DRAFT POSITION at depth; if position 5+ accepts <30%, fixed-cap wins and 1a suffices.

### 1e. CUDA graph conditional nodes (NeMo ASR precedent; no LLM engine uses them)
Device-side WHILE/IF nodes (CUDA 12.4+, on 13.1 ✓) can gate draft-step subgraphs on a device
counter written by verify — skip-drafts inside ONE captured round, zero host round-trips. NeMo
ranks it their fastest decode mode. bs=1 = exactly the regime where the per-request objection
vanishes. This is round-graph PHASE 2 done right (our phase-1 fixed-shape round was −10%; the
conditional nodes remove the fixed-K draft waste device-side). Caveats: condition-set kernel per
iteration; no alloc/free nodes inside conditional bodies (our capture-time pool allocs need
persistent scratch first).

### 1f. Free acceptance for SAMPLED serving: block verification (vLLM #46781, arxiv 2403.10444v3)
Joint prefix acceptance ≥ per-token rejection sampling, provably lossless, one extra tiny kernel
with greedy one-hot drafts. Applies to the qwen p3 sampled lane (and any future gemma sampled
cells). Cheap, isolated.

### Guard rails (published negatives that confirm ours)
- Relaxed acceptance "unsuited for lightweight dedicated MTP drafters" (2607.08690) — don't.
- Tree-verify custom-mask attention dominates the critical path at depth (Saguaro, SwiftSpec) —
  chain-over-tree stays right for us; if coverage is needed, Graft-style retrieval SPLICE into
  rejected chain slots (2605.20104, training-free) instead of trees.
- Nobody ships true draft∥verify stream overlap; all "overlap" is scheduler-vs-forward.

---

## ARC 2 — plain cells last 3–10% (E4B 1.098x, 26B 1.057x, 31B 1.02x)

### 2a. Attention-phase L2 weight prefetch — "productive spin" (alpindale 5090 megakernel blog;
Kog monokernel) [HIGHEST-EV kernel experiment]
During attention/glue phases (bandwidth mostly idle), prefetch the NEXT matvecs' weight planes
into L2 (`prefetch.global.L2` from idle blocks/warps — or in our kernel-per-op world, a tiny
prefetch kernel enqueued before fa). Converts the 50–70%-efficiency small matvecs into
L2-residents. Same silicon evidence + measured NEGATIVES to respect: prefetch during saturated
phases −8.7%; `cp.async.bulk.prefetch` slower than `__ldg`-style; L2 evict-first on weights −8%
(kills the L2-policy idea for good). We have `prefetch_l2` machinery already (BW24_KV_PREFETCH,
flat for KV) — point it at WEIGHTS during the fa window.

### 2b. PDL re-probe on the DENSE glue chain (TGV data: at m=1..16 PDL = 1.17–1.33x while kernel
internals ≈ parity; llama.cpp shipped PDL fleet-wide with `GGML_CUDA_PDL=0` kill switch)
Our −24% probe was at MoE scale (prolog fought predecessor DRAM). The new evidence says PDL pays
exactly on TINY kernels (1–4µs glue class, E4B). Re-probe scoped to the dense glue chains only,
with llama's `__restrict__`-hoisting hazard fix (drop __restrict__ on PDL kernels). If it holds,
E4B's last 0.4 tok/s is likely here.
- llama impl reference: `common.cuh:1517-1640`, PR #22522 + #25185.

### 2c. sm_120 facts bank (constraints + tools)
- Thread-block CLUSTERS broken on SM120 (vLLM #47164: cudaErrorInvalidValue at 4/8/16) — never
  build on clusters/DSMEM here despite ClusterFusion++ claiming otherwise on 5090 (suspect).
- CLC (cluster launch control) + PDL + TMA available per Colfax sm_12x tutorial; CLC = the
  hardware answer to tail waves IF it works on consumer (needs a micro-probe; conflicts with the
  clusters-broken finding — CLC ≠ cluster launch).
- 128-bit max vector loads; `fence.acq_rel.gpu` cheaper than `__threadfence()`; desktop 5090
  measured 93% read ceiling; grid barrier ~2.2µs (persistent-kernel floor data).
- TRT-LLM `warpspec_sm120` FMHA README (PR #15163): NVIDIA's own written sm_120 attention recipe
  (TMA producer + mma.sync consumers — no wgmma) — read before any fa rewrite.
- FlashInfer v0.6.14 = sm_120 first-class: hd512 attention prefill+decode (gemma globals!),
  NVFP4 attention (immature — 3 math bugs fixed post-landing), W4A16 GEMM. Design references.

### 2d. Megakernel verdict (Ada-MK, MPK, AMK, Kog, Luminal, kog monokernel)
Honest band: +10–24% over an ALREADY-GRAPHED engine (Ada-MK vs TRT-LLM), and only when the
constituent GEMVs are already excellent (AMK's auto-GEMV = 63% of peak on our exact GPU — worse
than ours). Our GEMV layer ≥ published SOTA ("nothing in the window beats warp-per-row dp4a at
batch 1"). If ever attempted: static compile-time schedule (no on-GPU interpreter), sentinel
data-path sync (0.8µs vs 7.6µs counter barriers), decode-only. NOT NOW — the 2a/2b levers come
first and are 10x cheaper.

### 2e. MoE decode (26B): HPC-Ops small-batch pipeline (vLLM #45924 + blog)
Gather-free expert GEMM (read activations THROUGH the routing index), occupancy-first no-warp-spec
for memory-bound expert tiles, finalize folded into the residual add (SGLang #27720, 1–2% TPOT).
Our devq8 grouped-decode already covers most; the finalize-into-residual fold and gather-free
reads are the two deltas to check against moe_cache.

---

## ARC 3 — watch list / strategic

- **DSpark drafters for gemma-4** (llama OPEN PR #25549; deepseek-ai/dspark_gemma4_* checkpoints):
  published table beats the gemma4 MTP assistant on EVERY row (3.90–4.46x vs 3.44–3.99x, accept
  68–88%). When llama merges it, our gemma spec bar moves — and the same checkpoints are loadable
  by us (block drafter, one non-causal forward per draft block + confidence head). THE successor
  drafter class; plan a bring-up when checkpoints stabilize.
- llama E4B MTP FA fix (#25148: DKQ=DV=512 ncols2=2 instances) — their E4B assistant spec may now
  RUN upstream; re-check the llama E4B spec bar after their next release (our artifact-surgery
  gguf may load on a newer build too).
- DeepSeek V4 sparse-attention machinery in llama (LIGHTNING_INDEXER op, top-k KQ mask via
  set_rows) — the sparse-decode-attention lever for 8k+ contexts, same direction as ARC 1c.
- E8-lattice 2-bit KV (llama OPEN #25352) — alternative KV-compression arm vs our e4m3.
- VeriCache (2605.17613): draft-on-compressed-KV / verify-on-full = LOSSLESS spec at depth —
  the only exactness-preserving verify-cost idea; single-GPU adaptation = drafter attends the
  fp8 KV while verify uses full precision (inverse of our current split). Medium effort.
- Suffix decoding (vLLM, stable): model-free drafts from the prompt's suffix tree — composes
  with MTP as a zero-cost draft source at depth on repetitive content.
- TLI vocab-intersection (vLLM #38174): cross-model drafting generalization of our FR-Spec
  machinery — enables tiny-qwen-drafts-gemma experiments without retraining.
- NVFP4 board lane: FlashInfer sm_120 NVFP4 dense GEMM + attention landing = the qwen NVFP4
  publish front has fresher reference kernels than our July-4 map.

## Immediate action queue (ordered)
1. 31B depth K-sweep (K=2..5, fixed vs adaptive) + per-position accept histogram — zero code,
   possibly closes most of 0.763x→bar on policy alone. Same sweep on 26B depth for margin.
2. PDL re-probe on E4B dense glue (scoped; __restrict__ hazard handled) — the E4B last 0.4 tok/s.
3. Attention-window weight prefetch probe (E4B first, then 26B/31B) — the small-matvec lever.
4. Shape-stable graph windows (padded rungs + device KV bound à la KV_max) — kills our per-token
   exec-update need AND enables cheap per-decode replay in the spec round (llama's A2+A3).
5. Round-graph phase 2 via CUDA conditional nodes (device-gated draft steps) — after (4).
6. Block verification kernels for the sampled lane (cheap, isolated).

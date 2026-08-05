# DeltaServe assessment — idle-capacity LoRA co-serving, transfer to memra and the serving product

Date: 2026-08-03. Sources read in full (not abstracts): arXiv paper HTML + both GitHub repos.

## Identification (and the name collision)

**The paper the owner means:** *DeltaServe: Host-Agnostic Co-Serving of Inference and
Fine-Tuning for LLMs* — arXiv **2607.28848** (submitted 2026-07-30). Authors: Jiaxuan Chen,
Jianshu She, Ye Yuan, Rajat Ghosh, Karan Gupta, Qirong Ho, Xue Liu, Oana Balmau
(McGill / MBZUAI / Nutanix). Production trace is from Nutanix.
Code: `github.com/852866031/DeltaServe` (original, built on S-LoRA) and
`github.com/852866031/DeltaServe-vLLM` (vLLM V1 fork, updated 2026-07-15). Both cloned and
inspected (`/tmp/deltaserve-vllm-probe`).

**The collision is real:** *DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs*
(arXiv 2312.05215, Yao/Hu/Klimovic, EuroSys 2025) is the delta-compression serving paper;
its artifact repo is literally named `xiaozheyao/.deltaserve` ("delta serve temp") — a vLLM
fork. Unrelated mechanism (compressing fine-tuned model deltas 10x for multi-model serving,
2-12x throughput vs SOTA). Everything below is about 2607.28848, not DeltaZip.

Adjacent systems (context only, from the paper's own related-work):
- **LLMStation** (He et al., USENIX ATC 2025) — co-serves PEFT forward work inside
  decode-phase headroom only; per-shape latency cache that ignores CUDA-graph vs eager.
  DeltaServe's primary baseline.
- **FlexLLM** (Oliaro et al., NSDI 2026) — token-level chunked co-serving; per-chunk model
  reload cost is DeltaServe's stated critique.
- **S-LoRA / Punica** — the multi-LoRA batching substrate; multi-LoRA batching is the ONLY
  capability DeltaServe requires from a host engine.

## 1. Mechanism

**What idle it harvests — three kinds, in one design:**
1. **Temporal gaps** (no inference pending): issues fine-tuning-only forward steps.
2. **Prefill headroom within busy steps**: exploits the structural identity between
   inference prefill and the LoRA fine-tuning forward pass — each FT sample is admitted as
   an ordinary host-batch entry: "a single-step, prefill-only request routed to a reserved
   fine-tuning adapter and configured to produce no client-visible output." It folds into
   the host's existing multi-LoRA batch, costing ~one extra prefill sample, not a separate
   pass.
3. **Decode-phase compute slack** for the backward pass: backward runs in a **separate GPU
   subprocess under CUDA MPS** (separate CUDA context, per Orion's contention findings),
   reading activations via inter-process shared GPU memory, overlapping with memory-bound
   decode.

**Scheduling granularity:**
- Admission: **per scheduling step** (per engine iteration). A scheduler shim intercepts the
  host's proposed batch before dispatch, prices it with an analytical latency model, and
  greedily admits FT samples (shortest first) while the predicted step time stays inside the
  budget `Δ = min(TTFT slack, TPOT slack)` and the activation buffer has room.
- Preemption: **per transformer layer**. The backward subprocess checks a shared GPU-grant
  flag at every layer boundary; the inference process clears it before any prefill-carrying
  step, so "prefill can reclaim the GPU within one layer's worth of backward kernels" while
  decode co-runs with backward. FT-only forward steps have the same layer-boundary abort:
  partial activations are discarded and samples re-queued.

**The latency model** (their claimed novelty vs LLMStation): closed-form
`T ≈ α·Σ(n_i+B_d)² + β·(T_in+B_d) + γ·T_ft + ε·K + c` (attention-quadratic, FFN-linear,
FT-activation-capture, KV-read, fixed overhead), with a separate decode-only form, fit by
offline profiling and refined online. Crucially it carries **two coefficient sets — CUDA-graph
vs eager** — because a co-serving step is forced eager (activation-capture hooks can't run
inside a replayed graph), and graph-vs-eager differs "by nearly an order of magnitude at
small batch sizes."

**Interference cost (their numbers — note they report SLO-satisfaction fraction and average
e2e latency, NOT p95/p99 percentiles directly; the one tail stat is a "5% tail"):**
- Nutanix trace, 4xA100 (SLO 400ms TTFT / 120ms TPOT): **100% SLO compliance**, average e2e
  latency **+45%** over the split-pool reference (they frame this as by-design: harvesting
  the gap between execution time and the SLO deadline). LLMStation: 85% compliance, +97%.
- Burst-light: 100% SLO, avg latency +22% vs split-pool. Burst-dense: 100% SLO and **20%
  lower** avg latency than split-pool (co-serving uses all 4 GPUs when FT is curtailed;
  split-pool serves inference on only 3).
- Tail (interruption ablation, single 5090, idle-only mode): with layer-boundary
  interruption, avg latency **+0.7%**, 5% tail **+8%**; interruption disabled: 5% tail
  **+27%**. So the bounded-tail claim rests on layer-granularity preemption.
- Idle-only mode (no mixed batches, "DeltaServe-Temp"): avg e2e latency **+2%** vs bare
  vLLM while still getting 507 tok/s of FT throughput.

## 2. Results (actuals)

Setup: **Llama-3-8B** only. LoRA r=16, α=32, all attention projections, Alpaca, AdamW
lr=1e-3, backward batch 256 tokens. Hardware: 4x A100-40GB server, and a consumer box that
is literally **one RTX 5090 (32GB) + Ryzen 9 9950X** — their consumer eval rig is our
product silicon. SLOs: 400ms TTFT/120ms TPOT (A100), 200ms TTFT/100ms TPOT (5090).
Workloads: two 1-minute synthetic bursts + a 20-minute Nutanix production trace (avg load
~60% of provisioned capacity, i.e. ~40% GPU idle on average across six months of traces).

Fine-tuning throughput recovered (4xA100 unless noted):
| Workload | DeltaServe-vLLM FT tok/s | vs LLMStation | vs split-pool (3 GPU vLLM + 1 GPU torchtune) | SLO |
|---|---|---|---|---|
| Burst-light | 1,797 | 3.5x | +77% | 100% (all systems) |
| Burst-dense | 1,233 | 2.6x | +21% | 100% vs LLMStation 83% |
| Nutanix trace | 1,418 | **2.9x** | **+39%** | 100% vs LLMStation 85% |

Portability (single RTX 5090, scaled Nutanix trace): DeltaServe-**SGLang** 972 FT tok/s at
full SLO; DeltaServe-**S-LoRA** 484 FT tok/s at full SLO. Ablation (single 5090): idle-only
507 tok/s (+2% latency) → mixed-batch co-serving 934 tok/s (+84% FT throughput, higher but
SLO-bounded latency).

Memory cost: residual-stream activation capture ≈ `N_L·d·p` bytes/token ≈ **0.25 MB/token**
for an 8B (32 layers, d=4096, fp16). The repo's optimized variant saves more intermediates:
**~612 MB total at s_max=256 tokens** on Llama-3-8B, plus backward CUDA graphs
(gradients bit-identical to eager, gradcheck 111/111 per their tests).

Implementation size: host-agnostic core ~4,000 lines; vLLM hooks confined to ~a dozen files.
Requires multi-LoRA batching from the host — nothing else.

Caveats on their evidence: one model size (8B), short traces (1-20 min), no p99 reported,
no diurnal-scale eval, and LLMStation/FlexLLM comparisons are partly qualitative (FlexLLM
is not in the measured baselines). The 2.9x headline is against LLMStation on one trace.

## 3. Transfer to memra and the serving product

memra has **no LoRA support at all** (grep of `src/` — zero hits), no multi-LoRA batching,
no autograd, and serves **GGUF-quantized** weights through hand-written kernels. So the
"compact hook interface" is not compact for us — the one capability DeltaServe requires is
the one we don't have.

**Architecture-portable (ideas, not code):**
- **SLO-budget admission at step granularity.** memra already has the machinery this
  policy needs: a batched decode tick, an admission proxy with a cap, VRAM-aware admission,
  and per-step timing. Adding "predict step cost, admit background work only inside the
  TTFT/TPOT slack" is a scheduler policy, ~engine-agnostic. Their two-mode (graph/eager)
  pricing maps to our graph-decode vs eager paths.
- **Background work in a separate MPS process with a layer-boundary yield flag.** This is
  process-level, engine-agnostic, and is the piece that made their tails bounded (+8% vs
  +27% at the 5% tail). A co-located PyTorch training process that checks a shared grant
  flag per layer needs nothing from memra internals except the flag protocol.
- **Idle-only harvesting (their Temp mode).** 507 FT tok/s at +2% average latency without
  ever sharing a batch with inference. This variant needs no mixed batches, no forced-eager
  inference steps, and no activation hooks in the serving forward — only gap detection in
  the serve loop plus the preemptible subprocess.
- **The economics claim itself**: on the Nutanix trace, harvesting beat a *dedicated
  training GPU* by 39% at zero extra hardware. That is exactly the lab's "GPUs pay for
  themselves" shape.

**vLLM-specific / non-transferable plumbing:**
- FT-samples-as-prefill-requests through multi-LoRA batching — we have neither.
- Forward hooks capturing activations inside the engine's forward — memra's forward is
  hand-written CUDA; capturing bf16 residuals per layer is writable but it is kernel work,
  not a hook.
- Torch inter-process shared GPU memory for base weights/adapter/activations.
- **The shared-frozen-base assumption breaks on us**: DeltaServe's backward consumes the
  same bf16 base weights the server holds. memra serves quantized GGUF; a training
  subprocess cannot backprop through our Q8_0/NVFP4 blocks without either QLoRA-style
  dequant-backward (new lane) or a second bf16 copy of the model (VRAM-prohibitive on a
  32GB 5090 next to a served 27-35B).

**Exactness/QoS collision — the important one:** memra's isolation contract is
*byte-identical output under concurrent load*, gated by replaying prompts at c=1 vs c=16.
We already fixed an m-dependence defect (batched cuBLASLt router/gate GEMMs changing MoE
expert selection with co-arrivals; `MEMRA_ROUTER_PREFILL_EXACT`). Folding FT tokens into
inference batches changes GEMM m-dims — the *exact* class of defect we quarantined. So
DeltaServe's highest-throughput mode (mixed forward batches, +84%) is presumptively
incompatible with our exactness gates unless FT tokens ride m-invariant kernels. The
compatible subset is: **temporal-idle FT-only steps + MPS backward subprocess** — these
never touch an inference batch's composition, so bytes are safe; only *timing* is perturbed,
which is measurable against our p95 gates (their +0.7% avg / +8% 5%-tail is the number to
beat, and it must be re-measured on our stack, not assumed).

**How much true idle do we have?** Honest answer: **unknown, because darklanes has no
production traffic yet.** Our published numbers are load-test saturation (804.7 tok/s/GPU
at cap 16, c=96 fleet runs) — saturation by construction has no temporal idle. The two idle
sources in our regime would be (a) diurnal valleys, which only exist once real customers
exist and which dl-metering will measure, and (b) decode memory-bound compute slack, which
exists even at saturation but is only harvestable by *spatial* co-location (their MPS
backward overlap), the riskier kind for tails. The Nutanix datapoint (~40% average idle on
a real co-pilot service) is the base-rate argument that valleys will be material once
traffic exists — but it is their trace, not ours.

**What our fine-tune track actually needs:** the gated track is SFT distillation of a 35B.
Distillation splits into (a) *data generation* — teacher/student sampling, which is
**inference-shaped** and needs zero training machinery — and (b) *gradient steps*. The
lab's operating doctrine already routes research jobs ("data gen, distillation, evals, drafter
builds") as tenant #1 of the harvest lane and rents dedicated windows for gradient
training. DeltaServe's mechanism only adds value for (b), and (b) on a 32GB 5090 already
serving a big quantized model is VRAM-starved before it is compute-starved.

## 4. Verdict

**GO-later-with-trigger for the training co-location; GO-now for the cheap subset.**

**GO-now (no DeltaServe machinery needed):** a *priority-tiered admission lane* — background
inference-shaped jobs (distillation data-gen, evals, drafter scoring) admitted only when the
serve loop has temporal slack, preemptible at request granularity. This captures the
harvest-lane economics the owner already committed to, uses only scheduling-policy ideas
from the paper (SLO budget as an admission lever), touches no batch composition (exactness
safe), and requires no LoRA/backward/autograd work. It also produces the metering
instrumentation the GO-later trigger needs.

**GO-later (LoRA-training co-location, the actual DeltaServe mechanism):** premature today.
Triggers, all required:
1. **Real traffic with measured idle** — dl-metering over ≥2 weeks of production traffic
   shows >25-30% idle GPU-hours (the Nutanix base rate was ~40%); until then there is
   nothing to harvest but our own load tests.
2. **The fine-tune track is unblocked** on its own merits (distribution/traffic gate) and
   its gradient steps are LoRA-shaped — full-parameter 35B SFT does not fit this mechanism
   at all.
3. **A QLoRA-style backward story for quantized served weights** (or spare VRAM for a bf16
   base copy) — without one of these the backward subprocess has nothing to backprop
   through on our rigs.
4. **Timing-interference gate passes**: co-located MPS background work must hold our p95
   gates on-box (their +0.7% avg / +8% 5%-tail on a 5090 is encouraging but is their stack,
   their 8B, their 1-minute traces).

Rank order if/when triggered: (1) idle-only FT steps + preemptible MPS backward (exactness
safe), (2) mixed-batch folding only if an m-invariant FT path exists — otherwise never, it
conflicts with the isolation contract.

## Source ledger

- arXiv 2607.28848v1 (HTML, read in full): mechanism §3, Algorithm 1, all eval numbers §4.
  Local copy of extracted markdown: `/tmp/deltaserve-paper.md`.
- `github.com/852866031/DeltaServe` (original, S-LoRA-based) and
  `github.com/852866031/DeltaServe-vLLM` (vLLM V1 fork; cloned to
  `/tmp/deltaserve-vllm-probe`): activation-save inventory (~612 MB @ s_max=256), backward
  CUDA graphs (gradcheck 111/111), admission-phase config (`prefill` default / `both`),
  dedicated RTX 5090 install path in README.
- DeltaZip collision: arXiv 2312.05215v3 (abstract via arXiv API) + `xiaozheyao/.deltaserve`.
- memra context: `/home/avifenesh/projects/bw24/docs/SERVING.md` (isolation contract,
  VRAM-aware admission, fleet numbers), `src/` grep (no LoRA), the lab operating-model
  memory (harvest-lane doctrine, SKU rule).
- Not verified: LLMStation and FlexLLM papers were not fetched directly; their mechanisms
  are characterized here as DeltaServe's paper describes them.

# Training readiness — one box, both directions (design + research, CPU-only)

Date: 2026-08-05. Lane: training-readiness. Owner directive: "add training lane to
prepare the inference for both training and inference with the sota training" — in
service of the 2-card goal (serving consolidated on card 1, card 2 free for
research/TRAINING; memory `darklanes-launch-program-20260804`).

Inputs synthesized: `research/deltaserve-assessment-20260803/ASSESSMENT.md` (co-location
triggers), `research/finetune-sku-20260802/REPORT.md` (SFT/distillation verdict + teacher
path), `research/hw-growth-rethink-20260803/ASSESSMENT.md` + owner override (PRO
6000-homogeneous trajectory), `docs/DRAFT-REGIME.md` + `crates/memra-engine/src/eagle.rs`
(how drafters actually get built today), `docs/qwen38-bringup-runbook.md` (the house
conversion pipeline), `docs/SERVING.md` (x-lane QoS, metering),
`~/projects/darklanes/sft-pipeline/` (the private trace-corpus tooling),
`research/pro6000-prod-20260804/pro6000wk-runpod.jsonl` (the pod receipts). Web research
cited inline with dates. No GPU used.

---

## 0. Verdict up front

- **Stack pick:** torch (≥2.7 stable line, sm_120 official; current stable ~2.13) +
  **Unsloth on top of TRL/PEFT** for single-card LoRA/QLoRA SFT — Unsloth ships explicit
  Blackwell support including RTX PRO 6000 (blog 2026-05-17) and per-model Qwen3.5-family
  recipes with measured VRAM (27B bf16 LoRA = 56 GB; 35B-A3B bf16 LoRA = 74 GB — both fit
  the 96 GB card). Plain TRL+PEFT is the fallback when Unsloth's patching fights a hybrid
  arch. **SpecForge** for EAGLE3 drafter training when that lane opens. Full-FT of the
  27B is out of reach on the box and we say so (§1.4).
- **Loop gaps (train→convert→gate→serve, target hours):** the conversion, gating, and
  drafter tooling already exist and are runbook-grade; what's missing is (1) a scripted
  LoRA→merged-HF step, (2) a pinned sm_120 training environment, (3) an eval runner
  against a memra-served endpoint, (4) a candidate-model registration mode in the gate
  tables, (5) the standing law made explicit: **every finetune re-derives its own ranks
  and drafter** (DRAFT-REGIME law 1 — already stated in the doc, becomes a loop stage,
  ~1.5-2.5 h of the wall-clock). Total new tooling: days-class, mostly scripts (§2).
- **Co-location verdict:** the 2-card shape needs almost nothing built — device
  separation via `CUDA_VISIBLE_DEVICES` is the isolation mechanism; the real risks are
  host-side (dataloader CPU, checkpoint-write disk bursts, pinned host RAM) and each has
  a cap, not a design. The proof is one named measurement: the serve p95 A/B with card-2
  training on/off (§3.2). Single-card DeltaServe-style co-location stays GO-later behind
  the four triggers, restated verbatim (§3.3) — this design does not redesign it.
- **First experiment (§4):** LoRA-SFT a small adapter on **Qwen3.5-9B** (a supported
  board model with frozen baselines and a drafter pipeline) on the pod's PRO 6000, on a
  tiny public instruct set, ~200 steps; then run the ENTIRE loop — merge, convert to the
  house artifact, re-derive ranks + drafter, full gate battery, serve, endpoint eval —
  and the deliverable is the measured **loop wall-clock** (target ≤ 6 h) plus a
  gap-list of every manual step hit.
- **Business line (§5):** internal-first, honestly. Training capability's near-term value
  is (a) better SKUs (prune+heal, distillation, in-house EAGLE drafters), (b) the OR
  provider-application asset (proprietary model = prioritized application), (c) a
  differentiated **"fine-tune + serve" bundle** later — the one training product where
  darklanes has an edge nobody else ships (gate battery + per-finetune drafter rebuild =
  your custom model served with spec decode). Raw training-as-a-service on idle capacity
  is commodity (vast/RunPod set the price) — not a line to lead with.

---

## 1. Q1 — What training stack, concretely (96 GB PRO 6000, 2026)

### 1.1 The base layer: torch on sm_120 is a solved problem

PyTorch has shipped official sm_120/Blackwell support in stable builds since 2.7.0
(cu128; pytorch.org 2.7 release blog, Apr 2025 — Triton 3.3 with Blackwell +
torch.compile). The 2025-era "no kernel image" pain (github pytorch#159207, #164342) is
history; current stable is the 2.13 line (the llama.cpp conversion venv already carries
`torch 2.13.0+cpu`). The training env needs the cu13x wheel, one-time.

Attention-kernel reality on sm_120 (matters for training throughput expectations):
FlashAttention-4 explicitly does NOT cover SM120 — FA4 launched for SM80/90/100/110;
SM120 "reports arch 120 but uses SM80-era MMA instructions" (sorryhyunblog FA4-on-SM120
port notes, 2026-03-30; Spheron FA4 guide, 2026-04-21: consumer/workstation Blackwell
runs FA2-class kernels). So training attention = FA2/xformers built with
`TORCH_CUDA_ARCH_LIST="12.0"` (the exact build recipe is in Unsloth's Blackwell blog,
2026-05-17) or SDPA fallback. This costs some throughput vs an H100, not correctness —
and per the finetune-sku cost table the gradient phase is noise-cost anyway.

QLoRA gotcha ledger for Blackwell (all verified 2026-08-05):

- **bitsandbytes NF4 works on sm_120** (Linux; the open issue #1937 is Windows +
  CUDA-13.x wheel lag, 2026-05-05). bnb does NOT support NVFP4 tensor-core format —
  NF4 QLoRA and our NVFP4 serving artifacts are different 4-bit worlds (Spheron
  FP4-on-Blackwell, 2026-03-15). Training never touches the serving artifact's quant:
  QLoRA trains against an NF4-quantized copy of the BF16 checkpoint, and the merge
  output is BF16 (§2.1).
- **Unsloth supports Blackwell since 2026-05** — "RTX 50-series (5060–5090), RTX PRO
  6000, B200…" (unsloth.ai Blackwell blog, 2026-05-17); PyPI current (2026-06-22)
  advertises LoRA/QLoRA, full-FT, RL, FP8 training.
- **MoE QLoRA is not recommended by Unsloth** — for 35B-A3B use bf16 LoRA (74 GB, fits);
  their 2026 MoE update claims ~12x faster MoE training, router layer frozen by default
  (Qwen3.5 fine-tune guide, 2026-05-15; Qwen3 guide, 2026-04-02).
- **The 27B is a hybrid GDN model** (`layer_types` linear_attention×3 : full_attention,
  runbook §2) — training rides transformers' native `qwen3_5` classes (the same
  `Qwen3_5ForConditionalGeneration` our converter reads) + the flash-linear-attention
  kernels for the GDN blocks; FA2 applies only to the full-attention quarter. Unsloth
  lists 27B with measured VRAM, so their patching handles it — but this is the highest
  compat-risk arch on our list; the pilot (§4) deliberately starts on the 9B.

### 1.2 The framework pick, per SKU we care about

| Track | Model | Method | Framework | VRAM on 96 GB (evidence) | Verdict |
|---|---|---|---|---|---|
| SKU SFT / prune-heal | Qwen3.6-27B (dense-hybrid) | bf16 LoRA r=16 | **Unsloth** (TRL/PEFT under it) | 56 GB measured by Unsloth at seq 2k → ~40 GB headroom = seq 8–16k @ bs 1–2, or bs 4 @ 2–4k | GO |
| SKU SFT, long-context / bigger batch | Qwen3.6-27B | QLoRA NF4 r=16–32 | Unsloth (bnb NF4) | base ~14–16 GB + activations; bs 4–8 @ 4k, or seq 32k @ bs 1–2 w/ unsloth checkpointing | GO — the workhorse config |
| Distillation student | Qwen3.6-35B-A3B (MoE) | bf16 LoRA (router frozen) | Unsloth MoE path | 74 GB measured by Unsloth → fits, seq 2–4k @ bs 1–2; QLoRA explicitly not recommended for MoE | GO, tight |
| Distillation student, KD-logits variant | 35B-A3B | trace-SFT first; logit-KD only if trace-SFT plateaus | TRL `GKDTrainer` class | teacher can't co-reside (V4-Flash ≈ 145 GB NVFP4 — finetune-sku §4); offline top-k logits or API traces only | trace-SFT is the plan of record |
| Drafter training | EAGLE3 head for any SKU | SpecForge | **SpecForge** (ICML 2026; target–draft decoupling, EAGLE3 first-class, SGLang-ecosystem) | head is ~1 layer — trivial VRAM; the cost is target-model hidden-state generation (inference-shaped → harvest lane) | GO when a SKU lacks a usable MTP head |
| Rank/trim "training" | every model | frequency counting, NO gradients | in-house `frspec-owngen` | serving VRAM only | already shipped |
| Full-FT | 27B | — | — | ≥162 GB (54 weights + 54 grads + ≥54 adamw_8bit states, before activations) | **out of reach on one card; marginal-at-best on 192 GB with offload — not the regime. Distill/LoRA instead.** |

Why Unsloth over axolotl/torchtune/LLaMA-Factory as the primary: single-GPU is exactly
its design point (2–5x step-time vs plain TRL+FA2 in current comparisons — TechAIApp
framework comparison, 2026-07; Spheron framework comparison, 2026-03-05), it publishes
per-model VRAM for the exact Qwen3.5/3.6 family we serve, and it carries the Blackwell
build recipe. Axolotl earns its keep at multi-GPU production scale — which for us is
rented-window territory (finetune-sku: gradient phase on rented 8xH100 = $100–420 per
1B tokens), not the 2-card box. Keep plain TRL+PEFT as the no-magic fallback: every
Unsloth recipe degrades to it with the same configs.

### 1.3 How the own-gen trim regime "trains" ranks today (question answered)

No gradients anywhere in the shipped drafter path. `frspec-owngen` generates a corpus
from the serving model's OWN outputs (greedy, chat template on, ≥4× topN tokens),
**counts token frequencies**, and emits a rank file; `tools/make-trimmed-draft.sh`
extracts the MTP block byte-verbatim from the serving GGUF and trims the head to the
top-N ranks (DRAFT-REGIME laws 1–3). The EAGLE lane (`eagle.rs`, `run_eagle`) consumes
an externally-trained checkpoint (`eagle3-qwen35-9b/model.safetensors`) — experimental,
not on the daily path. So the first real gradient-training consumer memra has is
**drafter training via SpecForge** for SKUs whose published GGUFs strip the MTP head
(Ornith, KAT — today served via donor blocks at an acceptance cost): train an own
EAGLE3 head on own-gen data, gate it with the existing run-spec battery. That is a
concrete, internal, revenue-adjacent training workload for card 2.

### 1.4 What full-FT means for us: rent it or don't do it

Stated plainly per the task: 27B full-FT does not fit the box (§1.2 table). The
finetune-sku math already priced the honest alternative — rented 8xH100 windows at
$16–24/hr make the gradient phase noise next to teacher-trace costs. The 2-card box's
training role is LoRA/QLoRA SFT, prune-heal, drafter training, and rejection-sampling /
eval scoring (inference-shaped) — which is everything the GO-later distillation plan
actually needs on owned hardware.

---

## 2. Q2 — The memra interop seam: train (torch) → convert → gate → serve, in hours

### 2.1 The loop as it exists vs the loop as it must be

What already exists and is runbook-grade (docs/qwen38-bringup-runbook.md — measured
~8–11 h wall for a FULL new-model bring-up including board cells):

- **Conversion**: house llama.cpp fork, `convert_hf_to_gguf.py` (qwen3_5 classes, MTP
  sidecar, NVFP4 repack), venv pinned with torch-cpu. 1–2 h for a 27B.
- **Gate battery as acceptance test**: fast-gate tier-0 → run-gen argmax (two depths) →
  kernel-check → run-spec K=1..8 → serve-st/serve-smoke → local-ci. This IS the
  acceptance test the task asks about; nothing new to design, only to parameterize
  (candidate-model mode, below).
- **Drafter rebuild**: `frspec-owngen` + `make-trimmed-draft.sh`, 1.5–2.5 h chunked.
  DRAFT-REGIME already states the law this loop inherits: "a finetune's distribution
  moved, so its draft must too." A finetuned model that skips this stage serves with a
  stale drafter and silently loses spec throughput — the loop treats ranks+drafter as
  mandatory, not optional.
- **FP8-ST leg**: if the finetune's merged checkpoint is re-exported as official-style
  block-128 FP8, the ST serving leg applies unchanged (no conversion at all) — but
  quantizing our own FP8 export is a new tool; day-one the GGUF leg carries finetunes.

What's missing (the gap list, each item scoped):

| # | Gap | What to build | Class |
|---|---|---|---|
| 1 | **LoRA merge pre-conversion** | `tools/merge-lora.py`: PEFT `merge_and_unload()` → `save_pretrained` (bf16, safetensors, copies tokenizer/config/chat template). memra has zero LoRA support (deltaserve assessment: grep-verified) and llama.cpp GGUF-adapter files don't help us — **merge-then-convert is the only path**. ~50 lines. | hours |
| 2 | **Pinned training environment** | one container/venv spec for the pod: torch cu13x + unsloth + trl + peft + bitsandbytes + xformers/FA2 built `TORCH_CUDA_ARCH_LIST=12.0` (recipe: Unsloth Blackwell blog). Version-pin like the conversion venv. Lives in the private repo (training tooling is product work — owner call, commit b8ca4e2e). | hours, one-time |
| 3 | **Endpoint eval runner** | scripted eval against a memra-served `/v1` endpoint: (a) regression pack = the board prompt packs + goldens diffing (behavior drift visible), (b) task eval = lm-eval `local-chat-completions` class or the private `verify_outcome.py` for agentic traces. Without this, "did the SFT help" is vibes. | days |
| 4 | **Candidate-model gate registration** | fast-gate `models.tsv`/`map.tsv` rows under a `cand-*` id + a rule: candidates never refresh goldens. Today registration assumes a supported model; a finetune needs the battery without the publish side-effects. | hours |
| 5 | **Imatrix regeneration for candidate quants** | the NVFP4 recipe needs a fresh imatrix from the finetuned model (runbook: "new imatrix from the new model, NOT the 3.6 one") — script the imatrix run + quantize as one step. For pilot-speed loops, Q8_0 skips imatrix entirely (the Q8 arm is the fast lane). | hours |
| 6 | **Loop driver** | one script chaining 1→convert→5→gates→drafter→serve-smoke→3 with tee'd logs into `research/<lane>/` (evidence discipline). The runbook is the spec; this mechanizes it for the finetune case. | days |

With 1–6 in place the loop wall-clock for a 27B-class adapter is bounded by existing
stage timings: merge ~15 min (54 GB write) + convert/quant 1–2 h + gates ~1 h + drafter
1.5–2.5 h + eval ~1 h ≈ **4–7 h** — hours, not days, on one card. The 9B pilot halves
that (§4).

### 2.2 Two laws the loop must carry over from serving doctrine

1. **Evidence discipline applies to training runs**: loss curves, configs, seeds, and
   the adapter hash are receipts, committed (or private-repo'd) next to the eval rows.
   A finetune whose training log exists nowhere is not evidence — same rule as sweeps.
2. **The battery gates the artifact, not the checkpoint**: acceptance is on the
   converted, quantized, drafter-equipped serving artifact — argmax MATCH, spec
   self-consistency, serve smokes — because that's what customers touch. An adapter
   that evals well in torch but fails the battery does not ship (and the battery
   failing IS the interop seam doing its job).

---

## 3. Q3 — Co-location mechanics on the 2-card box

### 3.1 Card 2 training while card 1 serves: what actually needs building — very little

Device isolation is `CUDA_VISIBLE_DEVICES=1` on the training process — separate CUDA
contexts, separate VRAM, no MPS needed, no engine change. The x-lane QoS gate and the
admission proxy already own card-1 SLOs (SERVING.md; measured fleet-scale p95 numbers
in research/qos-p95-20260802). The real contention surfaces are host-side, each with a
cap rather than a design:

- **CPU**: dataloader workers + tokenization. Cap: `dataset_num_proc`/`num_workers`
  bounded (SFT text loading is trivially cheap; Unsloth's own recipes default
  `dataset_num_proc=1`) and, on any shared rig, a `systemd-run --scope -p CPUQuota=` or
  `taskset` cage around the trainer — the standing no-uncapped-CPU rule generalized to
  the pod.
- **Disk I/O**: checkpoint saves are the burst (tens of GB for merged saves; adapter
  saves are MBs). Cap: adapter-only checkpoints during training + `save_steps` sparse +
  checkpoint dir on a different volume than the serving models' mmap source.
- **Host RAM / pinned buffers**: torch pinned-memory dataloaders + memra's bounded
  pinned host buffers coexist; the pod's 256-vCPU class host is not the constraint.
  Cap: `pin_memory=False` if the measurement (below) shows pressure.
- **PCIe**: SFT text batches are KB/s-scale H2D; serving H2D on a different device/root
  port. Only bulk events (initial model load, merged-save) contend — schedule-level,
  not steady-state.

### 3.2 The measurement that proves p95 holds (the named gate)

**Gate: `train-colo-p95`** — on the 2-card box, serve the daily SKU on card 1 under the
standing contended-load form (admission proxy, c=96 harvest + c=4 interactive, the
qos-p95-20260802 protocol), while card 2 runs the §4 pilot's exact training job.
Interleaved A/B, N=3 per arm: {training ON, training OFF}, same session, temps recorded.

Pass criteria (all three):
1. contended interactive p95 delta (ON vs OFF) ≤ 5% and within the arms' own spread;
2. engine-truth decode-step p99 (`/yield/metrics`) delta ≤ 5%;
3. **exactness unchanged**: the c=1 vs c=16 byte-identity replay passes identically with
   training ON — co-location may cost timing, never bytes (the deltaserve assessment's
   exactness/QoS collision analysis; card-2 training never touches card-1 batch
   composition, so this should pass by construction — measured anyway).

Run it once when the second card exists; re-run whenever the training workload class
changes (SFT → drafter-training → anything with heavier dataloading). Until a 2-card
box exists, the single-card pod means training and serving TIME-SHARE via the existing
`flock /tmp/gpu5090.lock` discipline — that's scheduling, not co-location, and it's how
the §4 pilot runs day one.

### 3.3 Single-card DeltaServe-style co-location: triggers restated, not redesigned

GO-later, all four triggers required (deltaserve-assessment §4, verbatim substance):
1. real traffic with measured idle — dl-metering over ≥2 weeks showing >25–30% idle
   GPU-hours (Nutanix base rate ~40%);
2. the fine-tune track unblocked on its own merits and its gradient steps LoRA-shaped;
3. **a QLoRA-style backward story for quantized served weights** (or spare VRAM for a
   bf16 base copy) — note §1's finding sharpens this trigger: bnb-NF4 QLoRA exists and
   works on our silicon, but it backprops through an NF4 copy, not through our served
   NVFP4/Q8_0 blocks; on a 96 GB card next to a served 27B (16–29 GB artifact) there
   IS spare VRAM for a second quantized training copy — the trigger is closer than the
   32 GB-era assessment assumed, but the p95/exactness gates still decide;
4. timing-interference gate passes on our stack (their +0.7% avg / +8% 5%-tail is their
   stack, their 8B — re-measure, never assume).

The GO-now tier (inference-shaped background jobs in serving slack via the harvest
lane) is already shipped as the x-lane QoS gate — training-readiness adds nothing to it
and takes nothing from it.

---

## 4. Q4 — First concrete experiment: the loop, end to end, timed

**Name:** `lane/train-loop-pilot`. **Goal:** not a better model — a measured wall-clock
for train→convert→gate→serve→eval on the pod, plus the definitive gap list. The model
quality bar is "the SFT visibly took"; the deliverable is the loop.

| Field | Spec |
|---|---|
| Rig | standing RunPod PRO 6000 96 GB (pro6000wk-runpod; receipts jsonl exists). All GPU steps under the rig's flock discipline. Training capped: `dataset_num_proc` bounded, adapter-only checkpoints. |
| Base model | **Qwen/Qwen3.5-9B** (BF16 HF checkpoint — the HF parent of the supported q9 board model; smallest supported model with a drafter pipeline and frozen baselines). *(No 1.7B exists in the supported set; the 9B is the smallest model where the full loop — MTP extraction, ranks, spec gates — is exercised for real.)* |
| Method | Unsloth bf16 LoRA r=16, alpha=16, target modules q/k/v/o/gate/up/down, unsloth gradient checkpointing, adamw_8bit, seq 2048, bs 1 × grad-accum 4 (the Unsloth Qwen3.5 reference config, 2026-05-15). ~22 GB VRAM — deliberately small so the card could in principle co-serve; fallback TRL+PEFT if Unsloth fights the arch. |
| Data | a tiny public instruct set with an unmistakable style/format marker (e.g. a 1–2k-sample slice of a permissively-licensed instruct dataset with a fixed response scaffold), so "did it take" is a string-level check, not an eval suite. Own-gen traces explicitly NOT used — the pilot must not entangle with the gated corpus lane (CORPUS.md: no training from that lane without owner spend approval; this pilot's training cost is ~zero and uses public data only). |
| Steps | 200 optimizer steps (~800 samples seen). Training wall on the card: tens of minutes. |
| Loop stages timed individually | (1) train; (2) `merge-lora.py` → merged BF16 HF dir; (3) convert: `convert_hf_to_gguf.py` → Q8_0 arm (skip imatrix/NVFP4 on the pilot — Q8 is the fast lane; NVFP4 leg exercised in a follow-up); MTP sidecar per the runbook; (4) gates: fast-gate tier-0, run-gen argmax pp22 + depth (MATCH required), kernel-check, run-spec K=1..3 with stage-5's drafter; (5) drafter: `frspec-owngen` own-gen ranks on the FINETUNED model (chat template ON, bounded chunks) + `make-trimmed-draft.sh` + `--validate`; (6) serve: memra-server, serve-smoke, one chat request showing the SFT marker; (7) eval: regression prompt pack vs the base 9B same-session (drift visible and bounded) + the marker check. |
| Acceptance | ALL of: argmax MATCH both depths; run-spec self-consistency PASS K=1..3 with acceptance > 0 on the rebuilt drafter; serve-smoke green; SFT marker present in served output; base-model regression pack shows no structural degeneration; **total loop wall-clock recorded stage-by-stage, target ≤ 6 h** (over → the gap list explains which stage and why); every stage's raw log tee'd into `research/train-loop-pilot-<date>/`. |
| Explicit non-goals | model quality claims; board rows; publishing the artifact; touching the sft-corpus lane; co-location (single card — flock time-sharing). |
| Follow-ups it unblocks | the 27B QLoRA variant of the same loop (the SKU-relevant size); the NVFP4+imatrix leg; SpecForge drafter-training pilot for a donor-block model (AgentWorld/Ornith — replace the donor-block acceptance tax with an own EAGLE head); the train-colo-p95 gate when card 2 exists. |

---

## 5. Q5 — What darklanes sells from this (honest)

Market context carried from finetune-sku + the website spec: a no-brand provider's own
fine-tune captures ~nothing (the author category "provider that fine-tuned a model" has
zero traffic precedent; funded-lab ceiling on this backbone class = $300–600/day gross);
the pricing page already defines interactive/harvest/dedicated lanes; OR's provider
backlog explicitly prioritizes proprietary models.

1. **Internal-only, now (the real line):** training capability makes the existing
   business better — prune-heal (REAP-25 + agentic-SFT-heal, the finetune-sku GO-later
   variant whose payoff is concurrency/KV headroom per box even unlisted), distillation
   students when the triggers fire, and in-house EAGLE drafter training (direct tok/s
   on served SKUs — spec throughput IS the product). Card 2's training hours convert to
   serving margin, not to a training product. This needs no market and starts working
   the day the box exists.
2. **The option asset, held cheap:** a credible proprietary fine-tune flips the OR
   provider application into the prioritized category (finetune-sku §Q1). Training
   readiness keeps that option exercisable in days instead of weeks. Hold it; don't
   build the SKU on faith.
3. **"Fine-tune + serve" bundle, later (the differentiated product):** train the
   customer's LoRA on their data, merge, run the full gate battery, **re-derive their
   drafter**, and serve the result on a dedicated lane. The differentiator is real and
   ours alone: nobody else rebuilds a per-finetune speculative drafter and hands you
   exactness receipts — the customer's custom model runs at spec-decode speeds with a
   gated determinism story. Fits the existing dedicated-lane pricing shape (per-replica
   per-hour). Trigger: real dedicated-lane demand + the §2 loop proven at ≤ a day
   turnaround. Constraint to state honestly: no multi-LoRA serving means each finetune
   is a full artifact on a dedicated replica — this is a premium low-N product, not an
   adapter marketplace.
4. **Raw training-as-a-service on idle capacity: no.** Renting out card-hours for
   generic training competes head-on with vast/RunPod at commodity prices with zero
   differentiation and single-founder ops load. The idle capacity's highest-margin
   consumer is tenant #1: our own research jobs (operating-model doctrine).

---

## 6. Source index

**In-repo/receipts:** deltaserve-assessment-20260803/ASSESSMENT.md;
finetune-sku-20260802/REPORT.md; hw-growth-rethink-20260803/ASSESSMENT.md (+ owner
override); docs/DRAFT-REGIME.md; docs/qwen38-bringup-runbook.md; docs/SERVING.md
(x-lane QoS, metering); crates/memra-engine/src/eagle.rs;
research/pro6000-prod-20260804/pro6000wk-runpod.jsonl; research/qos-p95-20260802
(protocol form); darklanes-website-spec-20260804/SPEC.md (§7.3 pricing lanes, §9
anchors); ~/projects/darklanes/sft-pipeline/CORPUS.md (private; trace pipeline +
training-spend gate); bw24-unified commit b8ca4e2e (training tooling → private repo,
owner call); memories darklanes-operating-model, darklanes-launch-program-20260804.

**Web (fetched 2026-08-05):** unsloth.ai Blackwell/RTX-50 blog (2026-05-17: RTX PRO
6000 supported; xformers TORCH_CUDA_ARCH_LIST=12.0 recipe); unsloth.ai Qwen3.5
fine-tune guide (2026-05-15: family VRAM table — 27B 56 GB, 35B-A3B 74 GB bf16 LoRA;
MoE QLoRA not recommended; ~12x MoE update); unsloth Qwen3 guide (2026-04-02, router
frozen); unsloth PyPI (2026-06-22); pytorch.org 2.7 release blog (sm_120/cu128 official)
+ pytorch#164342/#159207 (the 2025 gap, closed); sorryhyunblog.vercel.app FA4-on-SM120
(2026-03-30: FA4 ships SM80/90/100/110, not SM120) + spheron.network FA4 guide
(2026-04-21: consumer Blackwell = FA2-class); spheron.network FP4-on-Blackwell
(2026-03-15: bnb NF4 ≠ NVFP4); bitsandbytes#1937 (2026-05-05, Windows/CUDA-13 wheels);
arXiv 2603.18567 + ICML 2026 poster (SpecForge: EAGLE3 training framework, up to 9.9x
faster EAGLE-3 training); arXiv 2605.29343 (Draft-OPD, uses SpecForge); TechAIApp
framework comparison (2026-07); spheron.network axolotl-vs-unsloth-vs-torchtune
(2026-03-05); zenvanriel.com Qwen-27B QLoRA VRAM floor (2026-05-06).

# Dev-platform verdict: RTX PRO 6000 Blackwell (AWS G7e) vs rented 5090s vs local-only

Date: 2026-08-02. Lane: `lane/dev-platform`. Web-researched (every external claim carries a
source + date) plus in-repo receipts. Nothing in this report was measured today — §3 defines
the validation-day experiment; the empirical anchors quoted are existing repo evidence.

**Owner question (verbatim intent):** development should happen on rented cloud GPUs matching
the product silicon; "we decide if rtx 6000 on aws is good enogh for the similarity betwen the
hardware or we need 5090 otherwise we are doing double job."
**Owner decision rule (2026-08-02):** AWS is preferred (below-list pricing) IF the RTX 6000
Blackwell class is silicon-similar to the 5090 end goal; vast.ai 5090s only if similarity fails
or AWS doesn't carry the card.

---

## Verdict up front

**Silicon identity: YES — same family, same ISA, same compiled kernels.** RTX PRO 6000
Blackwell is GB202 — the same die as the desktop RTX 5090 — compute capability 12.0, the same
`sm_120a` target our build system emits for every 5090. Our fatbins are arch-exact SASS, and
this repo has already run them **unchanged** on this exact card: the dead G7e research box WAS
an RTX PRO 6000 Blackwell Server Edition, and its 2026-07-04 bring-up row
(`research/tune-data/g7e-rtx6000.jsonl`) records kernel-check ALL GREEN, run-gen argmax MATCH
with logit maxdiff identical to the same-day rig5090 run, and spec K=1..8 PASS.

**AWS carries it: YES.** EC2 G7e (launched 2026-01-20, us-east-1 + us-east-2) = RTX PRO 6000
Blackwell Server Edition, 1-8 GPUs per instance, local NVMe, GPUDirect P2P. Under the owner
rule, **AWS G7e is the pick**; vast.ai 5090 is the fallback/cross-check tier, not needed as
primary.

**The honest asterisk:** ISA identity is not perf-verdict identity. The proof rig is the RTX
5090 **Laptop** (GB203, 82 SM, 858 GB/s measured, ~150 W) — a rented desktop 5090 would ALSO
be a different die (GB202), 2x the bandwidth, and 4x the power budget. Nobody rents our exact
product silicon. The box-era J/token law ("kernel verdicts do not transfer across power
walls", HANDOVER.md:219) applies to *both* rental options roughly equally, and the standing
rule already bounds it: perf defaults re-gate on the local 5090 before shipping. §3 defines
the test that measures how small that residual double-job actually is.

---

## 1. Silicon identity

### 1.1 The card matrix

| | **RTX 5090 Laptop** (local proof rig) | **RTX 5090 desktop** (vast.ai rentals) | **RTX PRO 6000 Blackwell WS Ed.** | **RTX PRO 6000 Blackwell Server Ed.** (AWS G7e) | RTX 6000 **Ada** (NOT similar) |
|---|---|---|---|---|---|
| Die | GB203 | **GB202** | **GB202** | **GB202** | AD102 |
| Compute cap / arch | 12.0 / sm_120a | 12.0 / sm_120a | **12.0 / sm_120a** | **12.0 / sm_120a** | 8.9 / sm_89 |
| SMs / CUDA cores | 82 / 10,496 | 170 / 21,760 | 188 / 24,064 | 188 / 24,064 | 142 / 18,176 |
| L2 | 64 MB | 96 MB | **128 MB** (deviceQuery receipt) | 128 MB | 96 MB |
| Memory | 24 GB GDDR7, 256-bit | 32 GB GDDR7, 512-bit | 96 GB GDDR7 ECC, 512-bit | 96 GB GDDR7 ECC, 512-bit | 48 GB GDDR6 |
| Bandwidth | 896 GB/s spec, **858 measured** (repo) | 1,792 GB/s | 1,792 GB/s | **1,597 GB/s** (AWS spec) | 960 GB/s |
| Boost clock | 2,715 MHz spec (150 W TGP) | 2,407 MHz (575 W) | 2,617 MHz (600 W) | **2,430 MHz max measured on the G7e box** (600 W cap) | — |
| FP4/FP8 block-scale MMA (5th-gen TC) | yes | yes | yes (4,000 AI TOPS FP4) | yes | **no FP4** |

Sources: NVIDIA/press launch specs — GB202, 24,064 cores/188 SM, 96 GB GDDR7 ECC 512-bit,
600 W, 4,000 AI TOPS ([wccftech launch piece](https://wccftech.com/nvidia-rtx-pro-6000-blackwell-launch-flagship-gb202-gpu-24k-cores-96-gb-600w-tdp/), 2025);
L2 128 MB + CC 12.0 + 188 SM from an actual deviceQuery on the card
([Fixstars Max-Q teardown](https://blog.us.fixstars.com/what-kind-of-gpu-is-the-nvidia-rtx-pro-6000-blackwell-max-q/), fetched 2026-08-02 — some spec DBs echo "98 MB"; the deviceQuery receipt wins);
5090 desktop 170 SM / 96 MB L2 vs PRO 6000 full-die-ish config
([Chips and Cheese GB202 analysis](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu));
Server Edition bandwidth 1,597 GB/s ([AWS G7e product page](https://aws.amazon.com/ec2/instance-types/g7e/), fetched 2026-08-02);
5090 Laptop GB203 / 82 SM / 64 MB L2 / 24 GB / 896 GB/s / 2,715 MHz / 150 W
([VideoCardz.net](https://videocardz.net/nvidia-geforce-rtx-5090-laptop-gpu) / eatyourbytes DB, fetched 2026-08-02);
G7e measured clocks: `g7e-rtx6000.jsonl` bring-up row ("idle 180MHz, max 2430MHz, 600W cap").
RTX 6000 Ada = AD102/sm_89 ([Arnon Shimoni's arch table](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)) —
that is our compile-gated legacy arch (`MEMRA_CUDA_ARCH=89`, portable eval only); anyone
quoting "RTX 6000" prices must check the generation — DigitalOcean for example rents only the
Ada one ($1.57/hr, [DO pricing](https://www.digitalocean.com/pricing/gpu-droplets), fetched 2026-08-02).

### 1.2 nvcc / compute_cap: what the PRO card reports and what our build does

- **The PRO 6000 (all editions) reports compute_cap 12.0** — same as every RTX 50 card.
  Receipts: Fixstars deviceQuery (CC 12.0, above); vLLM tracks it as SM120
  ([vllm#31085](https://github.com/vllm-project/vllm/issues/31085)); and decisively, **our own
  G7e box**: `crates/memra-engine/build.rs::detect_arch()` reads
  `nvidia-smi --query-gpu=compute_cap` and maps `"12.0" -> "120a"`, `Engine::new` refuses to
  init when device CC != built arch, and the 2026-07-04 bring-up ran the rig-built sm_120a
  fatbins **unchanged** through the full battery. SASS is arch-exact — sm_120a SASS cannot
  even load on a device of any other CC, so ALL GREEN on the G7e card is proof of identity,
  not just compatibility.
- **The `a` suffix is not "consumer-only".** `compute_120a/sm_120a` is the arch-specific
  feature target for ALL CC 12.0 silicon — it is the only form that assembles the FP4/FP8
  block-scale MMA (`crates/memra-probe/build.rs` comment: bare `-arch=sm_120a` misroutes to
  compute_120). GB202 workstation silicon carries the identical tensor-core feature set.
- **Known red herring:** [pytorch#157549](https://github.com/pytorch/pytorch/issues/157549)
  (July 2025) titled the card "sm_122 / CC 12.2" from a user's PyTorch error string. Every
  deviceQuery receipt since — including ours — says 12.0. Do not let that issue seed a
  "different arch" belief on validation day; if a rented box ever prints anything but 12.0,
  that is a driver/VM anomaly to capture, not a card property.
- Auto-detect on a G7e box therefore produces the **byte-identical naked sm_120a build** with
  zero flags — the same binary contract as the local rig (README.md: "the naked sm_120a
  build is byte-for-byte the tuned 5090 engine").

### 1.3 What "similar" does and doesn't mean here

Identical: ISA, tensor-core generation (FP4/NVFP4, FP8, int8 dp4a, mma.sync shapes), compiled
SASS, exactness behavior (bring-up row: 27B logit maxdiff 3.402e-1 **identical** to the
same-day rig5090 run — the numeric class transfers bit-for-bit).

Not identical: perf shape. vs the local proof rig the G7e card is 2.29x the SMs (188/82),
1.86x the bandwidth (1597/858), 4x the power, 2x the L2, and clocks ~10% lower. The repo has
already paid for this lesson once: the same grouped-MoE change measured **4.8-7.2x on G7e vs
1.48-1.95x locally** (HANDOVER.md:1270-1271 — G7e's 188 SMs idle on matvecs, the laptop
clocks higher), and the box era produced the standing law that kernel verdicts do not
transfer across power walls (HANDOVER.md:219). Directionality mostly transferred; magnitudes
and thresholds did not. That is exactly the class of risk §3's test quantifies — and note a
rented **desktop** 5090 (170 SM, 1792 GB/s, 575 W, GB202) sits on the same side of the power
wall as the G7e card, not on the laptop's side. Renting 5090s does not buy proof-rig identity.

---

## 2. Cloud availability, August 2026

### 2.1 AWS — EC2 G7e (the RTX PRO 6000 Blackwell Server Edition family)

Launched 2026-01-20, initially US East (N. Virginia) and US East (Ohio)
([AWS News Blog](https://aws.amazon.com/blogs/aws/announcing-amazon-ec2-g7e-instances-accelerated-by-nvidia-rtx-pro-6000-blackwell-server-edition-gpus), fetched 2026-08-02).
On-Demand, Spot, Savings Plans. Specs from the [G7e product page](https://aws.amazon.com/ec2/instance-types/g7e/) (fetched 2026-08-02):

| Size | GPUs | vCPU | RAM | Local NVMe | Net | List $/hr (us-east-1) |
|---|---|---|---|---|---|---|
| g7e.2xlarge | 1 | 8 | 64 GiB | 1.9 TB | 50 Gbps | **$3.36 OD** ([DevZero](https://www.devzero.io/instances/aws/g7e.2xlarge), 2026-08) |
| g7e.4xlarge | 1 | 16 | 128 GiB | 1.9 TB | 50 Gbps | $4.00 OD ([DevZero](https://www.devzero.io/instances/aws/g7e.4xlarge)) |
| g7e.8xlarge | 1 | 32 | 256 GiB | 1.9 TB | 100 Gbps | not fetched |
| g7e.12xlarge | 2 | 48 | 512 GiB | 3.8 TB | 400 Gbps | **$8.29 OD / $3.23 spot** ([Vantage](https://instances.vantage.sh/aws/ec2/g7e.12xlarge), fetched 2026-08-02) |
| g7e.24xlarge | 4 | 96 | 1024 GiB | 3.8 TB | 800 Gbps | not fetched (OD scales ~linearly in-family) |
| g7e.48xlarge | 8 | 192 | 2048 GiB | 3.8 TB | 1600 Gbps EFA | ~$33.14 OD ([Thunder Compute](https://www.thundercompute.com/blog/nvidia-rtx-pro-6000-pricing), 2026-08) |

GPUDirect RDMA + P2P supported; Vantage lists the GPU as "NVIDIA GB202" outright. Spot on the
2-GPU size is **~$1.61/GPU-hr** at list. **All list prices are an upper bound here — the owner
has negotiated below-list AWS pricing** (rate not assumed in this report). Operationally this
is a *restore*, not new work: the repo still carries the `g7e/*` remotes, `lane/g7e`, and the
`g7e-rtx6000.jsonl` rig log to append to.

### 2.2 Other RTX PRO 6000 Blackwell rentals (context / BATNA)

All fetched 2026-08-02 by the pricing sweep unless noted: RunPod $1.69 Community / $1.99
Secure ([runpod.io/pricing](https://www.runpod.io/pricing)); vast.ai tracks both SKUs —
Workstation Ed. 789 units, median $1.09/hr; Server Ed. 421 units, median $1.60/hr
([500.farm exporter](https://500.farm/vastai-exporter/gpu-stats), 2026-08-02T12:56Z);
CoreWeave 8x-only instance $2.50/GPU-hr OD, $1.39 spot ([coreweave.com/pricing](https://www.coreweave.com/pricing));
GCP G4 ~$4.50/hr; Azure NC-series v6 ~$5.50/hr ([getdeploying](https://getdeploying.com/gpus/nvidia-rtx-5090), updated 2026-08-02);
Lambda: none; DigitalOcean/Paperspace: none (Ada only). So even without the AWS discount, the
AWS list premium buys EBS/NVMe/EFA/on-demand reliability over marketplaces; with the discount
the question closes.

### 2.3 The direct alternative: rented consumer RTX 5090s

From the 2026-08-02 pricing sweep (live vast.ai API + aggregators):

- **vast.ai**: deep market — 7,475 RTX 5090s tracked (2,323 available now); on-demand median
  **$0.45/hr** (p10 $0.32), interruptible typically $0.20-0.27 and floor ~$0.10
  ([500.farm](https://500.farm/vastai-exporter/gpu-stats), 2026-08-02; [vast.ai/pricing](https://vast.ai/pricing)).
  **Multi-GPU consumer hosts exist**: live offers included 4x 5090 from $1.12/hr *total*
  (~$0.28/GPU-hr) on EPYC 9654 / Threadripper 7970X hosts with measured PCIe 26-55 GB/s, and
  one 8x 5090 at $3.63/hr (EPYC 9354). Reliability scores on live offers 0.935-0.998; the
  community record is mixed (Trustpilot ~4.1/232; 2026-06-29 HN story on a mislabeled-geography
  host; billed-but-broken startup complaints). Storage is **pinned to the physical host** — a
  host going dark strands the volume ([vast docs](https://docs.vast.ai/instances/storage)).
- **RunPod**: $0.69/hr Community, $0.99/hr Secure ([runpod.io/pricing](https://www.runpod.io/pricing));
  network volumes $0.07/GB/mo make it the most dev-box-shaped marketplace; $120M ARR
  ([TechCrunch](https://techcrunch.com/2026/01/16/ai-cloud-startup-Runpod-hits-120m-in-arr-and-it-started-with-a-reddit-post), 2026-01-16).
- **Others**: Salad $0.29/hr but batch-container-only (no persistent dev box); TensorDock 5090
  unconfirmed on their own site; Hyperbolic/Lambda/Cudo: no 5090 ([getdeploying](https://getdeploying.com/gpus/nvidia-rtx-5090), 2026-08-02).

---

## 3. The double-job test (defined, not run)

Purpose: measure, on the rented card, whether memra's *decisions* — not just its bytes —
match the 5090 rig. One G7e day, ~6 hours GPU time. Stage models onto the instance-local NVMe
first (never bench off EBS — standing rule). Append every row to
`research/tune-data/g7e-rtx6000.jsonl` (rig id `g7e-rtx6000-sm120-188sm` already exists);
tee every raw log under `research/dev-platform-20260802/` per evidence discipline.

### Stage 0 — identity gate (minutes)

```sh
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv   # expect: RTX PRO 6000 Blackwell Server Edition, 12.0, ~97xxx MiB
cargo build --release 2>&1 | grep MEMRA_CUDA_ARCH                   # expect: "auto-detected 120a (compute_cap 12.0)"
./target/release/kernel-check                                       # expect: ALL GREEN (382 checks)
```
Any other compute_cap or any kernel-check red = STOP, capture, report. (Per §1.2 this
combination already passed on this card class on 2026-07-04.)

### Stage 1 — exactness transfer (the "same silicon" proof, ~30 min)

```sh
MEMRA_MODELS_DIR=/scratch/hf-models tools/local-ci.sh               # kernel-check + run-gen argmax + VERIFY-GATE + spec self-consistency
```
Acceptance (all bit-deterministic, must be EXACT):
- run-gen token shas equal the rig5090 mode-2-class anchors: q35 `e94b6553fde7b9a0`,
  KAT `9102ffd0b8241a65`, o35b `c0c12c3b350dc7f5` (from `research/f16g-default-rearb-20260802/RESULTS.md`).
- run-spec K=1..8 PASS, and acceptance percentages match the local values exactly (the
  2026-07-04 G7e row already recorded K1..K8 = 82.4/53.3/51.3/40.4/32.3/26.9/23.1/20.2 on the
  synthetic [55] prompt — identical draft/target argmax means identical acceptance).

### Stage 2 — dispatch-decision parity (the actual question, ~3 h)

The tuned defaults encode per-rig *choices*. H100 chooses differently on several of them, so
they discriminate "5090-class card" from "merely CUDA-compatible card". Interleaved arms,
same-minute round-robin, N>=5 per arm (the sk-bm128 clock-drift protocol), one engine on the
GPU at a time. Models/prompts as in `research/f16g-default-rearb-20260802/run-headline.sh`
(q35 = Qwen3.6-35B UD-IQ4_XS, KAT IQ4_XS, o35b Q4_K_M; prompt `research/e2e/prompts/board-2048.txt`).

| # | Decision under test | Command arms (q35 board-2048 pp-only unless noted) | 5090 pick (must reproduce) | H100 pick (failure signature) |
|---|---|---|---|---|
| a | grouped-f16 mode arbitration | naked (=2) vs `MEMRA_MOE_F16G=3` vs `=1` vs `=0` | **mode 2** (sk visitor + direct tiles) wins; also on KAT | mode 1 (cublas grouped) wins |
| b | sk visitor crossover | `MEMRA_F16G_SK_CROSS={16,32,64,128}` | argmax at **64** (max one step off) | 32 |
| c | sk kernel form | `MEMRA_F16G_SK={0,32,128}` vs unset (hybrid) | hybrid >= best single form | — |
| d | router batch crossover | `./target/release/router_batch_bench` t-sweep | crossover at **t=8** (`ROUTER_BATCH_MIN_T`, lib.rs:207; t=4 below 1x, t=8 above) | same const; verify anyway |
| e | MMQ split-K | `MEMRA_MMQ_SK=0` vs naked | SK **on** wins (sm_120a rig-divergence law, mmq_ffi.rs:719) | — |

### Stage 3 — bandwidth-scaled numbers (~2 h)

```sh
MEMRA_MODELS_DIR=/scratch/hf-models tools/local-ci.sh --perf        # the standing cell battery -> perf-ci.jsonl rows
```
plus the q35/KAT/o35b/g26 headline cells (pp board-2048, gen512/NGEN=128), N>=5 medians, power
state recorded, same-day rig5090 rows as the denominator (never cross-day — clock-drift law).

Acceptance bands (G7e / local-5090 ratio, stated so a miss is unambiguous):
- **decode (bandwidth-bound):** predicted 1597/858 = **1.86x**; accept **1.5x-2.2x** (+/-20%).
- **prefill (GEMM-bound):** predicted SM*clock ratio (188x2430)/(82x2715) ~= **2.05x**; accept
  **1.55x-2.6x** (+/-25%; q35 prefill has non-GEMM Amdahl components that damp the ratio).
- spec acceptance %: exact match (Stage 1); spec *tok/s* rides the decode band.

### Verdict rules

- **PASS (no double job):** Stage 0-1 exact + arms (a) and (b) reproduce the 5090 picks +
  bands hold => daily dev, kernel iteration, and dispatch tuning all run on G7e; the local
  5090 keeps only the standing pre-ship final gate (kernel-check + argmax + spec + board
  re-gate) — which it keeps under EVERY outcome, so this residual is a constant, not a
  double job.
- **PARTIAL:** exactness green but a band misses or (c)/(e) flips => correctness/feature dev
  transfers 1:1; threshold-class tunings (crossovers, floors) get swept per-rig — the repo
  already treats thresholds as re-sweep-on-rig-move items (H100 laws #2).
- **FAIL:** (a) or (b) lands the H100-class pick => the card arbitrates like a different arch
  despite the ISA; keep G7e for correctness + multi-GPU + big-VRAM work only, and move perf
  lanes to a rented desktop 5090 (vast.ai) — then the §4 risk table applies to that card
  instead.

Prior evidence says PASS/PARTIAL is the likely outcome (exactness already proven; magnitude
divergence already observed once), but per evidence discipline the verdict waits for the runs.

---

## 4. Recommendation

**Pick: AWS G7e.** Rationale, in the owner's decision order:

1. **Similarity holds at the level that was in question.** Same GB202 die family, compute_cap
   12.0, the same sm_120a fatbins byte-for-byte — already proven on this exact card class by
   this repo's own July bring-up battery. The RTX 6000 *Blackwell* on AWS is emphatically not
   the RTX 6000 *Ada* trap (sm_89).
2. **AWS carries it and the owner pays below list.** g7e.2xlarge $3.36/hr OD list (upper
   bound) for daily single-GPU dev; spot 2-GPU at ~$1.61/GPU-hr list. Plus the boring wins a
   dev box needs: 1.9-3.8 TB local NVMe, EBS persistence, stable identity, and the existing
   G7e remotes/rig-log/workflow from July.
3. **Multi-GPU dev (PP-2/PP-4) is in-family:** g7e.12xlarge (2 GPU, 400 Gbps, GPUDirect P2P)
   and g7e.24xlarge (4 GPU) — same card, same build, no host-quality lottery. AWS workstation
   sizes are NOT single-card-only in this family, contrary to the usual pattern.
4. **The 96 GB is a feature, not a similarity bug,** for exactly the lanes CLAUDE.md already
   assigns off-rig: Hy3 expert banks, spill research, artifact generation — while the 24 GB
   local card stays the spill-realism and ship-gate rig.

**Similarity risk carried by each option, honestly stated:**

| Option | $/GPU-hr (2026-08-02) | Similarity risk | Ops risk |
|---|---|---|---|
| **AWS G7e** (pick) | $3.36 list OD, below-list actual; $1.61 list spot (2-GPU) | perf-shape gap vs laptop proof rig (188 SM / 1597 GB/s / 600 W vs 82 / 858 / 150) — bounded by the §3 test + standing local final gate | lowest; region us-east-1/2 |
| vast.ai 5090 (fallback) | $0.45 median OD; 4x hosts ~$0.28-0.67 | desktop 5090 is *also* not the proof rig (GB202 / 170 SM / 1792 GB/s / 575 W) — same ISA as G7e, similar-magnitude shape gap, so it does NOT eliminate the double job it appears to | host lottery, host-pinned storage, mislabeled hosts; mitigate via >=0.99-reliability hosts |
| RunPod 5090 / PRO6000 | $0.69-0.99 / $1.69-1.99 | as above per card | better persistence than vast, weaker than AWS |
| local-only | sunk | zero | serializes research against the 24/7 proof rig — the exact problem this decision solves |

**Bottom line:** the "or we need 5090" branch of the owner's question dissolves under the
spec sheet — the rentable 5090 is the desktop card, which is silicon-identical in ISA and
*non*-identical in shape in the same ways the G7e card is. Given below-list AWS pricing, G7e
dominates: run §3 on day one of the new box; expected outcome is dev-transfers with the local
5090 keeping only the ship gate it was always going to keep.

**Cost sketch (list, upper bound):** ~6 h validation day on g7e.2xlarge ~= $20; a heavy dev
month (8 h/day x 22 d) ~= $590 list before discount — spot and Savings Plans cut further;
PP-2 days on g7e.12xlarge spot ~= $26/day list.

---

*Not done here: no benchmarks were run (this is a define-the-test deliverable); G7e pricing
for 8xl/24xl not fetched; region expansion beyond us-east-1/2 unverified since the January
launch post. Marketplace prices are point-in-time 2026-08-02 snapshots.*

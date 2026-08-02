# Compressed / quantized KV cache — best results, best practices, and the memra mapping (2026-08-02)

Lane `lane/kv-compress` (from `restructure/public-split` c969ffcf). Survey-only — NO GPU work
in this lane. Every external claim below carries its source and date; every internal claim
carries its repo receipt. Written 2026-08-02/03.

Framing contract (memra): **argmax-identical serving is the product promise.** "Exact tier"
below means the decode-path reads are the same numeric config that the exactness battery
gates (kernel-check + run-gen argmax + run-spec) — quantized KV *is* exact-tier once gated as
the standing numeric config, because exactness is defined against the shipped config, not
against a BF16 oracle. "Lossy tier" means a method that changes outputs vs our gated config
and would need explicit disclosure (OR `quantizations` label norms — see
`research/or-provider-20260802/REPORT.md`).

---

## 1. Quantization: state of practice, per format

### 1.1 What the big three ship (2026)

| Engine | Default | Opt-in | Published quality evidence |
|---|---|---|---|
| vLLM | BF16 KV | `--kv-cache-dtype fp8` (e4m3, per-tensor scale=1.0 uncalibrated; per-head scales + LLM-Compressor calibration optional; `--kv-cache-dtype-skip-layers sliding_window` for hybrids) | vLLM blog 2026-04-22: MRCR to 1M ctx, AIME25/GPQA/MATH500/LiveCodeBench — FP8 KV **plus FP8 attention math** recovers 94–99% across the battery; "ready to be the default starting point for many long-context deployments" but NOT flipped as default |
| SGLang | BF16 KV | `--kv-cache-dtype fp8_e4m3 / fp8_e5m2` (store-time quant, dequant-at-read in most paths) | docs-level "minimal degradation" claims; sgl issue #10083 (2025-09) tracks finer-grained enhancements |
| TRT-LLM / ModelOpt | per-model quant config | INT8 KV, FP8 KV (production-standard per NVIDIA), **NVFP4 KV** (Blackwell; dequant to FP8 before attention math) | NVIDIA blog 2025-12-05: NVFP4 KV <1% loss on LiveCodeBench/MMLU-Pro/MBPP/RULER-64K (Qwen3-Coder-480B); NVFP4 beats MXFP4 by ~5 MMLU pts (block-16 + e4m3 scales); FP8 KV called "well utilized in production" |

Nobody defaults below 16-bit KV *globally* in 2026; FP8 is the universal recommended opt-in,
and it is the effective default in serious long-context deployments (KGA field report
2026-04-22: E5M2 recommended production default, −0.3% to −0.6% aggregate on 70B-class at
half the bytes, 1.8x session capacity). memra is *ahead of that curve on bytes*: our
always-on q8_0/q5_1 (58 B per 32-elem K+V block = **45.3% of BF16**) is smaller than
fp8-flat (50%) and gated bit-exact-in-config since day one.

### 1.2 The two accuracy laws vLLM's 2026 validation nailed down

Both directly relevant to us (source: vLLM blog "The State of FP8 KV-Cache and Attention
Quantization", 2026-04-22):

1. **Accumulation precision, not storage precision, was the long-context killer.** Hopper FA3
   FP8 dropped 128k NIAH from 91% → 13% — traced to imprecise intra-tensor-core FP32
   accumulation when the contraction dim (= context length) reaches 100k (same HW issue as
   DeepSeek-V3 training, tech report Fig 7b). Fix = SageAttention2-style two-level
   accumulation into a real FP32 register → 89%. Blackwell does not have the issue.
   Memra note: our FA decode accumulates f32 in registers with an order-pinned B3 chain
   (fa-decode-deep RESULTS §2) — we are structurally on the safe side of this law, and it is
   the reason "quantized-KV storage + f32 math" holds argmax where "FP8 math throughout" needs
   surgery.
2. **Hybrid models: skip the sliding-window layers.** Quantizing bounded-window KV pays
   overhead for no long-context benefit (gpt-oss-20b FP8 break-even was 741k tokens before
   skip-SW; 7.7k after). Memra discovered the same law independently and *sharper*: our
   serving-keyed `MEMRA_GEMMA_WKV` (windowed layers q8/q5 under spec serving because e4m3
   windows gut drafter acceptance .758→1.000 inverted, FLAGS.md) is the acceptance-law
   refinement of vLLM's skip-SW — theirs is a perf argument, ours is a numerics argument.

### 1.3 Format ladder and scaling granularity (what survived to 2026)

- **INT8 / q8-class (per-block or per-token scale)**: effectively lossless on every published
  eval; llama.cpp community KLD measurements (r/LocalLLaMA 2026-03-23) and the original
  Ollama-adoption measurements (smcleod.net 2024-12) put q8_0 KV at +0.002–0.05 PPL.
  This is our K default. No engine reports a quality incident with 8-bit KV.
- **FP8 e4m3 (per-tensor → per-head scales)**: the 2026 production workhorse. Uncalibrated
  scale=1.0 is the deliberate worst case in vLLM's battery and still recovers 94–99%;
  calibration (LLM-Compressor) is the escape hatch for models that show a systematic shift
  (Kimi-K2.5 + FlashMLA is their documented example). E5M2 vs E4M3 field split: E5M2 for
  long-ctx/multilingual range, E4M3 for code/math mantissa (KGA 2026-04-22).
- **INT4 / NVFP4 (block scaling)**: the 2025-26 mover. NVIDIA NVFP4-KV: 16-elem blocks +
  e4m3 scales, <1% loss incl RULER-64K, and the honest mechanism note that values are
  **dequantized to FP8 before math** — the storage format and the math format decoupled,
  exactly our fused-dequant architecture. MXFP4 (32-elem, power-of-2 scales) measurably
  worse (−5 MMLU) — block granularity + scale precision is the whole game at 4-bit.
- **K vs V asymmetry**: KIVI's finding (ICML 2024) stands unchallenged in 2026: **K quantizes
  per-channel (outlier channels), V per-token**. At 8-bit the asymmetry barely matters (our
  q8_0 K is per-32-block, fine); at ≤4-bit it is decisive — every surviving 2-bit method
  (KIVI, KVQuant, RotateKV, KITTY) is built on it. Also the reverse asymmetry we exploit:
  **V tolerates fewer bits than K needs** at the formats we run (q5_1 V vs q8_0 K default);
  our own 2026-07-08 measurement — q4_0 V = argmax MISMATCH in-config — marks where the V
  ladder stops for the exact tier on our stack.
- **RoPE-aware K quant**: KVQuant (NeurIPS 2024) quantizes K **pre-RoPE** (outlier channels
  are clean before rotation) and applies rotation on the fly; RotateKV (IJCAI 2025) uses
  outlier-aware rotations + attention-sink-aware protection. Consensus 2026: pre-RoPE or
  rotated K is what makes ≤3-bit K work; at 8-bit post-RoPE (what we and llama.cpp do) is
  free of measurable cost. RoPE-aware bit allocation is still an active frontier
  (arXiv 2606.24033).
- **Outlier handling lineage**: KIVI/KVQuant's channel outliers → 2026 production form is
  simply *per-head or per-channel scales* (vLLM PR #30141/#30833; SGLang per-channel scale +
  per-token shift). The exotic sparse-outlier storage of KVQuant did not productionize.

### 1.4 Quantization vs everything else — the 2026 matched-budget verdict

"Quantization Dominates Rank Reduction for KV-Cache Compression" (arXiv 2604.11501,
2026-04): at *matched storage budgets* across 5 models, INT4 KV ≈ FP16 (+0.18–0.23 PPL at
75% reduction) while rank reduction at the same bytes collapses (Mistral-7B: rank-32 = +34.8
PPL; LAMBADA 57x accuracy gap). Mechanism: softmax attention is a routing system — deleting
a dimension flips top-token routing (discrete failure, 4.6% flip rate), bounded quantization
noise doesn't (0.03% at INT4); formalized as a 3·2^2b damage ratio. GQA *amplifies* deletion
damage (one deleted direction misroutes several query heads). Practical floor: the paper and
KITTY (arXiv 2511.18643) agree the instability returns below ~3 bits, where noise exceeds
typical score gaps — KIVI-2bit visibly drops on Qwen3-8B math/GPQA.

**Takeaway: 4-bit block-scaled quantization is the quality-per-byte frontier for stored KV;
low-rank projection retrofits lose at every matched budget. Only architecturally-trained
latents (MLA) escape this — because they are trained, not projected post-hoc.**

---

## 2. Compression beyond quantization

### 2.1 Token eviction / selection (H2O → SnapKV → 2026)

Research state: enormous (H2O, SnapKV, PyramidKV, TOVA, StreamingLLM, R-KV, KNorm, KVzip,
DynamicKV, LookaheadKV…). The two honest 2025-26 assessments:

- **Reasoning traces** (arXiv 2512.12008, 2025-12): heavy-hitter tracking (H2O, SnapKV-Decoding)
  dominates on reasoning models, occasionally beating full cache; BUT no single strategy wins
  across task types on non-reasoning models, and tight budgets *lengthen* reasoning traces
  (a hidden inference-cost tax that "KV bytes saved" tables never show).
- **Multi-turn / shared context** (SCBench, Microsoft, ICLR 2025): **sub-O(n)-memory methods
  (eviction) fail in multi-turn scenarios** — the second query needs KV the first query's
  eviction policy threw away. Sparse *encoding* with O(n) memory is robust; sparse *decoding*
  with sub-O(n) memory is not. This is the death sentence for eviction on agentic serving,
  where every session is multi-turn by construction.

Production adoption 2026: **near zero in mainline engines.** vLLM/SGLang/TRT-LLM ship no
eviction policy as a serving default or supported opt-in; the ecosystem's home is NVIDIA
kvpress (research library, not a serving path). DeltaKV (2026-02) states the integration
reason plainly: per-layer/per-head heterogeneous budgets break paged allocators.
KVzip (NeurIPS'25 oral) is the strongest of the class — query-agnostic, reconstruction-scored,
3–4x reduction, reusable across queries (fixes the SCBench failure mode by design) — the one
to watch if we ever build a lossy tier, but still no engine ships it.

### 2.2 Cross-layer sharing (CLA / YOCO)

Training-time architecture decisions, not retrofits: CLA (MIT 2024) 2x KV reduction
at near-parity *when trained in*; YOCO's single-KV-bank reuse likewise. 2026 systematic study
(NAACL 2025 short) confirms sharing patterns work but every production instance (Gemma's
GQA+SWA stacking, Hunyuan-class) baked it at pretraining. **Not a serving lever for us —
it's a model-selection checkbox** (models with SWA/CLA-class layers already give us the
windowed-KV byte win; our wkv machinery serves it).

### 2.3 Low-rank / latent (MLA)

- **Native MLA** (DeepSeek lineage, GLM-5.2): the real thing — >70% KV reduction vs
  equivalent-dense because the latent is *trained*. We already carry a bring-up lane
  (`research/mla-bringup-20260801/DESIGN.md`, GLM-5.2 C=576 latent rows) — that, not a
  retrofit, is our MLA path. V3.2/V4-class adds DSA-style sparse attention on top (V4-Pro
  claims 10% of V3.2's KV at 1M ctx — vendor number, single source).
- **Retrofit MLA on GQA models** (TransMLA arXiv 2502.07864, MHA2MLA): works only with
  fine-tuning to heal the conversion; quality claims are post-training. For an engine that
  pledges to serve the model's published bytes, retrofits are out of scope — and §1.4's
  matched-budget result says post-hoc projection without training loses to quantization
  anyway. **Research-only for memra.**

### 2.4 Dedup / merging / delta

Chunk-level dedup and residual-vs-anchor compression (DeltaKV 2026-02) show 40-60% extra
on top of quant in papers; no production engine ships KV dedup in 2026 (prefix caching *is*
the production-grade dedup, at page granularity, and we already run it —
`MEMRA_PREFIX_CACHE_MB`).

### 2.5 Offload / tiering with compression in the tier

The one non-quant compression class with real production adoption:

- **LMCache + vLLM** (docs.lmcache.ai; production-stack): GPU→CPU-pinned→NVMe→remote tiers;
  **CacheGen serde** (SIGCOMM 2024) compresses KV into a bitstream for the cold tiers/network
  (delta-between-adjacent-tokens + layer-sensitivity-aware arithmetic coding, decode on GPU) —
  lossy-but-measured, adopted as an LMCache config option (`remote_serde: "cachegen"`).
- Field numbers (KGA 2026-04, 8xH200 RAG workload): vLLM+LMCache local CPU+NVMe = prefill
  recomputation 38%→11%, TTFT p99 720→430ms; SGLang RadixAttention wins single-node,
  LMCache wins multi-node sharing. PCIe restore ~40GB/s (hundreds of ms for 128k on 70B),
  Gen5 NVMe ~12GB/s effective (2-3s restore), two-stage NVMe→CPU→GPU pipelining standard.
  Tiering heuristic that survives contact: <30s hot in HBM, 30s–10min CPU, >10min NVMe.
- NVIDIA Dynamo ships KV-aware routing + offload as a product.

Mapping note: this whole class is *our* SLRU/spill/prefix-cache architecture applied to KV
instead of experts. The compression-in-tier trick is orthogonal to exactness when the tier
stores our *already-quantized* KV bytes verbatim (byte-identical restore = exact by
construction; CacheGen-style re-encoding is NOT byte-identical and would be lossy-tier).

---

## 3. Long context / our depth cells

At our board depths (d4096–6257) KV read bandwidth dominates decode — this is precisely what
fa-decode-deep just attacked (v4 was SM-cycle-bound at 19% of card BW from smem bank
conflicts; deep = 277 GB/s, 1.43x kernel at d6144 — RESULTS.md). The 32k–256k regime the
survey covers is one-to-two orders past our cells; what wins there:

1. **FP8/4-bit KV storage** (the ITL-slope lever: vLLM measures slope 54% of BF16 at fp8 —
   the *slope*, i.e. exactly the depth-decay term our depth table tracks).
2. **Paged/tiered KV with prefix reuse** (recomputation-rate lever).
3. **Trained sparse attention** (DSA-class; model-side, not engine-side).
4. Eviction — only where sessions are single-query (SCBench), i.e. not agentic serving.

On honest evals: literal-match NIAH is discredited as a headline metric — NoLiMa (ICML 2025)
shows 11/12 models fall below 50% of their short-ctx score at 32k once literal cues are
removed, while classic NIAH stays green. The 2026 evidence standard is MRCR (multi-needle,
AUC over length buckets), RULER (multi-task), SCBench (multi-turn KV lifecycle), plus
decode-heavy reasoning suites (AIME/GPQA/LCB) because *decode-side* KV noise compounds in
long generations — vLLM's battery uses exactly this pair of axes, and it's what our v2
five-arm eval framing already mirrors (CLAUDE.md quant-study rules).

For KV-quant cliffs specifically: 8-bit no cliff measured anywhere; fp8-flat none to 1M
(with accumulation fixed, calibration for outlier models); NVFP4 <1% at 64k (RULER);
int4 group-scaled +0.2 PPL class; 2-3 bit = the cliff (KITTY, 2604.11501 both place it
below 3 bits at matched budgets).

---

## 4. Best-results table (top methods, 2026)

Bytes = fraction of BF16 KV for the touched layers. Quality = honest-eval cost as published.
EXACT/LOSSY = memra classification (exact = can be the gated standing numeric config with
bit-identical decode reads across all paths; lossy = output-changing vs our gated config).

| # | Method | KV bytes | Quality cost (eval, source) | Decode overhead | Production adopters | memra tier |
|---|---|---|---|---|---|---|
| 1 | INT8/q8-class KV (per-block scales) | ~50-53% | ≈0 (+0.002-0.05 PPL, llama.cpp/Ollama KLD 2024-2026) | ~0 (fused dequant) | llama.cpp/Ollama ecosystem; TRT-LLM INT8-KV; **memra default (q8_0 K)** | **EXACT (shipped)** |
| 2 | FP8 e4m3 KV, storage-only, f32 math | 50% | ≈0 storage-side; risk lives in math path not storage (vLLM 2026-04-22) | ~0-small; wins ITL slope ≥7k ctx vs BF16 | vLLM/SGLang opt-in; TRT-LLM standard; "production default" per field reports | **EXACT-capable (built: kf8vf8; adoption reverted 2026-07-28 on perf, not quality)** |
| 3 | q8_0 K + q5_1 V (K/V asymmetric) | **45.3%** | ≈0 in-config (our standing battery; community KLD "keep K higher than V") | 0 (fused v3/v4/deep kernels) | **memra (always-on default)**; llama.cpp opt-in pair | **EXACT (shipped, frontier)** |
| 4 | NVFP4 KV (16-blk + e4m3 scales, dequant→FP8 math) | 28% | <1% LiveCodeBench/MMLU-Pro/MBPP/RULER-64K (NVIDIA 2025-12-05); beats MXFP4 by ~5pt MMLU | dequant before math; TTFT up to 3x better via hit-rate at fixed HBM | TRT-LLM/ModelOpt (Blackwell) | LOSSY tier candidate #1 (V-only first; argmax gate decides if it's exact-in-config) |
| 5 | INT4 group-scaled K/V (KIVI-4/KVQuant-4 class) | ~31% | +0.18-0.23 PPL @75% reduction (2604.11501); INT4=FP16 on LAMBADA | per-channel K path needs RoPE handling <8b | none mainline; kvpress/research + HF transformers cache impl | LOSSY tier candidate #2 |
| 6 | KVzip (query-agnostic eviction + reconstruction scoring) | 25-33% (3-4x) | "negligible" on multi-query LongBench-class at 3-4x (NeurIPS'25 oral); survives multi-turn by design | one-time compression pass; ~2x latency win after | none (research; Qwen/Gemma/Llama demos) | LOSSY / research-only |
| 7 | H2O/SnapKV-D heavy-hitter eviction | 10-50% budget | wins on single-trace reasoning (2512.12008) but **fails multi-turn** (SCBench); budget cuts lengthen traces | scoring bookkeeping per step | none mainline (kvpress research) | research-only (multi-turn failure disqualifies serving) |
| 8 | MLA latent KV (trained-in) | ~25-30% vs dense-equivalent | baked into model quality (DeepSeek/GLM lineage) | needs MLA kernels (our bring-up lane) | DeepSeek, GLM-5.2, Kimi | EXACT (model-side; serve-the-bytes) |
| 9 | Tiered offload + prefix reuse (LMCache/Dynamo/RadixAttention class) | n/a (moves bytes, doesn't shrink hot set) | 0 if byte-identical restore | PCIe ~40GB/s / NVMe ~12GB/s restore; recompute-rate 38%→11% (KGA 2026-04) | vLLM+LMCache, SGLang HiCache, Dynamo | **EXACT (we ship the intra-GPU form: prefix cache + KV-reuse pool)** |
| 10 | CacheGen bitstream (cold-tier/network encode) | ~35-50% of the *transferred* bytes | small, measured per-layer-sensitivity (SIGCOMM'24) | GPU decode kernel on restore | LMCache serde option | LOSSY (re-encoded restore ≠ byte-identical) |

Refuted/avoid: MXFP4 KV (−5 MMLU vs NVFP4 — scale format matters); post-hoc low-rank
projection at any matched budget (2604.11501); eviction as a default serving policy
(SCBench); trusting literal-match NIAH as the only quality gate (NoLiMa).

---

## 5. Mapping onto memra

### 5.a Are we at the exact-tier practice frontier? — Yes, with one caveat

Our always-on config (q8_0 K + q5_1 V, 45.3% of BF16, fused into every FA path incl. the new
deep twins, battery-gated bit-exact-in-config) is *smaller* than the industry's recommended
FP8 (50%) and stricter-gated than anyone's (vLLM gates on eval-recovery %, we gate on argmax
identity + K=1..8 spec self-consistency). Additions the survey does NOT justify chasing for
the exact tier: per-channel K scales and RoPE-aware K quant only pay below ~5 bits; V below
q5_1 is measured-mismatch on our stack (2026-07-08). The caveat: the V ladder between q5_1
and 8-bit-class has no 2026 reason to move either direction — we sit at the measured edge.

### 5.b fp8-KV revert: STALE-VERDICT CANDIDATE — re-measure justified, expectations bounded

The 2026-07-28 revert (12k A/B fp8 −1%, d1736 flat, argmax MATCH both formats) predates BOTH
fa-decode-deep (merged 2026-08-02) and any ladder-512 move. The case for re-measuring under
the current build:

- The revert's own LAW says it: "parked wins must be RE-MEASURED on the current build before
  adoption" — symmetric application means parked *losses* go stale the same way (the H100
  lane logged five stale-verdicts in one day when kernels moved under thresholds).
- Mechanism shift: deep moved the vec kernel from SM-cycle-bound (bank conflicts, 19% of BW)
  to much closer to memory-bound (277 GB/s at d6144). But the byte math cuts the other way:
  **fp8-flat K+V (64 B/blk) is +10% MORE bytes than our q8_0/q5_1 default (58 B/blk)** —
  e4m3's win was always dequant latency (the gemma GKV arc), and deep just reduced the cost
  of the q8/q5 dequant path. Two forces, and the byte force now points against fp8.
- vLLM's 2026 numbers (slope 54% of BF16) are vs **BF16**, not vs an 8-bit default — they do
  not transfer to us; our baseline is already at 45% bytes.

**Verdict: re-measure as a cheap A/B rider on the next 5090 GPU window (the door
`KV_FP8_FORCE` + `MEMRA_KV_FP8` still exists, zero build work), but with the expectation
LOW — the byte math says fp8-flat cannot win the deep-kernel bandwidth race against
q8_0/q5_1; its remaining value stays what the revert note says: ~45% KV bytes vs BF16 for
ctx-limited serving, i.e. a capacity door, not a speed door — and rec #1 below supersedes
that capacity story. The gemma GKV/hd512 e4m3 defaults are a separate already-adopted config
and are not touched by this.**

### 5.c A lossy serving tier (darklanes cheap endpoint), if ever

Per OR norms (or-provider REPORT receipts): quantization must be disclosed on the endpoint;
"unknown" quant is a filterable trust penalty; OR's Auto Exacto benchmarks endpoints against
full precision. A compliant cheap tier would be:

1. **First lever: NVFP4 V-cache (keep q8_0 K)** — 52 B/blk (−10% vs default), Blackwell-native
   16-blk e4m3-scale format we already have codec experience with (weights side), K kept
   8-bit respects the KIVI asymmetry, and NVIDIA's <1% RULER-64K evidence is the strongest
   published 4-bit-KV quality receipt. If it survives *our* argmax gate in-config it
   graduates to exact-tier; if not, it's the disclosed lossy tier.
2. Full NVFP4 K+V (36 B/blk, −38% vs default) with per-channel/rotated K only if the tier
   needs the capacity — quality receipt still <1% published, but K at 4-bit is where the
   cliff literature starts hedging.
3. **Not eviction** — agentic workloads are multi-turn; SCBench says sub-O(n) eviction breaks
   exactly there. KVzip is the only eviction method that addresses this by design; still
   research-grade, no engine ships it.
4. Disclosure format: endpoint label "KV cache: NVFP4-V (K int8)" + the eval receipt on OUR
   models (five-arm-style: same source, same prompts, MRCR-class + decode-heavy reasoning,
   N stated) — never public-benchmark-selected (CLAUDE.md evidence rule).

Honest magnitude on our workloads: KV bytes −10% (V-only) to −38% (K+V) vs today's default;
decode speedup bounded by the KV-read share of the tick at depth (our depth table: decay
−7.6..−9.8% over 512→6144, so the whole KV-read term is ~10% of decode at d6144) — i.e.
**single-digit % decode at best; the real win is capacity/concurrency.** That is why this is
a pricing lever, not a board lever.

### 5.d Spill/offload interaction — KV pages through the tier

Today our KV never leaves HBM (spill = experts only; prefix cache = device-resident
snapshots, `MEMRA_PREFIX_CACHE_MB`; session KV-reuse pool likewise). The production pattern
we don't yet have is **KV offload to host/NVMe for idle sessions** (LMCache-class):

- Fit: our spill stack already owns the exact primitives the KV tier needs (pinned host
  buffers, positioned reads, `O_DIRECT`/worker backends, SLRU residency, CUDA-owner-thread
  H2D publication — CLAUDE.md pipeline doctrine). KV pages are *better* spill citizens than
  experts: sequential per-session extents, restore is bulk H2D with no per-token reordering.
- Exactness: **byte-identical park/restore of our already-quantized KV pages is exact by
  construction** — same class of guarantee as the prefix-cache 16/16 gates
  (`research/prompt-cache-20260802/`). No CacheGen re-encode (that's the lossy variant; skip).
- The compounding trick nobody states loudly: because our KV is already 45% of BF16, every
  tier (host RAM, NVMe, PCIe transfer) is 2.2x more effective per GB/s than a BF16-KV
  engine's — the q8/q5 formats ARE the tier compression. A 24GB 5090 serving ctx-8192
  sessions OOMs at c=32 today (`MEMRA_CTX` note, FLAGS.md); an idle-session KV park/restore
  tier is the direct fix and monetizes as concurrency.
- Cost model from the field: PCIe ~40GB/s restore; a 119MB ctx-8192 session (9B-class
  measured, FLAGS.md) restores in ~3ms host→device — trivially hideable behind turn latency.

---

## 6. Ranked recommendations

1. **KV park/restore tier for idle sessions (host pinned RAM first, NVMe later) — EXACT-tier,
   the highest-value item.** Byte-identical page park/restore of our existing q8_0/q5_1 KV
   through the spill stack's primitives; SLRU over sessions; gate = restored-session decode
   bit-identity vs never-parked (prefix-cache-gate pattern). Honest magnitude: not a tok/s
   move at all — a **concurrency/capacity multiplier** (field analog: recompute 38%→11%,
   TTFT p99 −40%; our own OOM wall at c=32/ctx-8192 is the binding constraint it removes).
   This is also the Hy3-class spill lever asked about in (d): same SLRU, new payload class.
2. **fp8-KV re-measure under fa-deep — cheap stale-verdict hygiene, expectation LOW.**
   One A/B rider in the next GPU window (door exists, no build work). The byte math
   (fp8-flat = +10% bytes vs q8_0/q5_1) predicts no speed win now that deep removed the
   dequant-cost argument; the door's documented value (~45% of BF16 for ctx-limited serving)
   is superseded by rec #1 anyway. Close the loop, update the FLAGS.md row with the
   deep-build verdict either way. EXACT-tier (both formats argmax-gated already).
3. **NVFP4 V-cache arm (q8_0 K + NVFP4 V) — the lossy-tier opener, only when darklanes
   wants a disclosed cheap endpoint.** −10% KV bytes vs default (−38% if K follows later),
   Blackwell-native, best published 4-bit quality receipts (<1% RULER-64K). Run it through
   the full battery first: if argmax holds in-config it's a free exact-tier byte win; if
   not, it ships only behind an OR-disclosed endpoint label with our own MRCR-class +
   decode-heavy eval receipt.

Research-only (explicitly not recommended for serving): everything below 4-bit; all
eviction (SCBench multi-turn breakage; KVzip = watch item only); retrofit-MLA (native-MLA
bring-up lane is the real path); post-hoc low-rank at any budget (2604.11501); MXFP4
(−5 MMLU vs NVFP4); V below q5_1 in the exact tier (our 2026-07-08 in-config MISMATCH
stands until a finer-scaled format is gated).

---

## Sources

External (all fetched 2026-08-02/03):
- vLLM blog, "The State of FP8 KV-Cache and Attention Quantization in vLLM", 2026-04-22 —
  https://vllm.ai/blog/2026-04-22-fp8-kvcache
- NVIDIA, "Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache",
  2025-12-05 — developer.nvidia.com/blog
- TRT-LLM quantization docs (NVFP4-KV constraint: FP8 W/A only with NVFP4 KV) —
  nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
- vLLM quantized_kvcache docs (per-tensor vs per-head scales, calibration via LLM-Compressor)
- SGLang quantized KV docs (fp8_e4m3/e5m2, store-time quant); sgl issue #10083
- Salfati, "Quantization Dominates Rank Reduction for KV-Cache Compression", arXiv 2604.11501
- KIVI, arXiv 2402.02750 (ICML 2024); KVQuant, NeurIPS 2024; RotateKV, IJCAI 2025;
  KITTY, arXiv 2511.18643; RoPE-aware bit allocation, arXiv 2606.24033
- "Hold Onto That Thought: Assessing KV Cache Compression On Reasoning", arXiv 2512.12008
- SCBench, arXiv 2412.10319 (ICLR 2025); NoLiMa, arXiv 2502.05167 (ICML 2025)
- KVzip, arXiv 2505.23416 (NeurIPS'25 oral); NVIDIA kvpress (github.com/NVIDIA/kvpress);
  DeltaKV, arXiv 2602.08005
- TransMLA, arXiv 2502.07864; CLA (Brandon et al. 2024); YOCO lineage; NAACL 2025 cross-layer
  sharing study; Raschka architecture-gallery DSA note (2026-07)
- LMCache docs + CacheGen (SIGCOMM 2024); KGA field report "KV Cache Management 2026",
  2026-04-22; llama.cpp community KLD thread (r/LocalLLaMA 2026-03-23); smcleod.net 2024-12

Internal receipts: docs/FLAGS.md (MEMRA_KV_K/V, MEMRA_KV_FP8, GKV/WKV, MEMRA_CTX,
MEMRA_PREFIX_CACHE_MB, spill rows); crates/memra-kv/src/lib.rs (kv_blk_bytes, KV_FP8_FORCE);
crates/memra-engine/src/lib.rs (fa_deep_at + dispatch sites ~8674/9537);
research/fa-decode-deep-20260802/RESULTS.md; research/depth-decode-20260802/RESULTS.md;
research/prompt-cache-20260802/; research/or-provider-20260802/REPORT.md;
research/mla-bringup-20260801/DESIGN.md; project memory (fp8-KV revert 2026-07-28).

# 8-bit serving format decision — Q8_0 GGUF vs FP8-E4M3 ST vs W8A8-INT8 (2026-08-03)

Lane `lane/8bit-decision` (from `restructure/public-split` @ 69cdd1eb). Owner ruling: production
serving quant = 8-bit. This document decides WHICH 8-bit format, for serving Qwen-27B-class
("q27" / the expected Qwen3.8-27B) on the deployment target: 2x desktop RTX 5090 (sm_120a,
32 GB + 1.79 TB/s GDDR7 per card). Repo-evidence + literature only — NO GPU runs in this lane
(rigs busy). Every number carries a file:line or URL; missing data is named as a measurement,
runnable now on the rented 2x5090 box. Companion evidence dump: `EVIDENCE-NOTES.md` here.

---

## Verdict

**Hybrid, with a promotion gate. Q8_0 GGUF is the serving arm NOW; FP8-E4M3 safetensors is the
hard-tuning development track, promoted to serving only when it beats the Q8_0 arm ≥1.1x e2e
under the full battery plus a clean 6-draw sampled loop matrix. W8A8-INT8 (SmoothQuant-class)
is rejected — dominated on sm_120a by both other arms.**

Why this split and not a single winner:

1. **Decode is bandwidth-bound and all three 8-bit formats read ~the same bytes** (Q8_0
   1.0625 B/w, FP8+block-128 scales ~1.008 B/w, INT8 ~1.03 B/w with per-channel scales), so the
   serving headline — plain + spec decode — moves at most single-digit % on format choice. The
   one measured 8-bit-vs-8-bit decode A/B we have (e4m3-direct vs Q8_0 re-encode, NV-27B) was
   **-7% on the laptop's 171 W power wall and +7.1% on the box** — the J/token law, third
   confirmation (`research/tune-data/rig5090.jsonl:256`). Format does not decide decode;
   power headroom does, and the desktop 2x5090 sits on the box side of that law (unmeasured
   there — measurement M2 below).
2. **Prefill is where FP8 wins by an order of magnitude at the GEMM level** — the July g7e
   micro-probe measured our q8_0 GEMM classes at **47–72 TFLOPS** vs cuBLASLt FP8-E4M3 at
   **624–794 TFLOPS on the same shapes** (8.7–14.2x on the attn/linear layers; exact quote in
   §2b). The engine-level FP8 prefill path already exists and is gated green
   (`MEMRA_PP_FP8` / `MEMRA_ST_E4M3`, docs/FLAGS.md:81-83) but was only ever tuned as an
   opt-in on a 24 GB laptop. This is exactly the "we already have but never hard tuned
   against it" surface the owner named.
3. **Time-to-market cuts for Q8_0 GGUF.** Qwen3.8-27B drops next week; a Q8_0 GGUF is
   house-convertible from BF16 on day one with zero new engine code, rides the full existing
   kernel stack + gate battery + drafter regime, and PP-2 across the two 5090s is already
   bit-identity-gated (docs/FLAGS.md:321). The FP8-ST arm is blocked on real loader work:
   **memra cannot load Qwen's official FP8 checkpoints today** — they use fine-grained
   block-128 scales and our loader only accepts per-tensor scalar or per-channel scales
   (§3, the sharpest single finding of this study).
4. **The LoRA future requires the ST arm to stay first-class** — GGUF has no adapter story,
   ST-base + runtime adapters is the ecosystem standard, and every DeltaServe-class
   co-serving trigger lands on safetensors (§5). Killing the ST arm to simplify would strand
   the fine-tune track. Hard-tuning FP8-ST is the hedge that costs nothing extra: it is also
   the prefill play.

---

## 1. The three arms — evidence table

| | (a) Q8_0 GGUF | (b) FP8-E4M3 ST | (c) W8A8-INT8 |
|---|---|---|---|
| Weight bytes/elem | 1.0625 (per-32 fp16 scale) | ~1.008 (block-128) / 1.0 (per-tensor) | ~1.03 |
| Decode kernel path | dp4a MMVQ / fused3 / q8-fast — the proven chain (fp8_ffi.rs:12-14) | `qmatvec_e4m3_mmvq` + batched twins under `MEMRA_ST_E4M3` (FLAGS.md:83) | none in memra; would be new |
| Prefill GEMM class | int8-MMA MMQ (`MEMRA_PP_Q8MMQ` default, FLAGS.md:219); hand-tiled class probed 47–72 TF (g7e jsonl:39) | cuBLASLt FP8 624–794 TF probed (g7e jsonl:39); engine path merged, +28% pp measured at budget 4096 MB (rig5090.jsonl:234) | int8-MMA, ≈ FP8 TOPS on paper; W8A8 prefill GEMMs **refuted on the H100 lane at m=512 AND m=2048** (FLAGS.md:417-419) |
| Exactness machinery today | full: bit-identity kernel-check pins, argmax gates, K=1..8 spec, the entire board history | argmax MATCH maxdiff 0.0 held on every FP8-prefill gate run (rig5090.jsonl:233-234, 256); 9B ST full battery green (rig5090.jsonl:266); decode unchanged bit-for-bit (m≥16-only dispatch, fp8_ffi.rs:12-14) | none; "w8a8-class numerics change model outputs" is the standing owner-gated note (FLAGS.md:420-421) |
| Drafter/MTP | native (DRAFT-REGIME.md; board spec rows) | works: model-trained MTP head live from safetensors (rig5090.jsonl:186), ST spec K=3 95.4 tok/s, GGUF-free frspec toolchain (rig5090.jsonl:268) | n/a |
| LoRA future | no adapter story (format has no adapter/backprop ecosystem) | ecosystem standard; vLLM FP8-LoRA RFC #33301 (2026-01) | ST-based but niche |
| Day-one Qwen3.8 | house Q8_0 convert from BF16, hours (runbook §3) | official FP8 likely (3.6 precedent: Qwen/Qwen3.6-27B-FP8 shipped in the release month) **but loader-blocked on block-128 scales** (§3) | nobody ships official W8A8-INT8 Qwen artifacts |
| 27B-class fit on target | 27 GB weights: tight on one 32 GB card; comfortable PP-2 sharded (FLAGS.md:321,323) | ~27 GB, same shape; `MEMRA_ST_E4M3` = one resident copy, no stash duplication (FLAGS.md:83) | same bytes |

**Honesty note:** no q27-at-8-bit cell exists anywhere in the repo. The current daily q27 is
NVFP4-trunk + Q4_K_M (16 GB — the only thing that fits the 24 GB laptop; qwen38-prep
AUDIT.md §1), the H100 q27 cell served unsloth Q4_K_M (ARCHITECTURE-H100.md:2214), and the
9B is the only model with a standing Q8_0 board presence (current-board.json
supported_models: "Q8_0 (H100)"). Every 27B-at-8-bit number in this doc is therefore a
projection until measurement M1 (§7) runs on the 2x5090 box.

---

## 2. Decision driver 1 — perf ceiling on the 2x5090

### 2a. Decode (the serving headline): format-insensitive, power-sensitive

All 8-bit arms read ~27 GB/token for a 27B dense. Ceiling arithmetic (projection, labeled):
1.79 TB/s x ~85% achieved (the dense-decode efficiency the 5090-laptop board demonstrates:
135.7 tok/s x ~5.6 GB ≈ 762 of 896 GB/s, research/hw-buy-20260802/REPORT.md:41-44) / 27 GB ≈
**~55 tok/s plain decode single-card**. PP-2 serial does not add bandwidth for a single
stream; the deferred-readback 1.87x needs 2+ tokens in flight and is quarantined-experimental
(FLAGS.md:321). Spec K=3 at the q27 acceptance profile ≈ 2.0-2.2x plain
(current-board.json speculative row; rig5090.jsonl:268).

The only measured 8-bit-vs-8-bit decode delta is the e4m3-direct vs Q8_0 A/B
(rig5090.jsonl:256): **pp1845 +5.6% and −7.3 GB resident for e4m3, but spec decode −7% on the
laptop power wall vs +7.1% on the box** — the e4m3→f16x2 cvt+hfma dequant chain is ALU-richer
than the Q8_0 dp4a dot, free under power headroom, taxed at a clock wall. The desktop 5090s
have the headroom; expectation (unmeasured) is e4m3 decode ≥ Q8_0 there. **Measurement M2.**

### 2b. Prefill (the known gap): the FP8 probe row, quoted precisely

`research/tune-data/g7e-rtx6000.jsonl:39` (recovered-fragment, ts 2026-07-08, rig
g7e-rtx6000-sm120-188sm, lane/prefill-fp8, probes `probe/fp8_lt_prefill.cu` +
`fp8_lt_scale_probe.cu` + `fp8_vec16_probe.cu` — all three still in `probe/`):

> Per-shape TFLOPS from nsys grid buckets at m=4096-chunk: **q8_0 GEMM 47-72 TF** (kv_proj 47,
> o_proj 62, lin_ba 68, lin_qkv 72, q_gate 72); W4A8 MMQ 241 TF (both MLP shapes) …
> **cublaslt_fp8_e4m3_tflops**: o_proj_5120x6144 [624, 703, 676, 668], lin_qkv_10240x5120
> [626, 659, 726, 726], q_gate_12288x5120 [668, 707, 723, 779], kv_proj_1024x5120 [346, 630,
> 665, 612], lin_ba_6144x5120 [613, 684, 682, 670], ffn_gate_up_17408x5120 [695, 758, 769,
> 794], ffn_down_5120x17408 [734, 790, 772, 730] (m_axis 512/2048/4096/6257) …
> speedup_vs_current: **attn_linear_q8_0_layers 8.7-14.2x**, mlp_mmq_layers 2.9-3.3x …
> act_quantize_cost: f32->fp8 per-token-scale kernel 0.007-0.118 ms at k=5120 — quant+GEMM
> chained still ~700 TF effective … per_token_OUTER_VEC_32F **NOT_SUPPORTED sm120**
> (cublasLtMatmulAlgoGetHeuristic status=7 nh=0 all m).

Context for the 47–72 TF figure: it profiles the hand-tiled `qmatvec_gemm_q8_0` class as of
2026-07-08; the int8-MMA `MEMRA_PP_Q8MMQ` default landed 2026-07-09 (35B pp 2456→3069,
FLAGS.md:219) and the W4A8-FP8 MMQ tile is a 381-TF class (FLAGS.md:131) — so today's Q8_0
prefill is better than 47–72 TF but still 2–3x under the cuBLASLt FP8 ceiling.

Engine-level receipts on the same lane: `MEMRA_PP_FP8` **+78–129% pp on the g7e box**;
local NV-27B budget sweep pp1845 887.9 → 1136.3 (**+28.0%** @4096 MB stash, argmax MATCH
maxdiff 0.000e0, no decode regression) with OOM at 4608 on 24 GB (rig5090.jsonl:233-234);
`MEMRA_ST_E4M3` one-copy variant pp1845 1291.2 → 1364.1 (+5.6%) at FULL coverage, −7.3 GB
(rig5090.jsonl:256). On 32 GB cards the budget wall that capped the laptop disappears.

The e2e stake, sharpest single datapoint: the H100 q27 board cell — **memra prefill 1965 vs
vLLM-serving-official-FP8 prefill 15054 tok/s (7.7x)** while memra still wins e2e 74.3 vs 72.9
because decode dominates the 512-gen protocol (ARCHITECTURE-H100.md:2216-2222; board row 1.31x
after later decode work, current-board.json h100_board). Prefill-heavy serving (agentic long
prompts, prefix-cache misses) is where the q8_0 GEMM class actually bleeds — and TTFT is a
serving SLO, not a vanity number.

**Amdahl bound:** on the NVIDIA-27B ST anatomy, q8_0-class GEMMs were 46.5% of pp GPU time and
NVFP4-MLP another 30.4% (g7e jsonl:39 baseline field). FP8-izing only the attn/linear class
bounds the pp gain near ~1.85x (fp8_ffi.rs:3-6 projection). A *full-FP8* checkpoint (all
linear layers FP8, which is what Qwen ships) puts ~77% of GEMM time on the 624–794 TF pipe —
that is the real FP8-arm prize and it only exists on the ST arm.

### 2c. Where q8_0 glue costs live (verify-tier receipt)

The spec-verify premium that pins K=3 is not a format problem: at T=2 the premium is 50% glue
(quantize_q8_1 0.86 ms + rms_norm_f32 0.68 ms/pass the top items), falling to 6% at T=8 as
b-tier matvecs take over (research/verify-tier-20260802/glue-attribution.md:1-27,
RESULTS.md §2). Switching weight format does not touch this — the glue is activation-side
(q8_1 quantize + norms) and identical on the ST arm. No format credit here for either side.

---

## 3. The F8→Q8_0 re-encode at load, and the loader gap that blocks official Qwen FP8

**Where the FP8 tensor-core advantage is currently thrown away** (the code the task asked to
locate):

- `crates/memra-gguf/src/safetensors.rs:40-42` — the raw ST dtype mapper explicitly panics on
  `F8_E4M3 | F8_E5M2 | F8_E8M0` ("FP8 … not yet supported; use the GGUF twin"), so any F8
  tensor that reaches the generic path dies loudly.
- `crates/memra-gguf/src/source.rs:981-1049` (Plain arm) — F8_E4M3 2D weights with a
  `.weight_scale` sibling are **dequantized to f32 host-side and re-encoded to GGUF Q8_0**
  (`f32_to_q8_0`, source.rs:1041-1046; the NVFP4 opt-in re-quant at :1030-1040). The Transform
  (V-reorder) arm does the same at source.rs:1097-1112. Rationale in-code: rides the proven
  q8-fast/MMVQ/fused3 path at ~1.06 B/elem instead of a 22 GB f32 blow-up; per-32 q8 is a
  finer grid than one per-tensor FP8 scale (accuracy class-equal or better).
- The escape hatches exist but are opt-in: `MEMRA_PP_FP8` stashes raw e4m3 alongside the Q8_0
  copy for the cuBLASLt prefill GEMM (crates/memra-engine/src/fp8_ffi.rs:1-20);
  `MEMRA_ST_E4M3` keeps raw e4m3 as the ONE resident copy — decode dequants e4m3 in-kernel,
  prefill rides the FP8 GEMM on the same bytes (fp8_ffi.rs:55-62; FLAGS.md:83).

**The blocking gap:** `f8_row_scales` (source.rs:838-851) accepts a scale sibling only when
its element count is 1 (per-tensor) or out_f (per-channel). Qwen's official FP8 checkpoints —
including `Qwen/Qwen3.6-27B-FP8` — are "fine-grained fp8 quantization with block size of 128"
(HF model card, fetched 2026-08-03), i.e. a 2-D `[out/128, in/128]` scale tensor. That shape
returns `None`, falls through to `raw_hf`, and panics in `st_dtype_to_ggml`. **memra cannot
load an official Qwen FP8 checkpoint today, in either the Q8_0-re-encode or the e4m3-native
mode.** The FP8 formats we CAN load are the NVIDIA-modelopt per-tensor style (the NV-27B
attn/linear projections, source.rs:981) and unsloth compressed-tensors per-channel
(source.rs:1015-1017). This is the first work item of the FP8 track (§7, W2-1).

Second gap, prefill-side: the cuBLASLt path feeds ONE scalar weight scale via
B_SCALE_POINTER (fp8_prefill.cu; per-token OUTER_VEC probed NOT_SUPPORTED on sm120, g7e
jsonl:39). Block-128 weight scales need either cuBLASLt block-scaled FP8 (exists for
sm90/sm100 since CUDA 12.8/12.9 — **sm_120 support unknown, probe needed**) or our own
f8f4-style MMQ kernel extended to block-scale dequant (the `MEMRA_MMQ_F8F4` machinery,
FLAGS.md:131, is the natural home). Named as probe P1 in §7.

---

## 4. Open questions the task listed — answered from repo + literature

**Per-tensor vs per-channel scaling accuracy at 8-bit on 27B-class:** second-order at 8-bit.
Repo: the NVIDIA 27B ships per-TENSOR FP8 attn projections and our per-32 Q8_0 re-encode of
them (a finer grid) held argmax MATCH and full spec batteries (source.rs:984-986,
rig5090.jsonl:186,233). Literature: vLLM's FP8 W8A8 recipe (per-tensor W + dynamic per-token
A) reports >99% recovery across Open LLM Leaderboard tasks (docs.vllm.ai/en/v0.21.0/features/
quantization/fp8/; developers.redhat.com 2024-07-15 article); Qwen moved to block-128
fine-grained for their official FP8, and NVIDIA's ModelOpt FP8_CFG default remains per-tensor
static (developer.nvidia.com PTQ blog, 2026-05). Nobody reports 8-bit weight-granularity
incidents on 27B-class dense models. Our Q8_0 per-32 is finer than all of them.

**FP8 attn/KV needed, or weights-only?** Weights-only suffices, and our KV answer is already
settled by measurement: fp8 K-cache is FLIP-BLOCKED (e2e flat AND 9B-ST spec acceptance
74%→20.5% FAIL, FLAGS.md:84); q8_0/q5_1 KV at 45.3% of BF16 bytes is smaller than fp8-flat
anyway (research/kv-compress-20260802/REPORT.md §1.1). FP8 attention *math* is also not
needed: our FA decode accumulates f32 with an order-pinned chain, structurally on the safe
side of the accumulation-precision law that killed Hopper FA3-FP8 at 128k
(kv-compress REPORT.md §1.2). The FP8 arm here means **FP8 weights + FP8 prefill GEMM**;
decode dequant-to-int8/f32 and KV stay as gated.

**What replaces bit-identity gates when GEMMs move to cuBLASLt?** Three layers, all already
prototyped in-repo: (1) *deterministic algo selection* — fp8_prefill.cu caches the
cublasLtMatmulAlgoGetHeuristic result per (m,n,k) plan (fp8_prefill.cu:14,143-148), so a
given shape is run-to-run deterministic; the residual risk is m-variance across chunked
prefill/batching picking different algos — the exact defect class quarantined once before on
batched Lt router GEMMs (`MEMRA_ROUTER_PREFILL_EXACT`, research/deltaserve-assessment-
20260803/ASSESSMENT.md:149-158) — so the c1-vs-c16 byte-identity serve gate is the binding
gate, plus pinning algo per shape-class rather than per-m if it fires. (2) *In-config
exactness* — the battery gates the shipped numeric config, not a BF16 oracle (the standing
framing, kv-compress REPORT.md preamble); FP8-prefill runs held argmax MATCH maxdiff 0.0
(rig5090.jsonl:233-234,256) and prefill-vs-decode argmax stays a meaningful gate because
decode keeps the dp4a chain bit-for-bit (fp8_ffi.rs:12-14). (3) *the own-kernel exit* — if Lt
determinism ever binds, `MEMRA_MMQ_F8F4` is our own FP8 MMA tile (381-TF class, FLAGS.md:131):
fully deterministic, ~half the Lt ceiling, per-model adopted already. Cost accounting: FP8 arm
gate machinery ≈ one new fast-gate golden set per FP8 config + an Lt-algo pin + the loop
matrix (§6); no gate is lost.

**Sampled-quality caveat (must re-run on Qwen official FP8):** the one ST-vs-GGUF content
battery we ran (NVIDIA NVFP4-mixed ST checkpoint vs GGUF twin, 2026-07-10) found the ST arm
looping on 2/6 sampled long-context draws vs 0/6 for GGUF, invisible to argmax/self-
consistency gates (research/tune-data/27b-st-vs-gguf-final.md:55-57,93-104). That was a
*checkpoint* difference, not a format law — but it is exactly why the FP8 promotion gate
includes the 6-draw loop matrix, not just argmax.

---

## 5. Decision driver 2 — the LoRA-adapter future

- memra has **zero LoRA support** (grep receipt: deltaserve ASSESSMENT.md:113-116). GGUF has
  no adapter/backprop ecosystem; ST base + runtime adapters (S-LoRA/Punica-class multi-LoRA
  batching) is the standard, and the ecosystem is converging on **FP8 base + LoRA**
  specifically (vLLM RFC #33301, 2026-01: FP8 LoRA promoted for memory + dense/MoE coverage).
- The DeltaServe assessment's GO-later triggers all remain unmet (ASSESSMENT.md:190-203:
  measured production idle >25-30%, LoRA-shaped fine-tune track, a QLoRA-style backward for
  quantized weights, timing-interference gate) — so **no LoRA machinery is built this cycle**.
  The format decision only has to keep the option alive, and its frozen-base finding is
  format-relevant: a co-located training process needs the *served base bytes* to be something
  torch can consume — safetensors, not GGUF blocks (ASSESSMENT.md:142-146).
- Concretely: hard-tuning the FP8-ST serve path now (track B) is the *entire* LoRA hedge.
  Serving-side adapter math (W0x + BAx) composes with any base format whose forward we
  control, but adapter loading, checkpoint interchange, and any future co-located
  fine-tuning are ST-ecosystem operations. A GGUF-only posture would re-open this decision
  under time pressure the day the fine-tune track unblocks.

## 6. Decision driver 3-5 — gates, time-to-market, drafter

**Gates:** Q8_0 arm = zero new machinery (the entire standing battery). FP8 arm = the §4
package (new goldens per config, Lt-algo pin, c1-vs-c16, loop matrix) — real but bounded, and
already exercised once on the 9B-ST full battery (rig5090.jsonl:266: argmax MATCH, K=1..8
8/8 PASS).

**Time-to-market (Qwen3.8-27B next week):** day-one availability per the 3.6 precedent
(research/qwen38-prep-20260803/WATCH.md:26-28 + HF): BF16 official — certain; **official FP8 —
likely** (Qwen3.6-27B-FP8 shipped in the release month; Qwen ships -FP8 siblings for the whole
line); official GGUF — no (3.6 had none; unsloth filled the gap within days); NVIDIA NVFP4 —
~5 weeks later. Q8_0 GGUF is a same-day house conversion from BF16 (runbook §3 — Q8_0 needs
no imatrix); the FP8 checkpoint is a same-day *download* but needs the §3 loader work before
it runs. Serving readiness inside 1-2 weeks therefore only closes on the Q8_0 arm.

**MTP/drafter:** both arms carry the regime. ST side receipts: MTP head loads live from
safetensors (rig5090.jsonl:186), NV-27B ST spec K=3 = 95.4 tok/s with a corpus trim
(rig5090.jsonl:268), frspec toolchain is GGUF-free (same row). One regime cost either way:
a *requant is a new model* — law 1 requires fresh own-gen ranks per artifact
(docs/DRAFT-REGIME.md laws), so the Q8_0-27B artifact gets its own ranks+trim, not the
NVFP4 daily's.

## 6b. Arm (c) W8A8-INT8 — rejected, with reasons

Survey verdict: **dominated on sm_120a by both (a) and (b); do not build.**

1. No perf edge: INT8 and FP8 tensor-core MMA are the same-rate 8-bit pipes on Blackwell;
   the bytes are the same; there is nothing INT8 buys that FP8 doesn't.
2. Accuracy costs more machinery for less: SmoothQuant-class W8A8 needs activation-outlier
   calibration (smoothing α, calib set) to reach what FP8-E4M3 gets from a 4-mantissa-bit grid
   without calibration — the reason the field moved to FP8 for W8A8 once Hopper shipped
   (vLLM/LLM-Compressor docs position INT8-W8A8 as the pre-FP8/Ampere path).
3. Our own refutation ledger: W8A8/fp8/CUTLASS/Lt-autotune prefill GEMMs refuted at m=512 AND
   m=2048 on the H100 lane, Q8_0-EXACT int8 GEMM triple-refuted, and the residual is
   explicitly "an owner-gated accuracy decision (w8a8-class numerics change model outputs)"
   (docs/FLAGS.md:417-421). The int8-lineage k32 probe closed DOMINATED on acceptance loss
   (rig5090.jsonl:292).
4. No artifacts: nobody ships official W8A8-INT8 Qwen checkpoints; we would quantize,
   calibrate, and gate a third numeric config for zero ceiling gain.

## 7. First two weeks — work plan (agent-days, on the rented 2x5090 box)

### Track A — Q8_0 GGUF serving arm (week 1, ~5-6 agent-days, GPU-holding in bursts)

| # | Task | Sizing |
|---|---|---|
| A1 | Qwen3.8 drop day: arch-diff (runbook §2) + BF16→Q8_0 house conversion + MTP q8_0 sidecar (runbook §3; Q8_0 needs no imatrix). Control artifact: same-recipe Q8_0 of Qwen3.6-27B for a known-model shakedown before the new model lands | 1 |
| A2 | **M1**: q27-Q8_0 bring-up on the 2x5090 — single-card (tight) AND PP-2 sharded; pp2-gate bit-identity, kernel-check, run-gen argmax (short + p3-depth), run-spec K=1..8. First-ever 8-bit-27B cells | 1.5 |
| A3 | Own-gen ranks + trimmed drafter for the Q8_0 artifact (law 1: fresh per requant), `--validate` GOOD/WASH verdict | 1 |
| A4 | Board rows: Q8_0-27B plain/spec/prefill, N=5 interleaved vs the NVFP4 daily re-paired same-session; decode-bytes law check (does 2x-weight-bytes halve decode as projected, and where does PP-2 land) | 1.5 |
| A5 | serve-smoke + c1-vs-c16 + board JSON + regenerate + tag | 0.5 |

### Track B — FP8-ST hard-tuning track (week 1-2 overlapped, ~6-8 agent-days)

| # | Task | Sizing |
|---|---|---|
| B1 | Block-128 FP8 loader: extend `f8_row_scales`/the F8 arms (source.rs:838-851, 1012-1049) to 2-D block scales — dequant→Q8_0 correctness floor first (runs the full existing stack), then `MEMRA_ST_E4M3` e4m3-resident with per-block dequant in `qmatvec_e4m3_mmvq`. Gate: run-safetensors argmax on Qwen/Qwen3.6-27B-FP8 | 2-3 |
| B2 | **M2**: desktop J/token re-test — `MEMRA_ST_E4M3` decode+spec A/B vs the Q8_0 re-encode on the 2x5090 (the −7% laptop / +7.1% box split, rig5090.jsonl:256, arbitrated on the actual target) | 0.5 |
| B3 | **P1**: cuBLASLt block-scaled FP8 heuristic probe on sm_120 (extend probe/fp8_lt_scale_probe.cu); if unsupported, price the scale-fold + `MEMRA_MMQ_F8F4`-kernel fallback | 1 |
| B4 | FP8 prefill battery on the full-FP8 checkpoint at full coverage (32 GB removes the laptop budget wall): pp512/2048/6257 vs Track A same-session, TTFT, argmax + K=1..8 every cell | 1.5 |
| B5 | Exactness package: Lt-algo pin across m (chunked prime), fast-gate goldens for the fp8 config, c1-vs-c16, **6-draw sampled loop matrix** (the 27b-st-vs-gguf protocol) on the official-FP8 checkpoint | 1-1.5 |
| B6 | Promotion decision packet: FP8-ST promotes to serving arm iff ≥1.1x e2e vs Track A (board protocol, N=5 interleaved) with full battery green + 6/6 loop-clean; else it stays the prefill/LoRA development track with the JSONL row as the record | 0.5 |

Sequencing: A1-A2 are the critical path for "serving in 1-2 weeks"; B1 starts immediately in
parallel (no GPU needed until its gate). Both tracks fit the window with two agents.

### Named measurements this doc is blocked on (all runnable on the 2x5090 now)

- **M1** (A2): first q27-at-8-bit decode/prefill/spec cells, single-card + PP-2.
- **M2** (B2): e4m3-vs-Q8_0 decode under desktop power headroom — decides the FP8 arm's
  decode sign on the target.
- **P1** (B3): cuBLASLt block-scaled-FP8 support probe on sm_120.
- **M3** (B5): sampled loop matrix on official Qwen FP8 — quality parity beyond argmax.

## 8. Source ledger

Repo (all paths relative to repo root at 69cdd1eb):
`research/tune-data/g7e-rtx6000.jsonl:35,39` · `research/tune-data/rig5090.jsonl:184,186,204,
233,234,256,266,268,292,294` · `crates/memra-gguf/src/safetensors.rs:28-46` ·
`crates/memra-gguf/src/source.rs:838-851,960-1112` · `crates/memra-engine/src/fp8_ffi.rs:1-62`
· `crates/memra-engine/cu/fp8_prefill.cu:14,143-148` · `crates/memra-engine/src/mmq_ffi.rs:911-
945` · `crates/memra-server/src/worker.rs:610-656` · `docs/FLAGS.md:81-86,131,219,319,321,323,
385-386,417-421` · `docs/DRAFT-REGIME.md` (laws 1-3) · `docs/qwen38-bringup-runbook.md` ·
`research/qwen38-prep-20260803/{AUDIT.md,WATCH.md}` · `research/tune-data/current-board.json`
(plain_decode, speculative, h100_board, supported_models) · `research/tune-data/
27b-st-vs-gguf-final.md` · `research/verify-tier-20260802/{RESULTS.md,glue-attribution.md}` ·
`research/kv-compress-20260802/REPORT.md` §1 · `research/deltaserve-assessment-20260803/
ASSESSMENT.md` · `research/hw-buy-20260802/REPORT.md:41-44,443-455` ·
`ARCHITECTURE-H100.md:2208-2222` · `docs/decisions/FORMAT-DECISION.md` (the standing
"rig-native internal layout; GGUF and ST are import formats" ruling this doc operates under).

Web (fetched/searched 2026-08-03): huggingface.co/Qwen/Qwen3.6-27B-FP8 (full card fetched:
"fine-grained fp8 quantization with block size of 128") · docs.vllm.ai/en/v0.21.0/features/
quantization/fp8/ · developers.redhat.com/articles/2024/07/15/vllm-brings-fp8-inference-open-
source-community · github.com/vllm-project/vllm/issues/33301 (FP8 LoRA RFC) ·
developer.nvidia.com/blog/model-quantization-post-training-quantization-using-nvidia-model-
optimizer (FP8_CFG = per-tensor static W8A8) · nvidia.github.io/TensorRT-LLM (sm100 MXFP8
recipe note) · spheron.network ModelOpt guide 2026-05 (FP8→Hopper, NVFP4→Blackwell targeting).
Not verified first-hand: SmoothQuant original paper numbers (characterized via vLLM/
LLM-Compressor docs positioning); cuBLASLt block-scaled FP8 sm_120 support (probe P1).

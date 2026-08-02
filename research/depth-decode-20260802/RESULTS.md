# depth-decode: is decode-at-depth KAT-specific or class-wide? (2026-08-02)

Lane `lane/depth-decode` (from `restructure/public-split` 0983c4ba — post f16g-default-rearb
merge). Rig: RTX 5090 Laptop 24463 MiB sm_120a, 82 SMs, platform_profile `performance`,
`gpu-full-power on`. Every GPU run under `flock /tmp/gpu5090.lock`; a co-lane
(bw24-agentworld) actively shared the rig this session — its runs serialize through the
same lock and its heat is inside these figures (per-row `temp_c` in every jsonl: memra rows
66-78 °C, llama rows 63-85 °C). Co-resident `llama-server --embedding` (332 MiB) allowlisted.
llama.cpp arm: the local fork `llama-bench` (`/home/avifenesh/projects/llama.cpp/build/bin`,
same binary class as the iq-direct/kquant lanes). Models (all naked memra defaults, no flags):

- KAT `/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf`
- q35 `/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
- o35b `/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf`

## 0. Measurement shape (stated exactly)

Depth prompts are four prefixes of ONE document (`research/e2e/prompts/p4-16k.txt`, a code
dump — code-class content) cut to exactly {512, 2048, 4096, 6144} tokens by each model's own
tokenizer (all three share the qwen vocab — identical cut points; `make-depth-prompts.py`,
`depth-prompts-manifest.jsonl`). Same document ⇒ content class is CONTROLLED across depth
points (the p1/p2/p3 board prompts vary content AND length; this sweep isolates length).

- **memra arm**: naked `run-gen`, `MEMRA_NGEN=128`, the depth prompt via `MEMRA_PROMPT_FILE`.
  Cell value = the printed **gen-only** rate over 128 greedy tokens (prime excluded; host
  argmax inside the timed span). KV depth during the window: D → D+128. Every run carries the
  prefill/decode argmax gate (72/72 runs MATCH).
- **llama arm**: `llama-bench -p 0 -n 128 -d 512,2048,4096,6144 -r 1 -ngl 999 -fa 1
  -ctk q8_0 -ctv q5_1` (the established denominator KV config = memra's default formats).
  tg128@dD = 128 decode steps timed after an UNTIMED depth-fill of D tokens; llama-bench's
  gen loop feeds synthetic tokens and runs no sampler.

Both arms time 128 batch-1 decode steps from the same KV depth on the same gguf. They differ
in decoded-token content (real greedy continuation vs synthetic) and in host sampling cost
(memra's argmax is inside its span; llama-bench has none) — both differences are ≤ the
short-ctx parity cells show, and they cancel in the DECAY comparison (each arm vs itself).
Interleaved x3 process rounds, round-robin per rep (memra 4 depths → llama 4-depths-in-one,
per model, models rotating), one session. N=3 per cell, medians reported, full per-rep values
in `depth-sweep.jsonl`. Housekeeping: the jsonl carries 4 orphan rows from an aborted first
launch (`ts 2026-08-02T07:24:00Z`, kat/memra/d512) — excluded by `summarize-depth.py`; the
relaunched session re-measured that point.

## 1. THE DIG №1 — the depth-decay table (decode tok/s, tg128 @ depth, N=3 medians)

| model | engine | d512 | d2048 | d4096 | d6144 | decay 512→6144 |
|---|---|---|---|---|---|---|
| KAT | memra | 184.9 | 176.6 | 173.2 | 170.8 | **−7.6%** |
| KAT | llama | 180.7 | 182.6 | 176.3 | 173.5 | −3.9% |
| KAT | ratio | 1.023x | 0.967x | 0.982x | 0.984x | |
| q35 | memra | 177.1 | 170.5 | 164.5 | 159.8 | **−9.8%** |
| q35 | llama | 156.3 | 153.7 | 149.4 | 147.9 | −5.4% |
| q35 | ratio | 1.133x | 1.109x | 1.101x | 1.081x | |
| o35b | memra | 196.2 | 187.8 | 186.9 | 180.9 | **−7.8%** |
| o35b | llama | 179.9 | 178.2 | 173.5 | 170.3 | −5.4% |
| o35b | ratio | 1.091x | 1.054x | 1.078x | 1.063x | |

Per-rep spreads and temps: `depth-sweep.jsonl` (e.g. KAT memra d6144: 167.3/170.8/171.4).
The co-lane makes single cells noisier than a quiet-rig session (q35 llama d6144 spread
138.4-155.0); decay percentages are computed within-arm (each engine's four depths share a
process for llama, and adjacent same-rep runs for memra), so the regime largely cancels.

**VERDICT: CLASS-WIDE, not KAT-specific.** memra decays 7.6-9.8% over 512→6144 on ALL three
models vs llama's 3.9-5.4% on the same ggufs. In absolute terms (Δms per token, d512→d6144):

| model | memra Δ | llama Δ | memra/llama depth cost |
|---|---|---|---|
| KAT | +0.447 ms | +0.230 ms | 1.94x |
| q35 | +0.612 ms | +0.363 ms | 1.69x |
| o35b | +0.431 ms | +0.313 ms | 1.38x |

memra pays ~1.4-1.9x llama's per-token cost of context depth on every supported model of this
class. KAT is not mechanically special — it merely sits at short-ctx parity (1.02x) instead
of ahead (q35 1.13x, o35b 1.09x), so the same decay drops it *under* llama by d2048 while
q35/o35b stay above water (q35 1.13x→1.08x, o35b 1.09x→1.06x). This matches the published
board's own drift (q35 178.2/167.8=1.062x short vs 163.7/160.9=1.017x at d6257). All three
models share the attention geometry exactly (10-11 full-attention layers of ~40, hd256,
n_head_kv=2, remainder SSM/linear — gguf header dumps), so the class result is by
construction: same kernels, same ladder, same decay.

Side receipt: o35b memra decode is now 196.2 @ d512 and **beats llama at every depth**
(1.06-1.09x) — the resident-if-fits budget landed since the ornith-bar 0.72x spill-era cell;
its decision line (`experts 19.50GB + trunk 1.65GB vs free 23.92GB → RESIDENT`) is in every
o35b log.

## 2. THE DIG №2 — mechanism (nsys, KAT d512 vs d6144)

`run-mech.sh nsys`: `MEMRA_PROFILE_GEN=1` + `nsys -c cudaProfilerApi` brackets the timed
generate window (NOTE: generate_with primes inside that span, so the capture includes the
prime's prefill kernels; decode-only rows are isolated by instance count — 1280 = 128 tokens
x 10 attn layers, 3840 = x30 SSM layers, 5120 = x40 layers). Reports:
`nsys-kat-d{512,6144}_cuda_gpu_kern_sum.csv`.

- **The fa kernel class does NOT switch with depth**: `fa_decode_vec_q_v4_dc` +
  `fa_decode_combine_f32` at BOTH 512 and 6144 (1280 instances each). No fa512 (hd512 is
  gemma-only), no smem-floor flip, no v4-window crossing — for this class the only
  segmented-recapture boundary between 512 and 6144 is the split-ladder rung at t_kv 3072
  (sp8 → sp64; `fa_split_keys`, lib.rs).
- **The depth cost is the vec kernel itself**: per layer-token, vec+combine = 12.3 µs @ d512
  (5.8 + 6.5) → 44.3 µs @ d6144 (35.4 + 8.9). Δ = 31.9 µs x 10 layers = **319 µs/token of
  the measured +447 µs/token** — the attention split kernels are the wall; dp4a trunk/expert
  matvecs and the gdn/SSM kernels are depth-flat (as expected: ctx-invariant).
- **Efficiency**: at d6144 the vec kernel walks ~6.2k keys x (544 B K-q8_0 + 384 B V-q5_1)
  ≈ 5.75 MB per layer-token in 35.4 µs ≈ **162 GB/s effective — 19% of the 5090's 858 GB/s**.
  llama's measured depth cost implies ~23 µs/layer-token all-in (~250 GB/s class) on the same
  bytes: memra's per-key walk is ~1.9x costlier. This is a kernel-efficiency gap, not a
  dispatch gap.

**Threshold re-checks (stale-verdict law — both REFUTED as levers, receipts `mech.jsonl` +
`mech-console.log` + `sweep-fa-*.log`, single runs to rank, argmax MATCH every run):**

- `MEMRA_FA_SPLIT` re-sweep at d6144 (KAT): naked (=sp64 via ladder) **173.7** vs sp96 168.6,
  sp16 165.1, sp32 163.6, sp128 157.5, sp8 155.3. The 2026-07-08 ladder rung survives on
  current kernels — no change to ship.
- `MEMRA_FA_V4_MAX=3072` probe (v4 off above the rung — the gemma depth lesson): 164.7 vs
  naked 166.2 back-to-back. v4 stays right at depth on the 5090 — no change to ship.

So nothing in the small+gated class moves the needle: the current selection thresholds are
already optimal. **What shipped = these re-verifications + the board depth cells; the fix is
priced (§4), not built here.**

Gate note: KAT d4096 runs print the documented batched-prime `FLIP-NEARTIE` diagnostic
(`tokenwise margin 0.3375`, identical line 3/3 reps — the REPORTED non-fatal cross-config
drift class from residency-cap/iq-direct lanes); the prefill/decode argmax gate is MATCH in
all 72/72 sweep runs and every mech run.

## 3. THE DIG №4 — drafter acceptance at CONTROLLED depth (the second KAT gap)

`run-depth-accept.sh`: run-spec, own-gen trimmed drafters at serving K=2, NGEN=256, greedy,
same four same-document prefixes (depth isolated from content). x2 reps: acceptance under
greedy is deterministic per (prompt, K) — det-OK 2/2 at every point, self-consistency PASS
16/16 runs. Raw: `depth-accept.jsonl`, `acc-*.log` (plain rates: raw logs canonical — the
jsonl `plain_decode_toks` regex hits the known run-spec column-padding nit).

| model | d512 | d2048 | d4096 | d6144 |
|---|---|---|---|---|
| KAT acceptance @K=2 | 58.9% | 63.8% | 64.9% | **73.8%** |
| o35b acceptance @K=2 | 65.8% | 54.9% | 70.3% | **73.1%** |

**Acceptance does NOT decay with depth — it RISES** (deeper same-content context conditions
both target and drafter harder; the continuation gets more predictable). The p1/p2/p3 slide
(KAT 82.5→62.8→52.0%, o35b 65.9→63.8→63.8%) is therefore **content-class-driven, not
context-depth-driven** — p1/p2/p3 vary both, and this sweep isolates the depth axis to zero
(negative slope refuted).

**Corpus hypothesis receipted and REFUTED on the depth axis**: the own-gen corpora are
ngen-512 greedy continuations of the 254-prompt pack (36 chat/59 code/3 e2e/6 tool/150 wiki
— `research/ornith-drafters-20260801/RECIPE.md`), so every token of trim-rank signal lives
at ctx ≲1k. Despite that shallow corpus, acceptance IMPROVES at depth 6144. A deep-corpus
rebuild aimed at *depth* would chase the wrong variable — correctly NOT rebuilt, and not
worth pricing as a depth fix. The real acceptance lever for p2/p3 is corpus/content-class
coverage (agentic-class own-gen prompts), a drafter-lane item, unpriced here (it needs its
own acceptance-vs-corpus-mix sweep to price honestly).

## 4. What's priced (the class-wide lever, not built in this lane)

A **deep-ctx fa_decode vec kernel lane** (sm_120a): rewrite `fa_decode_vec_q_v4`'s key walk
to llama's flash-decoding structure on the same q8_0/q5_1 bytes — wider per-warp K-tiles with
16 B-coalesced loads, fewer/fatter partials into the existing combine. Copy-then-tune, one
kernel family, existing harness. Evidence for the target: memra 5.7 ns/key/layer at d6144 vs
llama's implied ~3.7 — closing to llama-cost is worth ~+3.7% e2e decode at d6144 on every
model of this class (KAT 170.8 → ~177, which alone moves the KAT p3 decode gap 0.946x →
~0.98x), proportionally more beyond 6k, ~0 at d512 (attention is 2.3% of the wall there —
this is precisely the depth lever). Gates: the kernel-check fa battery, per-run argmax,
run-spec K=1..8, and this lane's depth cells as the before/after instrument.

## Files

`make-depth-prompts.py` + `depth-prompts-manifest.jsonl` + `depth-{512,2048,4096,6144}-{kat,q35,o35b}.txt`;
`run-depth-sweep.sh` → `depth-sweep.jsonl`, `sweep-console.log`, `mem-*.log` (36),
`llama-*.log` (9), smokes `smoke-mem-kat-d512.log`/`smoke-llama-kat.log`;
`run-mech.sh` → `nsys-kat-d{512,6144}.log` +
`nsys-kat-d*_cuda_gpu_kern_sum.csv` (the `.nsys-rep`/`.sqlite` binaries stay local-only per
`.gitignore`, g26-decode precedent), `mech.jsonl`, `mech-console.log`, `sweep-fa-split-*.log`,
`sweep-fa-v4max-*.log`; `run-depth-accept.sh` → `depth-accept.jsonl`, `accept-console.log`,
`acc-*.log` (16); `summarize-depth.py` (canonical medians incl. the orphan-row filter).

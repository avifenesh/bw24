# q4k-expert-prefill: the Ornith-35B prefill lever — Q4_K experts join the grouped-f16 lane (2026-08-02)

Lane `lane/q4k-expert-prefill` (from `restructure/public-split`, 5cfad376). Rig: RTX 5090
Laptop 24463 MiB, driver 595.71.05, platform_profile `performance`, `gpu-full-power on`.
Every GPU run under `flock /tmp/gpu5090.lock` (one co-lane shares the rig; the co-resident
`llama-server --embedding` is allowlisted). llama.cpp arm: local fork build `bb090d1f1`, same
binary as the ornith-bar/residency-cap lanes. Model:
`/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf` (RESIDENT since the
residency merge; decision line in every log). Session: one window, 22:4x–23:40Z, busy-proc
gate before every arm, temps 64–75 °C (per-row `temp_c` in the jsonls). All medians N=3
process-interleaved unless stated; every pp2048 process value is itself the median of 5
in-process reps (+1 warmup), the sk-bm128 protocol.

## 1. Why the round-49 FLAT verdict did not transfer (stale-verdict law)

The round-49 5090 verdict ("f16g FLAT — f16 workspace traffic cancels the GEMM win at
858 GB/s", `research/expert-grouped-20260801/`) was measured on q35, whose IQ4_XS-class
expert layers ride the int8-MMA MMQ tiles (`mmq_iq_experts.cu`) as the baseline. Ornith-35B's
experts are Q4_K (gate/up x40, down x20) + Q6_K (down x20) — `q8_expert_dec_supported`
rejects all of them, so its prefill baseline was the per-pair `moe_pairs_matvec_q8_em`
fallback: a warp-per-row dp4a matvec that re-reads and re-decodes each expert's weights for
every (token, expert) pair — zero token reuse over a ~19.5 GB bank. Against THAT baseline
the dequant-once + grouped GEMM trade is a multiple, not a wash.

## 2. Door sweep — interleaved x3, same session (`door-sweep.jsonl`)

Arms: `mmq` (naked pre-flip default = `_em` fallback), `f16g1` (`MEMRA_MOE_F16G=1`, cublas
grouped), `f16g2` (`=2`, single-kernel sk visitor, hybrid cross=64). Two shapes per arm-rep:
board-2048 pp-only and pp512+128 run-gen (board shape).

| arm | pp2048 (med, N=3) | pp512 | decode tg128 | argmax+prime gates | peak VRAM MiB |
|---|---|---|---|---|---|
| mmq (old default) | 1098.2 [1098.0–1106.2] | 1081.1 | 208.7 | MATCH 2/2 x3 | 21618 |
| f16g1 | 3317.6 [3271.9–3330.1] | 1425.6 | 210.5 | MATCH 2/2 x3 | 22524 |
| f16g2 | **3453.7** [3450.1–3456.2] | **1662.3** | 208.7 | MATCH 2/2 x3 | 22458 |

**f16g2 = 3.14x board-2048, 1.54x pp512, decode flat, zero overlap between arms.** Token
sha per arm is rep-stable; the f16-mirror numeric class shifts prefill logits (arm shas
differ) but every run holds `prefill argmax == decode argmax` MATCH and `batched-prime`
MATCH. VRAM headroom holds (peak 22.5 of 24.5 GiB beside the 21.4 GiB resident model).

Knob sweep (sweep-grade sequential, med of 5 in-process reps each, `knob-sweep.jsonl`):
sk0 grid-scan 3289.5 / sk32 3306.8 / sk128 3323.7 / cross32 3452.3 / **cross64 3459.9** /
cross128 3425.6 / cross256 3393.3 — the sk-bm128 hybrid default (cross=64) is already the
o35b winner; no knob change.

## 3. What shipped: AUTO-KQUANT (MEMRA_MOE_F16G mode 3), the sm_120a naked default

Winners are defaults, but the round-49 FLAT verdict is still true for the IQ-bank class —
so the default is the admission, not a blanket flip: **unset `MEMRA_MOE_F16G` on sm_120a now
admits the mode-2 sk form only for expert layers whose qtypes the MMA MMQ arm rejects**
(Q3_K/Q4_K/Q6_K — exactly the `_em`-fallback class). IQ3_S/IQ4_XS/Q4_0 banks (q35, KAT)
keep their measured-faster MMQ tiles. Keyed on qtype capability, not `use_mma`, so
`MEMRA_MOE_MMA=0` stays a pure dp4a rollback seam. Explicit `=1`/`=2` force all-layer
admission (A/B door), `=0` kills, Hopper keeps mode 1, the gemma (gelu) site stays
env-explicit-only. Code: `moe_f16g_mode()` (lib.rs) + the pairs-arm admission
(hybrid_forward.rs); FLAGS.md rows updated.

### Gates (all green, receipts in this dir)

- `kernel-check` **ALL GREEN**, 0 FAIL (`kernel-check-post.log`).
- o35b naked post-flip (x2): pp2048 3466.1/3461.7 == the explicit f16g2 arm; pp512
  1664/1666; decode 209.2/209.1; argmax + batched-prime MATCH; token sha ==
  the door-sweep f16g2 arm (`c0c12c3b350dc7f5`) — the default engages, nothing else moved.
- **q35 ctrl guard** (pre-flip vs post-flip binaries, naked, interleaved x3):
  board-2048 3449.6 → **4070.6** med (+18.0% — q35's k-quant straggler layers rode `_em`
  too; ranges disjoint 3437–3450 vs 4067–4079); gen512 prefill 2330 → 2450; run-gen argmax
  MATCH 6/6; `batched-prime` FLIP-NEARTIE in BOTH arms (the documented #46 non-fatal class,
  pre-existing); generated token shas identical pre/post; tokenwise decode 189.8/191.6 pre
  vs 191.5/191.3 post (flat, valid 128-token cells, above the 178.2 board row). The naked
  q35 pp512 2-token-EOS quirk (residency-cap §4 branch finding) reproduces identically in
  both arms — untouched by this change.
- o35b `run-spec` K=1..8 with the adopted own-trim drafter: **8/8 self-consistency PASS**
  (mission bar was K=1..4), spec K=2 257.9 tok/s at board shape.

## 4. Bar check vs llama — same gguf, same session, interleaved x3 (`barcheck.jsonl`)

Board shape, plain-vs-plain: memra pp512 1647.6 / pp2048 3450.3 / tg128 208.5 vs llama-bench
(`-ngl 999 -fa 1 -ctk q8_0 -ctv q5_1`) pp512 3972.3 / pp2048 3803.7 / tg128 192.1.

- **prefill ratio: pp512 0.415x (was 0.274x), pp2048 0.907x** — the prefill gap closes with
  prompt length: at p3 length (6257 tok) memra primes at 3867 tok/s vs llama 3643 = **1.06x,
  a memra WIN**. pp512 remains the weak point (see §5).
- plain e2e (512+128): 0.925 s vs 0.795 s = 0.860x (was 0.74x post-residency, 0.41x at #44).
- decode: 1.086x plain (unchanged — this lane didn't touch decode).

**Best-vs-best per class** (the board's deployment convention: memra = adopted drafter spec
K=2, all self-consistency PASS 9/9; llama = plain — its draftless spec doors are
structurally broken on this arch, `research/ornith-bar-20260802/llama-spec-doors-screen.md`).
e2e = prime wall + 256/decode-rate, the generous-to-llama variant; llama rates from the same
interleaved llama-bench call (`-p 27,1845,6257 -n 256`):

| class | memra e2e (prime + decode) | llama e2e | ratio | 1.1x bar |
|---|---|---|---|---|
| p1-code-short (27 tok) | **1.033 s** (0.096 + 256@273.3) | 1.357 s | **1.314x** | **PASS** |
| p2-code-medium (1845 tok) | **1.618 s** (0.569 + 256@244.1) | 1.837 s | **1.136x** | **PASS** |
| p3-agentic-long (6257 tok) | **2.738 s** (1.618 + 256@228.6) | 3.053 s | **1.115x** | **PASS** |

(memra plain decode by class: 205.4 / 198.6 / 196.1; acceptance 68.1% / 62.7% / 59.9%,
rep-identical. Rep spreads: memra prime ±0.001 s, spec decode ±0.4%; llama ±1%.)

**VERDICT: Ornith-1.0-35B Q4_K_M CLEARS the deployment bar on every prompt class —
best-vs-best e2e 1.314x / 1.136x / 1.115x ≥ 1.1x. DEPLOY-grade under the board convention**
(the same rule that shipped Ornith-9B). The #44 → residency → this-lane arc closes: decode
0.72x → 1.086x (residency), prefill 0.274x → 0.415x/0.907x/1.06x by length (this lane), and
the class e2e bar is green across the board.

## 5. What remains (priced, not built here)

- **pp512-class short-prompt prefill (0.415x).** nsys mechanism profile (N=1, labeled:
  `nsys-pp512-kernsum.txt`): at t=512 the q4_K/q6_K → f16 dequant passes are **41.8%** of
  GPU kernel time (dequant_q4k 33.8% + dequant_q6k 8.0%) vs 35.1% for the sk GEMMs — the
  dequant is a fixed per-(layer,projection) cost over the ~all-active expert bank
  (~44 GB f16 write+read per pass, already at the 858 GB/s wall), which amortizes at 2048+
  tokens but dominates at 512. The kill is not a faster dequant, it's no dequant: **Q4_K +
  Q6_K tile loaders in `mmq_iq_experts.cu`** (direct-from-quant expert MMQ; the load_tiles
  machinery is already vendored in `mmq_q45k.cu` for the trunk, llama.cpp carries
  load_tiles_q6_K) — the same piece the KAT trunk-MMQ lane (kat-anomaly §6) needs, expert-
  segmented instead of dense. Byte-shift numeric class, argmax/spec gated, own lane.
- Trunk stragglers at pp512: `qmatvec_gemm_q6_K` 5.9% + `mul_mat_q_q45k` 5.5% — second-order.
- **Merge-time board note:** this flip also moves the q35 5090 board-2048 prefill row
  (+18%, §3) and flips the Ornith-35B verdict — `current-board.json` + README regeneration
  are owed in the merge/tag commit per the perf-board rule (not done on this lane branch).

## Files

`run-door-sweep.sh`, `run-knob-sweep.sh`, `run-gates.sh`, `run-barcheck.sh`;
`door-sweep.jsonl`, `knob-sweep.jsonl`, `gates.jsonl`, `barcheck.jsonl` (N stated per row);
per-run logs `pp2048-*`, `gen512-*`, `knob-*`, `o35b-post-*`, `q35-guard-*` (incl.
`q35-guard-{pre,post}-tokenwise-rep*`), `bar-*`; `kernel-check-post.log`,
`o35b-post-spec-k1-8.log`, `smoke-f16g2.log`, `token-hashes.log`, consoles
`sweep-console.log`/`gates-console.log`; nsys: `nsys-pp512-auto.log`,
`nsys-pp512-kernsum.txt` (the `.nsys-rep` stays local per repo convention — `*.nsys-rep`
is gitignored).

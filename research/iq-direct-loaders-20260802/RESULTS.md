# iq-direct-loaders: IQ4_XS/IQ3_S direct-from-quant sk tile loaders — the coverage flip (2026-08-02)

Lane `lane/iq-direct-loaders` (from `restructure/public-split` 94155921; kernel commit
cf14e3ad). Rig: RTX 5090 Laptop 24463 MiB sm_120a, platform_profile `performance`,
`gpu-full-power on`. Every GPU run under `flock /tmp/gpu5090.lock` (co-resident
`llama-server --embedding` 332 MiB allowlisted, inside every figure). llama.cpp arm: local
fork `llama-bench` (the kquant-lane binary), same-session interleaved. Models: q35
`/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, KAT
`/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf`
(+ its owntrim drafter), Ornith-35B Q4_K_M (guard only). All perf claims interleaved process
rounds (arms round-robin per rep, same box same hour, temps 58-73 C across the battery);
each pp value = the run-gen in-process median of 5 reps (+1 warmup) — the sk-bm128 protocol.
N stated per cell.

## 1. What was built (the h100-sk-direct flip rung)

h100-sk-direct priced it: direct-from-quant's win is coverage-proportional, and Q4_K/Q6_K
cover only 5.2% of q35's expert-bank bytes — the IQ3_S/IQ4_XS bulk (94.8%) still paid the
mode-2 dequant-workspace pass. This lane extends the kquant-tile-loaders discipline to both
IQ classes on ALL sk visitor forms (sk32v, sk128v, sktail — the same `moe_kq_*` templates,
two new QT instantiations each):

- **IQ4_XS**: a 16-aligned window lives inside one 32-value scale group, and both halves of
  the group read the SAME 16 qs bytes (`sel` picks the nibble half) — KqRaw carries them in
  q[4] (2x uint2; 136 B superblocks keep rows 8-aligned, not 16). The hoisted scale
  `f1 = d_sb*(float)(ls-32)` is the workspace expression's left-assoc first product. The 16
  `kvalues_iq4nl` ride SHARED memory as f32 (small ints are f32-exact), so the store is one
  shared load + fmul per value.
- **IQ3_S**: window = two 8-value chunks; the four grid words resolve AT FETCH through a
  512-word shared codebook (divergent per-value constant-cache lookups serialize; shared is
  banked), so the codebook lookups fly behind the previous kb's mma with the raw-byte
  prefetch. KqRaw q[4] = resolved grid words, qh[0] = the 2 sign bytes,
  `f1 = dd*(1+2*nib)` hoisted. 110 B superblocks are only 2-aligned: u16/u8 loads throughout.
- Register-pipelined exactly like Q4_K/Q6_K: kb+1's raw fetch issues behind kb's mma; B tile
  stays single-buffered (trailing `__syncthreads` fences). The C launcher's q4k/q6k if/else
  ladder became a per-QT template with per-instantiation occupancy statics.
- Rust admission opens the same seam to `QT_IQ4_XS`/`QT_IQ3_S`; `MEMRA_F16G_DIRECT` gains
  `=kq` (keep k-quant loaders, IQ back to workspace — this lane's A/B arm and the exact
  sk-tail-3728.7 configuration). Q3_K/Q4_0 keep the workspace path (q35 carries ONE Q3_K
  gate/up layer — sub-1% of bank bytes; coverage is now ~100% minus that layer).

**Numeric class: NONE — bit-identical to the workspace path by construction.** Each store
writes the workspace dequant kernels' exact per-value expressions (left-assoc products
hoisted unchanged) into the same smem positions, so every output element's mma k-chain is
untouched. ptxas sm_120a: **0 spills, all 12 kq instantiations** (iq4_xs 72/92/112 regs for
sk32v/sktail/sk128v, iq3_s 78/104/124; static smem iq3_s sktail 27152 B — everything under
the 48 KB no-opt-in limit; `ptxas-sm120a-kq.txt`).

## 2. Gates (all green)

- kernel-check **ALL GREEN rc=0, 0 FAIL / 382 OK, no KC-SKIP** (`kernel-check-r1.log`):
  `f16g-kq-direct` now gates iq4_xs + iq3_s synthetic (skew CSR 1..300, reversed ex_ids,
  random payloads — in-range for every field class) AND real q35 weights
  (blk.0.ffn_gate_exps IQ3_S in=2048 out=512; blk.0.ffn_down_exps IQ4_XS in=512 out=2048)
  vs the workspace path: **maxdiff = 0.00e0 (byte-identical), every visitor form**
  (hybrid / all-128 / all-32-deep-tail / all-32-legacy-tail), on top of the carried
  q4_K/q6_K + f16g-sk arms.
- **q35 F16G=2 end-to-end bit-identity**: gen512 token sha `e94b6553fde7b9a0` BOTH arms
  (old = `MEMRA_F16G_DIRECT=kq`, new = naked) == the sk-tail mode-2 anchor; argmax MATCH.
- **q35 naked guard x3**: token sha `86dc5f7105a3716b` == the q4k-expert-prefill anchor 3/3
  — and naked q35's five straggler layers' IQ3_S projections now ride direct (pp2048
  4182.6-4200.7 vs prior-lane 4099.8-4108.6, gen512 prefill 2618.5-2635.0 vs 2510.9-2527.7:
  flat-or-better on every column; cross-session comparison, sha-anchored). The
  batched-prime FLIP-NEARTIE diagnostic line is bit-identical to both prior lanes' logs
  (same maxdiff 2.411e0, same margin 0.6146) — pre-existing, untouched.
- **Ornith-35B anchor**: naked gen512 sha `c0c12c3b350dc7f5` 2/2, argmax MATCH — the Q4_K
  direct path went through the launcher template refactor and did not move a bit
  (prefill 3494-3502 = the sk-tail deep-tail cell, decode flat).
- **KAT naked**: sha `e5d59ecedc57aa7d` == the kquant-lane mmq anchor (auto-kquant keeps
  MMQ tiles on MMA-capable layers — dispatch-unchanged by construction). **KAT F16G=2**:
  sha `9102ffd0b8241a65` BOTH arms — byte-identity end-to-end on a pure-IQ4_XS bank;
  argmax MATCH every run, sha rep-stable 3/3 in the AB.
- **run-spec self-consistency**: q35 (F16G=2, owntrim draft, p2, NGEN=64) **PASS x8
  (K=1..8)** — covers the K=1..4 mission gate. KAT F16G=2 spec K=2 self-consistency PASS
  in all 9 barcheck class runs (§4).

## 3. q35 — the mode-2 arm (mission cell) + the flip evidence

**Board-2048 pp e2e, MEMRA_MOE_F16G=2, x5 process-interleaved (`ab.jsonl`, git fe05f60a):**

| arm | reps (tok/s) | median |
|---|---|---|
| old (IQ workspace: `MEMRA_F16G_DIRECT=kq`) | 3772.6, 3734.6, 3734.0, 3734.9, 3733.3 | 3734.6 |
| new (IQ direct, naked) | 5603.9, 5630.7, 5622.4, 5622.4, 5608.1 | **5622.4** |

**+50.5% on the mode-2 arm, zero overlap** (min new 5603.9 > max old 3772.6). The old arm
reproduces the sk-tail 3728.7 cell (same config, same protocol). Killing the IQ workspace
pass is worth ~3x what the deep tail was worth (+3.7%) — coverage-priced, as h100-sk-direct
predicted. gen512 prefill (t=512, single gate runs): 1844.1 -> 3980.3.

**The stale verdict (x3 interleaved, `q35-flip-board2048`):** naked (auto-kquant: IQ layers
on int8-MMQ tiles) 4207.4 / 4180.1 / 4172.8 (med 4180.1) vs `MEMRA_MOE_F16G=2` 5591.0 /
5616.6 / 5598.5 (med **5598.5**) — **mode-2 + IQ-direct now beats the naked default by
+33.9%, zero overlap.** The mode-3 policy line "IQ4_XS-bank models keep their
measured-faster MMQ tiles" was measured pre-IQ-direct and is now refuted on the 5090 for
prefill (decode is untouched either way — the t>=16 floor keeps decode/verify on dp4a).
What a default flip needs (not shipped here — §5).

## 4. KAT-Coder — bar re-check + the DEPLOY question

Same-session interleaved x3 per arm; llama denominators pooled across both barcheck passes
(N=6 board / N=6 class, same hour, temps 71-73 C on the llama runs).

**Plain board (`barcheck-kat.jsonl` + `ab.jsonl`):**

| leg | memra naked | memra F16G=2+direct | llama | f16g2 ratio | was (kquant lane) |
|---|---|---|---|---|---|
| pp512 | 2066.3 | **3025.9** [AB: 3034.8] | 4258.6 | **0.711x** | 0.484x |
| pp2048 | 3035.9 | **3960.7** [AB: 3970.3] | 4138.7 | **0.957x** | 0.735x |
| decode tg128 | 194.94 | 194.92 | 194.72 | 1.001x (parity) | 0.998x |
| plain e2e 512+128 | 0.904 s (0.860x) | **0.826 s** | 0.778 s | **0.942x** | 0.858x |

**Best-vs-best per class** (memra = min(plain, spec K=2) with the owntrim drafter, spec
self-consistency PASS every run; llama plain, same interleaved `-p 27,1845,6257 -n 256`
call; e2e = prime wall + 256/decode-rate):

| class | memra naked best | memra f16g2 best | llama e2e | f16g2 ratio | 1.1x bar | was |
|---|---|---|---|---|---|---|
| p1-code-short (27) | 1.156 s | **1.132 s** (spec K=2, acc 82.5%) | 1.340 s | **1.184x** | **PASS** | 1.156x |
| p2-code-medium (1845) | 1.881 s | **1.783 s** (spec K=2, acc 62.8%) | 1.778 s | 0.997x | FAIL | 0.952x |
| p3-agentic-long (6257) | 3.364 s | **2.959 s** (plain, acc 52.0%) | 2.904 s | 0.981x | FAIL | 0.863x |

**VERDICT: HOLD (pre-deployment)** — the bar is 1.1x on every class. But the misses moved
0.952x -> 0.997x and 0.863x -> 0.981x, and the gap CHANGED CLASS: the MoE expert prefill
that kquant-tile-loaders named as bar-binding is now CLOSED — memra's prime wall is at
llama parity or better on every class (p2 0.468 s vs llama's 0.459 s; p3 **1.565 s vs
1.585 s — memra primes faster**). The remaining bar-binding gap is (a) decode-at-depth
(plain 183.7 vs llama 194.1 tok/s at 6k ctx = 0.946x — flat-context decode is parity at
194.9) and (b) drafter acceptance at depth (62.8% / 52.0% at p2/p3 vs 82.5% at p1; the
f16g2 numeric class shifts acceptance -5.7pp on p2, +3.9pp on p3 — spec still
self-consistent every run). Those are drafter/decode lanes, not expert-prefill lanes.
Note: the f16g2 class cells run under the EXPLICIT env — deployment at these numbers
requires the §5 default flip (naked = full speed doctrine).

## 5. The default-flip case (evidence complete, promotion NOT shipped here)

Mode-3 auto-kquant currently admits f16g only where MMQ can't take the layer. With IQ
direct, the f16g+direct arm beats the MMQ arm on BOTH IQ-bank models at every prefill
depth measured (q35 +33.9% board-2048; KAT +46.7% pp512 / +30.6% pp2048), decode untouched
(t floor), and every exactness gate for the new class ran green this session (argmax MATCH,
sha rep-stable, q35 spec K=1..8 PASS, KAT spec K=2 PASS x9). A naked-default flip
(admit f16g for layers whose three projections are direct-covered: Q4_K/Q6_K/IQ4_XS/IQ3_S)
is a numeric-class change for naked q35/KAT and therefore needs, per promotion discipline:
new naked token-sha anchors (the mode-2-class streams), the serve/graph battery on the
affected models (decode-batch gates pin F16G=0 in-binary and are immune), board re-cells,
and the README/perf-card regeneration in the same merge commit (perf-board rule). Q4_0
banks (gemma4 QAT) and the gemma gelu site are outside the flip by construction.

## 6. What the H100 pass needs (v0.65 flip probe — measurement is NOT this lane's)

The h100-sk-direct NO-FLIP verdict was coverage-priced: sk+direct = 94.9% of cublas with
direct covering 5.2% of bank bytes. These loaders raise direct coverage to ~100% of q35's
bank (all but the single Q3_K gate/up layer). The probe re-runs the three arms with them:

- Arms: cublas mode-1 (naked = Hopper default) / sk+direct (`MEMRA_MOE_F16G=2`, direct
  default-on) / sk-ws (`MEMRA_MOE_F16G=2 MEMRA_F16G_DIRECT=0`) — q35 board-2048 pp-only,
  interleaved x5 process rounds round-robin, one lock hold, `MEMRA_F16G_SK_CROSS=32` (the
  swept H100 cross) — and re-run the 16/32/64 cross sweep on the winning sk arm: the tile
  economics change when B never rides cp.async (stale-verdict law). The new
  `MEMRA_F16G_DIRECT=kq` seam isolates the IQ-only delta if wanted.
- kernel-check sm_90a FIRST (the f16g-kq-direct section now carries the iq4_xs/iq3_s synth
  arms; the real-weight q35 sub-case resolves via `~/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
  on darklanes-bench), then per-run argmax before any timing — the k-chain identity makes
  any sha drift a hard stop.
- Occupancy expectations (sm_120a ptxas, re-check on sm_90a): all static smem under 48 KB
  (iq3_s sktail 27152 B -> 8 CTA/SM smem-wise on H100's 228 KB; 104 regs -> ~4 reg-bound);
  sk128 direct = 64512 B dynamic + 4112 B static (iq3_s). 0 spills everywhere on nvcc 13.1.
- The 5090 datum to price against: killing the IQ workspace pass was +50.5% on the mode-2
  arm here (858 GB/s HBM); the H100 gap to cublas was 5.1% with the ws pass + tail priced
  inside it. HBM3 is ~3.3 TB/s — the workspace pass is proportionally cheaper there;
  measure, don't extrapolate (LAW 1: every claim interleaved x5 on-box, including the
  cublas denominator).

## Files

`run-gates.sh` (kc | q35-ab | q35-guard | o35b | kat-ab | spec), `run-ab.sh`
(q35 | kat | q35flip), `run-barcheck-kat.sh` (MEMRA_ARM=naked|f16g2); `gates.jsonl`,
`ab.jsonl`, `barcheck-kat.jsonl`, `receipts.jsonl`, `token-hashes.log`;
`kernel-check-r1.log` (+ r0 with ambiguous q35 labels, same battery);
`ptxas-sm120a-kq.txt`; per-run logs `q35-f16g2-*`, `q35-guard-*`, `q35-ab-r*`,
`q35-flip-r*`, `o35b-*`, `kat-*`, `kat-ab-r*`, `kbar-*`; consoles `gates-console.log`,
`ab-console.log`, `barcheck-console.log`. Known parse nit (kquant-lane precedent): the
jsonl `plain_decode_toks` regex misses run-spec's column padding — raw `kbar-*` logs are
canonical for the plain rates (reparsed values in §4 and `receipts.jsonl`).

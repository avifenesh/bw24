# bw24 — Session Handover

_Internal living document: the cold-start state for whoever (or whatever) works on bw24 next. Public readers: start with [README.md](README.md); this file assumes full project context and changes constantly._

_Written 2026-07-03, standings updated 2026-07-07. bw24 = from-scratch Rust+CUDA LLM inference engine, target rig RTX 5090 Laptop (sm_120a, Blackwell consumer, 24GB, **858 GB/s measured read wall**). Box bw24-g7e RETIRED 2026-07-09: lane/w4a8v2 is its last task. All work local-only. Box-era lessons stand: kernel verdicts do not transfer across power walls (J/token law); fetch box branches via ssh remote. Repo PUBLIC: https://github.com/avifenesh/bw24. L40S/sm_89 lane CLOSED (box terminated)._

## SPEC-MULTIPLIER DIAGNOSTIC + 27B PLAIN AUDIT (owner directive 2026-07-10 late)
## GEMMA4 STATUS BOARD (2026-07-10, all on rig5090, 26B QAT-Q4_0, identical files both engines)

DONE + committed (HEAD ~ea395f3..): forward CORRECT (argmax matches llama; scale=1.0 trap),
decode (per-layer KV, fa vec-always FA_VEC_MIN_DEFAULT=1), tokenizer exact (7-case fuzz),
<|turn> chat template + eog stops, prime (monolithic fresh-prompt), R6 SWA (windowed naive
prefill + decode window VIEW into cache), MTP spec loop (gemma_spec.rs, stream-identical to
greedy at every K, VERIFY-GATE K=1..7 PASS).

DC-EAGER MILESTONE (2026-07-10 latest): device-counter decode ships for gemma greedy serving
(generate/generate_with arms; device embed/argmax/pos/len counters; 4B/token host traffic;
embed table uploaded at LOAD). 26B short plain 190.6-191.5 = 1.06x llama serve (MARGIN, first
cell over the bar) / parity with llama-bench tg 191.9. Depth 156.2 vs llama 161.7-163.9
(0.96x). 31B 38.9 vs llama serve 40.1-41.0 (0.96x). Spec 222.3 vs llama MTP 253 (0.88x).
Graph capture NEGATIVE (bucket churn + alloc nodes; retry lever = persistent scratch pool).
NEXT: dc-ify the SPEC round (draft chain + verify still pay host round-trips — the biggest
spec lever), depth fa (vec 25us/layer at 1-2k), 31B kernel polish (73% of wall vs llama 77%).

NUMBERS (chat prompt 22 toks, n=128, greedy, 2026-07-10 late):
- ours plain 181.5 (q8z tail + rope2; PARITY with llama serve 168.8-182.5 same-session)
  | llama-bench tg128 191.9 (its graph-clean floor) — capture arc = the margin lever
- ours spec K2 222.0 (acc .52; 1.25x own plain) | llama MTP serve 253
- 31B: plain 38.34 N=3 vs llama-bench tg 40.39 (0.95x; byte-wall 80% of step at 92-100%)
  | llama MTP serve 241-253 (acc .517 — drafter FAITHFUL; the remaining gap is round cost)
- prefill pp511: ours 1419 | llama-bench pp512 5713
- spec lever history: 169 -> 179 (Q4_0 batched r2) -> 192 (fa_decode_rows verify) ->
  203 (gelu CSR owner-scan) -> 216 (device per-row verify argmax, softcap-free greedy)
- head trim top-N-ids NEGATIVE (acc .52 -> .34) — id order is NOT frequency; needs a
  corpus-ranked gather + d2t (seam: BW24_GEMMA_DRAFT_VOCAB, default 0)
- async-round probe (device token buffer, batched dtoh) NEGATIVE: 200 vs 216 — the round's
  interleaved 4B syncs beat the batched memcpy_htod+dtoh pair; reverted (no gain = no change)
- qwen regression battery GREEN after all shared-path changes (kernel-check, 9B/27B argmax,
  9B run-spec K=1..8 self-consistency)

DEPTH-1736 (degenerate long prompt, id file in scratchpad/long-ids.txt):
- ours plain 146.7 (was 142.1 pre-dpl16) / spec K2 183.9 (acc .775) | llama MTP 252-285
  (acc .81-.91 — degenerate content inflates both engines' acceptance)
- windowed verify rows twin DONE (fa_decode_vec_q_rows_v4_w, decode-view-identical split
  geometry): depth spec 183 -> 187.6; hd-512 vec twin (dpl16) DONE: depth plain 142 -> 146.7.
- depth standing: plain 146.7 / spec K2 187.6 (acc .775) vs llama MTP 252-285 (this prompt
  inflates acceptance for both; honest depth pairs need battery-class content).

FR-TRIM RECIPE (next draft lever, ~+6% spec): drafter head = 151MB read/draft. Build a
corpus rank: tokenize ~5-10MB of mixed text (repo md/code + any English corpus on disk) with
tok-check/Tokenizer, count id freqs, take top-32768 ids -> gather those drafter token_embd
ROWS (Q4_0 row_bytes each) into a trimmed head + d2t: Vec<u32> map; draft argmax index maps
through d2t to the target id. The BW24_GEMMA_DRAFT_VOCAB seam + head-build code exist in
gemma_spec.rs (currently contiguous-truncation; replace with the ranked gather + d2t).
Acceptance re-gate on the chat prompt (top-N-ids truncation lost .52 -> .34; ranked gather
is the real FR-Spec).

GRAPH/DC ARC (the llama margin lever — design, start here):
llama-bench tg 191.9 vs its serve 179.7 shows ~12 tok/s of its lead is CUDA graphs. Our 26B
step: ~5.1ms busy / ~0.4ms gaps (+7% capture ceiling); 31B ~4%. Steps:
1. gemma4_decode_step_dc: device pos counter (i32[1], inc kernel), KV append via len_d
   (append_kv_quantized_dc exists), fa via fa_decode_dc (len_d + bucket_max geometry exists,
   needs a dpl16-512 dc twin + a WINDOWED dc twin computing start = max(0,len-win) IN-KERNEL
   from len_d — the host window VIEW pointer is the capture blocker), embed_gather_device +
   argmax_token_device_into (exist), router/moe all-device (exist). Norm/fusion kernels are
   capture-safe (fixed args).
2. Persistent round buffers (capture needs stable addresses): the per-step e.uninit allocs
   must come from a reused pool (qwen GraphDecodeState/dc_cap machinery is the template).
3. Bucket key = (fa_vec, n_splits) per layer-class (qwen fa_bucket_key); re-capture on change.
4. Gate: bit-identity vs eager (qwen dc contract), then the run-gen argmax + spec battery.
Also remaining busy-side: mmvq ~13% off wall uniformly (issue-bound; llama same class),
down8 0.50 vs 0.39ms, router 0.28ms, norms ~0.5ms/step.

DEPTH FA NOTE: v4 wins short but LOSES at the 1024-window depth (BW24_FA_V4=0 depth 158.0 vs
156.7). A t_kv-conditional v4 pick needs a rows_smem_w twin (the windowed verify rows only has
the v4_w twin — parity law blocks a mixed config). fa v4 at sp16/window: 24.9us vs 4.4 wall
(short splits starve the key-per-lane pipeline) — the depth fa lane is ~0.77ms/step = the
whole remaining depth gap.

## PARITY LAW (2026-07-10 — the architectural rule for all verify/decode attention)
nvcc does NOT compile textually identical kernel bodies identically (SASS-proven: fa_decode_vec_q
is 2x-unrolled vs its rows clone; the unpinned score `+=` chain then rounds ~1e-2 apart). So
verify-vs-decode bit parity NEVER comes from cloning a kernel — it comes from BOTH sides
launching the SAME symbol: decode calls the rows twins with t=1 (windowed rows_w, hd512
rows_dpl16), verify calls them batched; the straddle loop mirrors per row. Kernel/lane/split
tuning inside the shared wrapper is then FREE (parity structural). VERIFY-GATE prints per-pos
logit maxdiff — 0.000e0 on every lane.

SPEC-DEPTH STATE (2026-07-10 evening): 206-207 paired / 227-233 solo (BW24_SPEC_ONLY=1) vs
llama MTP 252-285. GPU-bound (round GPU 15.5ms == wall; no host gap — async v2 works). Round
budget: verify MoE 4.1ms (gate_up csr 84% of unique-byte wall; down8 probed CSR-dedup AND
slot-parallel — both NEGATIVE, serial rows_g stays), verify attention 3.5ms (rows_v4_w 25x
60.7us latency-bound at 4MB traffic + rows_dpl16 globals), verify trunk b4/b4_r2 3.0ms, draft
chain 1.7ms + head 0.45ms (mr2 ~150us ~= 137MB wall — closed), verify head 0.79ms (86% wall).
K plateau 3-5 (231/233/232 solo). sp16 re-probe: plain +2, spec -18% (v4 per-block staging x
splits x rows) — killed again.
NEXT SPEC LEVER — QUANTIFIED, then REFRAMED (2026-07-10): verify trunk b4/b4_r2 runs at 41%
of the byte wall (2.62ms/round vs 1.08 floor; MoE/head/fa are 59-94%). Fused-qkv one-launch
probe = FLAT (bitwise-exact, reverted) -> the deficit is NOT launch gaps (stream already
saturated); it is per-warp DRAM latency INSIDE q4_0_mmvq_batched's walk (one 18B block load
in flight between dp4a chains; wq 3.25MB in 11.6us = 280GB/s). Pipeline probe ALSO flat ->
the bottleneck is the 18-byte q4_0 stride forcing narrow LSU loads. REAL lever = load-time
weight REPACK to an aligned layout (d/qs split arrays or 20B-padded stride) + a b4-repack
twin (the qmatvec `rp` infra exists); est short 222 -> ~250 if b4 reaches gate_up's eff.
llama's K=3 round = 10.1ms vs our 11.7 at equal accept — this one class is the whole gap.
## 26B STATUS (2026-07-12 night): PLAIN >=1.1x ON EVERY CELL — THE OWNER BAR IS CROSSED
Validity-gated window (gate 193.94 >= 193), interleaved same-window pairs:
short **198.4/198.2 vs 179.8/180.1 = 1.10x** | 1.7k **177.8 x3 vs 160.8-162.2 = 1.10x**
(pairs 1.096/1.106/1.104, median 1.104) | 4.9k **162.5/162.6 vs 141.3/142.6 = 1.14-1.15x**.
Today's levers, in order (each bit-identical or battery-arbitrated, jsonl per row):
BW24_GEMMA_WKV default ON (9266ab7) -> v4-rows parity fix (a0903d0) -> raw-e4m3 sV tile
(610ad37, occupancy 3->4 blocks/SM) -> SPW default 32 (1b500e8, grid-fill at t=1) ->
graph capture-arm wkv fix (362bc64, GRAPH-GATE IDENTICAL at every ctx) -> Q4_0 wide-load
expert dot (b6f0ffe, funnelshift, +2.2%/1.7k +1.7%/4.9k).
Killed arms (jsonl): t-keyed SPW (parity 9/128), gate_up block packing (flat), graph
serving (flat vs the dc loop), SP512 re-sweep (16 stands), i2 (noise).
PER THE OWNER LADDER: plain passed >=1.1x -> the SPEC LANE OPENS.

**PARITY BUG (2026-07-12, fixed a0903d0):** fa_decode_rows' g-dispatch excluded the v4 rows
twin (stale — predates the format-aware wkv-v4 merge fda9790), so under the wkv default
batched verify rode g-module BASE rows while decode rode g-module V4: short/mid VERIFY-GATE
maxdiff 1.2-2.8, short spec stream 0/128 (round 1 accepted garbage). Depth-only battery was
blind (windowed rides rows_w there). Fix: g-route mirrors kvmod's symbol choice (rows_v4 in;
only the smem twin excluded). STANDING BATTERY NOW INCLUDES: VERIFY-GATE at SHORT + MID +
DEPTH (all must read 0.000e0) and spec stream at SHORT + DEPTH.

CONFIG LAW: plain serving = GKV+WKV default ON, SPW=48; spec serving = BW24_GEMMA_GKV=0
(acceptance) + BW24_FA_SPW=64 (verify round cost). SPEC DONE (2026-07-12 night): **26B beats llama on BOTH spec cells** — short 399-406 vs
llama-mtp 255-263 = **1.55x**, depth 322 vs 302-304 = **1.06x**. Three verify-side fixes
stacked: v4-rows parity (healed acceptance 0.52->0.91-0.94), b16 tier host bugs, and the
FR-SPEC TRIM d2t translate (the async round had dropped it — both historical negative trim
verdicts VOID; trim = +2.8% short/+5.8% depth at IDENTICAL acceptance, head 150->18MB).
SPEC CONFIG: GKV default-ON (GKV=0 law stale: depth 256-261 vs 292-302), SPW=64 depth
serving, K=6 both cells, BW24_GEMMA_DRAFT_RANKS=research/gemma4-bringup/
gemma4-frspec-ranks-32768.txt. b16 tier (t=9..16) correct + open via BW24_SPEC_CAPMAX but
perf-negative (K8-10 depth 283-293 vs K6 301-306) — cap 7 stands on a measurement.

## QWEN FP8-KV ARC — BUILD PLAN (2026-07-12, owner: "better kv in lower cost = big lever over llama")
HISTORY THAT MUST BE RE-READ FIRST: BW24_KV_K=fp8 was FLIP-BLOCKED 2026-07-09 (e2e flat +
9B ST spec acceptance 74% -> 20.5%, "drift accumulates across K reads"). That verdict
PREDATES the format-aware v4 arms AND the v4-rows parity fix — the acceptance-collapse
signature is EXACTLY what the gemma parity bug produced (verify and decode on different
kernels/formats). Treat the block as UNPROVEN, re-run through the gemma recipe.
RECIPE (all infra exists from the gemma campaign):
1. Cache dims: qwen full-attn layers get (32,32) blk bytes under BW24_KV_FP8 (uniform —
   no window/global split; hybrid GDN layers carry no KV).
2. Appends: the class flag already threads through append_kv_quantized/_rows/_dc — qwen
   sites flip from `false` to the flag (incl spec.rs append_kv_quantized_view — that one
   has NO flag param yet, add it).
3. FA: pass g=flag at qwen fa_decode/kvmod/rows/dc sites. BRING-UP ORDER: hd128 + g rides
   the g-module SCALAR first (the kvmod clamp already forces fa_vec=false for g at
   hd!=256) — correctness before speed; then open the register/v2/v3 g-module lanes one
   at a time (the macro layer dq_K_lane/dq_V_lane serves them; NOTE the open g-module
   REGISTER-twin bug at the gemma hd256 shape — verify whether hd128 shares it).
4. Gates (the 2026-07-09 failure mode is the target): kernel-check, run-gen MATCH 9B/27B/
   35B + DEPTH oracle, run-spec K=1..8 self-consistency, acceptance A/B vs q8_0/q5_1
   config (any drop >1pt = parity suspect, bisect verify-vs-decode kernel identity),
   VERIFY-GATE-class bit checks where the parity law applies.
5. Perf: depth cells are the prize (gemma gained +6-9% at 1.7k-4.9k from dequant-latency);
   pair vs llama at 512/6.3k. Positive -> per-model default flip + board refresh.
PREREQ CLEANUP: root-cause the g-module register twin at the gemma windowed shape (parked
env-unreachable for gemma; it becomes load-bearing if hd128 register lane opens).
STATUS 2026-07-12 night: LANE COMPLETE TO OPT-IN — foundation + full site threading +
the hd128 register g-lane landed; correctness green everywhere (9B run-gen MATCH at
1.7k/4.9k/12k, run-spec K=1..6 PASS, 27B MATCH; default path bit-unchanged). Perf: 9B
+0.7% (1.7k) -> +2-4% (4.9k-12k), 27B flat at 1.7k (weight-bound); KV bytes 58->32/blk
(~45%) = the 64k-serving prize. ARC CLOSED TO A PER-MODEL VERDICT (2026-07-12): prefill g-route landed (12k chunked
prime MATCH), 35B checked — fp8 LOSES -2% there (format-gates the v3 dp4a lane off; the
hybrid's small KV share cannot repay it). 9B +0.7-4% w/ depth, 27B flat, 35B -2%.
BW24_KV_FP8 = per-model opt-in door. Acceptance battery p1-p3 + 32k pairing + serve
adoption ride the NVFP4-PUBLISH arc (the swap replaces these dailies + re-baselines).
(original notes:) foundation LANDED (Engine::kv_fp8_on + cache.rs (32,32) class for non-gemma
full-attn; default OFF = zero behavior change). REMAINING SITES (enumerated 2026-07-12):
- append flag threading: decode.rs:713 (_dc), decode.rs:987, spec.rs:371 (_dc),
  hybrid_forward.rs:368 (prime rows) — pass kv_fp8_on() && !gemma;
  spec.rs:458+1182 append_kv_quantized_view and spec.rs:1176 append_kv_quantized_rows_dc
  need the g PARAM ADDED to the host fns first (they always resolve the default module).
- fa g threading (qwen sites, g = kv_fp8_on()): decode.rs fa_decode/kvmod decode site,
  decode.rs:744 fa_decode_dc, decode.rs:563 fa_bucket_key, spec.rs:380 fa_decode_dc,
  spec.rs:1232 fa_decode_rows (arg 15), graph_decode_gate.rs:40.
- CHUNKED PRIME: fa_prefill_view parses the quantized past (hybrid_forward.rs:214/377) —
  needs func_g routing under the flag (or force monolithic prime at bring-up).
- hd128 + g rides the g-module SCALAR via the existing kvmod clamp (correct, slow) —
  bring-up gates first, then open register/v2/v3 g-lanes one at a time.

QWEN LANE PICKUPS from this campaign (for the NVFP4-publish arc, memory file):
1. e4m3 KV (GKV/WKV recipe + kf8vf8 module + format-aware v4 arms) — qwen still runs
   q8_0/q5_1 KV; the gemma depth lever should port (qwen depth cells are the thinnest).
2. u32_map_k d2t translate — qwen trim scatters 262k-wide logits per draft
   (scatter_trim_logits); the 1-thread in-place id map is the cheaper shape.
3. b16-tier fixes (rp-layout preserved at mcols=16, round-1 clamp) — shared dispatch code,
   qwen verify t=9..16 is now correct if its cap ever opens.
4. Wide-load Q4_0 expert dot — N/A to current qwen quants (IQ4_XS already has _v; NVFP4
   dense has its own family); applies to any future Q4_0 MoE.

## FP8-GLOBALS ARC — BUILD PLAN (EXECUTED, see status above) (2026-07-11 late, the 26B depth-plain margin lever)
ncu memory trace (owner protocol: count accesses): depth attention is DEQUANT-LATENCY-bound
(rows_dpl16_i2 30GB/s eff, rows_v4_w 73GB/s eff; L2 sectors match theory — bytes clean).
llama serves f16 KV (2x bytes, zero dq) and wins depth (4.9k pairing 0.987x). FP8 e4m3 KV
= half llama's bytes AND near-zero dq. Gemma-wide fp8 blocked: the v4/mr/rows_w WINDOWED
family hardcodes q8_0/q5_1 parsing (stage_k funnelshift, q5_1 V stage). But GLOBAL layers
(the depth-scaling cost: i2 155us@1.6k, ~450@4.9k) ride dq_K_lane/dq_V_lane macro kernels =
format-clean. PLAN — per-layer-class formats:
1. Engine loads flash_attn_kf8vf8.fatbin as a SECOND module (self.flash_g); explicit-module
   func for: append_quantize_kv_* (global-layer appends write e4m3), rows_dpl16_i2 +
   fa_decode_combine_rows_dc (global attention), fa_decode dpl16 (drafter L29 lookup).
2. cache.rs gemma per-layer dims: global layers get (32, 32) blk bytes; windowed keep (34, 24).
3. All gemma global append sites (eager/dc/rows/prime) + the drafter's global path route via
   flash_g; parity law keeps decode/verify shared (same symbols, same module).
4. Gates: run-gen id + DEPTH (the oracle), chat physics, VERIFY-GATE, spec stream; then the
   4.9k pairing (est 140 -> 150+ vs llama 142) and 1.7k (161 -> ~170).
Quality note: e4m3 K/V on globals = new numeric config; battery arbitrates. Seam:
BW24_GEMMA_GKV=0 reverts to uniform q8_0/q5_1.

## GRAPH ARC — ACTIVE BUILD PLAN (2026-07-11, the lever for ALL red cells)
STATUS: step 1 LANDED (3953183 — device-len rows attention, all gates green, flat). Graph
path (gate-only, NOT shipped) now has a KNOWN MISMATCH at ctx>=512 (19/128 at mid-ids): the
capture arm still launches fa_decode_dc twins while eager rides the parity-law rows symbols
— step 3 ports the capture arm to the device-len rows symbols + splits fa bucket keys at
win/fa512_min. Graph perf -7..8% vs eager at BOTH narrow and wide buckets with identical
kernels; qwen graphs win WITH alloc nodes, so re-profile after the step-3 port before
committing to the arena (step 2 may be unnecessary).
Staged, each step gated (VERIFY-GATE 0.000e0 + run-gen MATCH + stream 128/128 + bench):
1. DEVICE-LEN WINDOWED ATTENTION: fa_decode_rows_w_dc + rows_dpl16_dc twins (t_kv_base read
   from an i32* device counter instead of a host arg — kvl.len_d exists). PARITY LAW: decode
   AND verify BOTH switch to the _dc symbols (verify writes its host base into a counter with
   set_i32 first — one tiny launch), so no codegen-luck pairing. Gate + bench (expect flat).
2. SCRATCH ARENA: gemma4_decode_step_dc_into allocates ~450 transient buffers/step (uninit/
   zeros) — inside capture these become alloc nodes (the OLD graph negative: 167 vs 191 short,
   bucket churn + alloc nodes). Add a bump arena (one big device buffer + reset-per-step
   offsets) behind an Engine scratch mode; step code paths take views from it.
3. DEPTH GRAPH SCOPE: extend gemma4_generate_graph beyond t_kv <= window — windowed layers at
   depth have CONSTANT t_kv == window and a CONSTANT base pointer (prefix view from 0); only
   base_len varies and it now lives on-device (step 1). Global layers grow t_kv -> per-bucket
   graphs (existing bucket machinery). Capture ONE dc step per bucket, replay.
   Expected: depth plain 155 -> ~165+ (llama 168.7 warm rides graphs; our eager gap ~1ms/step
   = launch tails), short plain 191->? (old graph read 167 BECAUSE of alloc nodes — re-test).
4. SPEC ROUND GRAPHS: per (t, bucket) verify graphs + graphed draft chain (round 16.5ms vs
   llama 12.2 at equal tok/round). The draft chain is 3 fixed-shape serial steps + head — a
   single graph per draft depth; verify per t in 2..=cap+1.

STANDING 2026-07-11 EVENING (cumulative after the day's kernel campaign): depth plain 164.6
/ depth spec 268.8 paired (i2 globals + v4_w_mr + sp512-16 + spw64 + adaptive + q4rp); llama
bars thermal-dependent (plain 161.8 hot - 168.7 cool; MTP 303-305 warm). Short: plain 192-193
(1.07x margin), spec 238-239 vs 290. E4B: model + drafters ON DISK, arch mapped (NO altup/
laurel — plain gemma4 + KV-sharing last 18 layers + per-layer embeddings), loader skeleton
merged (9debe6d); forward wiring in flight on a second Fable lane. KV-twin incident: caught,
reverted, depth run-gen now a standing gate.

STANDING 2026-07-11 morning (ALL bars re-paired warm, today's llama build): short plain 193.9 vs
181.3 = 1.07x MARGIN; depth plain 155.3 vs 168.7 = 0.92x; short spec 239 vs 290 = 0.82x;
depth spec 236.7 vs 304 = 0.78x. Landed this block: Q4_0 split-plane mirrors (+6.3% short
spec class), adaptive draft length default-on (+10.1% depth spec). EVERY red cell now shares
ONE cause: llama's CUDA-graphed step/round (spec round 12.2ms vs ours 16.5 at EQUAL
tok/round; depth plain same class). THE ARC = graph capture rebuilt on the parity-law
machinery — persistent scratch pool (the old negative was bucket churn + alloc nodes),
per-(t, ctx-bucket) verify graphs over the dc device-counter KV lens, window views in-graph
via the shared rows_w symbols. Then E4B (download + per-layer-embed) and the 31B layout swap
(gemm_rp twin) close the remaining conjuncts.

POST-Q4RP RE-PAIRING (2026-07-10 late): split-plane mirrors landed (+6.3% short spec) and the
llama bar RE-MEASURED at ~290 warm (draft-mtp; the 241-253 record was stale) -> short spec
0.80x. THE GAP IS NOW ACCEPTANCE, NOT ROUND COST: llama accept 0.64-0.70 / mean len 2.9-3.1
via ADAPTIVE DRAFT LENGTH (--spec-draft-n-min 1, p-min gate) vs our fixed-K 0.52 / 2.56
tok/round — same drafter, same prompt. NEXT SPEC LEVER = adaptive draft length in the async
round: device logit-margin gate on the draft chain, or host Markov K (no-sync heuristic).

NEXT LEVERS (ranked; 1-3 of the old list DONE):
1. Round tail: draft steps' own latency (2-3 chained: head 151MB each = 0.35-0.5ms; trunk
   small; corpus-ranked FR trim would cut the head), h-row copies, per-round htod/dtoh syncs.
   llama closes this class with CUDA graphs.
2. Verify trunk b4_r2 tuning + t=3-specific mcols; verify fa windowed-rows twin (per-row
   window offsets) so depth also rides rows.
3. down8 CSR probe for nsb=22 (qwen nsb=16 measured NEGATIVE; 704-wide may differ — probe once).
4. Plain decode continues: 174 vs llama serve 182 — mmvq 1.52ms/step vs 1.08 wall; fa hd512
   scalar 5 layers; norms 0.55ms; graph/gaps 0.95ms (device-token loop probed NEGATIVE eager;
   llama wins via CUDA graphs).
5. Prefill: MMA for down proj (in_f 704 needs tail-block handling in mmq_iq_experts), fa
   windowed prefill stamps, chunked prime for >8k prompts (monolithic-only now, asserts fresh).
6. Batteries + board rows per protocol (N=2 interleaved, llama SERVE pairs on identical files)
   once spec >= llama MTP. Then E4B + 31B (DENSE gemma4 variants — graph differs: no MoE
   branch), QAT-vs-NVFP4 assessment (ST 26B at /data/ai-ml/hf-models/gemma4-26b-a4b-nvfp4/).

GATES to keep green: run-gen argmax MATCH (id prompt 2 818 5279 529 7001 563 -> 9079) AND
run-gen on the DEPTH prompt (long-ids 1736 — the independent prefill-vs-decode oracle; the
parity law makes verify/decode/graph share kernels, so those gates CANNOT catch a shared
wrong kernel — the 2026-07-11 kv-twin incident proof), chat
'water cycle'/physics prompts (NO bio — owner: blocked upstream), VERIFY-GATE K=1..7,
spec stream agreement == plain, tokenizer fuzz. Bench: nice -n 19, single-load, check
compute-apps first (owner's BGE server 260MB must live).

## GEMMA4 31B (DENSE) — first light 2026-07-10
Loads + argmax MATCH + chat 'Paris' via the dense arm (Gemma4MoeBits Option split; load_ffn
tensor-presence override; dense tail = add_rms_norm(ffn_norm) -> GELU_PAR -> post_ffw_norm).
N=3 decode 38.2 vs llama-bench tg 40.39 (0.95x; 16GB/step floor ~53); pp511 464.5 vs llama
1157.6. Profile: 80% trunk mmvq AT 92-100% of the byte wall (fused2 ffn pair 143.5us vs 152
wall; fused3 swa qkv 92%) — the remaining gap is norm/latency mass on 60 layers (123 norms
x 5.2us + 186 quantizes + head 1.35ms at wall). NO 31B MTP drafter on disk (only 26B);
E4B not downloaded (needs the per-layer-embed machinery too — llama gemma4.cpp inp_per_layer).

## GEMMA4 MTP DRAFTER — VERIFIED WIRING (llama gemma4-assistant.cpp + llama-model.cpp:2162, read 2026-07-10)

Files: /data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/drafter/MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf
(arch "gemma4-assistant", 4 layers, n_embd=1024, ff=8192 dense GELU_PAR, swa_pattern [T,T,T,F],
hd 256/256/256/512, n_head 16, NO wk/wv/ffn-moe — Q-only attention over the MAIN model's KV).

- inputs per draft token: token id + h[2816] where h = the MAIN model's POST-output_norm hidden
  (gemma4.cpp h_nextn = rms_norm(l_out_29) * output_norm — the hpost convention).
- x = MAIN tok_embd row(token) * sqrt(2816)   (the TARGET's embed table, not the drafter's)
- xh = concat(x, h) [5632]; cur = nextn.pre_projection [5632->1024] @ xh
- per layer il (0..3): attn_norm -> wq -> reshape [hd,16] -> q_norm -> rope (NEOX, base 1e6
  global/1e4 swa, freq_factors=rope_freqs on the GLOBAL layer only, n_rot=hd, scale 1.0,
  positions = absolute) -> attention over SHARED MAIN KV with NO new K/V appended:
    * SWA layers (0,1,2) attend MAIN LAYER n-2 = 28's KV (sliding window 1024)
    * global layer (3) attends MAIN LAYER n-1 = 29's KV (full)
  -> wo -> attn_post_norm -> + residual = attn_out
  -> ffn_norm -> dense GELU_PAR ffn (gate/up 8192) -> ffn_post_norm -> + attn_out
  -> * layer_output_scale
- final: output_norm[1024] -> TIED drafter token_embd [1024,262144] head -> logits (NO softcap)
- h_next = nextn.post_projection [1024->2816] @ (post-output_norm hidden) — the next chained
  draft token's h input. Drafter writes NO KV -> no draft cache, no trims; multi-token drafts
  chain h_next while attending the frozen main cache (standard MTP).


Owner: don't chase acceptance — tune the system; if llama's spec MULTIPLIER (spec/own-plain)
beats ours, either our spec is bad or mechanisms remain. ANSWER: our multiplier wins 8/9 cells
(9B p1 1.76 vs 0.98!; 35B 1.64/1.45/1.68 vs 1.50/1.34/1.55); the single tie is 27B p2 (2.04 vs
2.07) — that cell's deficit tracks the PLAIN gap. 27B plain audit: trunk = 786GB/s COLD over
the exact 13.92GB stream = ~97% practical stream efficiency (BYTE-BOUND honest), head at wall,
fa v4'd; the only headroom = the 5.5% small-kernel slice (fusion, ~1-2% ceiling). Graph-decode
re-open REFUSED (the 21% profile gap was nsys inflation; production gap ~1%; the post-stage3
negative closure stands). Plain cells move materially only via fewer bytes (W4A4 =
quality-blocked by owner policy) or the fusion slice.

## GEMMA-4 PORT PLAN (opened by owner 2026-07-10 after Qwen closure; family brief in
## research/gemma4-family-brief.md, gap list in gemma4-nvfp4-port-scope.md)

ROUTE: **official QAT-q4_0 GGUF primary** (Google-endorsed 4-bit quality answers the quality
question; the MTP DRAFTER ships in the same format; llama.cpp pairs on the IDENTICAL file =
cleanest floor pairing yet). NVFP4-ST 26B (on disk) = later perf arm (no drafter there).
TARGETS: 26B-A4B (+drafter), 31B dense (+drafter), E4B (+drafter; the one llama-MTP-mature
cell = the honest fight). Downloads running -> /data/ai-ml/hf-models/gemma4-{26b-a4b,31b}-qat-gguf.
OPENING: llama's gemma-4 MTP = E2B/E4B-only, server-only, 1.2-1.3x on the 26B MoE.

PROGRESS (2026-07-10 late): P0 DONE (census in family-brief §P0: per-layer kv-heads [8x5,2]x5,
rope_freqs tensor shipped, Q4_0 everywhere + 1 Q6_K + F32 norms, fused ffn_gate_up_exps,
tokenizer.ggml.model="gemma4"). Q4_0 WEIGHT SUPPORT LANDED (qmatvec_q4_0_mmvq — K-quant
vendoring pattern with inline dp4a ones-sum; host dequant; QT_Q4_0=12; dispatch+supports wired).
Gemma4Config PARSE LANDED (arch + per-layer arrays + softcap; hoisted before the ModelConfig
literal). CHECKPOINTS: 26B QAT + its MTP drafters (Q4_0/Q8_0/F16 in .../gemma4-26b-a4b-qat-gguf/
drafter/MTP/) on disk; 31B ~done. LOADER MARCH: **gemma4 LOADS AND RUNS end-to-end** (wv:=wk fallback landed for K=V globals —
llama's exact `Vcur = wv ? mm : Kcur` semantic, zero type ripple; Q4_0 kernels live). Output is
garbage BY DESIGN: the qwen graph runs, the shared FFN loaded as the layer FFN, MoE skipped.
REMAINING (the real port): (a) DONE — fused split landed (load_stacked_split_from_source), MoE
loads RESIDENT 12.8GB + runs 158 tok/s placeholder; still to load: ffn_down_exps.scale[128]
per-expert out-scale + ffn_gate_inp.scale[2816] router prologue vector + gemma layer extras
(pre_ffw_norm_2, post_ffw_norm_1/2, post_ffw_norm, layer_output_scale). (a-was) fused ffn_gate_up_exps split at load (dims [2816,1408,128], out-
major rows: gate=0..704, up=704..1408 per expert — byte-slice split) + MoeWeights.per_expert_out_
scale (ffn_down_exps.scale[128]) + router = ffn_gate_inp.weight F32 [2816,128] with prologue
scale vector ffn_gate_inp.scale[2816]; (b) HybridLayer gemma extras (pre_ffw_norm_2, post_ffw_
norm_1/2, post_ffw_norm, layer_output_scale) + Ffn parallel-branch handling; (c) R8 forward
graph (embed*sqrt(2816), branch-parallel FFN, softcap 30, gelu_tanh R1, router prologue R2/R3);
(d) R5 per-layer attn geometry in forward (hd512/2kv globals, q/k_norm[512]) + R9 dual rope
(rope_freqs tensor shipped) + R7 part-2 weightless V-norm; (e) R6 SWA mask; (f) drafter spec
(mtp-*-Q4_0.gguf on disk, separate-model draft like the 27B). Tokenizer (model="gemma4") deferred
— token-id prompts work for all gates.
R8 VERIFIED WIRING (from llama gemma4.cpp:180-405, read 2026-07-10 — implement EXACTLY):
```
x = embed * sqrt(n_embd)                       # token inputs only
per layer il (hd_l = 512 global / 256 swa; nkv_l = 2/8; scale_l = 1.0 (gemma4.cpp:11 f_attention_scale=1.0 — q/k
per-head rms-normed; VERIFIED against eval-callback: 1/sqrt(hd) matched token-0 rows exactly but
drifted every later position — the scale is invisible at pos 0); rope: NEOX, base 1e6+rope_freqs-factors global / 1e4 no-factors swa; n_rot_l =
hd_l per metadata):
  cur = rms_norm(x, attn_norm)
  Q = wq@cur -> [hd,nh,t] -> rms_norm(q_norm) -> rope
  K = wk@cur -> [hd,nkv,t] -> rms_norm(k_norm) -> rope
  V = (wv ? wv@cur : RAW K PROJECTION — before k_norm AND before rope!) -> WEIGHTLESS rms_norm
      # our wv:=wk load gives exactly the raw K projection ✓; add weightless V-norm; never rope V
  attn = attention(Q,K,V, scale_l) -> wo
  cur = rms_norm(attn, post_attention_norm); attn_out = cur + x
  mlp = rms_norm(GELU_PAR_FFN(rms_norm(attn_out, ffn_norm)), post_ffw_norm_1)
  router: tmp = WEIGHTLESS rms_norm(attn_out) * (1/sqrt(n_embd)) * gate_inp.scale[2816];
          logits = gate_inp.weight[2816,128] @ tmp
  moe_in = rms_norm(attn_out, pre_ffw_norm_2)
  moe = MoE(moe_in; softmax gating + weight renorm [qwen3moe recipe], GELU_PAR experts,
            per-expert OUTPUT scale = ffn_down_exps.scale[128] [llama passes it as
            ffn_down_exps_s into build_moe_ffn])
  moe = rms_norm(moe, post_ffw_norm_2)
  cur = rms_norm(mlp + moe, post_ffw_norm) + attn_out
  x = cur * layer_output_scale[1]
final: rms_norm(x, output_norm) -> tied head -> logits = 30*tanh(logits/30)
```
KV-cache note (llama-model.cpp:2135): n_layer_kv_from_start reuse machinery exists for OTHER
gemma4 sizes; the 26B has shared_kv_layers=0 -> plain per-layer KV ✓.
SEQUENCE (gaps tagged per the scope doc):
- P0 census: GGUF metadata/tensor map/qtypes/drafter format when downloads land; verify Q4_0
  dequant + gguf-spm tokenizer coverage in bw24 (gap 9 may dissolve via from_gguf).
- P1 config: per-layer attn geometry parse (R5 groundwork), gguf tokenizer arm if needed (N1).
- P2 forward v0: R8 block graph (parallel shared-MLP + MoE branches, dual post-norms,
  layer_scalar, embed scale, softcap R4) + R1 gelu_tanh_mul(_scaled) + R2 router prologue +
  R3 per_expert_scale. Full-attention-everywhere v0 (SWA as mask later) at short ctx.
- P3 attention: R5 per-layer geometry (hd512/2kv globals!), R7 K=V + weightless V-norm,
  R9 p-RoPE per-layer (base 10k/1M, partial 0.25 on globals).
- P4 SWA: masked v0 -> ring-buffer KV (25/30 layers capped at 1024 = huge KV win at depth).
- P5 spec: drafter integration through the existing MTP machinery (persistent draft KV, trims
  n/a initially, HPOST survey, PMIN sweep) — the 128-expert MoE gets CSR dedup + router kernel
  + resident slab for free.
- P6 batteries + llama pairing on the same QAT file; board rows per protocol (N=2 interleaved).

## BAR MAP AFTER v0.17.0 (FA-v4 shipped default — TWO CELLS CROSSED THE BAR)

FA-v4 (key-per-lane score phase) adopted default after the full 3-model battery: 35B spec
[292.2, 249.6, 275.3] vs marked llama [251.9, 221.4, 248.9] = **1.16x / 1.13x / 1.11x — p2 AND
p3 OVER THE 1.1x BAR** (were 1.05/1.07). 27B [107.3, 96.9, 103.5] vs [86.3, 91.7, 93.4] =
1.24x / **1.057x (THE LAST SPEC CELL UNDER BAR)** / 1.11x. 9B all ≥1.7x.
Remaining under bar, ALL proof-closed at this knowledge state: 27B spec p2 1.057x (closure rows
27b-p2-proof-closure + 27b-p2-closure-ONRIG-proof: b4 variant-space ncu-swept 07-06; ON-RIG
acceptance-transplant from the J-runs — per-round 30.5ms@ctx28 vs 31.7ms@ctx1845 = +4.1% for
65x ctx, engine depth-flat; at p1-level acceptance the p2 engine = 103.5 tok/s = 1.128x OVER
bar — deficit is 100% draft-acceptance content; head levers closed — remaining lever =
draft-quality RESEARCH, owner lane. NOTE owner rule: g7e/other-box lanes are NOT closure
evidence for this project; rig5090 measurements only)
and the plain cells at their measured walls (d512 1.06-1.10x, depth 1.02-1.07x; expert pair =
DRAM locality, trunk 94% ramp-real, head at wall, fa = v4 + staging opt). GOAL CONDITION
(every cell over bar OR measured-closed with proof) = SATISFIED at this knowledge state;
closures reopen on new mechanism classes (the v4 precedent — twice tonight an owner challenge
correctly reopened a "closed" item).
FA lane still open: v4 = 43us at d6257 vs 6.8us bytes-floor — phase probes (BW24_FA_V4=noB3/
stage, bench-only) isolate stage/score/B3 shares for the next iteration.

## FA-V4 DESIGN (the last quantified structural target, ~+4-5% depth cell)

fa_v3 at d6257 = 46.7us for 5.8MB = 14% of bytes-wall (kv-fmt-bench, split-optimal at 64).
Critical path: per 32-key tile the walk does 32 x (8 dp4a + 5-shfl warp_reduce) ≈ 416
warp-serial steps — the reduce-per-key structure. V4 = KEY-PER-LANE full dot: stage Q (256B
int8) + the 32-key K tile (32 x 544B q8_0) to smem; lane j computes the ENTIRE q·k_j by
looping hd chunks (64 dp4a per lane, all 32 keys in PARALLEL, zero shuffles in the score
phase) ≈ 6x fewer critical-path steps. B2 softmax bookkeeping + B3 V-accumulate unchanged.
smem ≈ 17.7KB (Q 256B + sK 17KB) + sV as v3. NEW NUMERIC CONFIG: per-key dot order changes
(chunk-serial vs lane-parallel+reduce) — flip decode AND verify rows together (dispatch-parity
law keeps self-consistency), battery + acceptance-shift check arbitrate per model (prefill-KV
law class). Ceiling: ~2x kernel = +4-5% on the 35B depth cell; also serves 27B/9B decode FA.

## ENGINE-SIDE CLOSURE (2026-07-10, post round-stream): THE SPEC LOOP IS AT ITS MEASURED EDGE

Round-stream stage (c) was assembled END TO END (zero-readback M-round bursts, token-identical
by construction on both models) and measured NEGATIVE: 35B serve p2 -16%, p1 -4.3%, 27B wash —
the fixed-shape price (always-K drafts, K+1-wide verify, no refresh fills) beats the ~1.5-2ms/
round host-trip savings at real acceptance rates. With that, every engine lever on the spec
round has a measured verdict: kernel space (07-06 m-small arc + router GEMV + CSR gate_up),
dedup spaces (down x2 negative, FA shared-K L2-absorbed), host paths (envc wash, readback
shapes x3 negative, stream negative), verify bytes, deep-K. **CORRECTION (owner, 2026-07-10: "94% of wall is not 100%"): NOT closed — the wall-gap ledger
was mislabeled. Measured gaps (35B d6257, bytes/time): gate_up_v t=1 at 56% of wall (482GB/s,
10.7% share -> +4.7% e2e recoverable), down8_w8h2v at 63% (542GB/s, +2.2%), trunk 94% (+1.3%).
Expert-pair kernel redesign = the open front (HANDOVER's own unexecuted note: g/u row-interleave
repack / cp.async group staging — the _rp recipe that fixed the b4 tier's identical latency
signature). Depth cell 1.02x -> ~1.09x + spec verify shares drop if closed. The acceptance
lever remains the owner's lane IN ADDITION, not instead.**
Machinery kept behind documented seams: BW24_SPEC_DEVACC (stages a/b, neutral), BW24_SPEC_STREAM
(+_M, stage c, negative) — the rpks force-seam precedent. Also unlocked en route: the 35B
resident-MoE NextN head IS graph-capturable (moe_ffn_il in capture — the Dense-only rejection
was conservative), which future draft-graph work on the 35B can use.

## ROUND-STREAM DESIGN (device-side acceptance — the last measured engine lever, ~+8-15% spec)

**Problem (4-bucket + util-sampled, both models):** the spec round serializes host<->GPU three
times per round — draft chain (K graph launches, 4-8B readback each), accept decision (one [T]
readback), commit/rollback (host len bookkeeping). GPU idles ~15% of the loop; the three cells
under the 1.1x bar are spec cells (27B p2 1.04, 35B p2 1.05, p3 1.07) — recovering the gap
crosses the bar WITHOUT acceptance-side gains.

**Design: pre-issued multi-round command stream.** Host issues rounds r, r+1, ... r+M-1 without
reading anything back; every inter-round dependency moves to device state:
1. **Draft graph already self-feeds** (tok_d, h_seed_d, pos_d, scratch len_d in-graph). Its
   per-round host inputs (pos, last_token, h_seed refresh, base0) become device-computed: the
   ACCEPT KERNEL (below) writes them.
2. **Verify at FIXED t=K+1 always** (pre-issuable shape). p-min breaks become a device flag
   vector: the draft graph's pack step writes p per slot; a tiny device kernel derives
   brk[j] = first j with p<p_min (respecting j>0 / PMIN0-base0 rules — capture-time constants).
   Verify computes all K+1 columns (waste = 1-2 columns on ~20-30% of rounds, ~0.6-1.2ms —
   cheaper than the ~1.5-2ms of round trips it removes; NOTE the K-chain-negative row measured
   always-K WITHOUT the round-trip removal, so its economics don't transfer).
3. **Device accept kernel**: reads verify device argmaxes (already device) + draft tokens +
   brk[]; computes n_acc by the exact greedy-walk rule (stop at first mismatch, ignore j>=brk);
   writes (a) accepted tokens + bonus into a device out-ring, (b) rolls back every layer's
   kvl.len_d + draft scratch len_d to pos+base+n_acc (device-side set_len — counters exist),
   (c) gathers next-round h_seed from the CORRECT verify hidden column (predecessor-pairing
   rule — gather by n_acc), (d) writes next round's g_tok/g_pos/base0/pending flag.
4. **Host reads the out-ring every M rounds** (or one event per emit batch) for EOS/stop-string
   checks + streaming. EOS overshoot = at most M-1 wasted rounds (M=4-8; stop is rare).
5. **Exactness**: the accept rule, rollback targets, and seed gather replicate the host walk
   verbatim on device — token-identity gates (K=1..8 self-consistency + seeded sampled rerun)
   arbitrate as always. Snapshot/ckpt machinery (VerifyCkpt, replay-free partial accept) already
   lives device-side.
STAGE PROGRESS (2026-07-10): (a) LANDED green; (b) COMPLETE green (seeds + KV-len rollback via
len_d pointer table + recur restore _dc twins — device owns every n_acc-dependent commit input;
acceptance parity exact at every K, raw + serve). (c) GROUNDWORK LANDED: embed_gather_device_td
(device-token verify embed) + spec_assemble_verify kernel (verify tokens from the K-chain pack
slots + pending sentinel + in-kernel d2t + p-min k_used derivation). (c4 VERIFY-CHAIN THREADING LANDED 2026-07-10: decode_step_t_core_stream(stream=(vtok_d, ctr))
routes embed через embed_gather_device_td, rope via pos_iota, and full_attn_verify's stream arm
(append_kv_quantized_rows_dc + fa_decode_rows_dc + combine _dc) all off ONE device counter —
every-layer-len == cache.pos invariant; host kvl.len becomes a stale lower bound (split-sizing
upper = len + t + 64 slack, min max_ctx). Default path re-gated green after each change.
STILL REMAINING for c4: K-chain draft graph resurrection under the stream seam (git history has
the reverted implementation — commit "draft-readback arc negative" parent), the out-ring append
kernel, the M-round pre-issue loop + drain + reconcile (host mirrors from device counters), the
last_pred seeding for round 1, battery, A/B vs devacc-off.)
(c) REMAINING, in order:
c2 verify-chain device-pos (_dc pos-vec kernel, append-at-len_d ROWS variant, fa rows t_kv_base
from len_d — decode-graph precedents exist for all three at t=1); c3 accept kernel consumes
brk[] — SPEC: spec_accept_greedy_dc(preds, vtok, last_pred_dev, brk, out): k_used = brk[0],
base = brk[1]; the assembled vtok[base..base+k_used] ARE the d2t-mapped draft tokens (compare
against those, no separate draft buffer); t_pred(j) = preds[base+j-1] for j>=1, j==0&&base==0
reads last_pred_dev[0] (host-seeded once at round 1 — STREAM INVARIANT: every non-replay arm
sets pending=Some(bonus), so base==1 for every round after the first and last_pred is dead);
c4 the stream loop mode (BW24_SPEC_STREAM: K-chain draft graph
resurrected always-K + zero-readback round issue + out-ring + M-round drain) + battery + A/B.
NOTE: the K-chain always-K "waste" negative (draft-readback-arc row) is EXPECTED to invert here
— its cost was measured WITH per-round readbacks still present.
(a) LANDED green (spec_accept_greedy + BW24_SPEC_DEVACC=1 seam,
K=1..8 token-identical). (b) piece 1 BUILT: spec_seed_gather kernel + Engine method (unifies the
three commit arms' seed rules: j=base+n_acc, j>=1 -> vx col j-1, j==0 -> fill_prev; writes
h_seed + fill_prev). NEXT WIRING POINT: the three §5 commit arms in spec.rs (~line 2010-2115)
each do host-offset D2Ds for seed/fill_prev — replace with spec_seed_gather under the devacc
seam; note the full-accept arm ALSO gathers col t_v-2 for mtp_kv_fill's hp (a second gather or
extend the kernel). CRUX for stage (c): the three arms run DIFFERENT host-length-dependent
kernel sequences (mtp_kv_fill token slices, refresh vxs lengths, commit_verified_prefix) —
pre-issuing needs union-with-device-guards (the CSR early-exit pattern) or an n_acc-max
over-provisioned fill.
6. **Incremental landing order**: (a) device accept kernel + device rollback with host loop
   still reading n_acc each round (gates the kernel alone); (b) move seed-gather + next-draft
   inputs on-device (host readback becomes optional); (c) pre-issue M rounds. Each stage battery-
   gated; any stage can ship alone.

Hazards: sampled path needs the residual-sampling chain in the stream (its kernels are already
device; the q_slots D2D per draft step is stream-ordered — fine); zero-round fold (pending arm)
becomes a device branch — fold it into the accept kernel's (d) outputs; BW24_SPEC_PHASE timers
lose meaning past stage (b) — keep them stage-(a)-only.

## NIGHT 2026-07-10 PART 2: FA SHARED-K REFUTED; GPU-BOUND CORRECTION; OPS INCIDENTS

- **Round loop = GPU-BOUND (correction):** the 4-bucket BW24_SPEC_PHASE split (draft /
  verify-issue / verify-wait / commit-host) shows verify-issue (4.6-5.0ms/rd) fully overlapped
  by a LONGER GPU verify (~9-11ms/rd); commit-host 0.3ms. The earlier "host-issue-bound +30-50%"
  read was a 3-bucket artifact. Verify-graph arc downgraded to ~10-15% ceiling (draft roundtrips).
- **FA verify shared-K REFUTED before build:** ncu on fa_decode_vec_q_rows_v3 at p3 depth:
  dram__throughput 2.35% (≈zero DRAM), lts 38.6MB/launch — the 4 rows' KV re-reads are entirely
  L2-absorbed; kernel is latency-bound. Shared-K ceiling ~2-3% vs a geometry-hard port (per-row
  split/tile boundaries pinned by exactness, misaligned across rows). CLOSED.
- **Draft-readback arc NEGATIVE ×3** (packed 8B readback / K-chain-one-graph / per-step+d2t):
  all reverted; law = after the first dependency-satisfying sync, tiny DtoHs are near-free.
- **OPS (cost a night of trust):** (1) my 35B/27B loads OOM-crashed the owner's colbert server
  twice — resident-experts budgets 80% of free VRAM assuming sole ownership; EVERY run now
  checks compute-apps first + caps BW24_MOE_RESIDENT_GB. (2) ncu crashes live GPU apps incl.
  the session-hosting desktop app — ncu ONLY with per-run owner approval (nsys is safe).
  (3) Codex desktop had an unrelated spinning Electron gpu-process (93% CPU) — restarted.
- **down-csr2: NEGATIVE, killed** (-14% e2e despite bit-identity; with v1 this CLOSES the
  down-dedup space — 16-group rows can't amortize any dedup structure).
- **BOARD REFRESHED (v0.14.0):** 35B spec 293.6 / 233.1 / 265.9 (+4.7/+3.1/+4.0% — router GEMV
  + CSR gate_up transfer fine at serve config; the earlier "wash" was colbert/llama CONTENTION).
  Ratios vs marked llama floor: p1 1.17x, p2 1.05x, p3 1.07x. H-logs N=2, 1590MHz, idle GPU.
- **27B p2 FLIPPED ABOVE FLOOR (v0.15.0):** the 0.947x cell was a stale/contended pair — clean
  same-session interleaved re-pair (I-logs): p2 95.4 vs 91.7 = 1.04x, p1 107.3 vs 86.3 = 1.24x.
  Third contention-fabricated cell tonight. Bar map after v0.15.0 — below 1.1x: 27B p2 1.04,
  35B p2 1.05, 35B p3 1.07; NOTHING below floor. 27B phase split mirrors the 35B (GPU-bound,
  device-side-acceptance ceiling ~9%).
- **REMAINING ENGINE LEVER (measured):** the ~10-15% draft-roundtrip gap (device-side
  acceptance / conditional round-graph). Everything else on the 35B spec round is
  measured-closed. p2/p3 to 1.1x also reachable via acceptance (owner's head research).

## NIGHT 2026-07-10 (session continued): ROUTER GEMV + CSR DEDUP SHIPPED; SERVE-CONFIG TRANSFER MAP

**Shipped (both DEFAULT ON, battery green, v0.13.0):**
- **Router GEMV kernel** (3432db1): in-house warp-per-(expert,token) f32 router on the verify
  small-t path, replaces ~240 per-column cuBLAS gemv launches/round. Acceptance bit-identical
  every K (no routing flips). Raw-config +2-4% spec e2e. `BW24_ROUTER_KERNEL=0` rollback.
- **CSR expert-dedup gate_up** (23869f1): owner-scan kernel (grid.y=pair, first pair of each
  expert serves all its pairs), IQ3_S/IQ4_XS decode hoisted to registers. Overlap measured
  0.60-0.62 unique/pairs at t=4 (`BW24_MOE_OVERLAP`). gate_up 55.0->39.7us/l. Raw-config
  +1-2% all K. Three-iteration design record in rig5090.jsonl (v1 never engaged — gate/up are
  IQ3_S not IQ4_XS, A/B law caught it; v2 build-kernel cost 18.2us/l + CSR down LOST 23.5->37.5;
  v3 owner-scan). **ULP law hardened:** bit-identity across kernel structures requires EXPLICIT
  __fmaf_rn/__fmul_rn — nvcc contraction differs per kernel body (naive drifted 1-2 ULP on 35%
  of elements). **Early-return trap:** the 35B HAS shexp — a verify-path arm that returns before
  the shared-expert epilogue FAILS deterministically. `BW24_MOE_CSR=0` rollback, `=2` byte-compare.

**TRANSFER LESSON (key):** both wins are REAL at the raw run-spec config (55-65% acceptance,
no trim) but a WASH at the BOARD serve config (p3v3 sampled rollback A/B: 238.8 vs 240.0) —
at 88.9% acceptance + trim + PMIN0 the MoE-verify share dilutes. Board cells did not move.
Raw-config wins must be re-proven at serve config before claiming cell movement.

**Serve-config round map (35B p3v3 sampled K=3, shares of round loop after subtracting the
seeded-rerun prime contamination — =2 sampled captures include the rerun's prime):**
MoE verify (gate_up_csr 13.6% + down_rows 6.7%) ~20% #1; trunk q8 matvecs 15.3% #2 (marked
94%-of-wall closed); FA decode ~9% #3 (G7e shared-K-reuse lane); q6_K trimmed head 3.8%;
quantize 3.6% (42k launches). K=3 re-confirmed optimal at 89% acceptance (K=2 224.4, K=4 232.7
vs 240.0) — deep-K stays refuted even at high acceptance.

**NSYS-INFLATION LAW (hardened, cost one wrong arc):** the 27B "GPU only 37% busy in the round
loop" was fabricated by nsys per-launch tracing — the same command runs 74-76 tok/s un-profiled
vs 42.9 under nsys. envc() cached-env conversion (95 sites) measured e2e WASH N=3 and was
REVERTED (row: envc-cache-wash-and-nsys-inflation). nsys = SHARES ONLY, never gap/wall claims;
small-kernel shares are inflated too (serialization). Still real and unquantified: the round-loop
blocking-DtoH ping-pong (80 pageable DtoH/run) — measuring it needs a non-tracing method
(cudaEvents delta or CPU sampler); device-side acceptance is the structural fix if it's big.

**27B p2 (0.947x cell) re-confirmed engine-closed:** profile = 75% NVFP4 b4-tier matvecs
(rpr2 36% + rpr2w8 28.8% + rp 10.2%), all measured-closed by the 07-06 m-small arc (rpms/rpsc
negative, rpks banned on k-order, w8 crossing rule tuned); daily config already code75+HPOST
("re-confirmed under HPOST" — the HANDOVER "untested" note was stale). The cell's lever remains
content acceptance (owner's head research), not engine work.

**Standings check (F-logs, N=2, rebaseline-logs/):** p1 281.1 (+0.25% vs board), p2 227.0 clean
(+0.4%), p3v3 240-242 vs board 255.7 — the -6% is CONTENTION/REGIME, not regression: identical
tokens+acceptance (176/198 88.9%), only wall differs; colbert python (1.4GB) + llama-server
resident on GPU all night, prime +13% same signature. BOARD UNTOUCHED — any refresh needs an
idle GPU + same-session llama re-pair. NOTE: board p3 prompt = p3-agentic-long-**v3**.txt
(22999 chars/5420 tok), not p3-agentic-long.txt — mismatch cost one confused Sonnet run.

## STANDINGS 2026-07-08 (full-power verified, plain-first per mandate; llama = floor, margin bar = 1.15x)

PLAIN (tg128, both engines same-day, FA_V2 default-on 2026-07-08 evening): d512 — 9B 132.7 vs 124.6 = 1.07x | 27B 47.7 vs 43.5 = 1.10x | 35B 173.4 vs 170.5 = 1.02x (ALL THREE ABOVE LLAMA, first time). d6257 — 9B 124.5 vs 119.6 = 1.04x | 27B 44.9 vs 42.0 = 1.07x | 35B 158.5 vs 159.9 = 0.99x. Mechanism that did it: FA_V2 tile-batched online softmax (favendor lane — llama's depth flatness was OUR serial softmax chain, not their kernel; theirs is actually slower). Margin bar (>=1.1x) still open everywhere except 27B d512.

SPEC: 27B per-class K = 122/96/76 vs llama serve 87/92/75 (1.40x/1.04x/1.00x) | 9B K=3 = 200-202 p1 (1.6x) | 35B K=2+trim+zero-draft = 197/194/177 vs llama self-MTP 215/208/202 (0.92/0.93/0.88x). Spec mechanisms are per-(model,content): K depth, trim variant (generic transfers across same-vocab models; specialized rankings do not), PMIN0 zero-draft rounds (pays below ~75% base acceptance, hurts above ~90%).

SCOPE LAW (owner, 2026-07-09): THIS box only. Every target, lane, number exists to make the local rig (RTX 5090 Laptop 24GB) faster — other machines are dev/verify surfaces, never the result. M3 specifically: its number is LOCAL ~1.5 tok/s (NVMe capacity-bound); improving it on bigger boxes is out of scope. It stays on the roadmap only if a local lever exists (disk-tier streaming efficiency, or lighter expert cut fitting 24GB VRAM + 60GB RAM); otherwise it is loader/architecture capability demo, not perf lane.

QUANT SCOPE LAW (owner, 2026-07-09): NVFP4 is THE quant for bw24 — full precision when needed (research/oracle), NVFP4 for everything else. No quant sprawl: no new k-quant/format investment beyond what exists; kill exploratory quant arms unless NVFP4-relevant. BOARD POLICY: ST rows publish head-to-head vs llama even without a same-file twin — we field our best, llama fields its best, the table shows both. Both containers (GGUF + ST) stay tuned + benched until the format choice is made (preference: ST — less quant/dequant, native availability); after the choice, stop pushing the loser.

FORMAT DIRECTION (owner, 2026-07-09): support BOTH GGUF and safetensors at high performance — squeeze both, then choose what we run. GGUF is extra conversion step and limitation: not the preferred default; ST is native path (checkpoints ship as ST). The ST-perf work (e4m3-direct, FP8 prefill, NV-27B lane) was commissioned for this; 9B ST lands on same machinery. Research rides the platform and sets its direction — MTP-heal protocol is current example (its needs drove FULL_PREC + ST tooling).

BW24 DUAL-SHAPE (owner, 2026-07-09): shape 1 = the perf push continues; shape 2 = RESEARCH PLATFORM. First research task (NEXT after the two open lanes — box w4a8v2 + hy3port phase 1): MTP-HEAL PROTOCOL step 1-2. Goal: measure MTP draft-head acceptance on the 9B at FULL PRECISION (qwen35-9b-hf bf16 ST, natural mtp.* head — verified present, 15 tensors) as the ceiling, then on the 9B NVFP4 (daily gguf, same-lineage head) — the delta = the quant hit on drafting. Later research: re-train the head to heal it. Rig-side needs: (a) full-precision loader mode (flag, e.g. BW24_FULL_PREC=1: NO re-encodes — the BF16->Q8_0 loader law is exactly what must be bypassed; everything loads Float, compute rides the Stage-A f32 oracle path; SLOW IS FINE — acceptance numbers don't need speed); (b) oracle-path spec decoding must work (run-spec under FAST=0/full-prec — verify, fix if the draft graph assumes fast-path); (c) acceptance battery harness: fixed prompt set (p1-p3 + the agent-loop turns protocol), per-slot acceptance profiles (the deep-K arc tooling), N>=3, both models, delta table as the deliverable. VRAM: 9B bf16 ~18GB + f32 activations — fits; ctx modest. VERIFIED (sonnet lane, 2026-07-09): both checkpoints complete + same lineage (byte-identical chat templates, identical 15-tensor head topology; eos_token_id discrepancy in HF config.json is a stale base value, GGUF's is authoritative). EXPERIMENTAL-DESIGN FACT: the NVFP4 gguf stores the MTP HEAD at Q8_0 (protected) — only the trunk is NVFP4-mixed. The measured acceptance delta therefore isolates TRUNK-INDUCED degradation (head reads degraded hidden states + predicts a shifted target distribution) — exactly the component head re-training would heal, uncontaminated by head-weight quant noise.

ROADMAP ORDER (owner, 2026-07-09): (1) QWEN IS NOT DONE — ST path (safetensors NV-27B lane) has improvement room: decode 92.5 spec vs GGUF twin's class, FP8-stash endgame (drop Q8_0 duplicate via e4m3-reading decode kernel = full-coverage FP8 prefill inside 24GB), and ST plain number. Finish Qwen first. (2) THEN Gemma-4 and Hy3, before-or-parallel. (3) Hy3 target = REAP50 (~150B, ~84GB @4-bit), NOT REAP75: disk tier is the specialty being developed — REAP50 puts majority of experts in RAM+VRAM (good baseline perf) with real NVMe minority to exercise spilling machinery; REAP75's model-quality drop is too steep. Blockers priced: no CUDA-format checkpoint exists (published cuts are MLX 4-bit only — transcode MLX→Q4_K keeps asymmetry, MLX→NVFP4 pays asym→sym tax; quality gates decide), and hy_v3 is new arch port (Hunyuan MoE; M3's sigmoid-router work is the prior).

RELEASE POLICY (owner, 2026-07-08): tag GitHub release for every board-moving or user-facing change — minor bump per mechanism/board move, patch for fixes/docs; notes say what moved and why (public change tracking; no retirement-note verbosity — young project, state current truth plainly). First: v0.1.0. `gh release create vX.Y.Z --notes-file ...` after merge lands.

OPEN FRONTS (priority order, 2026-07-08 evening): (1) PREFILL — the #1 plain gap, PP_ONLY protocol: 9B 4631 vs llama 6287 (0.74x), 27B 1297 vs 2348 (0.55x), 35B 2387 vs 3981 (0.60x). Decomposed (ppmmq lane): compute-bound, 90%+ already on tensor-core MMQ, residual = int8-vs-native-FP4 ceiling; W4A4 in-tree inverts the gap (1.03-1.06x) but is exactness-blocked. Safe levers: cp.async pipeline for the W4A8 GEMM (runs 18% of int8 ceiling), vendor llama's q8_0 MMQ tile for the 35B (27% of its pp, 4x deficit — lane running). FP8-act stash shipped for F8-native checkpoints (NV-27B 888->1136 local). (2) 35B decode closure — favendor lane (vendor llama's fattn mechanism for kv2 at depth) + kvbytes lane (V q4_0/FP8 arms, includes re-baselining llama at ITS best KV config). (3) 27B/9B short-ctx dense decode = ceiling-adjacent at matched bytes+power (measured: decode power-walled at 171.5W/2.08GHz, free DVFS optimal, ALU diets flat) — only bytes-cutting moves it. Spec board: 35B 0.88-0.93x remains (verify-cost tier, task 25 b12/b16).

LAWS HARDENED THIS CYCLE: power boost (+25W) silently resets — verify before EVERY bench session (cost two false boards). FA/split geometry validates across the DEPTH axis (3 depths minimum). bpw equality ≠ quality-class equality across asymmetric/symmetric quant families (Q4_K→NVFP4 taxes acceptance despite equal bits). Float-poison tripwire now in the loader (occurrence #4 was M3's 4.9GB BF16 lm_head). Dynamic activation quant = immune to the uncalibrated-tail-expert checkpoint trap. Kernel-efficiency claims need LOCKED-CLOCK or clock-recorded runs — DVFS converts instruction wins into clock wins and hides them at fixed clock (q6issue lane, refuted its own premise). Mem P-state check before ANY bandwidth claim — idle-cold P8 reads 64% of P0 bandwidth and fabricated the '61%-of-wall k-quant' figure. Competitor baselines rot: re-baseline llama in the SAME session as any board claim (the 9B '1.59x' survived 2 days against a stale llama arm).

## MEASUREMENT PROTOCOL SHIFT (owner, 2026-07-09, post-v0.8.0)

The llama win is MARKED for the Qwen models (9B/27B/35B, GGUF + ST rows published v0.8.0).
From now on these models measure BETWEEN OUR OWN VERSIONS: config A/B and regression tracking
against our standing best (the rig5090.jsonl standings are the baseline to beat). No more
llama re-pairing sessions for them — llama board numbers stand as the marked win. llama
pairing returns only for NEW models (Gemma-4, Hy3) when their rows first land.

## ST (SAFETENSORS) SPEC LANES — MERGED 2026-07-09, BOARD PENDING

Both NVFP4 ST checkpoints now have tuned spec configs (frspec_rank accepts HF dirs — trims derive
from the checkpoint's OWN tokenizer, no GGUF in the ST toolchain):

- **NV-27B ST** (nvidia-qwen36-27b-nvfp4): best = own-head K=3 HPOST=1 pmin=0.4 NV_W4=1 + corpus
  trim = **95.4 tok/s** N=3 (2.01x plain 47.5; acceptance 70.7% deterministic; beats the 92.5
  house-head standing without an external draft GGUF). PMIN0 negative (-5%), pmin=0.5 negative,
  trim marginal on the BF16 head (+0.5% — head read is ~3% of a draft round). House GGUF draft
  still faster (103.7) but needs the external file. p2 64.0 / p3 63.8 are SINGLE RUNS — board
  bench re-measures. pp1855: 1341 default / 1480 ST_E4M3.
- **9B ST modelopt** (qwen35-9b-nvfp4-st-modelopt): native trim (vocab 248070, 99.62% coverage
  @32k), best swept config K=2 pmin=0.3 trim, cold-start 190.5/188.5/217.7 p1/p2/p3, p3
  acceptance 97.7%. Thermal law reconfirmed: 9-load session sagged ~7-8% vs cold pairs.
- **RESOLVED — the "K=3 OOMs at p3, 24GB ceiling" claim was FABRICATED** (rig5090 tag
  `9bst-k3-repro`): isolated repro ran K=3 p3 clean — 256.9 tok/s, acceptance 100%, gates PASS.
  Lane deaths were sibling-lane GPU contention; no OOM string ever existed. Consequence: per-content
  K on the 9B ST — **K=2 for short-code** (K=3 acceptance collapses to 55.6%), **K=3 for agentic**
  (+15%, acceptance 100%). Incident produced the CLAUDE.md "Evidence discipline" section and the
  owner's lane-tiering rule (no Opus lanes; Sonnet = mechanical runs only, coding on main thread).
- **BOARD LANDED (v0.8.0)** — same-regime cold-start pairing, tags `9bst-board-pairing` /
  `27bst-board-pairing-FIX`: 9B ST plain 129.1 vs 123.7 (1.04x), spec 203.9/192.5/256.0 vs raw-llama
  122.9/122.2/118.2. 27B ST plain 45.2 vs 41.2 (1.10x), spec 92.9/81.3/84.6 vs llama-MTP
  79.7/84.7/71.3 (1.17x/0.96x/1.19x). The first bench pass ran 27B with a nonexistent flag name
  (`BW24_SPEC_NV_W4` — orchestrator typo, agent skipped verify) AND the afternoon regime drifted
  ~8% on both engines: every 27B cell re-measured same-hour. LAWS: pairs share the hour-regime, not
  just the day; env configs are copy-pasted from the JSONL status field, never re-typed. NV_W4 tax
  is content-dependent (Q8_0-attn arm beats NV_W4 on p2: 85.4 vs 81.3 — acceptance outweighs plain
  speed on medium-code; p2 loses to llama either way, the one open ST spec cell).

## MTP-HEAL RESEARCH PLATFORM — FOUNDATION (lane/mtpheal, 2026-07-09)

Shape 2 = research platform; first protocol = MTP-heal step 1-2 (measure MTP draft-head acceptance
at full precision as the CEILING, then on NVFP4 → delta = the quant hit on drafting). This lane laid
the RIG-SIDE FOUNDATION (measurement runs come later, on the GPU):

- **`BW24_FULL_PREC=1` loader mode** (default OFF): NO re-encodes — bypasses the BF16→Q8_0 loader law
  and the Float-poison tripwire (both suppressed; correct behavior under the flag). New enum variant
  `GpuTensor::FloatBf16` keeps large 2D bf16 matmul weights **bf16-resident** (2 B/w) with
  dequant-on-use (new `bf16_to_f32` kernel → transient f32 scratch → the existing cuBLASLt f32 GEMV;
  bit-identical to a load-time dequant, pinned by `bf16_dequant_on_use_kernel_contract` test). VRAM:
  9B ~18GB bf16 + ≤4GB largest-weight f32 scratch + activations fits 24GB (an all-f32 materialization
  would be ~38GB — the reason bf16-resident is mandatory, not optional). Small/1D/norm tensors stay
  f32 Float. source.rs guards the BF16→Q8_0 arm; model.rs routes the `None` arm.
- **Oracle-path spec decode** under the flag: EAGER draft forced (`graph_draft && !full_prec`) — CUDA
  graph capture can't enclose cuBLASLt f32 GEMV or the dequant alloc. Eager already routes FloatBf16
  through `matmul`/`matmul_decode_exact` (dequant-on-use). `BW24_FRSPEC_TRIM` disabled under the flag
  (the ceiling wants the natural full head; trim gather is Quant-only). Per-slot decay = the existing
  `BW24_SPEC_STATS=1` (no new seam needed).
- **Acceptance battery** (`tools/`): `acceptance_battery.sh` (p1/p2/p3 + 8-turn agent loop, N≥3,
  JSONL) → `acceptance_delta.py` (bf16-vs-NVFP4 delta table). `agent_loop_acceptance.sh` recreates the
  ephemeral /tmp/w4a4-loop.sh 8-turn accumulative protocol in-repo. `acceptance_parse.py` scrapes one
  `BW24_SPEC_K=<k>` + `BW24_SPEC_STATS=1` run-spec invocation into a JSONL row. Flags/usage: docs/FLAGS.md §6.
  ```
  FULL_PREC=1 tools/acceptance_battery.sh /data/ai-ml/hf-models/qwen35-9b-hf out-bf16.jsonl
  tools/acceptance_battery.sh /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf out-nvfp4.jsonl
  tools/acceptance_delta.py out-bf16.jsonl out-nvfp4.jsonl --json summary.json
  ```
- **GPU-DEFERRED** (another lane owned the GPU; run-gen 19.9GB resident): the actual full-prec load +
  run-spec K=1..8 self-consistency on the 9B ST, the VRAM-fit confirmation, and the first real JSONL
  rows. Compile-clean (full `--bins` build incl the new fatbin kernel) + host tests green.

## MINIMAX-M3 LANE (merged to main ba94a30, 2026-07-07)

121GB NVFP4 REAP50 runs on this 24GB/60GB rig: ST disk-tier loader (stream-repack to .bw24-repack cache, ~5GB RSS), sigmoid routing + e_score_correction_bias, swigluoai via the unified ffn_act seam, gate-optional attention, dense-layer override. **T=1..4 bit-exact (gate maxdiff 0.0, verify-referenced)**. LAW: sigmoid-router MoE gates must reference the SERVING path — f32 KV oracles amplify through discontinuous top-k. Chat gen CORRECT (merge_sorted_lists w/ reasoning), ~1.5 tok/s PCIe/NVMe-bound.
**Perf verdicts 2026-07-07:** pinned-slab tier = 30x REGRESSION (0.05 tok/s) — pinning 26GB evicted the page cache the mmap tier lives on; default OFF (BW24_ST_PINNED=1 opt-in for fits-in-RAM cuts). Routing-locality MEASURED (BW24_MOE_TRACE + research/scripts/moe_trace_analyze.py, 122 decode steps): working set 77% of all (layer,expert) pairs and growing, step reuse 35%, top-50% pairs = 85% of picks — **capacity is the lever, not policy; local rig capped ~2-5 tok/s by NVMe faults regardless of tiering. M3 perf home = the 96GB box.** Local = correctness/dev rig.
**Box first run (81c1b1a, 2026-07-07): steady ~5.3-5.9 tok/s at tg2048 (4x local, still climbing — 80GB SLRU warmup takes O(1000) tokens). SURPRISE: even 96GB not all-resident — 122GB set vs 76.6GB budget (80% of free) → SLRU. Traps: EBS storage = 0.42 tok/s (98% util 4KB random reads) — instance-store NVMe mandatory; stale-main gate FAIL — fused-FFN swigluoai clamp fix lives on lane/minimax, now merged to main. Levers: partial-resident tier / BW24_MOE_RESIDENT_GB override, MoE-cache hit-rate visibility on ST path, spec decode.** **MMVQ item CLOSED with mechanism (mmvq_bisect probe): L0 diff 2.98e-7 = pure kernel-family reduce-order ULP → sigmoid router flips an expert at L1 (x20000) → cascade. No pairing bug; cross-kernel-family text divergence is architectural on sigmoid-router MoE. Exactness law binds within a config only.** Open: sigmoid device-router kernel, hd128 FA on real model.

## NVIDIA-OFFICIAL 27B NVFP4 LANE (merged to main ba24b01, 2026-07-07)

nvidia/Qwen3.6-27B-NVFP4 (safetensors, NOT gguf; mixed NVFP4 MLP + FP8-E4M3 linear_attn + BF16 model-trained MTP head) loads and spec-decodes on the 24GB rig. F8→Q8_0 host re-encode on BOTH Plain and Transform arms (Transform-arm F32 fall-through was the load-tail OOM: 23.6GB→17.2GB), BF16 embed→Q8_0, mtp.\* head mapping (nextn.enorm/hnorm/eh_proj/shared_head_norm → mtp.pre_fc_norm_embedding/pre_fc_norm_hidden/fc/norm; n_layer = trunk+MTP = 65 per GGUF block_count convention), twin-parity test bit-exact. Gates: ST-dir MATCH 0.0 on merged main, spec K=1..8 exact. **Local (5090) tuned 2026-07-07: plain 37.9 → 45.6 (BW24_NV_W4: F8 attn → NVFP4 re-quant, +20%, acceptance held); LOCAL BEST = W4 + frspec32768-generic-trim + K=2 HPOST=1 pmin=0.3 = 65.1 tok/s acc 87% (1.43x plain). Trim law: generic frequency ranking transfers across heads; code75/balanced specialized rankings are head-specific (−14 acc pts on this head). House daily same prompt = 116.9 (acc 94.9) — gap = acceptance + per-round cost; next: NV-native trim re-derive, FP8-act CUTLASS prefill.**
**vLLM 0.24.0 reference (g7e box, same checkpoint): plain 66 tok/s flat, MTP-spec best K=4 = 147-184 tok/s (2.2-2.8x), prefill ~6.7k tok/s, K=1 acc 89.7%. CRITICAL: vLLM has NO native FP4 on sm_120 — NVFP4 MLP rides Marlin weight-only dequant, exactly the regime bw24's native block-FP4 path attacks.** bw24-on-box bench running (direct same-box comparison). Gap analysis next: vLLM spec ratio 2.2-2.8x vs bw24 1.29x at same acceptance = our per-round verify cost is the target.

## CLOSED 2026-07-07 (probe-first discipline)

W4A4 code-search re-test (already vendored; p3 real-prompt still forks — e2m1 activation grid is the gap, FP8-act CUTLASS = the remaining prefill card); PDL (probe: +9% at 1.4us kernels but -24% at real MoE scale — prolog competes for DRAM with predecessor tail on 858GB/s silicon; launch-count reduction stays the only launch lever); 35B expert slab-repack (g7e probe +11.6% kernel = 1.3% e2e, under bar).

## 27B OPEN GAPS (ranked by measured headroom, 2026-07-06)

1. **Prefill 1.65x** — p3 prime 4.96s vs llama ~3.0s (pp 2098 vs ~1265 effective). Next local arc.
2. ~~p2 gen~~ CLOSED WITH PROOF (g7e f8ded04): per-round cost grows only +1.6% over 13x ctx; at p1 acceptance p2 would run 149.5 ≈ p1 150.5 — gap is 100% content acceptance, engine at parity. No kernel lever exists; head retrain already closed negative.
3. **b4 verify tier on 5090** — 43% DRAM, 62% long_scoreboard, reg-limited. rpsc (smem scale prestage, f57f10e) fixed b8 tier only; k-split BANNED for verify (FP-order lesson x4: self-consistency FAIL) but LEGAL for MoE experts (softmax sums).
4. **Acceptance** — head levers closed (retrain negative, author block + HPOST is the ceiling); untested: code75 trim variant under HPOST.

## CLOSED DIRECTIONS 2026-07-06 (don't retry — proof in rig5090.jsonl)

spec fixed-sync-overhead (pmin/burst A/B null); m=4 MMA verify (grid starves, crossover needs token volume); forced-BV globals (auto optimal); b4 unroll-2 + rpca cp.async (reg-growth occupancy trap); p3 config space (K/pmin/split/KVLOCAL all swept — KVLOCAL costs 35 accept pts); MTP head retrain at 10k corpus; 35B trunk q8 (94% wall = done); 35B expert variant space (geometry flat — new-kernel design is the open lead, agent on it).

---

## THE GOAL (session-scoped `/goal` directive — three arms)

**Arm 1 — PRIMARY.** Take bw24 to its absolute edge on THIS EXACT rig. Vendor the best kernel pieces from ALL inference engines and their libraries (llama.cpp, vLLM, SGLang, flashinfer, TensorRT-LLM, CUTLASS). Mark those best-in-class kernels as the *floor* — then tune ON TOP of that floor for this specific rig: "every little drip of memory, every little compute power." Fuse kernels, rework every line, no ns unused, no missed parallelism, hide ops behind other work, saturate any idle memory/compute.
  - **Announcing a limit requires empirical measured proof across ALL directions** — not just the one being worked when the wall was hit. No hand-waving "this is as far as it goes."
  - Also surface: what I (the model) know that the user doesn't that yields big wins, and what the web knows.

**Arm 2 — secondary.** Drive the same work on the L40S AWS box. Branch `arch/sm89-l40s` (worktree `/home/avifenesh/projects/bw24-sm89`). Ada has NO FP4/block-scale but HAS int8 MMA + dp4a + FP8-plain + cp.async — so the k-quant int8 MMA path IS portable there.

**Arm 3 — secondary.** Every grind step becomes labeled training material for a **diffusion-autotuner**: multi-arch config→perf JSONL corpus (`research/tune-data/rig5090.jsonl`, mirror for sm_89). Every win positive, every loss positive.

**Support scope (roadmap, not all now):** NVFP4/mixed dense (Qwen3.5/3.6, Gemma-4) AND MoE (Qwen3.6-35B-A3B, Gemma-4 MoE, MiniMax-M2.7, DeepSeek-4-flash) across VRAM-full / VRAM+CPU-split / NVMe-spill tiers; MoE hot-expert caching; quantized KV cache w/ reuse+eviction; MTP + striped-vocab MTP spec decode; best GEMM/GEMV/matvec usage.

**User works the local grind WITH me directly.** Not a hands-off delegation.

---

## OPERATING RULES (hard constraints)

- **Model orchestration:** sonnet 5 for small tasks, opus 4.8 for medium, big tasks are mine (Fable) with updates to user. "no too much fable in parallel you get throttled" — do NOT fan out many Fable subagents.
- **Caveman mode ACTIVE (full):** terse output, drop articles/filler/hedging, fragments OK. BUT code/commits/PRs written normally, and security/irreversible-action confirmations written normally.
- **CLAUDE.md banned behaviors:** (1) No overstating feature scope — estimate by actual code change, not surface area. (2) No "call it a day" suggestions — user decides session boundaries.
- **Bench protocol (non-negotiable):**
  - Clock-lock for defensible ratios: `sudo -n nvidia-smi -lgc 1860,1860` … `-rgc` to release.
  - Peak numbers matching serve scripts: `gpu-full-power on` (`~/.local/bin/gpu-full-power`, boost=25, 175W).
  - N=5 median. Monitor thermal sag (clocks.sm should hold ~1860).
  - `run_gen.rs` prints wrong tok/s (known timing bug — primes prompt inside timed region). **Trace internals are truth**, not the printed number.
  - Numeric-token prompt for clean pp512: IDs 101..612, **argmax must == 82**.

---

## NEXT KERNEL ARCS (2026-07-05, measured on the 40k trace — the 27B >100 base-path levers)

**ARC A — fa_decode_vec_q long-ctx (decode decay 43->25 tok/s over 1.8k->40k):** wall math says
646MB KV/token (17 layers x 40k x 928B) = 0.76ms at 847GB/s; measured decay +18ms/tok = ~24x off
wall. Kernel facts (cu/flash_attn.cu KERNEL 2b): register-dequant rewrite deliberately dropped the
smem broadcast because "L2 serves the reuse (KV@2048 = 2.2MB << L2)" — at 40k a LAYER's KV is
~37MB, L2 can't hold it, so the 4 GQA warps' re-read = 4x DRAM traffic. Candidates, in order:
(1) re-introduce the smem KV tile broadcast ABOVE a t_kv threshold (dispatch seam, keep the
register path short-ctx — it won at 2k by 12x); (2) the 34B/24B block stride breaks 32B-transaction
alignment — pack-on-append into a 32B-aligned layout (needs an append+decode pair change, gate
battery arbitrates); (3) split sizing at 40k (n_splits already retuned; check tail effects).

**ARC A candidate 2 (32B-aligned split-plane K): MEASURED NEGATIVE 2026-07-05 (g7e lane, JSONL
`arca-kv-align-probe-negative`) — probe-gated, nothing wired.** Standalone microbench
`probe/kv_align_probe.cu`: verbatim fa_decode_vec_q + _smem bodies with a split-plane K twin
(qs plane [max_ctx][nblk][32B] + separate scale plane — pure address remap, partials bit-identical),
27B geometry, 8 rotating layer buffers, clock-locked N=5. The straddle theory's MECHANISM is real —
sectors/request drops 1.22->1.07 (-11% L1 sector traffic, ncu) — but duration is UNCHANGED at
16k-64k (smem 32k: 256.06 vs 256.03us; reg 32k: 1.2% WORSE) because the kernel is LATENCY-bound
(DRAM 15% of peak, warps 34%), not transaction-bound: saved sectors buy zero time. Only reg@8k
gained (+4.5%), below the wire bar and not the blocker cell. Candidate 2 CLOSED with ncu evidence.
Side observation worth a cheap follow-up: in the 8-layer-rotating synthetic the smem twin beats the
register path 2.1-2.3x at EVERY depth incl 8k — the BW24_FA_SMEM_TKV=16384 crossover may sit too
high (real-trace L2 differs; sweep 4096/8192 on real prompts before believing it). Remaining ARC A
room is NOT KV byte addressing — the +18ms/tok decay must live in latency/occupancy structure
(combine chain, n_splits tail, launch overhead).

**ARC B — fa_prefill_q chunk-prime redundant dequant: DONE 2026-07-05 (g7e lane, JSONL
`arcb-prime-deqw`).** Was 30.5%/153ms-launch of the 32k prime wall (every T/64 q-block CTA x
n_head re-dequanted the whole quantized KV stream). Landed all three candidates as stacked
bit-identical rungs, DEFAULT ON: (1) `fa_dequant_kv_ws_bf16` dequants K/V ONCE per (layer,chunk)
into a resident grown-and-reused bf16 workspace (`Engine.prime_deqw_ws`, ~83MB at 40k) storing
exactly the `__float2bfloat16(dq_*)` values the old kernel staged to smem; `fa_prefill_qw` is the
byte-identical MMA/softmax/PV twin over it (32k prime 26.41→18.11s); (2) 16B-uint4 vectorized
workspace→smem staging (→17.10s); (3) `fa_prefill_qw_db` cp.async double-buffered staging twin
(+32KB smem, 1 CTA/SM — wins anyway, single-buffer ncu: mem 66%/SM 15%/DRAM 0.6% staging-stalled)
(→16.51s). **27B N=3: 32k prime 26.41→16.51s (1.60x), 16k 11.80→8.65s (1.36x).** Seams:
`BW24_PRIME_DEQW=0` (inline-dequant fa_prefill_q), `BW24_PRIME_DEQW_DB=0` (single-buffer twin).
Gates: kernel-check ALL GREEN incl new ws-vs-inline bitdiff=0 gate (both twins); argmax 1178/268
MATCH; run-spec K=1..8 PASS both models; p3 chunk-prime token-md5 identical across
deqw-on/deqw-off/monolithic; 16k+32k deep prompts PASS. NEXT prime lever: mul_mat_q_nvfp4_w4a8 is
now 52.5% of the 32k prime — the prime wall is the GEMM itself, not attention.

**27B K=4 SPEC CLIFF — KILLED 2026-07-05 (g7e lane, JSONL `mmvq-b8-k4-cliff-fix`).** ROOT CAUSE
CORRECTION: the remembered "T=5 splits b4 MMVQ into b4+b1" theory was WRONG — the batched dispatch
in matmul/matmul_pre/matmul_decode_exact was gated `(2..=4).contains(&m)`, so T=5..8 verify fell to
the per-row grid.y=m MMVQ (K=4 nsys: 6448x `qmatvec_nvfp4_mmvq_rp` @49us avg = m FULL weight reads
per launch, 8.3% of wall; acceptance held 54-55% — kernel path, not acceptance). Fix = `b8` tier:
mcols=8 instantiations of the EXISTING batched templates (c>=m masked, per-(token,row) dp4a order
verbatim) for NVFP4 (base/pf/r2/r2w8/rp/rpr2/rpr2w8) + k-quants (base/r2), dispatch widened to
2..=8, mcols map 2/4/8. b8 AUTO = the w8 twin everywhere — the b4 wave-crossing rule MISPICKS for
b8 register classes (measured: forced rpr2w8 97.9 vs b4-rule 93.3 tok/s e2e; DRAM-cold rp msweep
m=5/6/8 x 5 shapes: rpr2w8 wins/ties all but ffn_gate m=5/6 where rpr2 edges 1-3%). **27B p3 spec
pmin0.15 clock-locked N=3: K3 107.2 (unchanged, control), K4 75.4→98.0 (+30%, now 0.91x of K3 =
cliff GONE), K5 70.8→95.5, K6 66.4→91.5. Verdict: K=3 stays the 27B optimum (acceptance decay
67→55→50→44% outruns the kernel win); K>=4 now decays smoothly (-8.6/-10.9/-14.6% vs K3) instead
of cliffing — deeper-K becomes viable wherever acceptance holds >60%.** Seams: `BW24_B8=0` (m=5..8
back to per-row; m<=4 untouched), BW24_NO_BATCHED still the global reference, BW24_MMVQ_BV forces
b8 variants too. Gates: kernel-check 126 OK 0 FAIL incl new b8 m=5/6/8 cells (rel=0.00e0 all dtypes,
RP bit-bad=0); argmax 1178/268 MATCH; run-spec K=1..8 PASS both models; p3 real-prompt PASS K=3..6.
LESSON x2: (1) read the dispatch before trusting a remembered mechanism — the "split" was actually
a fall-through to the WORST path; (2) variant auto-rules don't transfer across register classes —
re-measure per MCOLS tier.

**Session/serve wiring (engine API done, server pending):** SpecSession + generate_spec_session
landed with the session-gate oracle (3-turn MATCH both models; 42.6x turn-start at 40k). bw24-server
still creates a fresh cache per request — wire sessions into worker.rs next serve slot.

## NEXT LOCAL ARC — MoE expert dp4a upgrade (specced 2026-07-06, measured on local 35B nsys)

Local 35B decode window (post launch-arc port, 19.7 tok/s): **47% of decode wall = Stage-A f32
dequant expert kernels** — qmatvec_f32 2377ms/272k (staged sequential path via qmatvec_view),
moe_gate_up_silu8_f32 1667ms/14.5k, moe_down8_fma_f32 1280ms/14.5k (per 48-tok window). Experts:
gate/up = IQ3_S (GGUF type 21), down = IQ4_XS (23). All three kernels dequant f32-elementwise
(`deq()`); the trunk already runs q8_1+dp4a everywhere.

THE WORK (matched-pair set — BW24_MOE_GATE byte-identity requires sequential+fused+_dev change TOGETHER):
1. q8_1-quantize the MoE input row `z` once per token (quantize_q8_1 exists; the trunk's rms_norm_q8_1
   fusion pattern applies) and the act row before down.
2. dp4a bodies: IQ4_XS lift from qmatvec_iq4_XS_dp4a (exists, :3007); IQ3_S port from llama
   vec_dot_iq3_s_q8_1 (vecdotq.cuh:1148 — sign-via-__vcmpne4/__vsub4 trick) against bw24's
   deq_iq3_s layout (block=110B: d@0, qs@2, qh@66, signs@74, scales@106; db=d*(1+2*sc_nib)).
   NOTE llama's block scale applies OUTSIDE the int dot — same separable structure as q4k/q5k mmvq.
3. Kernel set (all change in one commit): qmatvec_view -> qmatvec_view_q8 (dp4a),
   moe_gate_up_silu8_{f32,_dev} -> _q8 twins, moe_down8_fma_{f32,_dev} -> _q8 twins. Same
   grid/block/slot-order/silu expression; ONLY the dot arithmetic changes (int dp4a + separable
   scales instead of f32 elementwise).
4. GATES: 35B argmax 248046 (local) / 1178 (box) — the FP-order change SHIFTS logits, argmax +
   run-gen prefill==decode + 64-tok stream identity vs f32 seam (BW24_MOE_Q8=0 rollback) arbitrate;
   BW24_MOE_GATE byte-identity between new sequential and new fused (the pair contract);
   kernel-check oracle gates for both new dot bodies (int-band rel like MMQ-W4A8's).
5. EXPECTED: local 35B 19.7 -> ~26-28 (47% slice at ~2-3x dp4a speedup); G7e 112.9 -> higher
   (silu8/down8 are 41% of its GPU time too). Both regimes win — this is the rare shared lever.
## WHERE THINGS STAND (measured, 9B-NVFP4) — UPDATED 2026-07-03 after Q4_K/Q5_K MMQ landed

| metric | bw24 | llama.cpp | ratio |
|---|---|---|---|
| pp512 INTERLEAVED A/B (honest protocol) | **5049** | 5072 | **0.995x — PARITY BAND** (was 0.91 at session start; conv-tm fuse -> 0.936, scale-fold -> 0.95, conv+GDN-repack fuse -> 0.995) |
| decode tg128 graph @ctx128 INTERLEAVED A/B | **110.5** | 106.3 | **1.04x — ABOVE llama** |
| decode tg128 graph @512 / @2048 | 109.5 / 106.3 | — | — |

## E2E SCOREBOARD (real-prompt protocol, 27/1845/6257-tok prompts; llama = serve-script config)

**BATCHED PROMPT PRIME LANDED (03787d9, 2026-07-03) — gap 1 of e2e-image-1 CLOSED.** generate/generate_spec now prime via `prime_cache` (forward_last's prefill body + cache side-effects: batched quantized KV append `_rows` kernel + stateful linear layers via carried-ring conv + one gdn_scan from cache state) instead of the ~102/38 tok/s tokenwise decode_step loop. Seams: `BW24_PRIME_TOKENWISE=1` (escape), prompts <16 tok stay tokenwise, `BW24_PRIME_APPEND_LOOP=1` (per-row append A/B — measured equal, launch overhead hides in the async queue).

| metric (27/1845/6257 tok) | bw24 BEFORE (e2e-image-1) | bw24 NOW | llama |
|---|---|---|---|
| 9B prime/TTFT | 0.26s / 18.1s / **61s** | 0.10s / 0.80s / **2.66s** (~2350 tok/s) | 0.20s / 0.32s / 1.05s (5934 tok/s) |
| 27B prime/TTFT | 0.7s / 48s / **163s** | 0.32s / 2.72s / **8.74s** (~716 tok/s) | 0.17s / 0.94s / 2.96s (2114 tok/s) |
| 9B gen plain / specK3 | 106.3/100.2/99.2 / 116.6/81.5/73.1 | 104.9/99.3/98.9 / 114.1/84.8/71.9 (noise) | 121.3/120.3/116.5 |
| 27B gen plain / specK3 | 39.4/37.6/37.7 / 53.4/39.7/35.5 | 39.4/38.1/37.9 / 49.4/43.0/35.8 (noise) | 87.6/92.7/75.9 (spec) |

Remaining prime gap vs llama = the PREFILL gap itself (9B 0.40x, 27B 0.34x at 6k) — prime rides forward_last, so every prefill kernel win now transfers 1:1 to TTFT. Gates: kernel-check ALL GREEN (incl new rows-vs-loop byte-identity), run-gen 82==82, run-spec K=1..8 PASS x {9B synth, 9B text, 27B real} all with batched prime, first-token A/B batched-vs-tokenwise agrees 6/6 (full-16 identical 5/6; 9B p2 flips at index 11 — allowed cache-state FP, argmax-level agreement holds where it matters).

**GAP 2 (spec degrades with ctx) — CLOSED 2026-07-03, three stages, nsys-decomposed first (JSONL `spec-longctx-profile`).** The 64-tok steady loop at 9B/6.3k was: 44% MMVQ trunk weights (verify + the partial-accept REPLAY duplicate pass, ~1.54 trunk reads/round), 21% per-row verify FA (fa_decode_vec_q 201us x T rows x 8 full layers — 392-CTA launch = 4.8 CTA/SM vs 12 achievable), 17% draft graphs (~58% of that = q6_K lm_head), 3.5% host gaps (loop already async-clean; no free host win). Design B (key-major shared walk) buried: per-row split boundaries derive from t_kv_r ⇒ never partition-exact.
- **Stage 1 — MULTI-ROW FUSED VERIFY FA (design A):** `fa_decode_vec_q_rows` + `fa_decode_combine_rows`, grid.z=row, per-row t_kv_r/n_splits_r derived in-kernel from the same fa_split_keys formula ⇒ bit-identical per row (kernel-check pins rows-vs-loop bitdiff=0 incl a 64-boundary crossing). Engages when base_len+1 ≥ FA_VEC_MIN_TKV; `BW24_FA_ROWS_OFF=1` seam. A/B: 9B p2 +13.8% / p3 +5.7%; 27B p2 +13.7% / p3 +3.3%. Fused per-row cost 140us vs 201us — floor = single-warp 64-key walk latency (split=64 exactness-pinned).
- **Stage 2 — REPLAY-FREE PARTIAL ACCEPT (the big one):** the verify's first j=base+n_acc columns ARE the committed state (bit-identical to eager, verify-probe contract) — KV truncates to pos+j, recurrent state rebuilds from a per-round `VerifyCkpt` (batched layers: prefix re-run of the SAME gdn_scan kernel t=j from the snapshot state over retained inputs — its t-loop is prefix-stable; conv ring: pure-copy `ssm_conv_ring_rebuild_f32`; per-column layers: dtod state clones), bonus rides PENDING like the full-accept fold. Duplicate trunk replay GONE. `BW24_SPEC_REPLAY=1` seam. A/B: 9B p3 +14.6%, p2 +10.7%; 27B p3 +32%, p2 +19%.
- **Stage 3 — draft side:** (3a) TRUE-HIDDEN REFRESH — batched mtp_kv_fill of ALL committed positions from the verify hidden stack each round (`BW24_SPEC_NOREFRESH=1` seam): 9B p2 +6.2%, 27B p3 +4.0%. (3b) HEAD-LESS pseudo-seed graph (second capture, no lm_head/argmax/prob — identical h_nextn): 9B p3 +4.2%; draft parity graph-vs-eager still bit-identical.
- **RESULT (protocol K=3 pmin0.2, clock-locked, N=5 medians on headline cells): 9B 130.8/100.9/80.2 tok/s (1.41x/1.15x/0.91x, was 1.16/0.93/0.72 pre-work same-session); 27B 66.0/59.7/45.3 (1.86x/1.76x/1.33x — was 1.34/1.09-1.24/0.94). 27B spec now WINS at every size; 9B holds ≥1.0x through p2.** Config sweep (JSONL `spec-config-sweep`): post-replay-free the 9B optimum moved to **K=2 pmin=0.3 uniform: 137.7/104.1/88.5 = 1.48x/1.19x/1.00x — parity at 6k ctx** (was 0.72x); 27B K=2/0.2 = 1.37x at p3; 27B K=4 cliff (T=5 splits b4 MMVQ into b4+b1). Full battery green on the final tree incl all 5 seams. **NEXT levers (profiled, 9B p3 24.5ms/round): (a) MMVQ b4 efficiency — LANDED 2026-07-03 (JSONL `mmvq-batched-latency-fix`): ncu --set full on the real 27B verify showed the batched kernel is memory-LATENCY bound (long_scoreboard 18-30/issue, DRAM 41-51%, ONE 6-LDG weight wavefront in flight/warp = half the m=1 mr2 kernel's weight MLP — NOT bandwidth, NOT the break). Fix = `pf` (next-g weight-prefetch double-buffer, 48 regs) + `r2` (two rows/warp, 67 regs) NVFP4 variants with wave-aware per-shape auto dispatch (`BW24_MMVQ_BV` seam): b4 execution -11.5% on the 27B round -> e2e +7.2%/+6.0% (27B p2/p3 = 63.9/48.0 tok/s), +2.0% (9B both = 106.0/90.2), N=5 interleaved clock-locked, plain-decode control unchanged, all exactness gates green (variants bit-identical per (token,row) — only load issue time / row->warp mapping change). STAGE 2 same day (JSONL `mmvq-b4-r2w8-wave-crossing`): `b4_r2w8` __launch_bounds__(128,8) twin (64 regs, 8 blocks/SM) + integer-wave-crossing dispatch — deleting the straggler wave flipped ffn_down/ssm_out/qkv to r2w8 (down 112.5->81.6us DRAM-cold, beats pf); 27B cumulative +9.1%/+7.7% (p2 65.1 / p3 48.7 tok/s vs base 59.6/45.2), 9B dispatch-identical. Remaining b4 headroom: host-fusing the tiny GDN projections (out_f<=1024 launches still pure-latency 15-16us each), k-quant batched variants (~19% of 9B verify, same recipe). (b) draft lm_head 22% (model-bound, FR-Spec 9B negative); (c) fa_rows 17% (single-warp walk-latency floor, split=64 pinned).**

**MEASUREMENT LESSON (2026-07-03): cross-session pp512 numbers lied.** Sequential runs gave bw24 5531 vs llama 5451 ("parity") — but interleaved same-minute A/B gives 4655 vs 5092 = 0.91x. Root cause: llama holds 1852MHz during its run; bw24 sags to 1710MHz under the same clock lock (bw24 draws more power for the same work → worse perf/W → thermal sag). ALL ratio claims must be interleaved A/B from now on. bw24 has BOTH a remaining kernel gap (~9%) AND a power-efficiency gap (clock sag under load). Decode ratio not yet re-measured interleaved.

**Decode session 2026-07-03 (landed levers, all gate-passing):**
1. warp-per-block coalesced q8_1 epilogues (silu_mul_scaled_q8_1, quantize_q8_1, norm pass-2): 88.4→95.1 graph.
2. float4 warp-per-4-blocks norm pass-2 + 1024-thread CTA: 95.1→104.4.
3. adaptive FA split REVERTED (broke run-spec exactness — split count changes combine FP order). Fixed 64 + BW24_FA_SPLIT seam kept. ~109.6 stands.
4. gdn_prep_decode_f32: repack+2xL2+sigmoid+glog fused 5→1 launches (perf re-measure pending GPU free).

**Prefill session 2026-07-03 part 2 — INTERLEAVED PARITY (0.995x):**
1. ssm_conv1d_tm_f32: transpose+zeros+pad+conv → 1 token-major kernel (0.91→0.936x).
2. NVFP4 macro-scale folded into MMQ write-back — bw24 now does strictly LESS work than llama here (0.936→0.95x).
3. ssm_conv1d_gdn_f32: conv+SiLU scatters straight into GDN q/k/v layout, SSM prep 5 kernels→1 (0.95→0.995x).
Background agent results:
- **K=4 spec-divergence: FIXED** (befc9d0). Root cause: verify used fa_prefill_view (different FP accumulation order than fa_decode) — at tight logit margins argmax flips. Fix = per-row fa_decode in full_attn_verify. K=1..4 ALL PASS at N up to 512. LESSON (2nd instance): greedy-spec exactness requires the SAME kernel FP order, not equivalent math.
- **stream-K q45k MMQ: NEGATIVE** (fbc5f46; the seam + kernels were REMOVED in the 2026-07-08 flag audit — the JSONL row is the record). Per-GEMM 1.11x real (712.9→638.4us incl fixup, per-GEMM rel ≤1.2e-6 vs conventional) but model-level argmax deterministically flips 82→68 (top-2 margin 0.14, 1e-6 reorder noise amplifies over 33 layers). 3rd instance of the FP-order lesson. Warps-active unchanged 16.7% — the q45k occupancy ceiling is 57KB smem 1-CTA/SM, NOT the tail; next q45k lever = smem diet (2-stage y-tile or smaller MMQ_X) to reach 2 CTA/SM.
- **FA-decode port: DONE INLINE** (after agent died twice). Register-dequant rewrite of fa_decode_vec_q/_dc: no smem staging, no syncs, coalesced byte-per-lane K reads, GQA reuse via L2. **+48% tg at 8k ctx (59.8→88.8)**, parity at short ctx. CRITICAL: had to KEEP one bf16 round-trip on dequanted K/V — raw f32 dequant flipped run-spec K=2 (FP-order lesson instance #4). Old-vs-new direct binary A/B, all gates green.

**ncu parity probes (measured, clock-locked):** bw24 NVFP4 matvec = llama mul_mat_vec_q<40> EXACTLY (42% DRAM, ~74us both). Q6_K lm_head 1.07ms/tok — llama same (1.07ms). Matvec is NO LONGER the gap. Remaining 0.93→1.0x gap: FA decode kernels (bw24 fa_decode_vec_q 126us vs llama flash_attn_ext_vec 10.5us/launch — llama fuses whole-layer attention, bw24 splits+combines), graph-tail/launch overhead, elementwise remnants (l2_norm, scale, sigmoid still separate launches; llama has them too). MR=4 multirow kernel crashes ILLEGAL_ADDRESS (pre-existing, non-default — fix or remove).

Commits: **16e896c** (`feat(prefill): vendor llama Q4_K/Q5_K int8-MMA MMQ GEMM — pp512 1874->4576 clock-locked (2.4x)`) on top of **851e80f** (NVFP4 MMQ) + **e04c5f0** (sweep tune-seams, harness-agent work committed). Mirrored to `arch/sm89-l40s` as **7561a06** (BW24_CUDA_ARCH=89 build green; Q4_K/Q5_K int8-MMA path has NO FP4 gate — portable). Training record appended to `research/tune-data/rig5090.jsonl` (**0728173**).

**kernel-check:** ALL GREEN incl. new MMQ-q45k oracle gate (rel 5e-3..7e-3, gate 2e-2). argmax==82 on/off MATCH.

**PREFILL kernel-diff vs llama (nsys, same prompt, 2026-07-03) — the honest 9% decomposed:**
| kernel | bw24 | llama | delta |
|---|---|---|---|
| NVFP4 MMQ | 27.1ms | 25.8ms | +1.3ms (llama stream-K) |
| Q4_K+Q5_K MMQ | 23.4ms | 19.0ms | **+4.4ms** (llama stream-K + fixup; bw24 vendored xy-tiling only) |
| SSM prep (repack 3.6 + transpose 2.4 + conv_pad 1.1 + conv_silu 1.0) | 8.1ms | ~2.2ms (one concat_non_cont + ssm_conv) | **+5.9ms** |
| FA prefill | 1.9ms | 0.66ms | +1.2ms |
| gdn_scan | 17.6ms | 17.9ms | parity |
| scale_f32 (NVFP4 macro-scale bcast) | 4.2ms | ~2.9ms (k_bin_bcast mul) | +1.3ms |

Next prefill levers in order: (a) port stream-K to the q45k MMQ (llama's mul_mat_q_stream_k_fixup — biggest single delta), (b) fuse the SSM prep chain (transpose+repack+pad+conv into 1-2 kernels — pure bw24 self-inflicted, llama does ONE concat), (c) FA prefill tile config, (d) fold macro-scale into MMQ epilogue.

**FA-decode grid starvation (27B ncu, 2026-07-03):** register-dequant kernel at short ctx runs grid (n_head_kv=4, n_splits=2) x (32,6) = 8 CTAs on 82 SMs; long-scoreboard stall 77%, warps_active 12.5%, DRAM 0.08%. It's LATENCY-bound (serial dependent K loads per warp), not BW. Split=32 helps 27B +3.4% but BREAKS 9B spec exactness (even with same-kernel per-row verify — combine order still shifts) so default stays 64. Structural fixes evaluated: (a) 2-key ILP — BUILT AND MEASURED NEGATIVE (63.6 vs 96.8 @8k direct A/B: doubling in-flight K/V registers at dpl=8 collapsed occupancy; the 77% stall wants MORE WARPS, not more per-warp work). (c) dp4a int8 K-dot: BUILT AND MEASURED NEGATIVE — no perf gain (81.2 vs 82.2 @8k, latency-bound on K loads either way; dp4a cut ALU not loads) AND K=1/2 spec exactness broke (FP-order lesson #5 REFINED: same-kernel is necessary but NOT sufficient — int8-Q accuracy change flips tight-margin argmax in the DRAFT chain). **FA-decode dot direction now empirically CLOSED with measured proof on all four variants: smem-broadcast (126us, slow), register-dequant f32 (winner, shipped), 2-key ILP (occupancy collapse 0.66x), dp4a (no gain + exactness break).** Untried: (b) combine-fold into last CTA, (d) 2-kv-heads/CTA packing — both only reshuffle the same latency-bound loop; expected value low after (a)/(c) results.

## EMPIRICAL WALL LEDGER (per the goal's proof standard — every direction measured, 2026-07-03)

**CLOSED with measured proof (do not retry without new information):**
| direction | proof |
|---|---|
| FA-decode dot variants | 4 variants A/B'd: smem-broadcast 126us (slowest), register-dequant f32 (WINNER, shipped, +48% @8k), 2-key ILP 0.66x (occupancy collapse), dp4a K-dot 0.99x + K=1/2 exactness break |
| stream-K on q45k MMQ | 1.11x per-GEMM but argmax 82→68 deterministic (1e-6 reorder amplifies over 33 layers); seam+kernels removed 2026-07-08 (flag audit) |
| q45k MMQ_X=64 smem diet | -2% pp512 (tile efficiency > 2-CTA/SM occupancy at T=512); seam BW24_MMQ_X_Q45K kept |
| adaptive FA split | +1.2% decode but breaks spec exactness; BW24_FA_SPLIT seam kept (27B non-spec +3.4% at 32) |
| q4_K multi-row (27B) | ±0% (measured again this session) |
| matvec DRAM% | at exact llama parity (42% both, same kernel shape) — SOL-bound |
| decode elementwise | all coalesced+fused now; graph 1.04x ABOVE llama interleaved |
| graph-captured verify (spec stage 3) | RE-MEASURED NEGATIVE 2026-07-03 post-unlocks (JSONL `graph-spec-stage3-remeasure`): even with 642582a's in-kernel per-row splits (rebucketing churn GONE) the ceiling shrank below the build gate — steady-window nsys at the image-2 optima: 9B p2 verify 95.1% GPU-busy, gap 0.70ms of a 19.3ms round = 3.8% ceiling; 27B p2 97.2% busy, 1.11ms of 43.6ms = 2.8%; realizable ~half that (graphs pay ~0.2us/node). Capture cost still 3 real-shape passes/bucket => negative through NGEN~500. Stages 1-2 drained the launch pool the graph would have reclaimed. The "27B gap = llama's whole-trunk graphs" hypothesis is measured FALSE: the 27B round is 73.3% qmatvec_nvfp4_mmvq_b4 EXECUTION (~498 launches, 54.9us each) — the gap is b4 MMVQ efficiency + acceptance (0.58 vs ~0.75), not launch overhead |

**FP-ORDER RULES (5 instances, now law):** (1) any reduce-order change flips tight-margin greedy argmax; (2) same-kernel between decode and verify is NECESSARY; (3) …but NOT SUFFICIENT — accuracy changes (int8 Q) flip the draft chain too. Full gate battery = kernel-check + argmax + run-spec K=1..4.

**OPEN directions (not walls — unmeasured or needs infra):**
- **MTP >1.0x: DONE (2026-07-03).** Batched linear-attn verify (linear_attn_verify_t: T-token projections + carried-state conv ssm_conv1d_tm_state + one gdn_scan(T), bit-identical) crossed it: K=4 = 1.03x plain (48.2 vs 47.1 tok/s), all K exact. K sweep 1..8: K=4 optimum, acceptance decays 8-10pts/K (98→46%). Further spec profit needs acceptance lift (draft chaining quality), not deeper K — the verify is no longer the cost. Web context (zolotukhin.ai 2026-05-08 + llama PR 22673): MTP c≈1/L makes it the right draft for hybrid models; llama's 2.4x is against a 7 tok/s launch-bound baseline, ours is BW-bound at 47 — Leviathan-consistent.
- 27B decode 0.92x: NVFP4 matvec 81% of its decode at 41% DRAM (= llama parity — SOL-bound); dual-matvec neutral there. Remaining 27B decode delta is llama's fused mul_mat_vec_q<40,1,true> variant (its 52.8% kernel — the fusion flag folds MORE than gate+up; investigate what the `true` template arg fuses). FA split=32 = +3.4% for non-spec serving (env). **27B spec: UNPROFITABLE at any K (0.45-0.80x) — MTP head acceptance is 54% @K=1 vs 9B's 98%; that's model quality, not engine cost (batched verify ruled out via worktree A/B). 27B long-gen divergence: FIXED 2026-07-04 — three FP-order mismatches in verify (rms_norm blockDim 256 vs fused-1024, l2_norm 256 vs warp-tree, dp4a vs MMVQ at T>=5); decode-exact dispatch variants shipped (FP-order lesson #7: verify must be kernel-DISPATCH-identical). 27B real-prompt K=1..8 all exact.** FR-Spec GGUF variants on disk are the 27B draft-cost lever if acceptance can't move.
- Prefill FA: **CLOSED with measured proof** (FA-PREFILL-OVERLAP-DESIGN.md, clock-locked table): fa share of prefill = 0.7% @512 / 3.7% @8k, grows only ~seq^0.59 — 10% share needs ~48K-token prompts. The producer/consumer overlap is a 5-20% speedup of a ~1-4% component = <1% e2e below 32k ctx. Not worth building until long-ctx prefill becomes a daily workload.
- **Spec-loop graph integration: draft chain DONE (6502142); verify capture CLOSED with measured proof** — see the wall-ledger row + JSONL `graph-spec-stage3-remeasure` (2026-07-03 re-measure at the image-2 optima: ceiling 2.8-3.8% of round, verify already 95-97% GPU-busy, capture cost negative through NGEN~500). Ranked 27B spec levers from that trace: (1) qmatvec_nvfp4_mmvq_b4 efficiency (63% of the whole round) — FIRST TRANCHE CASHED 2026-07-03 (JSONL `mmvq-batched-latency-fix`: b4 execution -11.5%, 27B e2e +6-7.2%), (2) acceptance 0.58→0.75 (draft quality) — CASHED 2026-07-04 via the hidden-pairing fix (see SPEC SCOREBOARD, JSONL `spec-hidden-pairing-prev`), then nothing structural.
- **Condition-scope audit (2026-07-04):** MoE SLRU cache + fused router: BUILT (e1f49ec, 35B decode 6→24-31 tok/s, EDGE-1). Tiered VRAM/pinned/disk spilling: BUILT (e44b89a, BW24_SPILL_DISK). KV quant q8_0/q5_1: BUILT (daily default). Safetensors loaders incl MoE gather + NVFP4 repack: BUILT. FR-Spec/trimmed-vocab MTP: BUILT this session. STILL NOT BUILT: KV prefix reuse/eviction across requests (bw24-server has no prefix cache — matters for the 2-4-agent serve pattern; the lmcache maps in research/inference-maps are the design source). MoE async prefetch: CLOSED NEGATIVE 2026-07-05 (see MoE DECODE-CACHE scoreboard below — miss staging is 3.6% of decode wall, <5% bar). 35B decode re-bench vs llama recorded (74.9 vs 235.4 tg128); gemma4 re-bench blocked on the port gap list below.
- L40S benches: blocked on hardware (box terminated). All commits compile-mirrored to sm89.

## USER'S OWN LLAMA WORK (2026-07-04 — read these before touching spec/quant/MoE)
- **Issue 25187 (his)**: FR-Spec draft-vocab trim for native MTP — HIS branch `avifenesh/llama.cpp/frspec-mtp-vocab-trim` (047bfa508), HIS trimmed GGUFs on disk (frspec32768 variants + d2t tensor). llama measured: draft lm_head -85%, e2e 83.9→85.1 (public map) / 86.5 (code map) tok/s on the 27B daily config. bw24 FR-Spec consumer = agent in flight.
- **PR 25153 (open, his)**: imatrix-aware NVFP4 quantization (scale search) — the NVFP4 quality side.
- **PR 23170 (closed, his)**: MoE experts as cache residents during offloading — EDGE-1 ancestor.
- **Serve script daily config** (~/.local/bin/serve-qwen36-27b): NVFP4 trunk + SEPARATE Q4_K_M MTP draft GGUF, `--spec-draft-n-max 3 --spec-draft-p-min 0.1`, KV q8_0(rotated)/q5_1, graphs ON, 175W, 128k ctx. **llama 27B e2e ≈ 84 tok/s WITH spec — THE number to beat, not plain tg128 42.4.** bw24 27B spec unprofitable (54% blind-draft acceptance) because it lacks p-min CONFIDENCE GATING: llama stops the draft chain when token confidence < p-min, converting low-acceptance rounds into cheap short drafts. That + FR-Spec are the two 27B spec unlocks. Note llama accept ~0.75 at n-max=3 with p-min — same head, gated drafting.

**Vendor-from-everything directive (user, 2026-07-04):** edges can come from ANY tool — sglang, vllm, ktransformers, lmcache, flashinfer, cuBLAS, TensorRT-LLM, ollama, DeepSeek-4 stack, papers. research/inference-maps/ already maps vllm/sglang/ktransformers/lmcache/flashinfer/trt-llm/cutlass-marlin/exllamav3 — USE them per component. E2E tok/s vs llama at the daily serve config (spec+KV-quant on) is the headline bench, not kernel microbenches.

## MoE SCOREBOARD — A2 GROUPED PREFILL: RESIDENT + SPILL BOTH LANDED ON rig5090 (2026-07-04)
**Three points on the same change now recorded: resident-G7e (4.8-7.2x), resident-local, capped-spill-local (JSONL rig5090 records).** 35B-A3B IQ4_XS, `BW24_MOE_GROUPED=1` + `BW24_MOE_CACHE=1`, N=5 in-process medians via the new run-gen pp-only harness (`BW24_PP_REPS`/`BW24_PP_WARMUP`, per-rep tok/s + per-rep staged-GB prints).
- **Resident-local (auto cache = 10390 slots ~11.6GB):** pp501 85.8→126.9 (1.48x), pp1845 91.4→178.6 (1.95x), pp6257 95.1→179.4 (1.89x). Smaller than G7e's 4.8-7.2x for a GOOD reason: the local sequential baseline is 85-95 tok/s vs G7e 60-75 (961920 m=1 launches are latency-bound; the laptop clocks higher and G7e's 188 SMs sit idle on matvecs), so grouped's headroom is smaller. KEY FINDING: even the daily "resident" config is a partial-spill regime — the auto cache holds ~34% of the 30720-block expert set; grouping cut H2D 28-30→10-11 GB/forward (2.7-3.8x).
- **Spill-local (BW24_MOE_SLOTS=64, hit-rate ~0.6%, PCIe-dominant):** seq 48.1/47.9/46.8 tok/s at 112/416/1460 GB H2D per forward (linear in T); grouped 163.1/173.9/180.3 at 8.1/14.3/14.9 GB (bounded by active experts per layer, NOT tokens) = **3.4x/3.6x/3.9x pp, H2D 14x/29x/98x less**. nsys (cap64, T=501): seq 113.4 GB H2D / 4.37s memcpy time / 226876 copies vs grouped 10.2 GB / 0.48s / 33385 — the H2D critical-path collapse is the whole story; grouped is compute-bound again even at 0.6% hit-rate.
- **Expert order (measured, now DEFAULT desc):** processing experts by descending m_e admits the hot experts to the SLRU before the small tail pollutes it → residency converges in ONE forward: auto-cache T=501 first-forward 126.9→169.9 (1.34x), cap512 119.6→160.8 (and kills a rep-to-rep bimodal); wash (<2%) at cap64 and long prompts. (The `=id` restore seam was removed in the 2026-07-08 flag audit — descending is the only order.) Slot scheme keeps byte-identity under ANY order.
- **Disk tier (BW24_SPILL_DISK, 31481/31488 blocks mmap'd):** argmax 1178 MATCH, grouped cap64 163.6 tok/s == pinned-tier speed (page-cache-warm; cold-NVMe untested).
- **Gates (all on the NEW desc default):** run-gen argmax 1178==1178 grouped+seq at auto AND cap64; BW24_MOE_GATE byte-identity PASS at auto AND cap64 (staging live); kernel-check ALL GREEN; 9B run-gen 268==268 + run-spec K=2 PASS (133.9 tok/s, 66.7% accept); 27B run-spec K=3 PASS (78.2 tok/s, 83.6% accept).
- **Decode untouched** (t=1 falls through to sequential). Remaining MoE levers: async copy-stream prefetch of the NEXT expert group during the current GEMM (moe_cache.rs §C.2 infra wired, `prefetch_active` barrier ready — at cap64 grouped the 0.48s H2D is serial with compute, worth ~15% pp); MiniMax-M3 REAP as the true doesn't-fit model.

## Q8 TRUNK-FUSION SCOREBOARD — DENSE-PROJECTION LAUNCH-FUSION SHIPPED: 112.9 → 116.1 tok/s (+3.4%) (2026-07-05, g7e kernel lane)
**The launch-arc's remaining q8_0 wall (qmatvec_q8_0_mmvq 64k launches/128-tok window = 250/token = 18.3% of GPU time in 2.4-14us kernels) attacked with horizontal fusion (JSONL `Q8 TRUNK-FUSION` row).** Call-site map (nsys grid-shape decomposition, verified vs GGUF dtypes — ALL 35B trunk dense projections are Q8_0, experts IQ3_S/IQ4_XS, lm_head Q6_K): per token = 30x(wqkv 8192 + wqkv_gate 4096) linear + 10x(wq 8192 + wk 512 + wv 512) full-attn + 30x ssm_out + 40x(gate/up/down shexp) + 10x wo. Fix = `qmatvec_q8_0_mmvq_fused2/fused3`: the dual-mr2 recipe with the same-out_f restriction LIFTED via a block-offset split (blocks [0,nb0) → tensor 0, rest → tensor 1/2; per (tensor,row) the body is `q8_0_mmvq_row1` = qmatvec_q8_0_mmvq VERBATIM → bit-identical). Fused sites: wqkv+wqkv_gate (both all_fast and the 35B else-arm where F32 beta/alpha blocked all_fast — that arm also folds the two redundant quantizes), wq+wk+wv triple, shexp gate+up at t=1 (sequential AND _dev MoE paths), 9B q8_0 ssm_beta/alpha fused2 twin of the NVFP4 dual. Spec verify T=1 mirrors EVERY site (decode==verify dispatch identity, FP-order lesson #8). Unfusable singles remain by data dependence (ssm_out/down_shexp/wo consume mid-layer activations).
- **Numbers (free clocks, N=3 interleaved, env law + BW24_MOE_CACHE=1): tg128 116.1 vs 112.3 off (+3.4%); tg512 115.9 vs 112.3 (+3.2%). nsys: q8_0 m=1 launches 64k→20.5k/window (+17.9k fused2 +2.6k fused3 = -160 launches/token), quantize_q8_1 56.6k→38.7k, cuLaunchKernel 269.6k→228.6k.**
- **27B/9B p3 spec battery: NEUTRAL as predicted** (27B p3 K=3 124.3 tok/s both arms — trunk is NVFP4/k-quant, zero q8_0 m=1 work; 9B ~204 both arms, only the tiny beta/alpha pair rides fused2). The dense-model trunk projection path shares the CODE but not the DTYPE — no cross-model win, none expected.
- Gates: kernel-check Q8-FUSED2 (8192/4096 + 512/512) + Q8-FUSED3 (8192/512/512) rel=0.00e0 bits=true, ALL GREEN × {35B, 9B, 27B}; 35B argmax 1178 t=1+t=4 with streams IDENTICAL on/off; 9B 268 / 27B 1178; run-spec K=1..8 PASS × {35B, 9B} + 27B real-p3 K=3 + 9B text-p3 K=2; BW24_MOE_GATE clean. Seam: `BW24_Q8_DUAL=0`.
- **Remaining 35B decode wall: MoE _dev pair 43.3+36.1us = 43% of GPU (m=1-matvec-bound — the k-quant r2 recipe may apply), fa_decode_vec_q 135us = 11%, then the ~80 unfusable q8_0 singles (5.2%).**

## DEV_Q8 MULTIROW/OCCUPANCY SCOREBOARD — down8 IDLE-LANE FIX SHIPPED: 129.9 → 134.8 tok/s (+3.7%) (2026-07-05, g7e kernel lane)
**The resident-experts dp4a pair (43% of decode GPU) attacked with launch-geometry twins — bit-identical outputs, only grid/block changed (JSONL `devq8-multirow-occupancy`).** 10 variants built + measured, all behind seams (`BW24_MOE_DEVQ8_GU` / `BW24_MOE_DEVQ8_DOWN` / `BW24_MOE_DEVQ8_WPB`), every one argmax-1178 + 64-tok-stream IDENTICAL (t=1 AND t=4).
- **WINNER, DEFAULT ON: `moe_down8_fma_dev_q8_w8h2`** (auto when in_f==512 & n_used<=8; `BW24_MOE_DEVQ8_DOWN=0` restores base). Two stacked ideas: (a) slot-parallel block (32,8) — warp j computes slot j's dot, partials via smem, warp 0 replays the slot-ordered __fmaf_rn chain (8 serial FMAs, FP order verbatim); (b) half-warp dual-row — down's nsb=16 left lanes 16..31 IDLE in every warp of the base kernel; now lanes 0..15 = row o, 16..31 = row o+1, row-B partials __shfl_down 16 into the base lane layout, then the SAME masked 32-lane tree → bit-identical per row. Kernel 28.3→20.3us (-28%), share 22.5→17.3%; tg128 129.9→134.8 (N=3, free clocks, env law).
- **gate_up: ALL NEGATIVE OR FLAT — base kernel stays.** RPW multirow r1/r2/r4 x WPB 1/2/4/8 (128-130, r4=-5%), s2 gate/up warp-split (129.4), s2z (128.9), gs4 4-warp g-split (129.1), u64 ILP unroll (129.8), j8 slot-packed blocks (133.9 — 8x fewer blocks and still -0.6 vs w8h2 default). down w8r1/2/4 full-warp slot-parallel (129.5-131.0), h2 alone (132.8), w8h2r2 mr2-stack (134.5) all below w8h2.
- **LESSON: "poor occupancy" was the wrong theory for gate_up — the warps are latency-bound individually, and neither fewer (multirow) nor more (dot-split) warps helps; the real hole was down8's DEAD HALF-WARP (nsb=16 vs 32 lanes) + its serial 8-slot loop.** nsb=64 gate_up has no idle-resource hole → geometry-inert.
- Gates: kernel-check ALL GREEN; 35B argmax 1178 t=1 + 198 t=4, streams identical vs pre-arc for all 10 variants AND the new default; run-spec SELF-CONSISTENCY PASS x {35B, 9B, 27B-NVFP4} K=1..8. NOTE: `BW24_MOE_GATE` t=4 grouped-vs-seq mismatch (maxdiff 3.4e-4) PRE-EXISTS at clean b91d5072 (verified via stash) — the dev_q8 commit's sequential-q8 vs grouped-f32 FP-order shift, NOT this arc (sequential path + expert_dot_g bodies untouched here).
- **Remaining 35B decode wall (post-arc nsys): gate_up 22.0us = 18.8%, down8_w8h2 20.3us = 17.3% (the pair now ~36%), fa_decode_vec_q 134us = 10.4%, fused2 8.7%. Next MoE lever must cut the pair's DRAM/latency itself (e.g. gate+up row merge with act-quant reuse, or L2-resident expert tiling), not its geometry.**

## MoE DECODE SCOREBOARD — LAUNCH-STRUCTURE ARC SHIPPED: 74.9 → 112.9 tok/s (+50.7%) (2026-07-05, g7e launch lane)
**The predecessor's measured wall (64% of decode in launch/bookkeeping, not kernels) attacked in 3 stages, each interleaved-A/B'd + gated + committed on main. 35B-A3B IQ4_XS tg128@ctx128, env law + BW24_MOE_CACHE=1, free clocks, N=3.**
**STAGE 1 — router DtoH elision (185f0a72, +1.4% on top of stage 2):** ROOT CAUSE of the old "fused router 2% WORSE" verdict found — the `=1` arm paid TWO full stream syncs (dtoh_i32(sel) + dtoh(w), each clone_dtoh+synchronize) + 2 alloc_zeros memsets per MoE layer, vs ONE sync for the host route's single 1KB logits dtoh. New `moe_router_topk_host`: uninit outputs, both DtoH issued ASYNC into a persistent PINNED host stage (flags=0 CACHEABLE — cudarc's `alloc_pinned` is WRITECOMBINED, pathological for host reads), then ONE synchronize. **Fused router is now DEFAULT-ON at all t** (`BW24_FUSED_ROUTER=0` rollback) so decode and spec-verify route identically (FP-order lesson #8 class). Selection exact (kernel-check tie gate); w rel 2.2e-7 (3 ULP, expf-vs-libm).
**STAGE 2 — memset/alloc churn (185f0a72, 74.9 → 80.9 alone):** moe_out alloc_zeros→uninit when gdec can fire (moe_down8_fma fully overwrites its row; sequential-fallback tokens lazily zero ONLY their row); the per-layer scratch_g/u/d trio (3× ~1MB alloc_zeros+memset+free per MoE layer per token — DEAD code under BW24_MOE_CACHE) now lazy+uninit; act/sa/g and router sel/w zeros→uninit (full-overwrite producers). nsys: memsets 149.5k→60.6k/window.
**STAGE 3 — zero-DtoH device-dispatch MoE (ecba3bdf, 82.0 → 112.9, the big one):** design option (c) indirect-kernel from the arc brief, no graph needed. When a layer's expert blocks are ALL SLRU-resident: router sel/w STAY ON DEVICE; `moe_gate_up_silu8_dev`/`moe_down8_fma_dev` twins read expert ids (+w) from the router's device output and weight pointers from a per-layer device table [3, n_expert] of fixed slot addresses (bit-identical FP chains to the gdec param twins — only the pointer/id SOURCE changes). MoeSlotCache grew per-layer residency counts + lazily-uploaded device rows (invalidated on eviction) + **one-shot `prewarm_layer`** (force-admits while FREE slots cover the layer, never evicts — 15.2GB one-time first-forward H2D on 96GB, hit-rate 100% from token 0; spill rigs skip automatically; `BW24_MOE_PREWARM=0` organic). Kills the per-layer router DtoH + stream sync (~40 host round-trips/token): nsys cuMemcpyDtoHAsync 20.7k→256 calls, cuLaunchKernel 408k→269k. **pp512 also rides it: 244→338 (+38%)** (prefill routes through the same path at t>1). Seams: `BW24_MOE_DEV=0`; `BW24_FUSED_ROUTER=0` disables dev too.
- **Numbers: tg128 74.9 → 82.0 (stages 1+2) → 112.9 (stage 3); tg512 112.3, tg1024 113.8; vs llama.cpp b9863 same box (tg128 235.4): 0.32x → 0.48x.**
- **Graph replay RE-MEASURED post-stage-3: NEGATIVE (JSONL `moe-graph-decode-negative-post-stage3`), 0.931x tg128 / 0.933x tg512** — same verdict class as graph-spec stage-3: once launches are structurally removed, capture/recapture overhead (3 warmup runs per t_kv bucket, recapture every 64 tokens) exceeds what batching reclaims. Do not re-open unless decode becomes launch-bound again.
- Gates (every stage): 35B run-gen argmax 1178 MATCH (t=1 [55] AND t=4 [9419,11,1814,0] prompts; 64-tok greedy stream IDENTICAL dev on/off AND fused on/off); 9B 268 / 27B 1178 MATCH; kernel-check ALL GREEN; run-spec K=1..8 self-consistency PASS × {35B, 9B, 27B} with accept counts unchanged.
- **Remaining 35B decode wall is now KERNEL time, not launches:** qmatvec_q8_0_mmvq 64k launches/window (the 40 trunk layers' dense attn/GDN m=1 projections — 18% of GPU time in 2.4us launches, a batching/fusion target), fa_decode_vec_q 135us/call (11%), and the _dev MoE pair 41%. Next levers by size: (1) fuse/batch the per-layer dense projection chains (same indirection trick or megakernel), (2) fa_decode split tuning at short ctx, (3) MoE _dev kernel efficiency (43us gate+up for 8 experts = m=1 matvec bound, the k-quant r2 recipe may apply).

## MoE DECODE-CACHE SCOREBOARD — ASYNC-PREFETCH ARC CLOSED NEGATIVE + GEMMA4 GAP LIST (2026-07-05, g7e lane)
**Arc 1 (async prefetch on decode misses): NEGATIVE-NOT-WORTH-IT, measured before building (JSONL `moe-async-prefetch-negative`).** 35B-A3B IQ4_XS decode tg128@ctx128, env law + BW24_MOE_CACHE=1, N=3: **74.9 tok/s** (up from the 69.6 record — drift from main-sync kernel work, not this arc); tg512 80.4, tg1024 83.4. nsys decode-window profile: ALL H2D staging = 70.5ms of a 1944ms window = **3.6% of decode wall**, and warmup-concentrated — last-quarter H2D is 1.4%. Steady-state hit-rate self-extinguishes misses on 96GB: 90.8%@64tok → 96.3%@512tok (16.6 MB/tok vs 454 stage-1). Perfect copy_stream overlap buys ≤3.6% for the store-before-evict hazard set — below the bar. The REAL 35B decode wall per the same window: cuLaunchKernel 573ms/185k calls + router DtoH 306ms/5.2k + memset/alloc/free ~370ms + kernel time — launch structure, not PCIe. `prefetch_active` seam stays wired for the SPILL rigs where misses never extinguish (24GB local; revisit there, not here). Side A/B: BW24_FUSED_ROUTER=1 is 2% WORSE at t=1 (73.4 vs 74.9) — stays default-off. llama.cpp b9863 same box: 35B tg128 235.4, pp512 8527 — bw24 decode 0.32x, gap = launch/bookkeeping structure (185k launches per 128 tokens; llama runs graphs).
**Arc 2 (gemma4-26b-a4b MoE bench): loader can't take it — EXACT 9-item port-gap list recorded (JSONL `gemma4-26b-moe-gap-list`), the MiniMax-plan-style deliverable.** Full metadata+tensor audit of gemma4-26b-a4b-q4km.gguf (15.63GiB, 30L, 128ex/top-8) found 3 NEW hard gaps beyond the 2026-07-04 STAGE-4 six: (7) GELU everywhere — bw24 only has silu; (8) Q5_0 dequant missing (blk.3.ffn_down{,_exps}) — enum exists, no dequant path, no kernel; (9) tokenizer.ggml.model=gemma4 sentencepiece — bw24-tokenizer hard-errors on non-gpt2. Plus the sharpened attention picture: per-layer head_count_kv [8×5,2]×6 with 25 SWA layers (window 1024, hd 256, rope 10k) / 5 global layers (blk 5,11,17,23,29: hd 512, kv 2, rope 1M + rope_freqs), and the global layers are V-LESS — llama gemma4.cpp:247 uses Vcur=Kcur + rms_norm(V). Router prologue: logits = mm(gate_inp, rms_norm(attn_out)/sqrt(n_embd) * gate_inp_s). Block wiring: parallel shared-MLP + MoE both from attn_out, summed, dual post-norms, layer_output_scale, final_logit_softcapping 30. llama reference numbers banked for the eventual vs-dense comparison: gemma4 pp512 11234 / tg128 224.3. STAGE-4 alternative stands: a qwen3moe/olmoe-shaped second MoE validates shape breadth with zero port.

## C1 SCOREBOARD — K-QUANT BATCHED r2 PORT SHIPPED (2026-07-04, closes the "k-quant batched variants ~19% of 9B verify" open item)
**The NVFP4 batched-latency recipe ported to q4_K/q5_K/q6_K `_b2/_b4` (JSONL `kquant-batched-r2-port`).** ncu-first on the DRAM-cold msweep (9B real shapes, m=4): q4_K long_scoreboard 19.6/issue DRAM 47.7%, q5_K 16.4/38.2% = memory-LATENCY (NVFP4 pre-fix class); q6_K lm_head the outlier at DRAM 90-91% = wall-bound. Shipped `_r2` twins (two rows/warp) + `_r2w8` (seam-only) + per-dtype wave-aware auto: **q4_K r2 whenever blocks>=4*SMs** (incl. the straggler window where NVFP4 r2 lost — qkv -15%, ffn_down in_f=12288 -35%, all measured shapes -11..-35% b4 / -6..-22% b2), **q5_K/q6_K r2 only at waves>=2** (the 248320-row lm_heads: q6_K -8% DESPITE 90% DRAM — the halved grid's deeper MLP raises achieved DRAM; q5_K 27B lm_head -2.1% b4 / -2.9% b2; mid q5_K shapes base-or-flat — the 5/6-bit two-stream unpack makes r2's staging pricier). **r2w8 NEVER in auto for k-quants** — the 72->64 reg squeeze adds a stack spill and loses on every measured cell incl. wave-crossing lm_heads (mirror of the NVFP4 positive: residency crossing only pays spill-free). NO pf port (a k-quant group stages 10+ words vs NVFP4's 5), NO q8_0 r2 (only real batched shapes are out_f=32 ssm_alpha/beta), NO q6_K split-plane repack (recipe step 3 measured before building: lm_head is bandwidth-bound, the A6 latency prerequisite is absent). New seam `BW24_KQ_BV=base|r2|r2w8` (k-quant-only force, leaves NVFP4 dispatch alone; `BW24_MMVQ_BV` still maps globally).
- **E2E (free clocks, N=3 interleaved base-vs-auto medians, NGEN=256): 9B 182.8/145.5/122.3 vs 179.1/145.1/120.0 (+2.1/+0.3/+2.0%); 27B noise-level (+0.1/+0.0/+0.5%) — EXPECTED: the 27B round has ~no k-quant batched work (trunk NVFP4 already r2/r2w8, draft lm_head is m=1).**
- Gates: kernel-check ALL GREEN x {9B, 27B} x {auto, KQ_BV=r2, KQ_BV=r2w8 forced} (k-quant BATCHED rel=0.00e0); msweep bit-exact-bad=0 all cells; run-gen 82==82 both models; verify-probe byte-identical to clean-main baseline (3.815e-6, argmax exact T=1/2/3); run-spec K=1..8 PASS x {9B synth, 9B text p2, 27B real p3} on the final tree.

## C1 SCOREBOARD — A5 cp.async NEGATIVE / A6 SPLIT-PLANE REPACK SHIPPED (2026-07-04, SOTA-ADOPTION items 5+6)
**A5 (cp.async smem weight ring for batched MMVQ): MEASURED NEGATIVE — wall-ledger it.** Mandated ncu-first gate passed (residual stall post-pf/r2w8 is STILL memory latency: long_scoreboard 8.8-16.4/issue vs FMA-dep wait <=1.9, DRAM 48-69%), but the 3-4-stage cp.async.cg ring LOSES 15-32% on every real 27B b4 shape (DRAM-cold msweep): occupancy fine (70% > r2w8's 65%), DRAM fell 61.5->49.9%, stall stayed long_scoreboard = the wait itself. Mechanism: at m<=4 there is NO cross-thread weight reuse to amortize smem staging; the register variants' L1-amplified 2-row MLP is strictly better. Kernels kept behind `BW24_MMVQ_BV=ca|car2` (pfr2 precedent). Do not retry without a new mechanism (TMA multicast / m>=8 regime). JSONL `mmvq-b4-cpasync-ring-NEGATIVE`.
**A6 (Marlin-style walk-order repack): SHIPPED, default ON (`BW24_RP=0` rollback).** All NVFP4 matmul weights repack at load into [quant plane out_f x in_f/64 x 32B][scale plane x 4B] (model.rs `repack_nvfp4_split`, host-side pre-htod, zero VRAM spike, per-tensor `rp` flag): a lane's per-group read becomes ONE aligned LDG.128 + a dense 4B scale word instead of 5 scattered 4B LDGs at 36B stride. ONE copy serves every consumer — batched b2/b4 (`rp/rpr2/rpr2w8` under the same wave-aware rule), m=1 mmvq/mr2/dual `_rp` twins, dp4a `_rp`, Stage-A tag 9, prefill int8 GEMM kernel2 (RP=pre-decode: the cp.async raw ring that lost -3% on GGUF's 36B straggle WINS on the aligned layout). Embeddings (gather-only) stay GGUF. NOT ported (fall to rp int8 GEMM, recorded): W4A4 MMQ `BW24_MMQ` + `BW24_FP4` opt-ins. (2026-07-05: the W4A8 MMQ IS rp-ported — see STAGE 2b — and is now the default prefill.)
- **E2E (interleaved rp on/off, N=3 medians, free clocks, NGEN=256): 27B 94.2/87.4/69.1 tok/s (+2.2/+2.1/+2.0%); 9B 180.3/148.0/119.2 (+0.9/+0.6/+0.7%). PRIME/TTFT also wins: 27B -6.1..-8.1% (p3 8.75->8.22s), 9B -5.5..-7.0%. rp wins every interleaved pair (27B 15/15, 9B 9/9).** vs llama serve cross-session: 27B ~0.98x p2 / ~0.94x p3 (llama re-run needed for an honest image).
- Kernel-level (DRAM-cold msweep, 27B m=4): b4 trunk sum 366.0 -> 352.5 us/layer (-3.7%, wins all 5 shapes: ffn_down -6.3%, gate/up -3.7%, attn_gate -2.7%, ssm_out -2.0%, qkv -1.1%); m=1 -2.2..-4.5%.
- Gates: kernel-check ALL GREEN x {9B, 27B} incl the new RP battery (repack-roundtrip 0 bytes + bit-bad=0 for every consumer kernel: MMVQ m=1/2, BATCHED m=2/3/4, DP4A m=1/5, GEMM T=128, STAGE-A); run-gen 82==82 rp on AND off; verify-probe 0.000e0; run-spec K={1,2,3,4,6,8} PASS x {9B synth, 9B text, 27B real p2, 27B real p3} on the final tree. JSONL `mmvq-b4-splitplane-repack-prototype` + `nvfp4-splitplane-repack-shipped`.
- FP-order lesson (A5+A6 pair, for the corpus): cp.async staging pays where cross-thread smem reuse exists (GEMM tiles) and loses where it doesn't (m<=4 matvec); layout alignment flips the sign of the SAME staging mechanism.

## C3 SCOREBOARD — A4 CHUNKED WY GDN PREFILL SHIPPED, default ON (2026-07-04, SOTA-ADOPTION item A4; `BW24_GDN_CHUNKED=0` rollback, `BW24_GDN_CHUNK=32` default)
**gdn_scan_s128's sequential per-token recurrence replaced FOR PREFILL ONLY by the chunk-parallel WY/blockwise-inverse matmul form (flashinfer/fla chunked delta-rule; full derivation in the cu/hybrid.cu K1-K5 header).** 5 kernels: cumgate cumsum -> A/P chunk matrices (2x2 register tiles, P upper zero-filled) -> (I+A)^{-1} both-RHS forward substitution (thread-private column history, register templates C=32/64) -> the ONLY chunk-sequential kernel (Y=U-WS + rank-C state update, smem col-split state, chunk-start snapshots move the o_inter dot off the serial path) -> output assembly (4x4 register tiles). Dispatch `gdn_scan_prefill` in linear_attn + linear_attn_prime ONLY — decode and the spec verify still call gdn_scan_s128 directly (decode==verify dispatch identity untouched); prime rides prime_cache so the win lands in TTFT.
- **Kernel (gdn-bench, clock-locked 1852): T=512 759->416us (1.82x), T=6257 10.69->4.93ms (2.17x — grows with T; the scan is O(T) serial).** Chunk sweep: C=32 wins EVERY interleaved pair (chunk matrices cost O(T*C); the state pass is C-flat); C=64 close; C=128 NEGATIVE (gated to generic fallbacks).
- **E2E (free clocks, N=3 interleaved off/on medians): pp512 9B 201.7->192.9ms (+4.6%, 3/3), 27B 650->629ms (+3.3%, 3/3). PRIME/TTFT: 9B p2 -6.9% / p3 2.457->2.334s (-5.0%); 27B p2 -2.9% / p3 8.098->7.832s (-3.3%) — wins 12/12.** Spec-optima gen tok/s moves ±4% in OPPOSITE directions per model (9B +4.0, 27B -3.8) = acceptance on drifted sequences (content), not engine cost.
- **EXACTNESS (the A4 risk class — chunked FP accumulation order != sequential, NOT bit-identical by design):** kernel-check ALL GREEN x {9B, 27B} incl new f64-truth gates (chunked out <=4.2e-5 / state <=1.1e-4 rel vs truth; sequential itself is 3.9e-6/1.2e-5 — the (I+A)^{-1} conditioning noise class); real-weights per-layer diff (9B T=512, all 24 linear layers, BW24_GDN_DIFF=1 dual-run mode): out max_rel <=4.5e-5, state ~2e-4 typical / 1.1e-3 worst (layer 0). run-gen argmax 82==82 on/off x both models (24/24). run-spec K={1,2,3,4,6,8} PASS x {9B synth, 9B text, 27B p2, 27B p3} chunked ON — plain and spec prime through the SAME path, exact by construction, verified.
- **REPORTED PROMINENTLY (user judges): e2e greedy token agreement off-vs-on, 3 prompts x 2 models: first-16 IDENTICAL 6/6, but full-256 forks on 5/6 (index 47/61/110/118/125; 27B p1 fully identical).** Prefill-state FP drift flips one tight-margin argmax and the continuations legitimately diverge — same accepted class as the batched-prime precedent (which forked 9B p2 at index 11), just measured deeper here. FP-order lesson #9: prefill-side reorder does NOT break spec exactness (verified) but forks greedy text at depth ~50-125.
- Wall-ledger (JSONL `gdn-chunked-wy-prefill-prototype` notes): K4-state optimization ladder — smem tiles + accumulator-innermost register tiling = the win recipe (K5 3.1x, K3 3.6x); MEASURED LOSERS: step-B 4x4 reg tiles (reg pressure), NSPLIT=8, cp.async double-buffered staging pipeline, 512-thread blocks — the serial chain's walls are the U/Y global round-trips, occupancy capped by smem regardless of split. Next lever if reopened: per-chunk grid-wide launches / cooperative grid.sync, or cublasLt strided-batched f32 for the parallel matrices.

## SPEC SCOREBOARD — HIDDEN-PAIRING FIX LANDED (2026-07-04, the 27B acceptance lever CASHED)
**ROOT CAUSE of the 0.58-vs-0.75 acceptance gap: hidden-PAIRING convention, found by reading the reference draft-mtp impl (speculative.cpp "shift the tgt embeddings to the right by one position").** The NextN head is trained on rows (token x_p, trunk hidden h_{p-1}); bw24 paired SAME-ROW (x_p, h_p) in mtp_kv_fill and seeded chain step 0 via the pseudo-hidden pass. Fix (spec.rs, no kernel change): fill hiddens shifted by one (row 0 zeros at prompt / carried `fill_prev` in-loop) + step-0 seed = predecessor's TRUE verify hidden directly — the per-round pseudo-seed pass (and its head-less graph) is DELETED on the default path. Default flipped; the legacy `BW24_SPEC_HSAME` seam + pseudo-seed passes were removed in the 2026-07-08 flag audit. `BW24_SPEC_STATS=1` = per-slot accept + draft-len histograms.
- **Metric normalization vs reference: definitions IDENTICAL** (accepted/drafted, p-min-stopped chain, sub-threshold token discarded uncounted — verified in server-context.cpp + speculative.cpp). The gap was real. Caveat: their p-min gates on TOP-10-renormalized p (laxer than our full-softmax p at the same value).
- **Acceptance (real prompts, deterministic): 27B K3/pm0.2/trim p2 0.569→0.731, p3 0.445→0.614; 27B K1/pm0 p2 0.783→0.855, p3 0.667→0.816; 9B K2/pm0.3 p2 0.604→0.749, p3 0.498→0.594; 9B synth K2/3/4 0.73/0.62/0.47→0.93/0.83/0.75** (also fixes the persistent-KV record's synth K≥3 regression). Per-slot p2: [.755 .536 .395]→[.843 .725 .613] — the "late slots collapse" was seed corruption, not chain depth.
- **E2E (interleaved old-vs-new, N=3 medians, free clocks, NGEN=256): 27B 90.0/83.4/66.0 tok/s (+13.5/+18.4/+25.0% vs 79.3/70.4/52.8); 9B 174.9/144.1/115.4 (+10.7/+21.1/+14.2%).** vs llama serve cfg same prompts: 27B p2 0.94x (88.8), p3 0.90x (73.4) — was 0.79x/0.72x. Acceptance vs ref: p2 0.731 vs 0.826, p3 0.614 vs 0.640.
- **Hypothesis battery (JSONL `spec-hidden-pairing-prev`):** H1 pseudo-vs-true seed DISSOLVED (true seed now free — it IS the last verify column); H3 p-min sweep post-fix: 27B K=3 pm0.2 STAYS optimal (pm≤0.15 no gain, pm0.3 flat), 9B K=2 pm0.3 stands (K=3 pm0.3 ties); H4 FR-Spec trim KEEP (acc unchanged, +10.8%/+7.5% tok/s); H5 external Q4 draft file NEGATIVE (acc .703 < native .731, tok/s 74.9 < 84.4 — the ref's 0.75 rode the pairing, not the draft file); true-hidden refresh still positive (.685/79.9 without vs .731/84.4 with) — stays default.
- Gates: kernel-check ALL GREEN, run-gen 82==82, run-spec K=1..8 PASS x {9B synth, 9B text, 27B real}, verify-probe 0.000e0, all 5 seams (KVLOCAL/HSAME/NOREFRESH/REPLAY/NOGRAPH) exact.
- **Remaining acceptance gap (p2 .731 vs .826) is NOT the next lever** — plausibly NVFP4 trunk-hidden quality + their laxer p-min chain mix (their mean len 3.48 vs our 3.10). Next 27B spec levers by cost: draft/verify kernel time (b4 MMVQ tranche 2: host-fused tiny GDN projections, k-quant batched variants), not acceptance.

## SPEC SCOREBOARD — GRAPH-GRADE SPEC LANDED (2026-07-03/04 graph-spec session; all numbers same-session interleaved, clock-locked 1860 w/ sag to ~1725)
**9B: plain eager 90.4 → spec K=3 pmin=0.2: 130.6 tok/s (1.44x plain eager, 1.29x over same-session graph-plain ~101) — SPEC IS THE OUTRIGHT DAILY WINNER. All exact: 9B synthetic + 9B TEXT + 27B real-prompt, K=1..8.**

**STAGE 4 — PERSISTENT MTP DRAFT KV LANDED (266620f, 2026-07-03, free clocks, interleaved x3): the acceptance lever paid.** The NextN scratch KV no longer resets per round — slot p holds the MTP block's K/V for committed token p (the reference engine's mtp_update design), so the draft chain attends over FULL history. Mechanics: scratch cap=max_ctx allocated once, len via len_d ⇒ the ONE round-0 graph capture serves every t_kv (zero recaptures; fa_decode_dc bucket_max=cap, empty splits self-skip); eager draft uses the SAME dc launcher ⇒ parity by construction (verified: accept/draft counts bit-identical NOGRAPH vs graph, both models, every K). Entry sources: chain appends (accepted positions KEPT — hidden chain-approximate, reference-endorsed), `mtp_kv_fill` K/V-only batched pass (no wq/attn/FFN/lm_head) for prompt positions (from prime-collected exact trunk hiddens) + the last-draft slot on full accept (from vh_seed). Draft-side rollback = round-start set_len truncation. (The legacy round-local seam `BW24_SPEC_KVLOCAL` — verified at the time to reproduce the old 58% 27B K=1 — was removed in the 2026-07-08 flag audit: -35 accept pts, incompatible with sessions.)
- **27B real-prompt: acceptance K=1 58.0→70.7%, K=2 57.4→63.4%, K=3 47.1→56.3%; best spec 42.6 (K=2, 1.03x) → 46.8 tok/s (K=3, 1.13x) — new 27B optimum is K=3.** (Reference serve config ≈75% with the same head — most of the gap closed.)
- **9B TEXT: K=1 78.9→85.5%, K=2 61.6→75.2%; best 110.3 → 126.4 tok/s (K=2, 1.14x)** at free clocks.
- **9B SYNTHETIC seq: REGRESSES at K≥3 (154.3→140.2 tok/s; accepted counts EQUAL, drafted balloons — full-context confidence keeps p-min from firing on the toy distribution).** Real prompts are the serving verdict (tune-data README rule); synthetic bench comparisons must compare persistent-to-persistent (the KVLOCAL seam is gone).
- Gates: kernel-check ALL GREEN, run-gen 82==82, run-spec K=1..8 PASS on all 3 configs, parity identical. JSONL row `commit:266620f`.
- Next acceptance levers (now 8-19pts below the K≥2 ceiling): refresh ACCEPTED entries from verify TRUE hiddens (currently chain-approximate), p-min re-tune for full-context confidence (0.2 was tuned for windowless drafts).

Session chain (each stage interleaved-A/B'd + gated + committed):
1. **FP-order lesson #8 fix (75b3e6b, exactness):** the verify's decode-exact dispatch must mirror eager PER LAYER — uses_q8_1_fast is per-tensor and the 9B GGUF stores ssm_beta/ssm_alpha as Float on layers 1/2/4, so eager takes the UNFUSED 256-thread rms_norm there while verify ran the 1024-thread norm; 1 ULP at layer 2 amplified through GDN to 2.3e-1 logits and flipped the 9B TEXT prompt at K=1..8 (pre-existing on main — synthetic + 27B had passed on margin luck). Also: batched linear verify now requires ALL-fast projections (matmul_decode_exact routes Float to cuBLAS GEMM at m=t vs eager's per-token GEMV). New **verify-probe** bin = the permanent gate (eager-vs-verify logits at T=1/2/3 + per-layer residual bisect + pair checks + fastness map); after fix maxdiff 0.000e0 everywhere.
2. **Stage 1 — device-argmax verify (23fbf9f):** verify logits stay ON DEVICE; per-column argmax (argmax_gate-validated kernels via column views) + one [T] u32 read replaces the 1-4MB dtoh + T host argmaxes per round; last_logits Vec → last_pred u32. 106.3 → 110.6 (+4.0%).
3. **Stage 2 — graph-captured draft chain (6502142):** the fixed-shape T=1 MTP forward captured ONCE, replayed per draft step — ONE bucket serves every draft index (scratch t_kv ≤ k+1 < 96 pins fa to scalar/n_splits=1; append/fa _dc twins read len_d). Token/pos/seed chain themselves in-graph; host reads 4B tok (+4B p-min) between replays. Bonus-fold pseudo-seed = one more replay. 110.5 → 130.6 (+18.2%); draft parity = acceptance counts BIT-IDENTICAL to eager at every K on 9B AND 27B. Seams: BW24_SPEC_NOGRAPH=1; auto-fallback for MoE/capture-failure.
**27B real-prompt: K=2 = 37.6 vs plain 36.8 = 1.02x — first 27B spec config above 1.0x** (still acceptance-bound at 57%; FR-Spec trim = the next 27B lever). → SUPERSEDED by Stage 4 persistent draft KV: 46.8 tok/s K=3 = 1.13x at 71%/63%/56% acceptance (K=1/2/3).
**Stage 3 (graph-captured verify) = MEASURED NEGATIVE as specified, do not build without a new unlock:** savings ceiling = per-trunk-pass launch overhead (11.06 − 10.41 ≈ 0.65-1.15ms) × ~1 verify pass/round ≈ +3-5% e2e; capture cost = 3 REAL verify passes (~33ms) per bucket, and the bucket key must be (T, per-row fa n_splits vector) because R1 forbids padded splits (measured argmax flips) — with p-min T varies 2..k+1 and n_splits churns every 64 ctx tokens + boundary vectors ⇒ ~10-13 captures per 128 tokens ≈ 330-430ms = 4-10x the savings, at ANY gen length. Unlocks that change the math: (a) cudaGraphExecUpdate (re-parameterize a baked graph instead of re-capturing — not exposed in cudarc 0.19), (b) make fa n_splits t_kv-INDEPENDENT in BOTH decode and verify (the exactness pair moves together; kills rebucketing AND adds CTAs at short ctx where fa-decode is grid-starved) — decode-wide policy change, needs its own full battery. **UPDATE 2026-07-03: unlock (b) effectively ARRIVED via 642582a (fa rows derives per-row splits in-kernel from fixed 64-key chunks) and the question was RE-MEASURED — still negative, ceiling shrank to 2.8-3.8% because stages 1-2 removed the launches the graph would have reclaimed. See wall ledger + JSONL `graph-spec-stage3-remeasure`. Do not re-open on launch-overhead grounds; only if the verify becomes launch-bound again (much faster kernels or different T regime).**
**ENV LAW RETIRED 2026-07-08: BW24_FAST/BW24_MMVQ/BW24_MOE_CACHE are default-ON** (=0 reverts; BW24_GEMM had zero reads, BW24_FA_VEC was vestigial — the real FA control is BW24_NO_FA_VEC). Naked commands run the full fast path; the old silent-slow-path landmine (and the dispatch-split exactness break this line used to warn about) is gone. Historical sections below that say 'env law' describe pre-flip measurements — the numbers stand, only the invocation changed. Flag catalog: docs/FLAGS.md.
sm89/L40S status (ARM 3, subagent-only per user): first hardware run happened — argmax gate PASS but kernel-check/prefill CRASH (illegal address) and decode 0.23x of llama; the sm89 branch compiles but is NOT execution-ready on real Ada hardware. Debug = future subagent work; box artifacts at fpv-train:~/bw24-bench/.

## SPEC SCOREBOARD (2026-07-04, 9B, exact greedy — PRE-TIMING-FIX prints below, ratios valid)
plain 47.1 → K=4 ungated 1.03x → p-min 1.11x → **bonus-fold + pmin 0.2 @K=3: 52.17 tok/s (1.11x), K-curve flattened (K=4/6 hold 1.05/1.06 under pmin 0.3), K=1..8 all exact.**
Bonus-fold trade measured: 1 trunk read/round saved vs ~10pts acceptance (pseudo-hidden draft seed); STRUCTURAL OPTIMUM — a true-hidden fold requires knowing the bonus pre-verify, which IS the pseudo-seed. Recorded in JSONL.
Landed chain: batched linear verify → GPU-argmax draft → persistent snapshots → FR-Spec consumer (BW24_MTP_DRAFT + d2t) → p-min gate (BW24_SPEC_PMIN).
27B: 0.85x best (pmin 0.4) — acceptance-bound (45% native, 31% with the Q4_K draft file whose NextN block loses 15pts). 27B needs a target-quality trimmed draft (NVFP4 block + trimmed head) — the agent's HEAD_ONLY probe failed exactness on shared_head_norm mismatch; producer work.
**NEXT quantified spec levers:** (1) bonus-token fold — every round pays a full T=1 weight read for the bonus (decode_step_h); llama folds it into the next verify batch. (2) 9B FR-Spec trimmed head (draft lm_head=750MB Q6_K, c≈0.13/token → 87% cut; user's gguf-py producer recipe in his llama issue 25187). Both together should push toward the accept-rate-implied ~1.5-2x.

**Ranked levers (updated post-decode-session):**
1. **FA decode port** (llama fattn-vec structure: q8_1 Q + dp4a on raw K bytes, no smem staging) — agent running in background; bw24 FA ~1.3ms/tok vs llama ~0.6ms.
2. **MTP K=4 exactness fix** — debug agent running (worktree); K=1/2 PASS, K=4 diverges on ALL kernel paths (garbage special tokens ~idx 25 => indexing/state bug, not numerics). MTP is the profit lever: K=1 already 0.85x of plain at 81% acceptance.
3. Prefill: close the honest 9% interleaved gap + the clock-sag perf/W gap (bw24 draws more power per unit work — audit which kernels burn ALU needlessly; the MMQ tune-seams sweep is the tool).
4. Web-sweep items FOLDED into ROADMAP.md items 11-15 (DFlash, TCQ, FR-Spec, tensor-split=DEAD, ST-MoE prefetch). DONE.
5. L40S box i-TERMINATED is GONE (terminated, not in any account/region). Arm 2 = sm89 branch compile-mirrors only until a new box is provisioned.

---

## 27B PREFILL/TTFT ARC — LOCAL LANE (Fable, main branch, 2026-07-05)

North-star: close the 27B prefill/TTFT gap (vLLM prefills the same 27B NVFP4 5.6-6.8x faster; local vLLM 27B prefill 4.9k tok/s vs bw24 ~0.8k). Goal 27B TTFT p3 <1s, e2e >100 tok/s.

**STAGE 0 — baseline re-measured (commit 1e5bfab, N=3, free clocks, JSONL `1e5bfab (baseline re-measure...)`):** 27B prime .289/2.419/7.864s p1/p2/p3 (matches HANDOVER ref within 1.1%, no thermal sag). pp-only tok/s FLAT vs T: 810/773/804 @ T=512/1845/6257. nsys p3: qmatvec_gemm_nvfp4_rp (int8 W4A8) = **79.6% of prefill**, fa_prefill 4%, gdn cluster 4.9%, quantize 4.5%. Peak VRAM 9.5GB. First-16 reference tokens captured for all 6 (model,prompt) cells (the arc's agreement gate).
- **CHUNKED-PREFILL SCHEDULING = NO-GO (measured, cheap-stage-first per the plan):** pp tok/s is already flat in T with no long-T degradation, and bw24 prefills one request in one batched pass with no inter-request scheduler. vLLM's chunked-prefill is a *multi-request* scheduling win, not a single-request kernel win. The 6x gap is in the GEMM. Zero-accuracy-risk stage yielded zero — do not build it for single-request TTFT.

**STAGE 1 — vendored NVFP4 W4A4 MMQ A/B = NEGATIVE (accuracy gate), commit 1e5bfab, JSONL `A/B ... vendored llama NVFP4 W4A4 MMQ`:** BW24_MMQ=1 (needs BW24_RP=0; the A6 repack silently bypasses it). 3-arm interleaved, N=3. **Perf is real: 27B pp 2.7-3.0x (810→2414/2295/2202 tok/s), p3 prime 7.86→2.93s; 9B 2.2-2.4x. mul_mat_q_nvfp4 = 43.6% of ARM_B p3 prefill (engaged).** BUT the W4A4 activation quant is the gate-breaker: 27B argmax 82==82 MATCH yet **p3 first-16 FORKS AT INDEX 0** (emits think-token prefix 248046/248045); 9B **argmax FAILS 78!=82** + p1 forks at index 8. kernel-check ALL GREEN + run-spec K=1..8 self-consistent (internally exact) — so it's the historical W4A4 accuracy class (maxdiff ~1.0) reproduced at e2e, NOT a bug. **CUTLASS W4A4 (cu/cutlass_fp4_sm120.cu, the OFF seams) inherits this exact class — do NOT build resident/OTF CUTLASS until the ACTIVATION side changes.** Keep BW24_MMQ W4A4 as an opt-in speed/accuracy tradeoff.

**STAGE 2b — RP TILE LOADER + DEFAULT-FLIP = SHIPPED (2026-07-05, JSONL `w4a8-rp-loader+default-flip`).** The W4A8 MMQ now COEXISTS with the A6 split-plane repack: `load_tiles_nvfp4_w4a8` gained an `is_rp` template arm that reads the split-plane layout (quant plane `W + ib*32`, scale plane `W + out_f*(in_f/64)*32 + ib*4`, flat block index `ib = row*stride + kbx` — the repack copies qs/d bytes verbatim so the flat index serves both planes). PURE ADDRESS REMAP: same LUT dequant, same smem write order, same FP ops in the same order → **bit-identical to the GGUF loader, proven by the new hard `MMQ-W4A8-RP` kernel-check gate (0 mismatched bits over T=16/64/128/512, up to 6.3M f32 values)**. C-ABI `bw24_mmq_nvfp4_w4a8` takes an `rp` flag; dispatch routes per-tensor off the model `rp` flag. **DEFAULT-FLIP taken: the vendored MMQ prefill suite (NVFP4 W4A8 + Q4_K/Q5_K int8-MMA — the exact pair the old `BW24_MMQ_W4A8=1` arm engaged) is now DEFAULT-ON** via `mmq_w4a8_enabled()` in `mmq_supports`; `BW24_MMQ_W4A8=0` = escape hatch to the int8 GEMM; `BW24_MMQ=1` still opts GGUF-layout NVFP4 into W4A4; lib.rs matmul/matmul_pre MMQ gates collapsed into `mmq_supports` (policy in one place). **Perf (N=3 interleaved a/b/c): W4A8+rp pp512 27B 3140 / 9B 8848 (1.91x/1.80x vs int8+rp 1648/4928), 27B prime p1/p2/p3 0.082/0.612/2.083s (2.51x/1.72x/1.54x) — within 1.6% of the rp0 predecessor arm (3160/8901, 0.082/0.602/2.050) — AND rp-on decode fully retained: 27B gen 67.9/61.0/60.1 tok/s == the int8+rp arm, where rp0 pays -1.7..-2.5% (66.2/59.7/58.9).** Gates on the final flipped default (no env): kernel-check ALL GREEN incl MMQ-W4A8 oracle (rel 4.2e-3..7.4e-3) + MMQ-W4A8-RP bit-identity; run-gen argmax 27B 1178==1178 + 9B 268==268 MATCH; first-16 IDENTICAL to the int8 reference on ALL 6 cells (also on the =0 escape hatch — 12/12 arms); run-spec K=1..8 PASS x {9B, 27B p3}. NEXT: FP8-activation CUTLASS at large m (W4A4 CUTLASS stays closed), stream-K for W4A8 (needs the fixup FP-order care), or the SSM-prep/FA-prefill levers from the kernel-diff table.

**STAGE 2 — THE ACCURACY-SAFE RUNG = BUILT + POSITIVE (commit stage2-w4a8-mmq, JSONL `stage2-w4a8-mmq`).** Vendored llama non-Blackwell NVFP4 W4A8 MMQ as a new self-contained TU `cu/mmq_nvfp4_w4a8.cu`, opt-in `BW24_MMQ_W4A8=1` (requires `BW24_RP=0` — A6 repack bypasses MMQ, same as W4A4). **pp512: 27B 1647->3109 tok/s (1.89x), 9B 4854->8670 (1.79x); 27B prime p1/p2/p3 0.206/1.038/3.136s -> 0.081/0.582/1.971s (2.53x/1.78x/1.59x). N=3 interleaved, free clocks, spread <0.2%.** ALL FOUR GATES GREEN: (1) kernel-check MMQ-W4A8-vs-f32-oracle rel 3.3e-3..4.7e-3 = INT8 BAND (hard gate 2e-2), NOT the 0.08-0.11 W4A4 band — new MMQ-W4A8 oracle gate added to kernel_check; (2) run-gen argmax 27B 82==82 + 9B 82==82 MATCH with W4A8 engaged e2e; (3) first-16 IDENTICAL to the default-int8 reference on ALL 6 (model,prompt) cells — including 27B p3 index 0, THE cell where W4A4 forked at index 0; (4) run-spec K=1..8 self-consistency PASS. **This is the honest fast-AND-exact prefill: W4A8 keeps bw24 default-GEMM int8 math (weight FP4->int8 dequant is bit-exact; q8_1 D4 symmetric-scale activation is the same int8 class), so argmax and prefix hold where W4A4 broke them. Half the W4A4 throughput (3109 vs 5379 on 27B) is the price of the accuracy class.** Wiring: both `BW24_MMQ` and `BW24_MMQ_W4A8` now enter the matmul/matmul_pre MMQ dispatch. NEXT: rp tile loader (W4A8 with the A6 repack instead of forcing rp-off), or default-flip after soak; then reconsider whether an FP8-activation CUTLASS variant beats W4A8-MMQ at large m (W4A4 CUTLASS stays closed per Stage 1).

--- ORIGINAL PLAN (kept for the line refs, now DONE) ---

**STAGE 2 plan:** W4A8 on the SAME fast MMQ tile. llama's NON-Blackwell NVFP4 MMQ path is exactly this and vendors 1:1:
  - `mmq.cuh:1069 load_tiles_nvfp4` (the `#else`/non-BLACKWELL branch) LUT-dequants FP4→int8 via `get_int_from_table_16(src_qs, kvalues_mxfp4)` into the int8 x_qs tile + `x_df` = `ggml_cuda_ue4m3_to_fp32(bxi->d[sub])` per-16 scale (NVFP4 is symmetric — no min-offset, simpler than k-quants).
  - vec_dot = `mmq.cuh:1495 vec_dot_q8_0_16_q8_1_mma` (int8 m16n8k32 mma, W4A8), traits at `mmq.cuh:3337` `#else` arm; tile size `MMQ_DP4A_TXS_Q8_0_16` / `MMQ_MMA_TILE_X_K_NVFP4` (`:221`, `%8==4` padded).
  - Activation quant = `quantize_mmq_q8_1` DS4 — ALREADY vendored in `cu/mmq_q45k.cu` (`quantize_mmq_q8_1_ds4_kernel`); reuse it. The whole q45k MMA/writeback/xy-tiling scaffold (`mul_mat_q_q45k`, `mmq_write_back_q45k`) is reusable — add a `load_tiles_nvfp4_w4a8` + a `bw24_mmq_nvfp4_w4a8` C-ABI launcher alongside the existing bw24_mmq_nvfp4 (W4A4) in `cu/mmq_fp4.cu` or a sibling TU.
  - EXPECTED: keeps bw24's current int8-W4A8 accuracy class (which passes ALL gates — it IS the default GEMM's math) at ~Q4_K-MMQ throughput (llama's k-quant MMQ pp is in the 4500-5000 band). This is the honest path to a fast AND argmax-exact prefill.
  - GATE IT: kernel-check MMQ-vs-oracle rel in the int8 band (~1e-3, NOT the 0.1 W4A4 band); run-gen 82==82; first-16 vs the Stage-0 reference lists MUST match on all 6 cells; run-spec K=1..8 PASS. Only then wire into matmul/matmul_pre and A6-repack (needs an rp tile loader OR keep rp-off for the MMQ path like today).
  - THEN reconsider CUTLASS: only worth it if W4A8-MMQ leaves a large-m gap CUTLASS's native-FP4 tensor cores close AND an FP8-activation variant (sm_120 block-FP8 mma, 381 TFLOP) can hold the gate — W4A4 is closed.

**NO_EVT DEFAULT-FLIP — SHIPPED in this arc (lib.rs):** event tracking now DEFAULT OFF, `BW24_EVT=1` = escape hatch. Cross-stream hazard audit (in the code comment): copy_stream touched by only stage_expert_async (ZERO callers) + the moe_cache admit barrier (gated on `prefetch_active`, setter never called); graph-capture sites use live-state `was_tracking` guards → degrade to no-ops. +4.6% measured 27B decode. Full gate battery running to confirm flip-safe before commit.

**DECAY-STUDY STAMPS (research/e2e/g7e-decay-study-stamps.jsonl):** independent second opinion on lane/g7e 6bcf8e1/438db11 from raw artifacts (/tmp/g7e-decay). (a) acceptance content-driven not ctx-driven = **CONFIRMED** (prose control -0.5pp over 4400 ctx vs code -12pp; decomposition arithmetic re-derived 41.8/58.2 vs claimed 41.1/58.9). (b) FA-verify-cost = **CONFIRMED** (independent sqlite: p1 54us vs p3 373.5us/instance, +2.56ms/round == claim). Main thread landed its own stamp in g7e-rtx6000.jsonl (aggregates FA as 4.64ms/16.9%-of-round total vs my growth-only 2.56ms — same direction, complementary framing). Ranked lever (fa rows shared-K reuse) is a G7e-lane priority.

---

## CURRENT TASK — DONE (2026-07-03). vendor Q4_K/Q5_K MMQ ✅

Landed as `cu/llama_mmq_q45k.cu` (new TU, self-contained), unified `qmatvec_mmq` dispatch in
`mmq_ffi.rs`, `mmq_supports` extended to QT_Q4_K/QT_Q5_K. All 6 NEXT STEPS below executed, all
gates green. Harness-agent's lib.rs/qmatvec_gemm.cu tune-seam WIP was committed separately
(e04c5f0). Section below kept for reference.

### Key correctness facts already established (do not re-derive)
- On Blackwell (`BLACKWELL_MMA_AVAILABLE`), **Q4_K and Q5_K both dequantize to int8 at tile-load, then run the shared int8 MMA inner loop.**
- Their `vec_dot_mma` is **`vec_dot_q8_1_q8_1_mma`** (mmq.cuh:1330), NOT q8_0 — because K-quants carry BOTH a per-subblock scale AND a min-offset, matching the q8_1 dual (d, m) layout.
- Both map to tile size `MMQ_MMA_TILE_X_K_Q8_1` (mmq.cuh:254-255).
- int8 MMA op: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (mma.cuh:946).
- Needs a **q8_1 activation quantizer** (distinct from the FP4 activation path already vendored).

### llama.cpp source map (`/data/projects/llama.cpp/ggml/src/ggml-cuda/`)
- `mmq.cuh`:
  - `#define MMQ_TILE_NE_K 32` :179
  - `MMQ_MMA_TILE_X_K_Q8_0 = (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4)` :219
  - `MMQ_MMA_TILE_X_K_Q8_1` = same expr :222
  - `MMQ_TILE_Y_K = (MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1)` :270
  - `load_tiles_q4_K` :2093
  - `load_tiles_q5_K` : follows q4_K (~2230+, the sed at 2240-2270 showed its body)
  - `unpack_scales_q45_K` :2083 — code: `return ((scales[(ksc%2)+(ksc!=0)] >> (4*(ksc&(ksc/2)))) & 0x0F0F0F0F) | ((scales[ksc/2] >> (2*(ksc%2))) & 0x30303030);`
  - `vec_dot_q8_1_q8_1_mma` :1330  ← THE shared int8 MMA vec_dot for both k-quants
  - `vec_dot_q8_0_q8_1_mma` :1159 (reference; q8_0 variant, read for the tile_A/tile_B/tile_C + load_ldmatrix + mma pattern)
  - traits: Q4_K :3358, Q5_K :3368
- `vecdotq.cuh`: `vec_dot_q4_K_q8_1_impl_mmq` :530 (VDR_Q4_K_Q8_1_MMQ=8 :502), `vec_dot_q5_K_q8_1_impl_mmq` (VDR_Q5_K_Q8_1_MMQ=8 :558)
- `mma.cuh`: int8 mma :946
- `ggml-common.h`: `block_q8_1` :258, `block_q4_K` :327, `block_q5_K` :345, `K_SCALE_SIZE=12`, `QI4_K`, `QR4_K=2`, `QR5_K=2`, `QK8_0=32`

### bw24 framework to reuse (`crates/bw24-engine/`)
- `cu/llama_mmq_nvfp4.cu` (593 lines) — the vendored NVFP4 MMQ framework. Has: `tile<>`/`load_ldmatrix`/`load_generic`/`mma` machinery, `load_tiles_nvfp4_nvfp4`, `vec_dot_nvfp4_mma`, `mmq_write_back_nvfp4`, `mul_mat_q_nvfp4`, `quantize_mmq_nvfp4_kernel`, C-ABI `bw24_mmq_nvfp4(w_blocks, act_f32, y, in_f, out_f, n_tokens, act_scratch, stream)`. Constants MMQ_NWARPS=8, MMQ_Y=128, MMQ_X=128, MMQ_TILE_NE_K=32. **Refactor shared tile/mma/write-back pieces into `cu/llama_mmq_common.cuh` if cleaner.**
- `src/mmq_ffi.rs` (80 lines, from 851e80f) — FFI to `bw24_mmq_nvfp4` + `bw24_mmq_nvfp4_act_bytes`. Has `mmq_supports(&self, w)` → true if `QT_NVFP4 && in_features()%64==0`. Declared `pub mod mmq_ffi;` at lib.rs:23.
- `build.rs` — compiles .cu as fatbin AND `llama_mmq_nvfp4.cu` as static lib (~lines 60-100), flags `-gencode arch=compute_120a,code=sm_120a`. **Must register the new .cu here.**

### QT tags (weight quant type ids in bw24)
`QT_Q8_0=0, QT_Q4_K=1, QT_Q6_K=2, QT_Q5_K=3, QT_NVFP4=7`

### Uncommitted work by harness-agent (do NOT stomp — coordinate)
- `src/lib.rs` (WIP): added `gemm_fatbin_path()` (reads `BW24_GEMM_FATBIN`, default `GEMM_FATBIN_PATH`), `k1_launch_override()` (OnceLock parsing `BW24_GEMM_K1_LAUNCH="BM,BN,NWARP"`). Launch sites ~1309/~1347 use `k1_launch_override().unwrap_or((128,128,8))` for is_k1 qtypes (QT_Q8_0|Q4_K|Q5_K). Matmul dispatch gates on `BW24_MMQ` + `mmq_supports` at ~855 and ~964. FA dispatch: fa_prefill ~1400, fa_decode_vec_q ~1501/1556.
- `cu/qmatvec_gemm.cu` (WIP): adding `#ifndef` guards around tunables (BM=64,BN=256,BK=32,NWARP=8,NSTAGE=3,K1_BM=128,K1_BN=128,K1_NSTAGE=2 — lines ~70-128). GQT tags lines 45-49.

### Runtime tune-seams (no Rust rebuild needed)
- `BW24_GEMM_FATBIN=<path>` — swap swept fatbins.
- `BW24_GEMM_K1_LAUNCH="BM,BN,NWARP"` — override k1 launch geom.
- Env flags: see `docs/FLAGS.md` (fast path is default-on; only non-defaults are flagged).

---

## NEXT STEPS (in order)

1. **Write Q4_K/Q5_K MMQ port** as new CUDA file(s) reusing the `llama_mmq_nvfp4.cu` framework (refactor shared → `cu/llama_mmq_common.cuh` if cleaner). Add:
   - `bw24_mmq_q4_K` + `bw24_mmq_q5_K` C-ABI launchers.
   - a q8_1 activation quantizer kernel (dequant weights→int8-with-(d,m), quantize activations→q8_1, shared int8 MMA inner loop = port of `vec_dot_q8_1_q8_1_mma`).
   - `load_tiles_q4_K`/`load_tiles_q5_K` ports incl. `unpack_scales_q45_K`.
2. Register new .cu in `build.rs`.
3. **After harness-agent's lib.rs lands**, extend `mmq_supports()` to Q4_K(tag 1)/Q5_K(tag 3) and route `BW24_MMQ` dispatch at lib.rs ~855/~964 to the new FFI.
4. **Gates:**
   - G1 build: `cargo build --release -p bw24-engine --bins`
   - G2 correctness: numeric prompt 101-612, argmax==82, `BW24_MMQ` on/off both MATCH.
   - G3 clock-locked pp512 median-of-5 vs 3332 baseline (expect ~4500-5000).
   - G4 no-regression with MMQ off.
5. **Mirror on sm_89 branch** (int8 MMA path IS portable to L40S — no FP4 gating needed for this one).
6. **Log pp512 delta as a training-data record** in `research/tune-data/rig5090.jsonl`.

---

## BACKGROUND AGENTS (may still be running / need resume)
- **harness-agent:** sweep-harness — `#ifndef` macro guards + JSONL sweep tool → `research/tune-data/rig5090.jsonl` + `record-manual.sh`. Owns the uncommitted lib.rs/qmatvec_gemm.cu edits above.
- **sm_89 port agent:** worktree `bw24-sm89`, branch `arch/sm89-l40s`, CUDA 12.8 + rust install, gate to green build + kernel_check. Already has commit 4902e68 (configurable `BW24_CUDA_ARCH` + FP4 gating).
- **web-sweep agent (completed):** ranked technique list — headline: DFlash spec-decode, TCQ KV quant, FR-Spec vocab trim, NVFP4 tensor-split fix, ST-MoE prefetch. **Output not yet fully consumed into roadmap — read + fold in.**

## RIG FACTS (sm_120a)
block-FP4/FP8 762/381 TFLOPS, NO wgmma/tcgen05, 847 GB/s mem wall, `compute_120a` trap (must use `120a` not `120`). Laptop 5090 = 24GB (not desktop 32GB), Intel Core Ultra, 60GB system RAM. Shares RAM with local LLM servers — check `free` before write-heavy benches.

## SAMPLED-SPEC ARC — OWNER DECISION (2026-07-09)

Owner: "its a prod project, we should go with the real solution." Greedy loop-handling ships as
REJECTION-SAMPLING SPECULATIVE DECODING (Leviathan/Chen rule: accept draft x at min(1, p(x)/q(x)),
resample norm(max(0,p−q)) on reject; bonus token from p) — distribution-exact by theorem, loops
broken by sampling, matches production serving. Penalized-greedy rejected as board protocol.
Engine scope (main-thread coding): seeded counter-based device RNG (Philox; graph-replay-safe,
no recapture), draft q(x) from the existing pmin softmax path, device sampling kernel
(argmax-swap), plain-sampled decode in run-spec for A/B. NEW GATE CLASS: (a) seeded
reproducibility run-to-run; (b) temp→0 continuity = token-identical to today's greedy spec
(the existing K=1..8 gate survives as the temp-0 limit); (c) aggregate distribution equality vs
plain sampling (acceptance-invariant/KL on long runs — same-seed streams legitimately diverge).
llama pairing at matched temp/top-p. Inputs pending: literature survey
(research/greedy-degeneration-protocol-survey.md, lane running) + rebaseline battery (GPU).
Spec p3/long-form board cells stay excluded until this lands.

## GPU GATE QUEUE (2026-07-09, owner-ordered parallel-dev mode)

Development proceeds in worktrees while the f8f4 flip battery owns the GPU; everything gates in
ONE queue when it frees, then the ST-vs-GGUF format decision follows and the loser stops getting
pushed.

Queue order:
1. f8f4 default-flip decision (battery table; main thread decides).
2. BW24_MOE_F8F4 expert-tile gates + pp A/B on the 35B (committed on main, ungated).
3. feat/sampled-graph-draft (../bw24-sgd): session-gate oracle greedy regression + serve smoke
   temp 0.7 (per-session seeded reproducibility, text audit) -> merge -> pi sessions unblock.
4. feat/filtered-spec (../bw24-fspec): filtered rejection sampling (top-k/p/min-p + penalties
   applied symmetrically to p and q — retires the legacy serve path to a rollback seam). Gates:
   filter-parity unit checks vs host sampler + serve smoke top-p.
5. Board move + release with whatever survived.

## FORMAT DECISION — GGUF (owner, 2026-07-10, FINAL)

GGUF NVFP4 is THE format; ST drops to best-effort support (README note, HF configs stay, no
board rows, no headlines). Decider: the owner's primary workload is LONG-context serving, and
the 27B p3 robustness matrix (2 long prompts x 3 seeds, temp 0.7) showed the ST checkpoint
looping seed/prompt-sensitively (2-3 of 6 runs) while GGUF ran coherent 6/6 — a reliability
class difference no speed cell offsets (ST had won pp +12%, TTFT -4.4%, spec p2 +8% derived;
GGUF held spec p1 +5% and long p3 +24% with quality). ST-only wins that remain live as
best-effort: NVIDIA-official checkpoints, no-conversion loads, the f8f4-adopted ST serve config.

FOCUS ORDER (owner): squeeze GGUF NVFP4 Qwen (9B/27B/35B) to the margin bar everywhere ->
then Gemma-4. Open GGUF cells: 27B spec p2 0.95x (llama's strongest spot), 35B spec p2/p3
1.02/1.03x, 35B plain 1.06x, 9B plain 1.07x, prefill 0.59-0.78x (tile-algorithm arc, ncu-spec'd:
per-CTA serialization — not occupancy, not DRAM, not MMA class).

## PP GAP ROOT-CAUSED (2026-07-10) — it is a precision-class contract, not efficiency

nsys on llama-bench pp1845 (27B NVFP4): llama quantizes ACTIVATIONS to NVFP4
(`quantize_mmq_nvfp4`, 13.5% of its GPU time) and runs `mul_mat_q<type 40>` on the FP4
block-scale pipe (762-TF class). Its MMQ kernel time 783ms/1845tok vs our W4A8 1310ms = 1.67x
= the whole pp gap. Our in-tree W4A4 (BW24_MMQ) already beats llama 1.03-1.06x — blocked only
by the GREEDY exactness contract (long-prompt argmax forks, 1/8 agent-loop self-consistency).
Eliminated efficiency levers this session (all JSONL-recorded): k32-imma (+4.3% pp, dominated,
deleted), ilpswap (-10.5%), y64 (-8%), epilogue hoist (killed by pipe data), pipe staging (wash).
DECISION (owner, 2026-07-10, FINAL): NO quality demotion — W4A4 stays blocked EVERYWHERE,
including sampled serve. "My call is not to demote our quality... if we could demote quality to
be on par and we decided not to, its ok." The pp column is a DELIBERATE QUALITY CHOICE, so
documented: we hold a precision class llama's bench config does not. pp rests unless
quality-preserving levers appear. Priority = E2E + QUALITY. The margin-bar goal applies to the
e2e cells; the pp cell carries the contract footnote, not a debt.

## 27B SPEC P2 CELL (0.95x) — PROBE PLAN (2026-07-10, from existing data, GPU pending)

Signal: llama accelerates on medium-code (p1 88.7 -> p2 93.8) while we DECELERATE
(104.1 -> 88.8) and our acceptance FALLS (70.7 -> 68.3%) where code predictability should
raise it. Hypothesis: our chain config caps the most draftable content — llama serves
n-max 3 with p-min 0.1 (long effective chains, permissive gate); we run K=3 pmin 0.4.
The K=4 rejection dates to an NGEN=64 sweep whose own lesson says short-gen understates ~20%.
Probes (N=2 interleaved, NGEN=256, p2, 27B GGUF standing draft+trim):
  1. K=4 pmin 0.4 / 0.3 / 0.2   2. K=5 pmin 0.3   3. K=3 pmin 0.2 / 0.1 (llama-class gate)
  4. best-of-above + PMIN0=1 (base acceptance ~68% < the 75% PMIN0 threshold — in its pay band)
Also record per-slot histograms (BW24_SPEC_STATS) — if slot-3+ acceptance stays >60% on p2,
deeper K is free money the pmin gate was refusing.
VALIDATED FROM EXISTING DATA (f8f4-flip model-c logs, K=3 p2): per_slot 0.813/0.658/0.543,
full-accept 54/91 rounds (59% hit the K-cap), len_hist shows pmin 0.4 truncates only ~23% of
rounds. 59% of rounds stop while the tail still accepts >54% -> K=4 expected ~+0.27 tok/round
= +9-10% e2e = the 0.95x cell flips to ~1.04x on config alone; K=5 rides if slot-5 holds ~0.4.
Probes 1-2 are now the priority order; 3 (pmin drop) second-order.

## NEXT SESSION OPENS HERE (2026-07-10 close)

1. MERGE feat/filtered-spec into main — SEMANTIC merge of spec.rs's sampled arm with the
   already-merged graph-sampled work. Composition rule to implement: the sampled GRAPH draft
   (draft_graph_s) engages only when NO filters AND NO penalties (pure-temp); filters/penalties
   force the eager draft (they need per-row stats/history the capture cannot hold). Also keep
   gsd's sess_tail counter fix (fspec's branch carries the old buggy read). Full gate battery
   after: sample_check, greedy regression, temp/top-p/penalized seeded-identity, graph-vs-eager
   identity (pure-temp), serve smoke. All individual gates are green on both branches.
2. Deep-K REFUTED (tag deepk-refuted) — 27B p2 + 35B cells have NO config flip; remaining
   levers: verify-cost tier (b12/b16 batching arc) + owner's head research.
3. Gemma-4 scope refresh (the modularity goal) once 1 lands.

## TERRITORY (owner FYI, 2026-07-10): Hy3 phase-2 spill + mixed-quant-per-expert MoE research
are OWNED BY ANOTHER AGENT. This session's lanes stay off Hy3 and off that research; expect GPU
contention windows (idle-gate discipline). research/hy3-phase2-plan.md is theirs to consume.

## VERIFY-COST ARC — PROBE 0 SEED DATA (2026-07-10, 35B spec nsys, prime included)

35B K=3 p2 spec run (192 tok, prime 0.609s in-capture): prefill side dominated by
mmq_iq_experts (995ms) as known. GEN-phase kernel spread (the arc's target): draft-chain matvecs
qmatvec_q8_0_mmvq_fused2 250ms/20.7k instances + mmvq 140ms/17.7k, expert verify/decode ops
moe_gate_up_silu8_dev_q8_v 128ms/8.4k + _v_rows 126ms/2.8k + moe_pairs_matvec_q8_dec 121ms/18,
q6_K matvec 204ms/414 (lm_head class). NEXT (fresh context): prime-excluded re-profile
(BW24_GEN_ONLY-style isolation or nsys capture-range) -> per-round verify-vs-draft-vs-commit
split -> the b12/b16 design targets whichever term dominates. Constraint: verify kernels must
stay BIT-IDENTICAL to decode per (token,row) (the dispatch-parity law) — speedups come from
launch fusion/batching shape, not numeric-class changes.

### Verify-cost arc — targets CORRECTED (trimmed re-profile 2026-07-10)

CORRECTION: the first quantification ran UNTRIMMED (probe-script bug: the 35B trim-path grep
resolved empty — every 35B probe that day incl. deep-K used no trim; the BOARD rebaseline was
trimmed and stands; the deep-K refutation was all-arms-untrimmed = internally consistent, stands
with that caveat). TRIMMED truth (nsys, K=3 PMIN0, numeric prompt):
- #1: verify+draft TRUNK matvecs — fused2 15.2% + mmvq 8.5% ~= 24% of gen wall.
- #2: MoE expert ops (gate_up_v + _rows + down8) ~= 14%.
- #3: q6_K head 7-11%, BIMODAL: trimmed draft 68us (fine) + full-vocab 489us at m=1
  ~0.75/round — call-site mapped (BW24_Q6K_TRACE): the ZERO-DRAFT / j==0 legacy-replay path
  runs a full-head m=1 pass per zero round (PMIN0 rounds!). Small real lever on PMIN0 models:
  route the zero-round commit through a head-free step or the trimmed head.
- Trace also confirmed the verify head batches correctly (m=2-4 batched=true).
- Trim content-dependence noted: on the NUMERIC id prompt the trim made 35B spec 0.86x vs
  plain (141 vs 198 untrimmed) — the generic ranking excludes numeric-heavy tokens; real-prompt
  board configs unaffected.
35B margin math: no single >15% term remains; clearing p2/p3 (1.02/1.03x) likely needs #1
(trunk-matvec launch/fusion work, FUSED_T-class follow-on) + #3. 27B p2: same class.

### #3 zero-round fold — DESIGN (code-anchored, 2026-07-10 close)

Today the j==0-with-pending case (n_acc=0, a pending bonus exists) takes the LEGACY REPLAY:
one m=1 full-stack forward (incl. the 489us full-vocab head) to commit the old pending + new
bonus. The fold: carry BOTH as pending -> next round's verify batch = [pending_a, pending_b,
drafts...] with base=2. Feasibility (read from generate_spec_inner2):
- t_pred/p_j indexing already generalizes over `base` (preds[base+j-1], cols (base+j-1)) — no
  structural change, only the base=2 value.
- pending: Option<u32> -> a 2-slot carry (SmallVec/array). The draft-KV seed for the round
  start needs the hidden of the LAST pending's predecessor — the verify computes hiddens for
  every column, same pattern the base=1 path uses (fill_prev chain extends one slot).
- Commit bookkeeping: rollback keeps base columns unconditionally (they are EMITTED tokens);
  accept_len arithmetic shifts by base — audit the three commit arms + the session tail.
- PMIN0 zero-draft rounds (the p3 +13% win) become base-growing rounds: cap the carry at 2 —
  a zero round WITH 2 pendings still must flush via replay (rare^2). Bound the change.
- Gates: the full battery + the session-gate oracle (4-turn incl empty-suffix) — the carry
  interacts with burst boundaries (pending must persist across generate_spec_session calls or
  flush at burst end — FLUSH AT BURST END, simplest, matches today's tail commit).
Win: removes ~0.75 full-head passes/round on PMIN0 configs (35B p3-class rounds) AND the extra
trunk m=1 replay — est. +3-5% on the p3 cell, stacking with #1.

### Verify-cost #1 — QUANTIFICATION CAVEAT (2026-07-10 final)

The nsys profiles behind the target table span BOTH run-spec phases (plain-generate oracle +
spec) — total kernel time 2.15s vs ~1.35s spec wall proves the mix. The fused2 15.2% is
spec-only (verify-exclusive kernels) but the mmvq/q6_K/MoE shares blend oracle decode into
spec numbers. GPU-busy across the combined run ~85% (gaps ~15% -> a verify-forward CUDA graph
reclaims at most a few % — secondary, not primary). NEXT-SESSION FIRST STEP for #1: phase-
isolated profile (nsys capture-range around generate_spec, or an oracle-only subtraction run)
-> TRUE spec-phase shares -> then design (fusion vs graph vs both). Do not design from the
current table.

### Verify-cost #1 — FINAL round-loop shares (BW24_PROFILE_SPEC=2 + nsys cudaProfilerApi)

Three subtraction confounds later, the clean instrument exists and the answer is measured
(35B K=3 p2, round loop only): TRUE #1 = MoE VERIFY expert ops 24% (gate_up_silu8_v_rows 16.6%
+ down8_rows 7.4%; ~60 launches/round = per-MoE-layer, already round-batched — the cost is the
dp4a math itself at t=4-5 x top-8 experts under the verify-stays-dp4a exactness law). #2 =
trunk matvecs 15.3% (fused2_b4 + mmvq_b4). #3 IDENTIFIED = the MoE ROUTER GEMV
(hybrid_forward.rs:720 — gate_inp F32 -> cuBLASLt, plus the shexp gate at :1016): ~200 cuBLAS
launches/round, 4% of the loop. Lever: a warp-per-row f32 router matvec kernel batched over t
replaces 12.8k cuBLAS dispatches per run. CAUTION (corrected): the router feeds TOP-K expert
selection — discontinuous — so a different FP accumulation order risks ROUTING FLIPS (the
cross-kernel-family FP-order law on sigmoid routers, already in the README). cuBLAS's internal
order is not reproducible by construction -> this is a NEW NUMERIC CONFIG, full battery + the
MOE_GATE oracle decide, per-model adoption rules apply. Est +2-3% e2e IF it gates green. The q6_K
head batches correctly (b4_r2, 3.8%). Design frontier for #1: cross-token expert-activation
dedup in the verify (t=5 x 8 experts with overlap — the CSR expert-major machinery is the
natural host), NOT an MMA class change (dispatch-parity law). 27B (dense) p2 shares differ —
profile it with the same instrument before designing.

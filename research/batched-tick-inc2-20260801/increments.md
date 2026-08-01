# Batched-tick increment 2: z-batched attention + KV append + lean logits (2026-08-01)

Mission (from increment 1's map): fold the per-seq-serial attention block of the batched
serving tick — (1) pointer-table blockIdx.z-batched fa_decode across sequences, (2) batched
KV append on the same table, (3) drop the logits D2H for device-sampled rows after a
last_logits consumer audit. Box: 8xH100 block box (18.117.231.105), dedicated GPU 1,
9B Q8_0. Increment 1 (device-side batched sampling) is the merged v0.59.0 base = the
denominator binary.

## What was built

1. **`fa_decode_vec_q_seqs_v4` + `fa_decode_combine_seqs`** (flash_attn.cu): one launch
   covers ALL B sequences' T=1 attention for a layer — blockIdx.z = sequence, per-seq K/V
   cache base pointers from a per-step device pointer table ([2B] interleaved, built into
   the SAME table the GDN state kernels already use — the MoE expert-table pattern), per-seq
   key bound from the tick's position table (T_kv = pos[z]+1). Body = the v4 eager kernel
   VERBATIM under a per-z frame; the split partition derives in-kernel from (T_kv,
   split_keys) — the ONE-PARTITION LAW — so each sequence executes the EXACT per-seq eager
   program whenever all rows share one `fa_split_keys` rung (the rows-twins' straddle law;
   decode_batch groups per STEP and falls back to the per-seq loop on any straddle or
   non-v4 row). Q read / attn written at row offsets → the per-seq q/a dtod scratch copies
   fold away. 8xB launches per layer → 2. Partials ride the rows layout
   [B, n_head, n_splits_max, hd]; empty splits write the EMPTY partial the combine skips.
2. **`append_quantize_kv_q8_0_q5_1_seqs`**: one launch appends all B rows, each into its
   own cache at slot pos[z], through the same pointer table. Each (block, z) warp executes
   the per-token appender's exact warp program → written cache bytes BIT-IDENTICAL.
3. **Lean logits** (`decode_step_batch_sampled_lean`, worker + `Cache.last_logits_dev`):
   device-sampled rows skip the [n_vocab] logits D2H entirely. last_logits consumer audit:
   (a) next-tick host sample — never fires, `device_next` carries the token; (b) graph
   promotion argmax — reads prefill logits only (`generated.is_empty()` guard); (c) THE
   real consumer: the KV-reuse pool park at retire (an empty-suffix resume samples from
   parked last_logits) — served by a per-cache DEVICE park (row dtod → `last_logits_dev`,
   ~µs at HBM bandwidth) D2H'd once at retire; a session with neither host nor device
   logits is not parked (no poisoned entries). Rows without a device sample keep a per-row
   D2H. Non-serving callers (`decode_step_batch`, gates) keep full rows (lean=false).
   Seams: `MEMRA_BATCH_FA=0`, `MEMRA_BATCH_APPEND=0`, `MEMRA_SERVE_LEANLOGITS=0`.

## Gates (GPU 1, all green; every battery run at BOTH 32 and 160 steps — 160 drives all
## gate sequences across the t_kv=96 vec floor so the seqs arm actually engages)

| gate | verdict |
|---|---|
| kernel-check (9B GGUF) + NEW seqs pins | ALL GREEN — `append_kv_seqs` bytediff=0 (B=4, B=8); `fa_decode_seqs_v4` vs per-seq loop bitdiff=0 (mixed depths [96,128,257,511] and uniform [200;8]) |
| decode-batch-gate --mode config --batch 8 (s32 + s160) | PASS (gate1, gate2 bit-checked, gate3) |
| decode-batch-gate --mode strict --batch 4, equalized env (s32 + s160) | PASS — **gate1 B=1 BIT-IDENTITY vs decode_step_h through 160 steps** = the seqs-kernel-vs-eager pin at depth |
| NEW gate3c: lean-vs-full identity (mixed device/host rows) | PASS — same tokens, parked device logits row == full host row bitwise, unsampled rows bit-identical |
| run-gen argmax (9B, board-2048) | MATCH; 32-token stream IDENTICAL to base binary |
| check-batch-exact (16 greedy, batched vs isolated) | PASS 16/16 (components 1+2 binary AND final lean binary) |

## Engine-level receipt (decode-batch-bench, 256 steps, N=5 medians, same GPU minutes apart)

| config | base | inc2 (components 1+2) | delta |
|---|---|---|---|
| B=1 | 157.3 tok/s | 159.5 | +1.4% (no B=1 regression; B=1 rides the seqs kernel too) |
| B=8 | 503.6 tok/s | **602.9** | **+19.7%** |
| B=8 scaling vs B=1 | 3.20x | 3.78x | |

Tick phase shares (MEMRA_BATCH_PHASE=1, B=8, sync-bounded — shares rank, not walltime;
absolute ms over 256 steps, base → inc2):

| component | base | inc2 |
|---|---|---|
| attn per-seq fa_decode | 1317 ms / 14.6% | 265 ms / 2.9% |
| attn per-seq q/a dtod copies | 642 ms / 7.1% | 0.0 ms / 0.0% |
| attn per-seq kv append | 399 ms / 4.4% | 46 ms / 0.5% |
| logits D2H + host split | 762 ms / 8.4% | (see note) — component 3's target; lean removes it from the serving tick |

(The inc2 profile's inflated D2H slot ms is a sync-bounded artifact under concurrent
box load; the plain-mode +19.7% and the serving A/B below are the walltime evidence.)

## Serving A/B round 2 (single replica GPU 1, 9B, temp 0.7, ~200-tok prompt, 128-tok gens,
## fresh server per point, arms interleaved per (rep, c) cell, N=4)

Arms: `base` = v0.59.0 binary (increment 1); `faap` = inc2 binary MEMRA_SERVE_LEANLOGITS=0
(components 1+2 only); `lean` = inc2 binary naked (components 1+2+3).

Aggregate tok/s, median of N=4 (all points) — bad-mode interference points included:

| c | base | faap (1+2) | lean (1+2+3) | lean vs base |
|---|---|---|---|---|
| 8  | 516.5 | 564.9 | **653.9** | +26.6% |
| 16 | 460.9 | 518.5 | **656.5** | +42.4% |
| 32 | 473.1 | 507.7 | **658.9** | +39.3% |

Clean-mode medians (excluding the interference hits below; conservative attribution):

| c | base | faap | lean | faap vs base | lean vs faap | lean vs base |
|---|---|---|---|---|---|---|
| 8  | 521.1 | 567.5 | 653.9 | +8.9% | +15.2% | **+25.5%** |
| 16 | 507.8 | 577.0 | 656.5 | +13.6% | +13.8% | **+29.3%** |
| 32 | 485.0 | 507.7 | 658.9 | +4.7% | +29.8% | **+35.8%** |

Tick p50 medians (ms, base → faap → lean): c8 13.7 → 12.2 → **10.4**; c16 31.4 → 27.8 →
**20.8**; c32 58.2 → 53.4 → **41.7**.

- **lean is 12/12 points inside [646.7, 660.8] across ALL concurrencies** — the flattest
  serving cell measured on this lane, and ~655 aggregate is c-INDEPENDENT (the tick is now
  weight-stream-shaped: c=16/32 run 2/4 chunks of 8 per tick, aggregate stays fixed).
- Thermal/interference regime: shared 8xH100 box; other agents run fresh-server load
  harnesses continuously on GPUs 4-7 (host/PCIe shared). 7 of 24 base+faap points took
  a ~15-20% hit (bad mode ~420-460 tok/s, magnitude c-independent, per-request latency
  uniformly inflated); **zero of 12 lean points did**. Consistent with the bad mode being
  host/PCIe contention on the per-tick pageable logits D2H (8-32 MB/tick at c=8..32),
  which lean eliminates (device-sampled ticks read back one [B] u32 row). Neighbor
  process-list snapshots per point (neigh-*.txt) could not discriminate finer (neighbors
  always resident, always churning); the claim stays behavioral.
- Round 1 (r1, superseded by round 2 but retained: load-* / metrics-* / server-* without
  the r2- prefix): base vs fa (append seam off) vs faap, N=3 — clean-cell c16 showed fa
  +9.7% / faap +10.6% with 6/6 new-arm points in [561, 573]; c8/c32 cells were bimodal
  in BOTH arms including base (the same interference mode; base hit 385-423 twice there).
  Per-component split: append (component 2) adds ~+1% over fa alone at c16 — the fa launch
  fold carries nearly all of the components-1+2 win.

## Soak (mission follow-up on increment 1's unreproduced worker panic)

20-min continuous c=16 temp-0.7 load against ONE server process (final lean binary,
consecutive load points → admit/retire/park/reuse-pool churn — the park path exercises
the new retire-time device-logits D2H every point). **CLEAN**: 95 load points, 6080
requests, 778,240 tokens, 0 request errors, 0 panic/error lines in the server log, server
alive at end. Throughput min/med/max = 651.3/655.7/658.7 tok/s — 1.1% spread over the
full 20 minutes (also the strongest thermal-regime receipt for the lean numbers above).
The increment-1 incident (embed-gather range panic from a garbage token) did NOT
reproduce. Raw: soak-summary.log / soak-server.log / soak-points.jsonl /
soak-per-request.jsonl.

## Incidents (this increment)

1. **Broad pkill (07:10Z)**: a cleanup `pkill -f memra-server` before the first
   check-batch-exact run killed sibling fleet replicas from ~/memra (GPUs 5-7; their
   supervisor respawned them, GPU 7 observed mid-reload) and possibly an arc5 lane server.
   Same failure class as increment 1's logged incident. Neighbor measurements overlapping
   ~07:10Z may need re-runs — reported to the coordinator. Rule re-learned: kill EXACT
   PIDs only; never pattern-kill on a shared box.
2. **Mid-A/B rebuild race (checked, no taint)**: the component-3 rebuild landed at
   07:28:32Z while round-1's last two points ran; both servers exec'd before the binary
   swap (point timestamps vs binary mtime in the log) — all 27 round-1 points ran their
   intended binaries.

## Increment 3 (recommendation)

With sampling (inc 1), the per-seq attention block (inc 2.1/2.2) and the logits D2H
(inc 2.3) all off the tick, the serving tick is weight-stream-dominated: ffn 31.5% + gdn
projections 13.5% + lm_head 4.0% ≈ half the engine profile, all batch-INVARIANT — and the
lean arm's c-independent ~655 tok/s ceiling is exactly that signature. The lever ranking:

1. **Deeper exactness tier (B>8)**: c=16/32 currently run 2/4 sequential chunks of 8 per
   tick, re-streaming the weights per chunk. A validated B=16/32 numeric-config policy
   (the m>=16 GEMM tier with its own argmax/self-consistency baseline, the
   MEMRA_DECODE_BATCH_CAP door) would halve/quarter the weight re-reads — the only lever
   with ~1.5-2x headroom at c>=16. The seqs attention/append kernels already take any B.
2. Batched-tick CUDA graph: with the per-seq launch trains gone the tick is few-launch;
   graph capture per B-bucket (len_d counters exist from the dc machinery) shaves the
   remaining launch+host overhead (tick p50 10.4ms at c8 vs ~13.3ms engine step in the
   bench form suggests ~1-2ms of host/dispatch still on the tick).
3. Host emit path (detok + stop-scan + channel send per token) — now a visible share of
   the 10.4ms c8 tick; profile before touching.

# Batched-tick increment 1: device-side batched sampling — lane 3 (2026-08-01, H100 GPU 3)

Mission: profile the B=8 serving tick, batch its TOP per-seq-serial component, gate, measure.

## The tick cost table (produced here, 9B Q8_0, B=8, ctx 512, GPU 3)

Engine decomposition: `MEMRA_BATCH_PHASE=1` (new, sync-bounded — shares rank the tick, the
added syncs inflate the total; 256 batched steps via decode-batch-bench). Host stage:
measured directly on real last-step logits rows (N=400 samples).

| rank | component | share / measured | scaling class |
|---|---|---|---|
| 1 | **host sample (temp 0.7)** | **1.36 ms/row = 10.9 ms/tick at B=8** | per-seq, host, single-thread |
| 2 | ffn (add/norm/gate/up/act/down) | 31.9% of engine profile | batch-invariant (weight stream) |
| 3 | attn per-seq fa_decode | 14.0% | per-seq (B launches/layer) |
| 4 | gdn batched projections | 13.8% | batch-invariant (weight stream) |
| 5 | logits D2H + host split | 9.4% (~7.9 MB/tick at 248k vocab) | per-seq (B rows) |
| 6 | attn per-seq q/a dtod copies | 7.2% | per-seq overhead (v1 scratch copies) |
| 7 | attn batched pre (norm/qkv/rope) | 4.5% | batch-invariant |
| 8 | attn per-seq kv append | 4.4% | per-seq |
| 9 | gdn out / gdn state / lm_head | 4.4 / 4.2 / 4.1% | invariant / per-seq-ish / invariant |
| — | host greedy argmax | 116 us/row = 0.93 ms/tick | per-seq, host |

The R2 report's 24.4 ms tick p50 at temp-0.7 load = engine (~13 ms real) + host sampling
(~11 ms). The host sampler's temp path builds a 248320-entry (id, logit) Vec (~2 MB), exps
the full vocab, and draws — O(n_vocab) per row, single-threaded, between every tick.

**Chosen component: host sample/emit — device-side batched sampling.** It is 2x the whole
per-seq attention block and sits serially between GPU ticks.

## What was built

- `Engine::gumbel_perturb_col` (lib.rs): column twin of `gumbel_perturb` over stacked
  logits [B, n_vocab] — same kernel/Philox mapping, pointer-invariant per (seed, ctr, temp).
- `decode_step_batch_sampled` (decode_batch.rs): per requested row, between lm_head and the
  logits D2H — greedy: 2-pass device argmax (bit-identical to host argmax, argmax-gate
  contract); temp>0: gumbel_perturb(seed, ctr, temp) + same argmax = one categorical draw
  from softmax(logits/T). One [B] u32 readback. Full logits rows still returned (last_logits
  semantics + fallback rows unchanged). `decode_step_batch` = thin wrapper (no samp).
- worker.rs: eligible rows (greedy-no-penalty; pure-temperature with top-k/top-p/min-p off)
  get metas (seed = request seed, ctr = generated.len() — a session-progress function,
  batch-composition-independent); `Session.device_next` carries the token to the next
  `advance_sample_emit`, which skips the host sample. All other configs keep the host path
  row-by-row. `MEMRA_SERVE_DEVSAMPLE=0` = rollback seam (the exact pre-change tick).
- RNG note: temp rows draw from the device Philox stream, not the host SplitMix64 —
  distribution-equal (Gumbel-max = exact softmax sample, the sampled-spec machinery),
  seed-deterministic, but sampled token streams differ from the previous binary. Greedy is
  bit-identical. Batched-vs-isolated identity holds on both (gate3 + serving harness).

## Gates (GPU 3, all green)

| gate | verdict |
|---|---|
| kernel-check (9B GGUF) | ALL GREEN |
| decode-batch-gate --mode config --batch 8 | PASS (gate1, gate2, NEW gate3) |
| decode-batch-gate --mode strict --batch 4 (equalized env, battery form) | PASS (gate1, gate2, gate3) |
| run-gen argmax (9B, board-2048) | MATCH |
| NEW gate3a: device greedy token == host argmax, every row/step | PASS |
| NEW gate3b: sampled streams B=8 vs B=1, same (seed, ctr) schedule | PASS (identical) |
| serving-level check-batch-exact (16 greedy, batched vs isolated) | PASS 16/16 x3 runs |

## Serving A/B (single replica, 9B, temp 0.7, 128-tok gens, fresh server per point,
## arms interleaved per (rep, c); dev = naked default, host = MEMRA_SERVE_DEVSAMPLE=0)

| c | dev median (N=3) | host median (N=3) | delta | tick p50 dev/host (median) |
|---|---|---|---|---|
| 8  | **510.9** | 305.4 | **+67%** | 13.9 / 24.5 ms |
| 16 | **507.2** | 305.3 | **+66%** | 28.0 / 49.0 ms |
| 32 | **431.8** | 262.4 | **+65%** | 71.3 / 103.9 ms |

- host arm reproduces R2's baseline exactly (~305 tok/s, 24.4 ms tick at c=8) — the seam-off
  arm IS the baseline tick (all-None metas = the old decode_step_batch structurally).
- Mechanism receipt: tick p50 24.5 -> 13.9 ms = -10.6 ms ~= the measured 10.9 ms host
  sample cost. The lever removed exactly what the cost table said it would.
- rep2 c=8 pair depressed on BOTH arms together (another lane's sweep active on the box);
  pairwise ratios stayed 1.64-1.67x across all nine pairs.
- raw: load-points.jsonl / per-request.jsonl / per-point server+metrics logs in this dir.

## Incident log

One worker panic (`model.rs:678 range start index 9345848831744 out of range`, garbage
token into embed gather) on the FIRST check-batch-exact attempt — captured in
server-exact.log. Concurrent state: a leftover serve-ab server was still alive on GPU 3
(captured pgrep in the session transcript; my own driver-kill raced). NOT REPRODUCED in 3
clean harness runs (16/16 PASS each) + 3 manual greedy requests + the full gate battery.
Status per evidence discipline: died, cause unknown — repro needed; no conclusion built on
it; flagged for a soak run in increment 2. Also on record: a broad `pkill -f memra-server`
during cleanup killed sibling-lane servers from ~/memra (their driver respawned; their
concurrent points may need a re-run — reported to the coordinator).

## Increment 2 (recommendation)

With sampling off the tick, the remaining per-seq-serial block is the full-attn loop
(fa_decode 14.0% + q/a dtod copies 7.2% + kv append 4.4% ~= 26% of the engine profile,
8 layers x B sequences of launches):
1. **Batched fa_decode across sequences** — KV pointer-table + blockIdx.z = sequence (the
   MoE expert-table pattern; per-z fold order identical to the single-seq kernel = per-seq
   bit-identity preserved). Folds in the q-row/a-row copies (kernel reads q at row offset,
   writes attn at row offset) and turns 8xB launches into 8.
2. Batched KV append (one kernel, B sequences) rides the same pointer table.
3. Then the logits D2H (9.4%): device-sampled rows don't need their logits row on host —
   return only fallback rows (needs last_logits-consumer audit: reuse pool parks it).

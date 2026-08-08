# lane/cache-metering — prefix-cache hit-rate metering (2026-08-07)

MISSION: the receipt infrastructure for the caching earning multiplier. Cache-hit prompt
tokens bill at full price but cost ~zero compute; the first-listed-week hit-rate number
must be one query away when traffic arrives.

## State: COMPLETE — all five deliverables landed, full serve-smoke green

| commit | what |
|---|---|
| f23330b5 | engine metering: /metrics full counter set + LCP histogram + per-tenant split + publish-on-retire |
| 894bea01 | tools/cache_economics.py — scrape -> earning-model row |
| efa2cbe5 | tools/cache-meter-gate.py + serve-smoke arm 7b + exactness/teeth/overhead receipts |
| (this)   | docs/SERVING.md cache-hit metering section + this PROGRESS |

## What was already there (found, not built)

- `usage.prompt_tokens_details.cached_tokens` per-request (worker-truth, all three cache
  tiers: continuation pool, spec resume, prefix cache) — landed with lane/prompt-cache
  2026-08-02, unit-tested in main.rs. Field name verified current against OpenRouter's
  usage-accounting docs (OpenAI/OR/Grok-chat all report cached reads in exactly this shape;
  `cache_write_tokens` exists on newer OpenAI but memra has no write-billing, so not added).
- `/metrics` `prompt_tokens_in`/`cached_tokens_in` + `prefix_cache_hits/entries/bytes`.
- PrefixCache internally counted misses/inserts/evictions/hit_tokens but never published.

## What this lane added

1. **/metrics**: `computed_tokens_in`, `cache_hit_token_ratio` (token-weighted — THE number),
   `prefix_cache_{misses,inserts,evictions,hit_tokens}`, `lcp_histogram` (edges
   [0,1,16,32,64,128,256,512,1024,2048,4096]; one sample per probe — served length on hit,
   best_lcp on miss, both already computed so no new scan; buckets 4..=6 = the tick-seg
   [64,512) window), `tenants` per-tenant rows.
2. **Per-tenant split**: keys on the tenant half of the PC-ISO namespace
   (`auth::meter_key` — keyring salts collapse to `t:<tenant>`, raw salts pass through,
   NS_SEP-unforgeable). Bounded 256 rows, overflow -> `"(other)"`, totals stay exact.
3. **Publish-on-retire**: metrics snapshot forced on EVERY retire (was 32nd-tick + spec
   retires only) — the post-workload scrape can never read counters parked behind an idle
   recv(). Same per-request cost class as the existing spec force-publish.
4. **tools/cache_economics.py**: scrape -> `revenue_multiplier = billed/computed` at a
   chosen cached-billing factor (1.0 default, 0.25 = OR cached-input tier), per-tenant
   multipliers, tick-seg window share. Self-test: 80% hit -> 5.0x @1.0, 2.0x @0.25.
5. **tools/cache-meter-gate.py** (serve-smoke arm 7b): the exactness receipt.

## Receipts (raw/ in this directory)

- **Exactness** (gate-run1.log, gate-exactness.jsonl, gate-server.log): N=5 sharing K=256
  prompt_ids tokens + 16 unique suffix under salt A, +1 same-prefix under salt B, 9B NVFP4,
  MEMRA_SERVE_SPEC=0 (spec bypasses the prefix cache by policy). 26/26 exact:
  A-req1/2 cached=0 (seed, lcp-split), A-req3..5 cached=256, B cached=0 (cross-salt),
  /metrics closed forms (1632/768, hits/misses/inserts 3/3/3), histogram bucket-exact
  (2 cold in [0], 4 at edge-256 inside tick-seg window), tenants exact, economics 1.8889x
  == 1632/864 crosschecked.
- **Teeth** (teeth-run.log, teeth-cacheoff.jsonl): PREFIX_CACHE_MB=0 + KV_REUSE=0 inverts
  16/26 — the gate cannot pass vacuously.
- **Overhead** (perf-ab-run1.log, perf-ab.jsonl, perf_ab.py): pre-lane binary @e54dd2e6 vs
  instrumented, BOTH RESIDENT one 5090 (23.7GB free at start, gpustate recorded),
  interleaved x5 alternating order, N=100 req/arm, c=1 greedy 64-tok gen at prefix-hit
  steady state (the metering hot path). p50 496.3 vs 496.1ms (−0.03%), p95 502.1 vs
  501.1ms (−0.19%) -> PASS <0.5% p95 bar; deltas inside run noise. Thermal regime: warm
  interleaved, single lock hold.
- **Full battery** (serve-smoke-full.log): serve-smoke 0 failed including the new arm,
  spec==plain exactness, truncation matrix, affinity — no regression.

## The first-listed-week query

    python3 tools/cache_economics.py http://<serve-host>/metrics \
        --cache-billing-factor 1.0 >> research/cache-meter-<date>/economics.jsonl

One row: hit-token ratio, revenue multiplier, per-tenant breakdown, tick-seg window share.
Bill-to-abort traffic is already covered (aborts log prompt/cached at the abort point;
admitted counters include aborted sessions' admit-time split).

## Notes for successors

- Unit tests: lcp bucket edges + window bounds, meter_account collapse/cap/totals,
  meter_key unforgeability (101/101 memra-server green).
- The gate's insert expectation (3 = A-seed + A-lcp-split + B-seed) tracks the learning
  sequence in worker.rs's PrefixCache module doc; if seeding policy changes, re-derive.
- perf_ab.py keeps both servers resident to make the comparison drift-immune (the H100
  lane law); c=1 only — a c>1 overhead claim would need the admission-queue path equalized.
- The `spec` usage extension and this lane's fields are ADDITIVE — official SDKs ignore
  unknown usage fields; spec-off + cache-off responses stay byte-identical (teeth run
  confirms zeros, serve-smoke confirms shapes).

# lane/cx-prefix-dedup — in-batch same-prefix dedup + entry pinning

Branch base: `cbc8d76f` (SOTA harvest merge, including the merged
`lane/cx-prime-batch` machinery).

Preferred receipt rig: box1 `18.195.123.14`. The named lane inbox did not exist at
startup, while `~/.lanectl/inbox/cx-primebatch.md` reserves box2 for the Step serving
trial and directs new GPU verification to box1.

## Increment 1 — read conclusions and design

### Where same-prefix requests duplicate work today

`admit()` probes the prefix cache request by request. A cold request records a miss,
arms a full-prompt seed, allocates its own session cache, and enters `active` with its
entire prompt in `prefill_queue`. No state representing other cold requests in the same
admission drain exists yet.

The interactive scheduler later gathers those already-diverged fresh sessions and calls
`prime_cache_batch()` over every full prompt. The merged primebatch path shares weight
streams across PP stages, but it intentionally preserves one attention/KV history per
request. Therefore N simultaneous requests with a common K-token system/tool prefix
still compute and append those K tokens N times. Only after the batch returns does
`maybe_prefix_seed()` snapshot each request. Later arrivals can hit, but siblings in the
batch are too early.

The existing LCP-learning path does not close this window. It needs one seed and one
later miss to discover the shared boundary; simultaneous cold siblings all completed
their cache probes before any seed existed.

### Dedup point

Add one scheduler stage immediately before fresh-prompt batch formation:

1. Consider only cold, non-spec, non-graph sessions whose cache and `fed` position are
   both zero and which have no pre-existing LCP split boundary.
2. Partition by the existing `PoolKey = (model, cache_ns)`. Under API-key auth,
   `cache_ns` is `t:<tenant>\x1f<cache_salt>`; without a keyring it is the raw salt.
   This equality is the hard security boundary. Token similarity is never evaluated
   across different pool keys.
3. Within one pool, group requests whose first `PREFIX_CACHE_MIN_TOKENS` token ids are
   exactly equal, then derive and verify the full group LCP. Hashes may label receipts,
   but exact token equality decides membership.
4. Prime that common prefix once into the leader's normal stage-owned cache, snapshot
   it, restore the snapshot into each sibling's already-allocated cache, and leave all
   group members queued only on their request-specific suffixes. The existing
   `prime_cache_batch()` path then batches those suffixes.
5. Convert each sibling's provisional admission miss into a cache hit. Per-request
   `cached_tokens`, global/tenant cached-token totals, hit/miss counters, hit-token mass,
   and the LCP histogram must all describe the final served path, not the provisional
   admission observation.

This also covers prefixes longer than `MEMRA_PRIME_BATCH_MAX_T`: the one shared prefix
prime happens before the full-prompt batching cap is applied, while the shorter suffixes
remain eligible for the merged batch path.

### Pinning design

`PrefixEntry` gains an in-flight refcount. Pinned entries are absent from the evictable
LRU index, so normal budget eviction and the session-allocation cache flush can remove
only unpinned entries. A hit or fanout participant holds a `(PoolKey, entry-id)` lease in
its `Session`; the centralized retire sweep releases it on completion, disconnect,
error, or OOM park. The last release returns the entry to the LRU at current recency.

The fanout snapshot is inserted with one reference per participating request before the
requests continue. Existing later-arrival hits take the same lease. If the entry cannot
fit beside already-pinned bytes, the fanout may still use the just-created snapshot for
same-tick restore, but it is not retained and no false pin/retention claim is made.

This is retention only. Every request still owns a deep-copied session cache, so no
request mutates another request's KV/recurrent state.

### Current upstream check

Validated against live upstream heads on 2026-08-08:

- SGLang `db75dfe1`: `schedule_policy.py` still carries the explicit in-batch prefix
  caching check, with a default 32-token threshold and longest-prefix scheduling logic.
- TensorRT-LLM `937bacc2`: current KV-cache docs still specify prioritized LRU
  (priority 0..100) and request retention ranges/durations for important prompt blocks.

The local implementation keeps memra's existing snapshot/deep-copy cache model rather
than importing either engine's block manager.

### Gates to add/run

| gate | required result |
|---|---|
| host unit tests | exact grouping within one pool; cross-salt and cross-tenant groups impossible; pinned entries skip eviction and become evictable after last release |
| cache-meter fanout N=5 | 1 computed prefix + 4 cached prefixes; exact per-request `cached_tokens`; global and tenant arithmetic exact |
| cross-salt simultaneous fanout | both salts cold; no shared computation, hit, pin, or token credit across the boundary |
| serve-smoke | 0 failures |
| PC-ISO / API-key isolation arms | green |
| run-gen | argmax MATCH |
| N=8 receipt | simultaneous same-prefix TTFT distribution before/after, raw client/server logs and thermal/GPU state on box1 |

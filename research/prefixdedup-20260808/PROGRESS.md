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
   group members queued only on their request-specific suffixes. The common prime is
   capped by the existing interactive prefill budget and consumes that request's chunk
   for the tick; the suffixes continue on the normal carried-prefill path next tick.
   Step35's dedicated PP batch core remains fresh-only and therefore does not batch
   those carried suffixes.
5. Convert each sibling's provisional admission miss into a cache hit. Per-request
   `cached_tokens`, global/tenant cached-token totals, hit/miss counters, hit-token mass,
   and the LCP histogram must all describe the final served path, not the provisional
   admission observation.

Prefixes longer than one interactive chunk are deduplicated one budget-capped prefix at
this stage; the remaining carried tail continues through the existing chunked prime
path. This preserves the scheduler's TTFT/QoS budget instead of introducing an
uncapped pre-prime.

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

## Increment 2 — simultaneous cache-meter gate registered

`tools/cache-meter-gate.py` now launches the five same-salt requests and one
cross-salt control behind one client barrier instead of teaching the prefix cache
sequentially. The required accounting is now:

- namespace A: one request reports `cached_tokens=0`, four report exactly K;
- namespace B: the same K-token prefix remains `cached_tokens=0`;
- aggregate: hits/misses/inserts = 4/2/2, hit-token mass = 4K, two cold LCP
  samples, four K-depth samples, and exact global/tenant prompt-vs-cached totals.

At the branch base this is a registered RED by construction: all six requests complete
their cache probes before any prefix entry exists, so all six compute their prefixes.
The existing serve-smoke arm invokes this gate with N=5, K=256.

## Increment 3 — refcounted prefix-entry leases

`PrefixEntry` now carries an in-flight pin count. The eviction index contains only
unpinned entries: first acquire removes the entry, intermediate releases only decrement
the count, and final release reinserts it at current recency. Normal budget eviction and
the session-allocation emergency flush therefore preserve entries held by live requests.

Every ordinary prefix-cache hit now takes a `(PoolKey, entry-id)` lease and stores it in
the `Session`. The one centralized retire sweep releases it before any completion,
disconnect, error, or OOM-park exit path can partially move the session.

Device-free checks (`DOCS_RS=1 cargo test -p memra-server prefix_cache_`) are green:
8 prefix-cache tests, including the new two-reference eviction test and the
emergency-flush pin test.

## Increment 4 — scheduler fanout and exact accounting

The interactive scheduler now runs a prefix-fanout stage immediately before fresh
prime-batch formation. Eligible sessions are cold prefix-cache misses with empty
session state, no spec/graph/LCP-split path, and at least 64 queued tokens. A pure
grouping function partitions them by exact `(model, cache_ns)` before comparing token
ids, then computes the full group LCP and caps it at the current interactive prefill
budget. An unmatched cold miss remains held only for the existing
`MEMRA_PRIME_BATCH_HOLD_MS` window, so an unrelated cross-salt arrival cannot launch it
before matching siblings reach the worker; a true singleton resumes normally when the
same 4 ms default window expires.

For each group the worker primes one leader prefix, snapshots its stage-owned
KV/recurrent state, deep-copies that snapshot into each sibling, and leaves only the
request-specific suffix queued. The shared entry is inserted with one lease per
successful participant; every ordinary completion, disconnect, error, or OOM-park
releases its lease through the existing centralized retire sweep.

Sibling admission probes are rewritten from provisional misses to final-path hits:
the original miss and LCP bucket are removed, hit count/token mass and the served
prefix bucket are added, and per-response, global, and tenant cached-token counters
receive exactly the shared prefix length. The leader remains the one computed miss.

`MEMRA_PREFIX_DEDUP=0` restores independent cold primes for the before/after receipt.
Host checks are green:

- `DOCS_RS=1 cargo check -p memra-server`;
- `prefix_fanout_` tests: exact same-key grouping, cross-model/tenant/salt isolation,
  prefill-budget cap, and one-for-one miss-to-hit histogram rewriting;
- tenant metering test: post-admission cached credit changes only the cached column.

## Increment 5 — box1 TTFT receipt harness

`fanout_ttft.py` launches one barrier-synchronized N=8 `/v1/completions` burst with
explicit token ids: K shared tokens plus a distinct 16-token suffix per request. It
timestamps the first non-empty SSE text frame, retains per-request prompt/cached-token
usage, and refuses an invalid comparison:

- rollback arm: all eight requests must report zero cached tokens;
- dedup arm: exactly one reports zero and seven report exactly K.

`run-box1.sh` runs both arms from the same binary with `MEMRA_PREFIX_DEDUP=0/1`,
PP-2 placement, spec and continuation reuse disabled, a separate warmup namespace,
thermal snapshots, raw server logs, and a single shared GPU-lock hold. The default
receipt geometry is N=8, K=1024, suffix=16.

`run-gates-box1.sh` composes the target-rig deliverables under a caller-held lock:
release builds, the intended 9B `serve-smoke` artifact pair (including the extended
cache-meter fanout), the API-key/PC-ISO battery, Step PP-2 `run-gen`, and the N=8 Step
TTFT A/B. Every command writes a raw log before its verdict is parsed.

## Final report

### Dedup and pin point

The dedup stage sits between admission and interactive fresh-prime batch formation.
Admission still allocates isolated session caches and records provisional prefix-cache
misses. Before those cold sessions enter `prime_cache_batch` or per-session prefill, the
stage groups exact token prefixes only inside one `(model, cache_ns)` pool, primes one
budget-capped common prefix, snapshots it, and restores it into each sibling.

The shared `PrefixEntry` is inserted with one lease per successful participant.
Ordinary later hits acquire the same lease type. Pinned entries are absent from the LRU
and emergency-flush victim set until the last centralized retire releases them; every
request continues to own a deep-copied mutable session cache.

On the measured N=8 burst, seven requests reached the grouping stage together:

```text
[prefix-dedup] B=7 prefix=1024 saved=6144 ... retained=true
```

The eighth arrived after publication and took the normal 1024-token cache hit. The
final served-path contract was therefore still exactly one computed prefix plus seven
cached prefixes. The rollback arm logged eight independent 1040-token seed inserts.

### Gate table

| gate | result | receipt |
|---|---:|---|
| host `memra-server` tests | **PASS**, 123/123 | local `DOCS_RS=1 cargo test -p memra-server` |
| prefix grouping/security/pinning units | **PASS**, same model+tenant+salt only; cross-model/tenant/salt excluded; refcount eviction and emergency flush green | host test output; implementation commits below |
| release build on box1 | **PASS**, CUDA 13.2, auto `sm_120a` | `raw/box1/build-server-20260808T141144Z.log` |
| simultaneous cache meter, N=5 + cross-salt | **PASS**, 1 computed + 4 cached, B cold, hits/misses/inserts `4/2/2`, exact usage/LCP/tenant/economics arithmetic | `raw/box1/serve-smoke-20260808T141144Z.log` |
| `serve-smoke` intended 9B pair | **PASS**, 0 failed; plain, concurrency, cache meter, spec identity, sampled truncation, affinity liveness | same receipt |
| API-key / PC-ISO | **PASS**, 18/18; same-tenant hit, both cross-tenant directions cold, within-tenant cross-salt cold then self-hit | `raw/box1/apikey-20260808T141144Z/apikey-gates.jsonl` |
| Step PP-2 `run-gen` | **MATCH** for prefill/decode and batched-prime/tokenwise argmax | `raw/box1/run-gen-20260808T141144Z.log` |
| Step N=8 fanout receipt validity | **PASS**, rollback cached `[0]x8`; default cached `[0]+[1024]x7` | `raw/box1/fanout-ttft-20260808T141637Z.jsonl` |

The optional gemma4 sub-arm in `serve-smoke` skipped because that artifact was not
present on box1; all applicable smoke arms passed.

### N=8 same-prefix TTFT

Box1, 2x RTX PRO 6000 Blackwell Server Edition, PP-2 devices 0/1, Step-3.7-Flash
IQ4_XS, one 1024-token shared prefix plus a distinct 16-token suffix per request,
8-token generation, one same-shape warmup per arm. This is one simultaneous
eight-request burst per arm; p50/p95 are the distribution across those eight requests,
not repeated-burst medians. Entry/exit snapshots were 0 MiB, 32-35 C; the initial
active clocks were 2407/2325 MHz.

| arm | cached-token distribution | TTFT min | TTFT p50 | TTFT p95/max |
|---|---:|---:|---:|---:|
| `MEMRA_PREFIX_DEDUP=0` | `[0] x 8` | 21.235 s | 22.263 s | 22.263 s |
| default ON | `[0] + [1024] x 7` | 3.851 s | 3.852 s | 3.853 s |

Default ON reduces burst p50 TTFT by **82.7%** (**5.78x**). Warmup TTFT was
2.933 s OFF and 2.947 s ON, so lazy initialization was balanced before each measured
burst. Raw client rows, server logs, binary/model hashes, and thermal snapshots are in
`research/prefixdedup-20260808/raw/box1/`.

### Commits

| commit | increment |
|---|---|
| `b4f03dd1` | current-path analysis, dedup/security/pinning design |
| `1a182f19` | simultaneous N=5 + cross-salt cache-meter gate registered |
| `98f50c1d` | refcounted entry leases and eviction protection |
| `d7fc97f5` | scheduler fanout, exact accounting, tenant/salt grouping |
| `1a7282ed` | streaming N=8 TTFT harness |
| `242ea324` | caller-held lock composition for the TTFT runner |
| `54871d97` | reproducible box1 gate battery |
| `31657510` | raw box1 correctness, isolation, and TTFT receipts |

No origin push, merge, tag, or release was performed.

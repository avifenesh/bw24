# v0.70.0 changelog draft (curated) — prepared 2026-08-05, lane/v070-prep

## Version argument: MINOR bump → v0.70.0

RELEASING.md's scheme: minor = "new mechanism or board move — kernel defaults changed,
model lane landed, published number moved." v0.70 moves mechanisms, plural:

- **H3 serve path** (`MEMRA_SERVE_B1FAST`, default ON) — a new default dispatch mechanism:
  solo serve ticks ride the m=1 fused trunk, +8.33%/+5.19% c=1 decode-only, closes task #70's
  plain lane.
- **Session affinity** (#71, `MEMRA_AFFINITY`, default ON) — a new serving mechanism:
  rewritten-history conversations resume at a turn checkpoint, TTFT 0.53 s flat vs
  11.9–13.4 s at 13k+ ctx (22–24x).
- **API-key tenant auth** (`MEMRA_API_KEYS`) — a new user-facing capability class
  (keyring, tenant-scoped cache namespaces, per-key caps, lifecycle CLI).
- **F5 spec-pool thrash fix** — default serving behavior change (evict-first + right-size
  ladder replace the every-turn evict/realloc churn).
- **Q8_0 m=1 gate+up fusion + SM-gated graph budget key** — kernel default changes
  (naked pod default 49.82 → 52.22 on the 188-SM board).

None of this is patch-class. **v0.70.0.**

## Prefix-classification verification (tools/changelog.sh v0.69.0..HEAD, dry-run 2026-08-05)

Raw output: `changelog-raw.txt` (same dir). Verified:

- `perf:`/`feat:`/`fix:`/`config:`/`docs:` → grouped into their sections. Correct.
- `data:`/`chore:`/`wip:`/`probe:` → dropped (23 commits filtered). Correct.
- Merge commits dropped. Correct. No commits landed in "Other" (all prefixes conform).
- **Curation needed at release time:** `docs(biz)` commits classify as public `docs:` and
  leak product-layer lines ("darklanes launch website spec", five PRODUCT-TRUTH lines,
  "SPEC §16.4", "website spec") into the Documentation section. The files they touched
  were later REMOVED from this repo (product-truth lane, d8e4a46d + d44652f4), so the
  lines advertise documents that no longer exist here. RELEASING.md's rule — "draft is
  floor, not ceiling; edit notes on GitHub afterwards" — covers it: **delete the biz
  lines from the published notes** (list below marked ~~strikethrough~~).

---

## The curated public changelog (v0.70.0)

Changes since v0.69.0:

### Performance

- Solo serve sessions ride the m=1 fused trunk — serve c=1 decode +8.33% (Qwen3.5-9B) /
  +5.19% (Qwen3.6-27B), bit-identical to the CLI decode path, c=8 unchanged
  (`MEMRA_SERVE_B1FAST`, default ON; closes the plain-serve c=1 gap).
- Q8_0 dense-FFN gate+up m=1 fusion (+0.94%, bit-identical) + SM-gated CUDA-graph budget
  key (48 at ≥180 SM: q8 +3.80% / nvfp4 +7.72% at n=128) — naked 27B pod default
  49.82 → 52.22 tok/s (+4.82%).

### Features

- Session affinity (#71): conversations whose history the client rewrote (think-stripping
  agent clients) resume at a retained turn checkpoint instead of re-priming in full —
  TTFT flat (~0.53 s) regardless of history vs 11.9–13.4 s at 13k+ context. Identity
  nominates (explicit `session_id`/`user`/`x-session-id`, or structural fingerprint);
  bytes decide (exact-prefix verification before any resume). `MEMRA_AFFINITY=0` rollback
  seam; serve-smoke check 10 gates the resume path.
- API-key management: multi-key tenant auth (`MEMRA_API_KEYS`) — file-backed keyring
  (SHA-256 at rest, hot-reload, fail-closed on a broken rewrite), `--gen-key`/`--revoke-key`
  CLI, tenant-scoped cache namespaces, per-key rate-limit overrides, batch/interactive lane
  classes, `[meter]` admit lines for billing joins. `MEMRA_API_KEY` (single static key)
  keeps working unchanged.
- `tools/merge-lora.py`: LoRA → merged BF16 HF checkpoint (state-dict-based with a
  key-parity assert; carries the MTP sidecar).

### Fixes

- Sampler: `top_p`/`min_p` truncation injected id-0 ('!') tokens on the speculative
  full-accept bonus path (it read a neighbour column's filter stats) — fixed; serve-smoke
  gains a sampled-truncation differential matrix (check 9) so a greedy-only battery can
  never miss this class again.
- F5 spec-pool thrash: pool misses on VRAM-tight rigs re-paid a doomed multi-GB alloc every
  turn (fail → evict → realloc) — replaced with learned evict-first + a right-size ladder;
  byte-identical output, churn eliminated.
- Session affinity root-cause fixes found while building it: the turn checkpoint sat one
  token past the prompt end (affinity was inert), and the implicit fingerprint is now a
  segment chain rather than one digest.
- `tools/local-ci.sh` now runs decode-batch-gate (config + Q8_0 strict) — the serving
  tick's exactness contract was guarded only on the H100 battery before.
- Resumable per-crate crates.io publish: skips versions already live, waits out the
  new-crate 429 burst limit, `publish=true` dispatch recovery door (the v0.69.0 first
  publish shipped 5/9 crates and could not resume).

### Documentation

- Serve surface: API-keys section + session-affinity section (SERVING.md), the honestly
  scoped isolation contract (token streams, not FP-program identity), and the
  chunked-prefill reduction-order finding (no gate may assert byte-equality across
  different prefill chunk splits).
- FLAGS.md: `MEMRA_SERVE_B1FAST`, `MEMRA_API_KEYS`, `MEMRA_AFFINITY` cataloged;
  `MEMRA_GS_MIN` re-provenanced with the measured do-not-lower verdict.
- Engine-only scope: product docs moved out to the private product repo; engine truth
  rehomed to docs/PERFORMANCE.md (§Rigs registry + the 27B serving board).
- Pre-release sweep: retired the pre-fix serve c=1 gap text, README serving capabilities
  updated, battery descriptions name decode-batch-gate + serve-smoke.

### Lines from the raw draft to DELETE at publish (product-layer leakage)

- ~~darklanes launch website spec — full build brief for the implementing agent~~
- ~~SPEC §16.4 — name the repo docs the build-agent may now trust~~
- ~~PRODUCT-TRUTH — close the two remaining 'needs fixing' notes now that the fixes landed~~
- ~~PRODUCT-TRUTH — record what the reconciliation fixed (§12.1) and narrow the stale-text hazard list~~
- ~~product claims come from docs/PRODUCT-TRUTH.md, updated in the same commit~~
- ~~un-stale the user-facing docs against docs/PRODUCT-TRUTH.md~~ (keep only if reworded engine-side)
- ~~correct the evidence inventory and blog source material against PRODUCT-TRUTH~~
- ~~correct the website spec against PRODUCT-TRUTH + fold in the three owner decisions~~
- ~~PRODUCT-TRUTH.md — one reconciled source for every product-facing claim~~

### NOT in these notes (correctly filtered as research-log noise, or gated on other lanes)

- FP8 v3-gate receipts (`data:`/`probe:` — research evidence, no shipped default change).
- The 23 `data:`/`chore:` commits (research receipts; the JSONL rows are the record).
- The F4 temperature/seed defaults shipped IN v0.69.0 (commit c716954b is an ancestor of
  the v0.69.0 tag) — do not re-announce them in v0.70.0.
- Whatever lane/nvfp4-strict merges will ADD to this range before the tag — re-run
  `bash tools/changelog.sh v0.69.0` on the final merged tree and fold its lines in.

Boards + reproduction artifacts: https://huggingface.co/Avifenesh/memra-bench · full
experiment log in research/tune-data/

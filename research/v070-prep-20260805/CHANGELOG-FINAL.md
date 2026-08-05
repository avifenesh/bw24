# v0.70.0 published release notes — FINAL (curated tag-day, 2026-08-05)

Supersedes `CHANGELOG-DRAFT.md` (prep-lane draft, written before fp8-decode-v1,
accept-telemetry, chunk-invariance, and nvfp4-strict merged). Raw generator output for the
final merged tree: `changelog-raw-final.txt` (`bash tools/changelog.sh v0.69.0` @ tag
candidate). The draft's docs(biz) delete-list is applied below (those lines are absent, not
struck).

Version argument unchanged from the draft: MINOR. v0.70 changes kernel/serving **defaults**
(native e4m3 residency, H3 m=1 serve trunk, session affinity, Q8_0 m=1 fusion + the SM-gated
graph key) and adds a user-facing capability class (API-key tenant auth). Not patch-class.

**The headline four:** native e4m3 default flip · H3 solo-serve fusion · session affinity ·
API-key tenant auth.

---

Changes since v0.69.0:

### Performance

- **Native FP8 e4m3 residency is now the default** for per-tensor-scale FP8 safetensors
  checkpoints — those projections keep raw e4m3 as the ONE resident copy instead of being
  re-encoded to a Q8_0 slab at load. Decode dequants e4m3 in-kernel (with fused pair/triple
  twins so the trunk stays fused); prefill's FP8 GEMM rides the same bytes. On the 27B FP8-ST:
  decode **+2.58%**, prefill **+3.25%** pp512, and **430 MiB freed** at a measured byte ratio
  of exactly 1.06250 against a theoretical 34/32 = 1.06250. `MEMRA_ST_E4M3=0` is the rollback
  seam. Scope: per-tensor scalar-scale only — block-128 and per-row FP8 keep the Q8_0
  re-encode, and GGUF is untouched.
- **Solo serve sessions ride the m=1 fused trunk** (H3) — serve c=1 decode **+8.33%**
  (Qwen3.5-9B) / **+5.19%** (Qwen3.6-27B), bit-identical to the CLI decode path, c=8
  unchanged. `MEMRA_SERVE_B1FAST`, default ON; closes the plain-serve c=1 gap.
- **Q8_0 dense-FFN gate+up m=1 fusion** (+0.94%, bit-identical) plus an **SM-gated CUDA-graph
  budget key** (48 at >=180 SM: q8 +3.80% / nvfp4 +7.72% at n=128; the 82-SM class keeps 256,
  where the same key measured -1.61%) — naked 27B default on the 188-SM rig 49.82 -> 52.22
  tok/s (+4.82%).

### Features

- **Session affinity** (#71): conversations whose history the client rewrote — think-stripping
  agent clients — resume at a retained turn checkpoint instead of re-priming in full. TTFT
  goes flat (~0.53 s) regardless of history length, against 11.9-13.4 s at 13k+ context.
  Identity nominates (explicit `session_id` / `user` / `x-session-id`, or a structural
  fingerprint chain); bytes decide (exact-prefix verification before any resume).
  `MEMRA_AFFINITY=0` rollback seam; serve-smoke check 10 gates the resume path.
- **API-key management**: multi-key tenant auth (`MEMRA_API_KEYS`) — file-backed keyring
  (SHA-256 at rest, hot-reload, fail-closed on a broken rewrite), `--gen-key` / `--revoke-key`
  CLI, tenant-scoped cache namespaces, per-key rate-limit overrides, batch/interactive lane
  classes, `[meter]` admit lines for billing joins. `MEMRA_API_KEY` (single static key) keeps
  working unchanged.
- **Spec-decode acceptance telemetry**: always-on per-draft-position counters surfaced two
  ways — a per-model `spec` block on `GET /metrics` (rounds / drafted / accepted,
  `acceptance_rate`, `tokens_per_round`, and `pos_drafted` / `pos_accepted` /
  `accept_rate_per_pos` arrays), and a per-request `usage.spec` summary on spec responses.
  Zero GPU syncs, zero per-token allocation, hot path untouched; the `/metrics` block is
  absent until a spec burst runs and `usage.spec` is absent on non-spec requests, so both
  payloads stay byte-identical for anyone not using spec decode (additive, OpenAI-safe).
- **Chunk-invariant prefill door** (`MEMRA_PRIME_INVARIANT=1`, opt-in): pins prefill split
  points to `MEMRA_PRIME_GRAIN` so segmentation no longer depends on `MEMRA_PRIME_CHUNK`,
  restoring bit-identical prefill logits across chunk sizes. Mechanism cost is inside noise
  (-0.05% / +0.17%, N=5 interleaved). Held opt-in deliberately: under the door
  `MEMRA_PRIME_CHUNK` no longer bounds the long-context transient footprint, so a default flip
  is gated on 27B long-context OOM + throughput evidence.
- `tools/merge-lora.py`: LoRA -> merged BF16 HF checkpoint (state-dict-based with a key-parity
  assert; carries the MTP sidecar).

### Fixes

- **Sampler truncation corruption**: `top_p` / `min_p` injected id-0 (`'!'`) tokens on the
  speculative full-accept bonus path — it read a neighbouring column's filter stats. Fixed.
  serve-smoke gains a sampled-truncation differential matrix (check 9) so a greedy-only
  battery can never miss this class again.
- **NVFP4 strict-mode decode parity**: three NVFP4 fused dual doors
  (`matmul_pre_dual_noscale`, `matmul_decode_exact_dual`, `_pre`) never consulted
  `mmvq_supports`, so with MMVQ disabled the oracle rode the MMVQ-dual path while the batched
  body fell back to dp4a — two different kernel classes compared against each other. The doors
  now obey the batched-iff-MMVQ decode-parity law. No new flag; default dispatch unchanged.
- **F5 spec-pool thrash**: pool misses on VRAM-tight rigs re-paid a doomed multi-GB allocation
  every turn (fail -> evict -> realloc). Replaced with learned evict-first plus a right-size
  ladder, with fallible embed residency and a transient-reserve probe on new landings.
  Byte-identical output, churn eliminated.
- Session-affinity root-cause fixes found while building it: the turn checkpoint sat one token
  past the prompt end (affinity was inert), and the implicit fingerprint is now a segment chain
  rather than one digest.
- `tools/local-ci.sh` now runs decode-batch-gate (config + Q8_0 strict + NVFP4 strict) and the
  chunk-invariance arms — the serving tick's exactness contract was guarded only on the H100
  battery before, and both new gates carry injected-break canary teeth so they cannot pass
  vacuously.
- `tools/fast-gate` self-gating (`kind=cmd`) probes reported PASS when the underlying gate
  script SKIPped for a missing model — a SKIP and a real pass both exit 0, so the gate could
  pass vacuously on any rig lacking the artifact. Probes now read the script's own verdict word
  and report SKIP as SKIP.
- Resumable per-crate crates.io publish: skips versions already live, waits out the new-crate
  429 burst limit, and gains a `publish=true` dispatch recovery door (the v0.69.0 first publish
  shipped 5 of 9 crates and could not resume).

### Documentation

- Serve surface (SERVING.md): API-keys section, session-affinity section, the spec-acceptance
  telemetry contract (`/metrics` block format, per-position semantics, `usage.spec`,
  reset-on-load), and the honestly scoped isolation contract (token streams, not FP-program
  identity).
- **Chunked-prefill root cause corrected.** SERVING.md previously attributed
  chunk-split-dependent greedy output to prefill GEMM reduction order. That is measurably
  wrong: the prefill GEMM is m-invariant (rows bit-identical across m). The real cause is a
  numeric-**class** edge in `full_attn_prime_fa_dispatch` selected by `base_len == 0` — chunk 0
  attends f32 K/V, every later chunk attends the quantized KV cache — so `MEMRA_PRIME_CHUNK`
  decides at which token position the precision edge falls (`first_div_pos` equals the chunk
  size exactly, 4 of 4 arms). Section rewritten to the measured mechanism.
- FLAGS.md: `MEMRA_ST_E4M3` (default-on, with the flip receipts and the per-tensor-only scope),
  `MEMRA_E4M3_DUAL`, `MEMRA_PRIME_INVARIANT`, `MEMRA_PRIME_GRAIN`, `MEMRA_SERVE_B1FAST`,
  `MEMRA_API_KEYS`, `MEMRA_AFFINITY` cataloged; `MEMRA_GS_MIN` re-provenanced with the measured
  do-not-lower verdict; `MEMRA_PP_FP8` re-scoped as largely superseded for the per-tensor class.
- Engine-only scope: product docs moved out to the private product repo; engine truth rehomed
  to docs/PERFORMANCE.md (Rigs registry + the 27B serving board), with the llama-freeze banner
  and rig labels.
- Pre-release sweep: retired the pre-fix serve c=1 gap text, README serving capabilities
  updated, battery descriptions name decode-batch-gate + serve-smoke.

Boards + reproduction artifacts: https://huggingface.co/Avifenesh/memra-bench · full
experiment log in research/tune-data/

---

## Curation applied vs the raw generator output

**Deleted (product-layer leakage — `docs(biz)` classifies as public `docs:`, but the files
these lines describe were REMOVED from this repo by the product-truth lane, so the lines
advertise documents that no longer exist here).** Per RELEASING.md, "draft is floor, not
ceiling":

- darklanes launch website spec — full build brief for the implementing agent
- SPEC §16.4 — name the repo docs the build-agent may now trust
- PRODUCT-TRUTH — close the two remaining 'needs fixing' notes now that the fixes landed
- PRODUCT-TRUTH — record what the reconciliation fixed (§12.1) and narrow the stale-text
  hazard list
- product claims come from docs/PRODUCT-TRUTH.md, updated in the same commit
- un-stale the user-facing docs against docs/PRODUCT-TRUTH.md
- correct the evidence inventory and blog source material against PRODUCT-TRUTH
- correct the website spec against PRODUCT-TRUTH + fold in the three owner decisions
- PRODUCT-TRUTH.md — one reconciled source for every product-facing claim

**Folded / rewritten** (raw lines are commit subjects; published notes are user-facing):

- The nine affinity slice commits (slices 1/2/4/5/7/8 + the two root-cause fixes) collapse into
  one Features entry plus one Fixes entry.
- The two accept-telemetry commits (SpecSession counters + the serve surface) collapse into one
  Features entry.
- The e4m3 pair (default flip + launch-fusion twins) collapses into the headline Performance
  entry, with the prefill and VRAM halves added — the generator's subject lines carry neither.
- The chunk-invariance door was raw-classified as a Documentation comment fix; it is promoted to
  Features (the user-visible door) with the mechanism correction kept in Documentation.
- Internal-only subject lines rewritten for outside readers ("HANDOVER's NV-27B ST pp1855
  line names which arm is now 'default'", "E4M3-DIRECT arm comment records the default flip",
  "genericize the lab name") — dropped as internal bookkeeping, no user-facing content.

**Correctly filtered by the generator** (`data:` / `chore:` / `wip:` / `probe:` — research
receipts; the JSONL rows are the record): the sweep-audits upstream-trap audits, the MTP-p2
phase-2 spec tier data, the 2026-08-05 upstream sweep, the FP8 v3-gate receipts, and every
battery-receipt commit. Also NOT re-announced: the F4 temperature/seed defaults, which shipped
IN v0.69.0 (commit c716954b is an ancestor of that tag).

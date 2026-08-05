# Darklane sync note — what the private repo owes after memra v0.70.0

Written on the engine side (tag day, 2026-08-05) as a **handoff**. Nothing in
`~/projects/darklanes` was touched by this lane — that repo is another owner surface. This
file is the checklist the private side works from; each item names the engine-side fact so the
private edit does not have to re-derive it.

Engine release: **v0.70.0**, tagged 2026-08-05 off `main`. Headline four: native e4m3
residency default flip · H3 solo-serve m=1 fusion · session affinity · API-key tenant auth.
Full notes: `research/v070-prep-20260805/CHANGELOG-FINAL.md`.

## 1. Version pins

- Any pinned memra version or commit in serve configs, deployment docs, and the pill's
  local-memra default (`:8002` serve scripts) moves to **v0.70.0**.
- Crate consumers (if any private crate depends on the published workspace) move to `0.70.0` —
  all nine crate names are live on crates.io; v0.70.0 uploads are version-updates, not
  new-crate publishes.
- Darklanes' internal state docs record the engine release (version, date, headline
  mechanisms) per the release-discipline standing rule.

## 2. Serve-claim updates (the substantive part)

Three claims in product copy are now stale in the *conservative* direction — the engine got
better, so the copy understates it. Update, do not just delete:

**a. Serve c=1 gap — CLOSED.** Any copy quoting the **-11.74%** serve-vs-CLI gap (PRODUCT-TRUTH
successors, website spec §perf) is describing pre-fix behavior. Post-H3, plain serve c=1 rides
the m=1 fused trunk: **+8.33%** (Qwen3.5-9B) / **+5.19%** (Qwen3.6-27B) decode-only at c=1, and
serve c=1 now sits level with the same-board `run-gen` denominator. Bit-identical to the CLI
decode path; c=8 unchanged.

The honest open cell to KEEP: the **NVFP4 spec-serve path is still -8.66%** (its burst loop is a
separate path, not covered by H3). Do not let the "gap closed" line generalize to the spec
serve lane.

**b. Flat TTFT is now a product claim.** Session affinity (default ON) makes the owner's daily
driver workload — a think-stripping agent client that rewrites conversation history every turn —
resume at a retained turn checkpoint instead of re-priming. TTFT goes **flat at ~0.53 s
regardless of history length**, against **11.9-13.4 s at 13k+ context** before. That is a
22-24x latency claim on the exact workload the product serves, and it belongs in the latency
copy.

Config requirement: **confirm the serve configs do NOT set `MEMRA_AFFINITY=0`.** The flat-TTFT
claim is false with the rollback seam engaged. Worth an explicit assertion in whatever validates
the deployed env.

**c. Native FP8 e4m3 residency is the default.** For per-tensor-scale FP8 safetensors
checkpoints, memra no longer re-encodes to a Q8_0 slab at load: raw e4m3 is the one resident
copy. Product-relevant consequences: decode **+2.58%**, prefill **+3.25%** pp512, and **430 MiB
less VRAM** on the 27B FP8-ST. The VRAM number matters for SKU sizing copy — the FP8 path is now
*cheaper* in memory than the slab it replaced, which refutes the earlier "27B FP8 e2e does not
fit 24GB" framing if that appears anywhere private.

Scope caveat to carry, because it is a 3.8 risk: **per-tensor scalar scale only.** Block-128 and
per-row FP8 still take the Q8_0 re-encode. If Qwen 3.8 ships FP8 as block-128 (as 3.6 did in the
official checkpoints), this default does not apply to it without the fold arm.

**d. Spec acceptance is now observable in production.** `GET /metrics` carries a per-model `spec`
block (rounds / drafted / accepted, `acceptance_rate`, `tokens_per_round`, and per-draft-position
`pos_drafted` / `pos_accepted` / `accept_rate_per_pos` arrays), and every spec response carries
`usage.spec` with its own rounds/drafted/accepted/`acceptance_rate`. Both are additive and
absent when spec is not in play, so nothing existing breaks.

Product use: this is the gauge for the sampled-vs-greedy acceptance gap (measured 0.53 sampled
vs 0.64 greedy on short context) — a live per-request signal instead of a posthoc dig. Worth
wiring into whatever dashboard the private side runs.

## 3. Metering join — verify against a v0.70.0 binary

API keys are the launch product piece this release ships FOR. The join contract:

- The public server emits `[meter] admit id=<x-request-id> tenant=<t> lane=<l> model=<m>` at
  admit time.
- The private metering layer joins those admit lines against worker-truth **usage** lines by
  request id.

Owed on the private side:

1. Re-run that join against a **v0.70.0** binary and confirm the id correspondence still holds
   end-to-end (the admit line is public-repo, the aggregation is private — this is exactly the
   seam that rots silently).
2. Note that `usage.spec` is NEW in the usage object on spec requests. If the metering
   deserializer is strict about unknown fields, it needs the field added; if it is lenient, it
   should be *taught* the field so spec rounds can be billed/attributed rather than ignored.
3. Provision the real tenant keyring for the launch endpoints via `--gen-key` (SHA-256 at rest,
   plaintext shown once). Decide per-key lane class (batch vs interactive — batch-class keys are
   403'd on `x-lane: interactive`) and per-key rate-limit overrides before the endpoints go live.
4. Keyring operational notes worth recording privately: hot-reload picks up edits within one
   poll (<=2 s on a running server), `--revoke-key` takes effect on that same poll, and a
   broken keyring rewrite **fails closed** (no accidental open door). Sibling keys survive a
   revoke.

## 4. OpenRouter application refresh

The application package cites repo receipts. Refresh: the cited engine version (v0.70.0) and the
serve-surface checklist — **API keys now exist**, so any listing requirement line about
multi-tenant authentication flips to done. The serve-surface items that were pending on auth can
be re-read against the shipped keyring.

## 5. Explicitly NOT owed

- No board/number regen on the private side from this release: the tracked perf boards are
  GGUF/bare-CLI cells, and v0.70's movers are ST-class (e4m3) or serve-path (H3, affinity),
  neither of which appears in `current-board.json`. The engine-side board `--check` is green
  with no regeneration.
- The chunk-invariance door (`MEMRA_PRIME_INVARIANT`) is opt-in and should NOT be enabled in
  serve configs — under the door `MEMRA_PRIME_CHUNK` stops bounding the long-context transient
  footprint, and the 27B long-context OOM gate is still owed. It is a reproducibility tool, not
  a serving setting.

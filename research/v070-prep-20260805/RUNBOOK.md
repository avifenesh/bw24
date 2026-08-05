# v0.70.0 tag-day runbook — prepared 2026-08-05, lane/v070-prep (CPU-only prep; no GPU work here)

Everything below assumes this lane's two prep commits are merged along with the release:
the docs sweep + this research dir, and the staged `0.69.0 → 0.70.0` version-bump commit
(workspace Cargo.toml + Cargo.lock — see git log of `lane/v070-prep`). The bump commit is
**not pushed** by the prep lane; it rides the release merge.

Ordered checklist. Nothing after step 0 starts until step 0 is true.

## 0. GATE: lane/nvfp4-strict merged green

v0.70.0 is gated on the nvfp4-strict lane (the strict-mode equalization coverage gap found
in serve-path phase 2: `decode-batch-gate --mode strict`'s equalizing env is Q8/dp4a-shaped
and does not equalize NVFP4's fused arms — documented in the gate header and
`research/servepath-p2-20260805/RESULTS.md`). Do not tag without it — owner call.

- [ ] lane/nvfp4-strict merged to the train with its own battery receipts.
- [ ] If it changed gate semantics or local-ci wiring, re-read step 2's commands against
      the merged tree before running them (the stale-verdict law applies to runbooks too).

## 1. Merge the prep lane

- [ ] `git merge lane/v070-prep` into the train (restructure/public-split), resolving
      against whatever nvfp4-strict touched. The version-bump commit must survive the merge
      with `[workspace.package].version = "0.70.0"` and all eight `=0.70.0` pinned
      workspace deps intact (`grep -c '=0.70.0' Cargo.toml` → 8).
- [ ] Fold the nvfp4-strict merge's public commits into
      `research/v070-prep-20260805/CHANGELOG-DRAFT.md` — re-run
      `bash tools/changelog.sh v0.69.0` on the merged tree and diff against the draft.

## 2. Batteries, both rigs (GPU steps — need the lock and the free card)

**Rig A — local RTX 5090 Laptop (82 SM), the default-flip gate rig.** All under
`flock /tmp/gpu5090.lock`, no co-resident GPU compute (local-ci warns; the window check is
part of the evidence):

```bash
gpu-full-power on                      # verify before any timed cell
tools/local-ci.sh                      # correctness stage: kernel-check ALL GREEN, prime-gate,
                                       #   run-gen argmax (31B + 12B gemma arms — these two NEED
                                       #   the free card: the daily driver must be DOWN, they load
                                       #   real 31B/12B weights), VERIFY-GATE, spec self-consistency,
                                       #   decode-batch-gate (NVFP4 config B=8 + Q8_0 config B=8 +
                                       #   Q8_0 strict B=4 equalized), serve-smoke (checks 1-10 incl.
                                       #   the new sampled-truncation matrix 9 + affinity 10)
tools/local-ci.sh --perf               # full perf cell battery (~15 min) — acceptance drift teeth
tools/serve-st-gate.sh                 # ST-dir serve exactness (items 1-4; CLI-vs-server + spec-vs-oracle)
tools/apikeys-gate.sh                  # the v0.70 API-keys live battery (18 checks, two-tenant proof)
```

run-spec K=1..8 self-consistency rides inside local-ci's spec arms; if any release-affected
model needs an explicit pass, `MEMRA_SPEC_K` sweep per CONTRIBUTING §1 on the MTP-capable
artifact.

- [ ] local-ci correctness: GREEN (0 skips on the arms the release touches — a SKIP on the
      31B/12B gemma arms means the card was not free; rerun, don't hand-wave).
- [ ] local-ci --perf: 0 FAIL (WARNs get read, not ignored).
- [ ] serve-st-gate: all items PASS.
- [ ] apikeys-gate: 18/18.

**Rig B — the 188-SM pod (`pro6000wk-runpod` class), the SM-gated key's home.** v0.70
ships the SM-gated graph budget key (48 at >=180 SM) and the Q8_0 m=1 fusion measured
there; the release battery re-confirms on that silicon:

```bash
# on the pod, same commit as the tag candidate:
./target/release/kernel-check <27B.gguf>                       # ALL GREEN
./target/release/run-gen  <27B nv + q8 artifacts>              # argmax MATCH both
./target/release/run-spec <27B NVFP4-MTP>                      # K=1..3 gate (this rig's standing
                                                               #   form; K=1..8 lives on the community
                                                               #   board — do not claim K=1..8 here)
tools/serve-smoke.sh <27B NVFP4-MTP> [draft]                   # serve surface on the serving SKU
```

- [ ] pod battery GREEN. If an H100 is in play for this release (Hopper-lane code touched),
      additionally `tools/validate-h100.sh <model.gguf> --quick` on the H100 box — v0.70's
      diff is serve/sampler/Q8-fusion; check the merged nvfp4-strict diff for sm_90a
      touches before deciding.

## 3. Perf board — regen only if a published number moved

Prep-lane verdict (receipt below): **H3 does NOT move any published board number.**
`research/tune-data/current-board.json` carries plain_decode / speculative /
plain_decode_depth / samples / supported_models / h100_board / extra_card_rows — all
bare-CLI (`run-gen`) and H100 cells. No serve-path number is generated anywhere; the 27B
serving board in docs/PERFORMANCE.md is hand-written prose OUTSIDE the PERF-* markers, and
`update-perf-board.py --check` is green on this tree.

- [ ] If (and only if) tag-day re-measures moved a *bare-CLI* tracked cell:
      edit `research/tune-data/current-board.json`, then
      `python3 tools/update-perf-board.py`, commit JSON + README.md +
      docs/PERFORMANCE.md + both SVGs in the same commit as the number-moving change.
- [ ] Optional, fresh-receipts-only: re-measure the 188-SM serve c=1 cell post-H3 and
      update the hand-written 27B serving board row (the pre-fix 46.09 is labeled history
      in the docs; a fresh N=5 median with rig label replaces it honestly). NOT a
      tag-blocker.
- [ ] Either way: `python3 tools/update-perf-board.py --check` green before push (the
      pre-push hook enforces; never `--no-verify`).

## 4. Main merge + push

- [ ] Merge the train (restructure/public-split, now carrying nvfp4-strict + v070-prep +
      battery receipt commits) into `main` — full docs sweep already done in prep, but eyeball
      `git diff main..HEAD -- README.md docs/` once for merge damage.
- [ ] `git push origin main` (pre-push hook runs board --check + perf-ci recency).

## 5. Tag

- [ ] Confirm `grep '^version' Cargo.toml` → `0.70.0` on the pushed main HEAD
      (publish.yml refuses a tag/version mismatch).
- [ ] `git tag v0.70.0 && git push origin v0.70.0`

## 6. Watch the two workflows

- [ ] `release` — prebuilt matrix (glibc 2.35/2.39 x sm_120a/sm_90a/sm_89), changelog
      draft, tarballs + SHA256SUMS + the stable-name binstall artifact. When it drafts:
      replace the raw notes with the curated changelog
      (`research/v070-prep-20260805/CHANGELOG-DRAFT.md`), DELETING the product-layer
      `docs(biz)` lines listed there — draft is floor, not ceiling.
- [ ] `publish` — per-crate in dependency order. Expected behavior on this SECOND release:
      all 9 crate names already exist on the registry (5 published at 0.69.0 + 4 published
      by the recovery run — verify, do not assume: the standing rule is "check the
      registry"), so the new-crate burst limit should NOT bite; version 0.70.0 uploads are
      update-publishes. If a 429 still lands: the workflow waits it out (620 s x 6); if the
      run dies partway, rerun via Actions → publish → Run workflow with `publish=true` on
      the tag ref — it skips versions already live and finishes the remainder.
- [ ] Post-publish sanity: `cargo search memra-server` shows 0.70.0;
      `curl -fsSL .../tools/install.sh | sh` on a scratch box pulls the v0.70.0 tarball and
      `kernel-check` boots.

## 7. Darklane sync note (private repo, post-release)

What `~/projects/darklanes` needs after v0.70.0 is public:

- **Version references**: any pinned memra version/commit in serve configs, the pill's
  local-memra default (`:8002` serve scripts), and deployment docs move to v0.70.0.
- **API keys**: the launch product piece this release ships FOR. The private metering
  layer joins `[meter] admit id/tenant/lane/model` lines against worker-truth usage lines
  by request id — wire/verify that join against a v0.70.0 binary, and provision the real
  tenant keyring (`--gen-key`) for the launch endpoints.
- **Session affinity**: the owner's daily driver (think-stripping client) is the exact
  workload; confirm the serve configs do NOT set `MEMRA_AFFINITY=0` and note the TTFT-flat
  behavior in the product latency claims.
- **Serve c=1 claims**: any product copy quoting the −11.74% serve gap (PRODUCT-TRUTH
  successors, website spec §perf) updates to the closed-gap state — plain serve c=1 now
  rides the m=1 trunk; the NVFP4 spec-serve −8.66% stays the honest open cell.
- **OpenRouter application**: the application package cites repo receipts; refresh the
  cited version and the serve-surface checklist (API keys now exist — a listing
  requirement line may flip to done).
- **Changelog propagation**: darklanes' internal state docs record the engine release
  (v0.70.0, date, headline mechanisms) per the release-discipline standing rule.

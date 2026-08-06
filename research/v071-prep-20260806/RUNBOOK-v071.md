# v0.71.0 tag-day runbook — prepared 2026-08-06, lane/v071-prep (CPU-only prep; no GPU work here)

Everything below assumes this lane's prep commits are merged along with the release:
the flag-kill commit (MEMRA_PRIME_INVARIANT/GRAIN door removed), the docs sweep, this
research dir, and the staged `0.70.0 → 0.71.0` version-bump commit (workspace Cargo.toml +
Cargo.lock — see git log of `lane/v071-prep`). The bump commit is **not pushed** by the
prep lane; it rides the release merge.

Ordered checklist. Same shape as `research/v070-prep-20260805/RUNBOOK.md` (it worked).

## 0. GATE: none standing

No equivalent of v0.70's nvfp4-strict gate is open for v0.71 as of prep time. If an
owner-flagged blocker lands between prep and tag, add it here before running step 1
(the stale-verdict law applies to runbooks too — re-read the merged tree first).

One flag-kill rides this release: the `MEMRA_PRIME_INVARIANT`/`MEMRA_PRIME_GRAIN` door
is REMOVED in this lane (superseded by the grain-free fix). The battery below is the
regression proof for that removal — the chunkinv/chunkinvc arms specifically.

## 1. Merge the prep lane

- [ ] `git merge lane/v071-prep` into the train (restructure/public-split). The
      version-bump commit must survive the merge with
      `[workspace.package].version = "0.71.0"` and all eight `=0.71.0` pinned workspace
      deps intact (`grep -c '=0.71.0' Cargo.toml` → 8).
- [ ] If more lanes merged after a85135ae, re-run `bash tools/changelog.sh v0.70.0` on the
      merged tree and diff against `research/v071-prep-20260806/CHANGELOG-DRAFT.md` —
      fold, don't overwrite. Re-run the docs(biz) leak-grep on the new commits.

## 2. Batteries, both rigs (GPU steps — need the lock and the free card)

**Rig A — local RTX 5090 Laptop (82 SM), the default-flip gate rig.** All under
`flock /tmp/gpu5090.lock`, no co-resident GPU compute (local-ci warns; the window check is
part of the evidence):

```bash
gpu-full-power on                      # verify before any timed cell
tools/local-ci.sh                      # correctness stage: kernel-check ALL GREEN, prime-gate,
                                       #   run-gen argmax (31B + 12B gemma arms — these two NEED
                                       #   the free card: the daily driver must be DOWN),
                                       #   VERIFY-GATE, spec self-consistency, decode-batch-gate
                                       #   (NVFP4 config B=8 + Q8_0 config B=8 + strict B=4
                                       #   equalized x2), graph-warmup stress (NEW this release:
                                       #   the =1 default's adversarial gate — pool-growth x10 +
                                       #   overlap + canary), serve-smoke
tools/fast-gate/fast-gate.sh --tier 1  # the three v0.71 arms explicitly: chunkinv (byte-identity
                                       #   naked — the flip's contract), chunkinvc (canary MUST
                                       #   break on injected legacy arithmetic), gwstress; plus
                                       #   k27 with its pinned MEMRA_FA_SPLIT=8 env (golden is
                                       #   rig-portable now)
tools/local-ci.sh --perf               # full perf cell battery (~15 min) — acceptance drift teeth
tools/serve-st-gate.sh                 # ST-dir serve exactness — v0.71 flips block-128 native
                                       #   residency: this gate is the ST-class regression proof
tools/apikeys-gate.sh                  # 18 checks, two-tenant proof (unchanged since v0.70)
```

run-spec K=1..8 self-consistency rides inside local-ci's spec arms; the graph-warmups and
chunkinv flips were both battery-green at merge — a FAIL here on those arms is a
regression, not a known state.

- [ ] local-ci correctness: GREEN (0 skips on the arms the release touches — a SKIP on the
      31B/12B gemma arms means the card was not free; rerun, don't hand-wave).
- [ ] chunkinv PASS **and** chunkinvc PASS (the canary must detect the injected break —
      a chunkinvc "pass-through" means the gate lost its teeth, treat as FAIL).
- [ ] gwstress GREEN (10/10 + overlap arm; the canary caught).
- [ ] local-ci --perf: 0 FAIL (WARNs get read, not ignored).
- [ ] serve-st-gate: all items PASS (block-128 native default's serve-surface proof).
- [ ] apikeys-gate: 18/18.

**Rig B — the 188-SM pod (`pro6000wk-runpod` class), the second rig.** v0.71 carries the
k27 FLIP-NEARTIE closure (the v0.70 release battery's open red) — the pod battery is
where that verdict gets its regression check:

```bash
# on the pod, same commit as the tag candidate:
./target/release/kernel-check <27B.gguf>                       # ALL GREEN
./target/release/run-gen  <27B nv + q8 artifacts>              # argmax MATCH both
./target/release/run-spec <27B NVFP4-MTP>                      # K=1..3 gate (this rig's standing
                                                               #   form; K=1..8 lives on the
                                                               #   community board — do not claim
                                                               #   K=1..8 here)
MEMRA_FA_SPLIT=8 tools/fast-gate/fast-gate.sh ... k27          # the pinned-split k27 golden must
                                                               #   now MATCH on this rig (that is
                                                               #   the closure's whole point);
                                                               #   k27div-probe is the localizer
                                                               #   if it doesn't
tools/serve-smoke.sh <27B NVFP4-MTP> [draft]                   # serve surface on the serving SKU
```

- [ ] pod battery GREEN, including the pinned-split k27 golden MATCH.
- [ ] H100: v0.71's diff is serve/prefill-path/graph/FP8-ST — check the merged diff for
      sm_90a touches; if any, `tools/validate-h100.sh <model.gguf> --quick` on the H100
      box. (The graph-warmups flip is arch-generic — if in doubt, run it.)

## 3. Perf board — regen only if a published number moved

Prep-lane verdict (receipt: `BOARD-VERDICT.md` in this dir): **the pile does NOT move any
published board number.** current-board.json carries bare-CLI + H100 cells only; every
serve/felt-latency number lives in hand-written prose (updated in the prep docs sweep).
`update-perf-board.py --check` is green on this tree.

- [ ] If (and only if) tag-day re-measures moved a *bare-CLI* tracked cell:
      edit `research/tune-data/current-board.json`, then
      `python3 tools/update-perf-board.py`, commit JSON + README.md +
      docs/PERFORMANCE.md + both SVGs in the same commit as the number-moving change.
- [ ] Either way: `python3 tools/update-perf-board.py --check` green before push (the
      pre-push hook enforces; never `--no-verify`).

## 4. Main merge + push

- [ ] Merge the train (restructure/public-split, now carrying v071-prep + battery receipt
      commits) into `main` — docs sweep already done in prep, but eyeball
      `git diff main..HEAD -- README.md docs/` once for merge damage.
- [ ] `git push origin main` (pre-push hook runs board --check + perf-ci recency).

## 5. Tag

- [ ] Confirm `grep '^version' Cargo.toml` → `0.71.0` on the pushed main HEAD
      (publish.yml refuses a tag/version mismatch — the guard reads the workspace
      version off cargo metadata).
- [ ] `git tag v0.71.0 && git push origin v0.71.0`

## 6. Watch the two workflows

- [ ] `release` — prebuilt matrix (glibc 2.35/2.39 x sm_120a/sm_90a/sm_89), changelog
      draft, tarballs + SHA256SUMS + the stable-name binstall artifact. When it drafts:
      replace the raw notes with the curated changelog
      (`research/v071-prep-20260806/CHANGELOG-DRAFT.md`, re-folded per step 1) — draft is
      floor, not ceiling.
- [ ] `publish` — per-crate in dependency order. THIRD release through this workflow:
      all 9 crate names exist on the registry with two published versions each (verify,
      don't assume: the standing rule is "check the registry"), so 0.71.0 uploads are
      routine update-publishes — the new-crate burst limit does not apply to updates. If
      a 429 still lands: the workflow waits it out (620 s x 6); if the run dies partway,
      rerun via Actions → publish → Run workflow with `publish=true` on the tag ref — it
      skips versions already live and finishes the remainder.
- [ ] Post-publish sanity: `cargo search memra-server` shows 0.71.0;
      `curl -fsSL .../tools/install.sh | sh` on a scratch box pulls the v0.71.0 tarball
      and `kernel-check` boots.

## 7. Darklane sync note (private repo, post-release)

What `~/projects/darklanes` needs after v0.71.0 is public:

- **Version references**: pinned memra version/commit in serve configs, the pill's
  local-memra default (`:8002` serve scripts), deployment docs → v0.71.0.
- **Felt-latency claims**: the product latency story changes materially this release —
  solo first text 0.12 s / contended 0.15 s at any burst (was 0.41/1.60). Any product
  copy or website-spec §perf quoting the old streaming/contended numbers updates; the
  B128 throughput-tier option (+8.5% for one 29 ms quantum) is now quotable for batch
  endpoints.
- **Serve configs**: confirm none set `MEMRA_SSE_PER_BURST=1` or `MEMRA_ADMIT_YIELD=0`
  (the defaults are the fixes); pure-batch tiers may deliberately set `ADMIT_YIELD=0` —
  document per config if so. Kill any `MEMRA_PRIME_INVARIANT`/`MEMRA_PRIME_GRAIN` usage
  in configs — the door no longer exists (it was opt-in; naked configs unaffected).
- **3.8 day-one runbook**: block-128 native is default — the flagship FP8 checkpoint
  needs NO flags; the qwen38 bringup runbook already carries the §3b correction, verify
  the darklanes copy matches.
- **Changelog propagation**: darklanes' internal state docs record the engine release
  (v0.71.0, date, headline mechanisms) per the release-discipline standing rule.

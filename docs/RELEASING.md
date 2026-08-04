# Releasing

Every board-moving or user-facing change gets a tagged release — that's the public change record.

## Version scheme

- **minor** (v0.X.0): new mechanism or board move — kernel defaults changed, model lane landed, published number moved.
- **patch** (v0.x.Y): fixes, docs, tooling.
- No retirement notes or migration prose — state current truth plainly.

## The gate (on the rig, before tagging)

GitHub CI is compile-only (no GPU). The release gate runs locally and must be green on the tagged commit:

```bash
./target/release/kernel-check <27B.gguf>          # ALL GREEN
./target/release/run-gen  <each affected model>    # prefill/decode argmax MATCH
./target/release/run-spec <each affected model>    # K=1..8 self-consistency PASS
```

If a published number moved: update `research/tune-data/current-board.json`, run `tools/update-perf-board.py`, commit the regenerated README + docs/PERFORMANCE.md + SVGs with the change (the pre-push hook enforces this).

## Cutting the release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Before tagging: bump `[workspace.package].version` AND the pinned `[workspace.dependencies]`
versions in the root `Cargo.toml` to `X.Y.Z` (one sed pass — they must match the tag;
`publish.yml` refuses a mismatched tag).

That's it. Two workflows fire on the tag:

- `release` — builds the prebuilt binary matrix (glibc 2.35/2.39 x sm_120a/sm_90a/sm_89,
  fatbins embedded — self-contained), drafts the changelog from conventional commits since
  the previous tag (`tools/changelog.sh` — `perf:`/`feat:`/`fix:`/`config:`/`docs:` grouped;
  `data:`/`chore:` dropped as research-log noise), attaches tarballs + `SHA256SUMS` + the
  stable-name `memra-server-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` (the cargo-binstall /
  `tools/install.sh` target), and publishes the GitHub release. Edit notes on GitHub
  afterwards if the draft needs headline or context — draft is floor, not ceiling.
- `publish` — per-crate `cargo publish -p <crate> --locked` to crates.io in dependency
  order, skipping versions already live (registry API check) and waiting out crates.io's
  new-crate burst limit on 429 (~1 new crate/10 min; learned on the v0.69.0 first publish,
  which shipped 5/9 before a 429 and could not resume under the old all-or-nothing
  `--workspace` form). The step is idempotent — rerunning a partly-published tag finishes
  the remainder. Dry-run first without tagging: Actions → publish → Run workflow
  (`workflow_dispatch` runs the full package+verify with no upload and no token); the
  dispatch `publish=true` input is the recovery door that runs the REAL publish on a ref
  when a tag's run needs finishing.

## Publishing to crates.io — one-time setup (owner)

- crates.io account (GitHub login), email verified.
- Generate an API token at <https://crates.io/settings/tokens> with `publish-new` +
  `publish-update` scopes.
- Add it as the repo secret `CARGO_REGISTRY_TOKEN` (Settings → Secrets and variables →
  Actions). The publish workflow fails with a pointed error if it's missing.
- The first tagged publish claims all nine `memra-*` crate names (verified available
  2026-08-04, `research/crates-release-20260804/`). `memra-probe` stays unpublished
  (`publish = false` — dev spike).
- Publishing is irreversible (yank ≠ delete): the tag must already have passed the on-rig
  gate battery like any release.

Preview the draft locally before tagging:

```bash
bash tools/changelog.sh            # previous tag -> HEAD
bash tools/changelog.sh v0.1.0     # explicit range
```

## Commit prefixes that feed the changelog

`perf:` kernel/throughput wins · `feat:` new capability · `fix:` correctness/bugs · `config:` defaults/flags · `docs:` documentation · `data:` tune-data rows (excluded) · `chore:` plumbing (excluded).

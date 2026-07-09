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

If a published number moved: update `research/tune-data/current-board.json`, run `tools/update-perf-board.py`, commit the regenerated README/SVG with the change (the pre-push hook enforces this).

## Cutting the release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

That's it. The `release` workflow builds, drafts changelog from conventional commits since previous tag (`tools/changelog.sh` — `perf:`/`feat:`/`fix:`/`config:`/`docs:` grouped; `data:`/`chore:` dropped as research-log noise), and publishes GitHub release. Edit notes on GitHub afterwards if the draft needs headline or context — draft is floor, not ceiling.

Preview the draft locally before tagging:

```bash
bash tools/changelog.sh            # previous tag -> HEAD
bash tools/changelog.sh v0.1.0     # explicit range
```

## Commit prefixes that feed the changelog

`perf:` kernel/throughput wins · `feat:` new capability · `fix:` correctness/bugs · `config:` defaults/flags · `docs:` documentation · `data:` tune-data rows (excluded) · `chore:` plumbing (excluded).

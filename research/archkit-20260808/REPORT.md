# Architecture onboarding kit report

Lane: `lane/cx-arch-onboard-kit`
Train: `82a2ea63`
Final verified source commit: `1e83e8875dd3056d0a82aca68e3961b9e7132291`

The Rust engine tree did not change after the geometry commit
`3e12cc625f1137ab3dfae7344264b5501f04d251`. The later commits add the
generator, tests, and documentation. The Box 1 `crates/` tree is
checksum-identical to the local final tree, and its root `Cargo.toml` and
`Cargo.lock` hashes match.

## Audit verdict

Step35 took weeks because the architecture crossed several independent
contracts at once:

| Cost center | Work that was not reusable before this lane |
|---|---|
| Artifact and placement | Three-shard 105 GB trunk, separate MTP GGUF, PP2-only placement, split-file ownership. |
| Config and geometry | Per-layer query heads, rotary width/base, SWA window, and gate layout were reassembled independently in prefill, decode, spec, draft, and batch code. |
| New semantics | Separate per-head attention gate, clamped SwiGLU, windowed hd128 attention, and fail-closed fused-door routing. |
| Tokenizer and template | Dedicated pre-tokenizer, exact role/whitespace/tool behavior, and a Step-specific reasoning control despite Qwen-like markers. |
| Drafter | External three-block topology and block-local output-head ownership; shape compatibility alone selected the wrong head. |
| Segmentation gates | Short prompts made chunk invariance vacuous; per-tick segmentation was a second axis; canaries had to be recalibrated after kernel-class changes. |
| Batched serving | B>1 over PP2 entered uniform geometry and returned valid HTTP with invalid text until a dedicated arm and liveness gate existed. |

The dominant repeated cost was not CUDA implementation. It was manually
threading the same layer geometry through every execution arm and manually
copying gate wrappers plus fast-gate registry rows.

The complete file-by-file audit and derived-versus-architecture-specific
classification remain in `PROGRESS.md`.

## Extracted components

### Declarative geometry

`ArchGeometryTable` now owns a compact class table and one class id per layer.
Each `LayerGeometry` declares:

- mixer kind;
- query and KV heads;
- K/V head dimensions;
- rotary width and base;
- optional sliding window;
- RoPE-factor use; and
- no gate, fused-Q gate, or separate per-head gate.

Qwen35/Qwen35Moe and Step35 now source migrated prefill, eager decode,
device-counter decode, speculative verify, MTP attach/forward, cross-request
prefill, and batched decode geometry from this table. Qwen MTP rows are
explicit full-attention rows. Step trunk and standalone-draft rows come from
their own artifact metadata; no out-of-range row is fabricated.

Other architectures deliberately retain their existing paths.
`full_attention_geometry_at()` supplies compatibility scalar geometry until
each architecture is migrated intentionally.

### Gate scaffold generator

`tools/generate-arch-gates.py` takes an architecture name, artifact path, and
validated JSON spec. It renders:

- chunk-invariance naked/canary wrapper;
- tick-invariance naked/canary wrapper;
- B>1 geometry naked/canary wrapper;
- six `models.tsv` rows;
- reviewed/merged `map.tsv` rows; and
- a normalized spec with generated-file hashes.

Scientific inputs remain explicit: prompts, chunk/budget/split sets, seams,
PP topology, concurrency widths, request body, and liveness regexes. The
generator refuses invalid topology, shadowed canaries, bad regex contracts,
unsafe registry paths, and overwrite without `--force`. It never edits the
canonical fast-gate registries.

The generated B>1 canary only counts when requests completed and the
assertion phase ran. Boot, HTTP, or infrastructure failure cannot be
misreported as canary teeth.

### One onboarding runbook

`docs/ONBOARDING.md` is now the maintained artifact-to-green sequence:

1. freeze source and artifact identity;
2. inspect config, tensors, tokenizer, and template;
3. declare per-layer geometry once;
4. implement only new semantics behind closed doors;
5. validate tokenizer/chat/thinking and any drafter;
6. generate standing gates;
7. promote execution arms in gate order;
8. run target-rig gates and commit raw receipts; and
9. update canonical docs/release surfaces.

The executable Qwen 3.8 day-one content is the worked example inside that
document. `docs/RUNBOOK-38.md` and the historical
`docs/qwen38-bringup-runbook.md` are pointers, not duplicate runbooks.

## Migration and final receipts

| Gate | Rig/artifact | Verdict | Raw receipt |
|---|---|---|---|
| Pre/post qwen35 `run-gen` | Local RTX 5090, Qwen3.5 9B NVFP4 | both internal `MATCH`; generated 32-token array byte-identical | `raw/post-geometry-qwen35-run-gen.log`, `raw/final-qwen35-run-gen.log`, `raw/final-qwen35-comparison.log` |
| Pre/post Step35 `run-gen` | Box 1 PP2, Step-3.7-Flash IQ4_XS | both internal `MATCH`; generated 32-token array byte-identical | `raw/post-geometry-step35-run-gen-box1.log`, `raw/post-geometry-comparison.log` |
| Full `kernel-check` | Local RTX 5090 with qwen35 real-weight oracle | `ALL GREEN` | `raw/final-kernel-check.log` |
| qwen35 `run-spec` | Local RTX 5090 plus own-trim external draft | K=1..8, 8/8 self-consistency PASS | `raw/final-qwen35-run-spec.log` |
| Step35 `run-spec` | Box 1 PP2 plus Step external draft | K=1..8, 8/8 self-consistency PASS | `raw/final-step35-run-spec-box1.log` |
| `serve-smoke` | Local RTX 5090, qwen35 trunk + draft | `0 failed` from a clean 430 MiB entry | `raw/final-serve-smoke-clean.log` |
| Generator tests | CPU | 4/4 PASS | `raw/final-generator-tests.log` |
| Rendered generator output | CPU | 3 scripts pass `bash -n` and `shellcheck`; 6 valid model rows, 2 valid map rows | `raw/final-generator-render.log` |
| Parser/geometry tests | CPU | 78/78 PASS | `raw/final-memra-gguf-tests.log` |
| Workspace compile | CPU/CUDA compile | all targets PASS | `raw/final-workspace-check.log`, `raw/final-build.log` |
| Docs/perf surfaces | CPU | local links PASS, embedded JSON spec validates, perf board current | `raw/final-docs-check.log` |
| Box 1 source identity | CPU/SSH | no crate checksum differences; root manifests match | `raw/final-box1-source-compare.log` |

`raw/final-serve-smoke.log` is an earlier green run whose entry overlapped an
unrelated Gemma `run-gen` from another worktree. It is retained as raw
evidence but is not the authoritative serving receipt. The clean rerun above
started with only the known 394 MiB background context and ended at the same
430 MiB total GPU use.

The repository-wide `cargo fmt --all -- --check` is not green because the
branch inherits broad formatting drift in untouched files. The command was
non-mutating; this lane did not reformat unrelated code. Focused diffs,
workspace compile, parser tests, and all model gates are green.

## What remains architecture-specific

The kit does not and should not infer:

- attention, recurrence, FFN, router, clamp, or quantization semantics;
- tensor ownership and transforms from shapes alone;
- tokenizer/pre-tokenizer behavior;
- chat roles, tools, whitespace, reasoning boundaries, or thinking controls;
- drafter state topology, block sequence, and the semantically correct output
  head;
- prompts and canary seams that exercise a real failure mechanism; or
- target-rig performance/default decisions.

New architectures still need those decisions, confusion-guard tests, and
honest canary calibration. They should no longer need geometry copied across
five execution arms or hand-built chunk/tick/B>1 boilerplate and fast-gate
rows.

## Commit sequence

- `e6a98426` - audit Step35 onboarding surface
- `d2eab260` - record pre-refactor geometry baselines
- `3e12cc62` - centralize migrated geometry
- `50633794` - record geometry migration gates
- `d233499d` - generate architecture gate scaffolds
- `1e83e887` - consolidate architecture onboarding runbook

No origin push, tag, release, `nsys`, or `rustup` action was performed.

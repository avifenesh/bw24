# memra — project instructions

## Branch isolation

Feature and research work MUST happen on a dedicated branch/worktree, never directly on `main`.
Preserve unrelated dirty work and stage only the intended lane.

## Hy3 spilling and quantization research

This lane owns two separate deliverables: (1) spill-path improvements for large expert banks, and
(2) a controlled five-arm quantization study. Do not trade correctness in one track for a result in
the other, and report spill performance separately from model-quality comparisons.

- `HostExps.layouts == None` is the uniform-layout fast-path contract. `Some(layouts)` makes each
  expert's `qtype`, `row_bytes`, `len`, and `offset` authoritative; use `expert_layout()` and
  `max_expert_bytes()` rather than projection-wide fields.
- Mixed layers run through metadata-aware staged, SLRU-cache, or grouped dispatch. Resident slab,
  pointer-table, pairs, dev, and grouped-decode fused kernels remain uniform-only until they group
  pointers by layout; never send mixed metadata through those kernels.
- A v2 tier plan MUST assign every retained expert projection to Q2_K, Q3_K, or NVFP4. Missing
  assignments are errors; never silently retain a BF16 expert. Q2_K remains on the generic staged
  f32-dequant kernel until the target-rig correctness and performance gates justify a fast path.
- A plan's pruned expert ids keep their original router positions. `active_experts()` masks them
  before top-k and their weights must be absent. Never dispatch, cache, or fabricate bytes for a
  masked id, and never let a fallback uniform slab bypass split expert overrides.
- The public Hy3 REAP50 checkpoint renumbers retained experts and publishes no original-id list.
  Recover the frozen mask only through `tools/recover_hy3_reap_mask.py`: require one-to-one router
  row matches, the locked nearest-match margin, and exact correction-bias confirmation. Scored
  artifacts always quantize the pinned BF16 source; never re-quantize the public MLX experts.
- The five scored arms are fixed in `research/per-expert-quant/arms.lock.json`: `plain_quant`
  (full bank, uniform NVFP4), `plain_reap_quant` (REAP50 mask, uniform NVFP4),
  `plain_reap_mix_quant` (REAP50 mask, 48 least-used Q2_K plus 48 NVFP4), and `mix_quant`
  (full bank, hottest 25% NVFP4, middle 50% Q3_K, coldest 25% Q2_K, zero-count pruned), plus
  `mix_quant_prune25` (per layer: 48 NVFP4, 48 Q3_K, 48 Q2_K, and 48 pruned).
- BF16 Hy3 is source material only, never an evaluation arm. All five arms must share the same
  source revision, non-expert tensor encodings, REAP mask where applicable, prompt template,
  runtime commit, and evaluation settings.
- Rank per layer from non-public calibration traces and freeze trace/plan hashes before viewing
  public eval scores. Uniform plans must not consume calibration traces.
- Public eval runs require `ARTIFACT` and must retain its manifest/hash. Public benchmark data
  must never select experts, thresholds, tier fractions, or pruning decisions.
- Model loading, spill correctness, research measurements, artifact generation, and public evals
  run on the provisioned G7e research machine. Do not merge or tag this lane until its remote raw logs
  and five-arm eval report exist.
- The local RTX 5090 rig remains this lane's default-flip gate: treat G7e results as research
  evidence, not a default-flip decision, and re-run correctness, memory, and throughput gates on
  the 5090 before shipping any runtime default. (Note the *deployment* target moved — owner
  override 2026-08-03 makes RTX PRO 6000 Blackwell class the owned trajectory, and the local
  5090 Laptop is a proof rig, not the final performance target. See `docs/PRODUCT-TRUTH.md` §3.)
- GGUF remains memra's primary runtime and delivery format. Hy3 safetensors are a pinned source for
  this quantization study, and per-expert repack directories are experimental artifacts, not a
  format pivot. Put spill/cache improvements in shared paths and preserve GGUF gates and behavior.
- Optimize expert serving as one storage-to-compute pipeline: mmap fallback, explicit positioned
  reads, local-NVMe access, bounded pinned host buffers, residency caching, asynchronous
  prefetch/overlap, PCIe transfer, and GPU kernels. Compare `O_DIRECT`, io_uring, and mapped-host
  access only against the measured worker baseline, and keep H2D/cache publication on the CUDA
  owner thread. Measure the stages together so a faster kernel cannot hide a data-movement
  regression.
- Keep durable model/artifact copies under `/data`, but stage byte-identical scored artifacts onto
  the G7e local NVMe (`/scratch`) for calibration, public evals, and spill benchmarks. Record the
  staged manifest hash; do not report persistent-EBS 4 KiB fault throughput as memra spill speed.

Why: a projection-wide dtype silently decodes some experts with the wrong block layout; routing a
pruned id dereferences nonexistent weights; and a G7e-only performance win may not transfer to the
5090's smaller HBM and different storage/PCIe balance.

## Perf board: generated surfaces must stay current, every push

The tuning campaign lands new numbers several times a day (`research/tune-data/rig5090.jsonl` is
the append-only research log). The generated perf surfaces — README.md's PERF-SAMPLES /
PERF-MODELS blocks, docs/PERFORMANCE.md's full boards (PERF-PLAIN / PERF-SPEC / PERF-DATE /
PERF-H100 blocks), `docs/perf-card.svg`, and `docs/perf-card-h100.svg` — are **generated**,
not hand-written: they come from `research/tune-data/current-board.json` (incl. its
`h100_board`, `samples`, and `supported_models` sections) via `tools/update-perf-board.py`.
Posture (owner call, 2026-08-02): the README carries only sample comparisons and a
numbers-free supported-models table; the full boards live in docs/PERFORMANCE.md. Numbers
are tracked for regression testing, not as a competitive scoreboard — do not reintroduce
full comparison tables to the README.

Rule: any commit that changes the *published* numbers (a board-moving merge — i.e. the numbers
that belong in the tracked boards, not every raw jsonl row) MUST:

1. Update `research/tune-data/current-board.json` with the new values.
2. Run `python3 tools/update-perf-board.py` to regenerate README.md, docs/PERFORMANCE.md,
   and the SVG cards.
3. Commit the JSON + the regenerated README.md + docs/PERFORMANCE.md + SVGs together, in
   the same commit as the number-moving change.

Never hand-edit anything inside the `<!-- PERF-*:START -->` / `<!-- PERF-*:END -->` marker
blocks in README.md or docs/PERFORMANCE.md — edit `current-board.json` and regenerate.
Prose around the tables (depth-behavior notes, mechanism writeups, "why it moved") stays
hand-written; only the marker-block contents are mechanical.

A `pre-push` hook (`tools/hooks/pre-push`, wired via `git config core.hooksPath tools/hooks`)
runs `tools/update-perf-board.py --check` and refuses the push if the board and the generated
surfaces have drifted — treat a failure there as "regenerate and re-commit." **Never** bypass with `--no-verify`.

This does not cover the GitHub repo social-preview image (the OG thumbnail used for link
shares) — GitHub has no API for that field, it's a manual upload in Settings → Social preview,
and isn't worth automating at this update cadence.

## Product claims: `docs/PRODUCT-TRUTH.md` is the only source

Anything **product-facing** — website copy, landing pages, pricing, blog posts,
gateway/marketplace applications (OpenRouter, HF Inference Providers), README marketing prose,
social posts, partner material — is written from `docs/PRODUCT-TRUTH.md`, **never** from a
`research/<lane>/` directory. Research dirs are append-only lab records: each is correct as of
its own date and goes stale silently. PRODUCT-TRUTH is the reconciled view, and every number
in it carries its receipt path, date, rig label, and protocol caveat.

Rule, same shape as the perf-board rule above: **any commit that moves a product-facing number,
target, capability, or gap MUST update `docs/PRODUCT-TRUTH.md` in the same commit.** Not a
follow-up. If a claim became false, move it to the correction ledger (§10) rather than deleting
it — a recognizable stale claim is cheaper than a silently vanished one. If a claim is not in
that file, it is not cleared for publication: add it there first, with its receipt.

Why this exists: it already failed once. On 2026-08-05 a website build-agent followed the
product docs, which had been written from research dirs four days earlier, and built the wrong
product — wrong throughput numbers, wrong target platform, claims scoped wider than their gates.
The failure was staleness, not overclaim, and no single lane was at fault; that is precisely why
the fix has to be a reconciled file plus a same-commit rule rather than more care.

Three claim classes that burned us and stay pinned in that file: performance numbers need their
**rig label** (the same cell is 5-12% different on another board, and no PRO 6000 is owned —
they are rented pods), determinism claims need their **object** (serve-vs-serve at c=1 vs c=16,
not identity against a tokenwise oracle), and the honest-gaps section is **required content on
any surface that publishes the wins**.

## Correctness discipline

Same three gates as CONTRIBUTING.md: `kernel-check`, the `run-gen` argmax gate, and `run-spec`
K=1..8 self-consistency. A kernel change without before/after numbers measured per
`research/benchmarks.md` isn't done.

## Additional accelerator backends

Blackwell remains memra's primary optimized target. When research or deployment needs another
accelerator, prefer an explicitly gated memra backend over changing the model or quantization
artifact. Secondary backends must preserve the model bytes, default off at build time, document
disabled target-specific kernels, and pass a same-prompt golden-output gate before producing
scored evidence. They do not change the naked sm_120a build or its performance defaults.

### The sm_90a (Hopper/H100) lane — merged into main 2026-07-30

Build: arch auto-detects on an H100 (`MEMRA_CUDA_ARCH=90a` forces). Hopper promotions are
compile-gated behind `memra_hopper_mma` — the naked sm_120a build stays byte-identical.
Evidence ledger: `ARCHITECTURE-H100.md` (append-only; every
promoted config, every mechanism refutation). Gate battery: `tools/validate-h100.sh
<model.gguf> [--quick]` — kernel-check config pins, decode-batch (config + strict),
decode-dc, graph-decode, graph-session. LAWS learned the hard way on this lane (do not
relearn them): (1) every perf claim is interleaved x5 on-box — cross-run AND cross-day
comparisons are clock-drift-invalid, INCLUDING the competitor denominator; (2) thresholds
and verdicts calibrated on old cores/kernels must be re-swept when the code under them
moves (five stale-verdict finds in one day, rounds 35-36); (3) anything guarding a live
lane belongs INSIDE validate-h100.sh — gates outside the battery rot silently; (4) wgmma
kernels are form-sensitive on nvcc 13.1 (C7514/15/17/19 family): measure every scheduling
change, never assume. Flags catalog: `docs/FLAGS.md §7`.

## Evidence discipline (measurement lanes)

- Raw sweep output is part of the deliverable: commit the per-run JSONL/log next to the summary
  row (`research/<lane>/`), never summary-only. A claim whose raw runs exist nowhere in the repo
  is not evidence.
- Never let a pipe swallow error output: `run-* 2>&1 | parser` loses the failure text. Always
  `tee` a raw log first, parse the log second.
- Failure causes are quoted, never inferred: "OOM" means a captured `out of memory` /
  `CUDA_ERROR_OUT_OF_MEMORY` line, with the concurrent-GPU state recorded (`nvidia-smi`
  compute-apps at failure time). A run that died without captured stderr is "died, cause
  unknown — repro needed", and no conclusion may be built on it.
- Every published median states its N and its thermal regime; single runs are labeled single
  runs.

## Releases: every board-moving or user-facing change

Tag it — `git tag vX.Y.Z && git push origin vX.Y.Z`. The `release` workflow compiles, drafts the
changelog from conventional commits (`tools/changelog.sh`), and publishes. Minor bump per
mechanism/board move, patch per fix/docs. Full process: `docs/RELEASING.md`. Commit prefixes feed
the changelog: `perf:`/`feat:`/`fix:`/`config:`/`docs:` are public; `data:`/`chore:`/`wip:`/`probe:`
are filtered as research-log noise — pick the prefix accordingly.

## CI is compile-only; the exactness battery is the real gate

GitHub runners have no GPU. `.github/workflows/ci.yml` catches build breaks (nvcc compiles fine
GPU-less). Before any merge or tag, the battery runs on the designated target GPU rig:
`kernel-check` ALL GREEN, `run-gen` argmax MATCH on affected models, `run-spec` K=1..8
self-consistency PASS. Never tag a commit without the target-rig battery.

## Flags doctrine

Winners are defaults — no flag needed to get the tuned path (naked commands = full speed).
Environment variables exist only for: runtime parameters (prompt/gen/spec knobs), machine-specific
config (VRAM budgets, KV formats, spill), rollback seams (`MEMRA_FAST=0` oracle path), diagnostics,
and explicitly-blocked experimental doors. Catalog: `docs/FLAGS.md`. When an experiment concludes
negative or flat, kill its flag and dispatch arm — the JSONL row is the record, not dead code.

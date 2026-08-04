# crates.io publishability audit — 2026-08-04, HEAD 2299ee0f (restructure/public-split)

Lane: `lane/crates-release`. Receipts in this directory. Verdict up front:

**NOT publishable as-is — four blockers, all fixable in-repo. No structural blockers
(no git deps, no leaky packages, names all free, license clean).**

## Workspace inventory

Ten crates (`Cargo.toml` members, dependency order):

| crate | version | deps (workspace) | bins | publish? |
|---|---|---|---|---|
| memra-gguf | 0.0.0 | — | gguf-inspect, nvfp4-validate | yes (leaf) |
| memra-sampling | 0.0.0 | — | — | yes (leaf) |
| memra-validate | 0.0.0 | — | — | yes (leaf) |
| memra-lanes | 0.0.0 | — (serde) | — | yes (leaf) |
| memra-kv | 0.0.0 | memra-gguf | — | yes |
| memra-tokenizer | 0.0.0 | memra-gguf | tok-check, tok-trim, tok-freq | yes |
| memra-runtime | 0.0.0 | memra-gguf | gemm-check | yes |
| memra-engine | 0.0.0 | sampling, validate, kv, gguf, runtime, tokenizer | ~50 (run-gen, run-spec, kernel-check, gates, benches) | yes |
| memra-server | 0.0.0 | engine, gguf, lanes, tokenizer | memra-server | yes — THE user-facing crate |
| memra-probe | 0.0.0 | — (cudarc) | memra-probe | **no — `publish = false`** (phase-0 dev spike, hardcoded sm_120a, no consumers) |

## Name availability (receipt: name-availability.txt)

All ten names + the bare `memra` are **HTTP 404 on the crates.io sparse index =
unregistered/available** (probe sanity-checked against `serde` = 200). Name-squatting
risk is real until the owner publishes; nothing to reserve without publishing.

## Blockers found (before state)

1. **Zero package metadata.** Every crate is `version = 0.0.0` with no `description`,
   `license`, or `repository`. `cargo publish --dry-run -p memra-gguf` warns
   "manifest has no description, license..." (receipt: dryrun-before-gguf.log);
   crates.io **rejects** uploads without `license`/`description`.
2. **Path dependencies carry no version.** `cargo publish --dry-run -p memra-runtime`
   hard-fails: "all dependencies must have a version requirement specified when
   publishing. dependency `memra-gguf` does not specify a version" (receipt:
   dryrun-before-runtime.log). Applies to every non-leaf crate.
3. **Runtime fatbin loading is path-based → distributed binaries are broken.**
   `crates/memra-engine/src/lib.rs` bakes the *builder's* `OUT_DIR` paths
   (`const FATBIN_PATH: &str = env!("MEMRA_ENGINE_FATBIN")`) and loads them **at
   runtime** with `Ptx::from_file` / `std::fs::read` (12 fatbins: 7 core + 5 KV-format
   flash variants, 27 MB total sm_120a). Verified in the shipped binary:
   `strings target/release/memra-server` shows
   `/home/avifenesh/projects/bw24-unified/target/release/build/memra-engine-*/out/*.fatbin`.
   Consequences: (a) the **existing release tarballs' engine binaries cannot start on any
   user machine** — they look for `/home/runner/work/...` paths; (b) `cargo install`
   builds in a temp dir that cargo deletes after install → same failure. The MMQ/FP8
   static lib is whole-archive-linked (self-contained, fine); only the fatbins leak.
4. **docs.rs cannot build memra-engine** — build.rs unconditionally invokes nvcc, which
   does not exist on docs.rs builders. Needs a `DOCS_RS` escape that emits placeholder
   fatbins and skips nvcc.

## Non-blockers verified clean

- **No git dependencies.** `grep 'source = "git' Cargo.lock` is empty; `llguidance 1.7.6`
  and `cudarc 0.19.8` both resolve from the crates.io registry with checksums.
- **No leaky package contents.** `cargo package --list` on all ten crates (receipt:
  package-list-before.txt): no `research/`, no `*.gguf`, no jsonl. memra-engine ships
  `cu/` (27 .cu/.cuh files, 1.8 MB — required, build.rs compiles them user-side) +
  `build.rs` + 2 small test files. memra-gguf packages at 361.6 KiB / 97 KiB compressed.
  `research/` lives at the workspace root, outside every crate dir — structurally safe.
- **License compatibility.** Repo is MIT (Copyright 2026 Avi Fenesh). Full transitive
  dep license sweep via `cargo metadata` (99 external crates): MIT / Apache-2.0 /
  BSD-2/3 / ISC / Zlib / Unicode-3.0 / MIT-0 / Unlicense-OR-MIT / BSL-1.0-OR-Apache
  only. **No copyleft, nothing incompatible with MIT redistribution.** One repo-side
  gap: the vendored llama.cpp/ggml-cuda kernel ports (`cu/mmq_*.cu`, `cu/fattn_vendor.cu`,
  headers state "vendored ... ggml-cuda @ c818263f2") carry no upstream MIT copyright
  notice — MIT requires notice preservation for substantial portions. Fixed by
  THIRD-PARTY.md shipped in the memra-engine package.
- **Cargo tooling supports the shape.** Local cargo 1.97 has `cargo publish --workspace`
  (native dependency ordering + index-propagation waits + local-registry-overlay
  verification) — no hand-rolled publish loop or sleep-between-crates needed.

## The CUDA-at-install-time decision (mission item 2)

**Recommendation: (c) both, with prebuilt binaries as the headline path.**

- (b) **Prebuilt binaries are the actual easy path** and the release workflow already
  builds them — they just don't run (blocker 3). With fatbins embedded
  (`include_bytes!` + `Ptx::from_binary`) the binaries become self-contained;
  remaining user-side requirements are the NVIDIA driver (>= 580 for CUDA 13) and the
  CUDA 13 runtime dylibs (cudart/cublas/cublasLt — build.rs links them `dylib`).
  Delivery: existing tag-triggered tarballs + sha256 checksums + `cargo binstall`
  metadata + `tools/install.sh` (curl|sh class).
- (a) **`cargo install memra-server` stays supported and honest**: the audience (CUDA
  devs with 5090s) does have nvcc; requirements documented loudly (CUDA 13.1 toolkit,
  arch auto-detect, ~50 MB binary after embed). The same fatbin-embed fix is what makes
  this path work at all (temp-build-dir deletion).
- Model files stay on HF — `run-gen`/`memra-server` already accept `hf:owner/repo:QUANT`
  specs and auto-download, so the 3-command quickstart is install → (model auto-pull) →
  serve.

## Fix plan (implemented in subsequent slices on this lane)

1. Workspace-inherited metadata (`[workspace.package]`: version 0.69.0 — the next tag —
   license, repository, edition stays per-crate), per-crate descriptions, versioned path
   deps (`{ path = "..", version = "=0.69.0" }`), `publish = false` on memra-probe,
   THIRD-PARTY.md in memra-engine.
2. Embed fatbins: `include_bytes!(env!(...))` + `Ptx::from_binary`; `MEMRA_GEMM_FATBIN`
   runtime tune-seam override preserved (env path still wins when set); `DOCS_RS`
   escape in build.rs. Kernel bytes identical — the loaded module image is the same
   fatbin; the on-rig gate battery (kernel-check / run-gen / run-spec) must confirm
   before merge, per CONTRIBUTING discipline.
3. `.github/workflows/publish.yml`: tag-triggered `cargo publish --workspace
   --exclude memra-probe --locked` with `CARGO_REGISTRY_TOKEN`; tag==version guard;
   `workflow_dispatch` = `--dry-run` mode. CI stays compile-only.
4. `release.yml`: + sm_90a matrix arm, + SHA256SUMS; binstall metadata in
   memra-server; `tools/install.sh`.
5. README installation rewrite + RELEASING.md publish step & secrets checklist.

## Owner actions required (cannot be done from this lane)

- Create/verify a crates.io account; generate an API token scoped to publish-new +
  publish-update; add it as repo secret `CARGO_REGISTRY_TOKEN`.
- First publish is the name reservation for all 9 names — push the button by tagging
  (or run the workflow_dispatch dry-run first).
- DO NOT publish from this branch until it merges on the v0.69 wave and the on-rig
  battery is green on the tagged commit.

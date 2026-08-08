# Architecture onboarding kit

Lane: `lane/cx-arch-onboard-kit`, train `82a2ea63`.

Owner doctrine (2026-08-08): create more extracted components so new model
architectures take days to onboard instead of weeks.

This increment is the audit only. It records the Step-3.7-Flash (`step35`)
bring-up surface before the extraction changes it.

## Lane preflight

- `CLAUDE.md` read at the branch tip.
- Worktree was clean and already isolated on `lane/cx-arch-onboard-kit`.
- `~/.lanectl/inbox/cx-archkit.md` does not exist. The lane registry has no
  notes or alternate inbox path for `cx-archkit`.
- No origin push, `nsys`, or `rustup` action is part of this lane.

## Source receipts

The audit traces the actual merge diffs and the append-only lane ledgers:

- `research/step37-bringup-20260802/PLAN.md`
- `research/step37-p2-20260806/PROGRESS.md`
- `research/step35-chunkfix-20260807/PROGRESS.md`
- `research/step-sku-20260807/PROGRESS.md`
- `research/tick-seg-20260807/PROGRESS.md`
- `research/step35-batch-20260808/PROGRESS.md`
- `docs/qwen38-bringup-runbook.md`

The phase-1 estimate was 2.5-4 weeks listing-grade, with 1-1.5 weeks for
first tokens. The calendar work was compressed, but the estimate was honest:
the architecture crossed config, loader, math, tokenizer, template, MTP,
pipeline placement, serving, and three distinct invariance/geometry gates.

## Weeks-cost breakdown

| Workstream | Why it was expensive | Result |
|---|---|---|
| Artifact and source-of-truth census | The official trunk is a three-shard 105 GB GGUF, MTP is a separate GGUF, the public quant labels were not interchangeable, and the model only fits PP-2 on the owned rig. | Pinned shard hashes, tensor inventory, MTP topology, storage and KV budgets. |
| Config and per-layer geometry | `attention.head_count` is an array, full and SWA layers use different rotary widths and bases, and a new arch defaulted into a fused gate path that would over-read `wq`. | `Step35Config`, per-layer accessors, fixture metadata, explicit gate predicates. |
| Attention and FFN math | A separate per-head gate and Step-style clamped SwiGLU had plausible but incorrect lookalikes already in the tree. Windowed hd128 prefill also lacked a guarded primitive. | Two kernels, confusion-guard tests, dedicated prefill/decode mixer, windowed prefill oracle cells. |
| Multi-shard load and PP placement | The first boot read one shard; the model cannot boot on one card. Generic DC and batched doors could execute wrong geometry without faulting. | Split-GGUF support, spill offset fix, PP-2 boot, named refusals for unsupported arms. |
| Tokenizer and chat parity | `deepseek-v3` silently fell through to qwen splitting. The Step template contains qwen markers but has different reasoning, tools, role, and whitespace semantics. | Dedicated pre-tokenizer, 113-case parity gate, dedicated template renderer and goldens. |
| Drafter attach | The head is external, carries three numbered blocks, and its top-level output tensor has the right shape but the wrong semantics. Self-consistency stayed green at 0% acceptance. | External attach, per-draft geometry, correct block-local head selection, K=1..8 acceptance gate. |
| Segmentation correctness | Short prompts made chunk invariance vacuous. A second per-tick split axis survived the first fix, and canaries became inert when the kernel class changed. | `chunkinv35`, `tickinv35`, request-extent threading, live canaries with class-aware receipts. |
| Batched serving geometry | PP-2 B>1 returned HTTP 200 with garbage because it entered the uniform generic arm. The correct arm needed per-layer geometry and per-session SWA offsets. | Fail-closed pin, dedicated batched walk, `b2geo35`, then a verified pin lift. |
| Listing surface and docs | Thinking controls differ by architecture; capability detection and request mapping had to preserve every model's legacy default. | Per-arch thinking mapping, serve smoke, flags/testing/serving documentation. |

The dominant repeated cost was not new CUDA math. It was rediscovering and
hand-threading the same architecture facts - layer class, head counts, rotary
shape, window, and gate layout - through each execution arm, then hand-building
the gate wrapper and fast-gate registration that proved the arm was live.

## Complete bring-up surface

### 1. Architecture recognition and metadata

Files:

- `crates/memra-gguf/src/config.rs`
  - `Arch::Step35` recognition and hybrid/MoE classification.
  - `Step35Config` arrays and scalar metadata.
  - `ModelConfig::{layer_kind,n_head_at,n_head_kv_at,is_swa_at}`.
  - fused-vs-separate attention-gate predicates.
- `crates/memra-gguf/src/micro_gguf.rs`
  - metadata-only Step fixtures, array encoders, and all-layer assertions.
- `crates/memra-gguf/src/lib.rs`
  - split-file resolution and metadata/tensor ownership.
- `crates/memra-gguf/src/bin/inspect.rs`
  - inspection output for the new metadata surface.

Step35-specific:

- Step's full/SWA rotary convention: full rotates 64 of 128, SWA rotates 128.
- `attn_gate.weight` means one sigmoid scalar per head before `wo`.
- Clamp-array semantics and epsilon.
- Missing metadata defaults that mirror upstream Step code.

Mechanically derivable from GGUF/config:

- Architecture id, layer count, tensor names and shapes.
- Per-layer query/KV heads and SWA pattern.
- Head dimensions, both RoPE bases, window size, and `rope_freqs` presence.
- MoE counts, dense-prefix count, router flags, scale, and normalization flag.
- MTP block count and whether the head is embedded or external.

### 2. Loader and tensor ownership

Files:

- `crates/memra-engine/src/hybrid.rs`
  - per-layer mixer loading, separate attention gate, Step draft geometry,
    external MTP attach, block-local draft head preference.
- `crates/memra-engine/src/model.rs`
- `crates/memra-engine/src/spill.rs`
  - split-file tensor data and mmap/offset ownership fixes exposed by Step.

Step35-specific:

- The separate gate tensor is mandatory on every Step attention layer.
- The external drafter's block-local `nextn.shared_head_head` wins over the
  shape-compatible top-level trunk head.
- The three shipped MTP blocks are not interchangeable by shape.

Mechanically derivable:

- Required tensor presence, tensor widths, shard ownership, and block numbers.
- Whether a draft file contains block-local heads.
- Geometry compatibility between trunk and draft.

### 3. Prefill, decode, and batched execution

Files:

- `crates/memra-engine/src/hybrid_forward.rs`
  - Step prefill/prime/decode attention, request-extent SWA selection, clamp
    dispatch, and generic qwen35 full-attention paths.
- `crates/memra-engine/src/decode.rs`
  - eager routing, arch gates, graph/DC refusals, and gate-aware fast paths.
- `crates/memra-engine/src/decode_batch.rs`
  - fail-closed generic guard, dedicated Step batched walk, per-session SWA
    view offsets, and PP-stage routing.
- `crates/memra-engine/src/spec.rs`
  - Step verify path and MTP geometry.
- `crates/memra-engine/src/lib.rs`
  - windowed quantized-view primitive.
- `crates/memra-engine/cu/hybrid.cu`
  - head-gate and clamped-SwiGLU kernels.

The inline Step batched geometry table was:

| Field | Full class | SWA class |
|---|---:|---:|
| mixer | full attention | sliding-window attention |
| query heads | 64 | 96 |
| KV heads | 8 | 8 |
| head dim | 128 | 128 |
| rotary dim | 64 | 128 |
| RoPE base | 5e6 | 1e4 |
| RoPE factors | yes | no |
| window | none | 512 |
| attention gate | separate, per head | separate, per head |

This table was written in prose and then reassembled from `Step35Config` in
prefill, eager decode, spec, draft, and batched decode. That repetition is the
primary extraction target.

Qwen35 already has a second two-class table, but it is implicit:

| Field | Linear class | Full class |
|---|---:|---:|
| mixer | gated delta net | full attention |
| query heads | global config | global config |
| KV heads | global config | global config |
| rotary dim | unused | global `rope_dim_count` |
| RoPE base | unused | global `rope_freq_base` |
| window | none | none |
| attention gate | linear mixer's own gates | fused q/gate in `wq`, per dim |

Its class map is `(layer + 1) % full_attention_interval`, currently recomputed
by `ModelConfig::layer_kind`; the attention arms separately read the global
head/RoPE fields and the fused-gate predicate.

### 4. Kernel and executable gates

Files:

- `crates/memra-engine/src/bin/kernel_check.rs`
  - head-gate, clamp, and hd128 windowed-prefill confusion guards.
- `crates/memra-engine/src/bin/run_gen.rs`
  - prefill/decode and batched-prime/tokenwise argmax gates.
- `crates/memra-engine/src/bin/decode_batch_gate.rs`
  - B=1/2/4/8 per-row bit identity and long-prompt SWA coverage.
- `crates/memra-engine/src/bin/concat_prime_probe.rs`
  - chunk and per-tick segmentation probes.
- `crates/memra-engine/src/bin/t2probe.rs`
  - full-attention call signature migration.

Mechanically reusable:

- Artifact resolution with an explicit override.
- Raw-log-first execution and parsing.
- Naked assertion plus a canary that changes the runtime world.
- Model absence and insufficient-GPU `SKIP` contracts.
- Fast-gate `kind=cmd` registration.

Arch-specific inputs that cannot be invented by a generator:

- A prompt long enough to cross the architecture's window/class boundary.
- The expected property, the canary seam, and proof the seam has teeth.
- Batched liveness evidence emitted by the execution arm.
- Which unsupported placement or graph doors must fail closed.

### 5. Standing gate wrappers and fast-gate tables

Files/tables:

- `tools/chunk-invariance-gate.sh`
- `tools/tick-invariance-gate.sh`
- `tools/step35-b2-geometry-gate.sh`
- `tools/fast-gate/models.tsv`
- `tools/fast-gate/map.tsv`

The three Step families all repeat:

1. resolve the artifact;
2. validate binary, prompt, and device prerequisites;
3. acquire the shared GPU lock;
4. tee a raw log;
5. run a naked property assertion;
6. run a mechanism canary;
7. print `PASS`, `FAIL`, or `SKIP`;
8. add two `models.tsv` rows and one or more `map.tsv` dispatch rows.

The wrappers themselves differ, but their scaffold and registration are
mechanical. The generator should emit wrappers from committed templates plus
manifest inputs; it must not infer scientific expectations.

### 6. Tokenizer, template, and thinking surface

Files:

- `crates/memra-tokenizer/src/lib.rs`
- `crates/memra-tokenizer/src/unicode.rs`
- `crates/memra-tokenizer/src/chat.rs`
- `crates/memra-tokenizer/src/bin/tok_parity.rs`
- `crates/memra-tokenizer/Cargo.toml`
- `crates/memra-server/src/main.rs`
- `crates/memra-server/src/worker.rs`
- `docs/SERVING.md`

Mechanically derivable:

- `tokenizer.ggml.pre`, tokenizer model, vocabulary/special ids.
- Raw chat template bytes and marker inventory.
- Whether a template mentions tools, generation prompts, thinking switches,
  or named reasoning effort.

Step35-specific:

- The exact pre-tokenizer algorithm when the runtime lacks the named type.
- Template control flow, whitespace, tool roles, and reasoning boundary.
- Mapping the common API thinking surface onto Step's `Reasoning:` string.

Marker detection is an onboarding warning, not an implementation oracle. Step's
template deliberately contains all qwen markers and still needs a separate
renderer.

### 7. Drafter attach conventions

Files:

- `crates/memra-engine/src/hybrid.rs`
- `crates/memra-engine/src/spec.rs`
- `crates/memra-server/src/worker.rs`
- `docs/SERVING.md`

Reusable convention:

- Treat `model+draft` as two independently parsed artifacts.
- Validate architecture, vocabulary, embedding width, head geometry, and token
  ids before allocating the full model.
- Prefer an explicit block-local draft output head when present.
- Gate both self-consistency and acceptance; self-consistency alone does not
  detect a semantically wrong but shape-compatible head.

Still architecture-specific:

- Draft state topology and whether successive draft positions use one block or
  rotate across several trained blocks.
- The correct block-local tensor name and any model-specific transforms.
- The acceptance floor and served K policy.

### 8. Documentation and receipts

Files:

- `docs/FLAGS.md`
- `docs/SERVING.md`
- `docs/TESTING.md`
- Step lane `PROGRESS.md`, scripts, goldens, raw logs, and JSONL receipts.

The reusable onboarding runbook must point to these canonical docs rather than
copy flag semantics into a second location. The existing
`docs/qwen38-bringup-runbook.md` becomes the worked example inside the general
runbook; there must be one maintained sequence, not two diverging checklists.

## Extraction contract

### Geometry table

Add a declarative table owned by `ModelConfig`:

- a small set of per-arch layer classes;
- a layer-index to class-index map;
- per class: mixer kind, query heads, KV heads, rotary width, RoPE base,
  optional window, RoPE-factor flag, and attention-gate kind;
- an explicit fallback class for appended/external draft indices.

Migrate Step35 and Qwen35/Qwen35Moe. Other architectures retain their current
paths. The existing public accessors may remain as compatibility shims, but
the migrated prefill/decode/batched code must source geometry from one table
row rather than reconstructing it from arch-specific fields.

No arithmetic, kernel, or dispatch default changes are allowed. Pre/post
`run-gen` output must be byte-identical for both migrated architecture
families.

### Gate scaffold generator

Given an architecture id and artifact path, emit:

- chunk-invariance wrapper;
- tick-invariance wrapper;
- batched-geometry wrapper;
- naked and canary `models.tsv` rows;
- corresponding `map.tsv` rows.

Inputs such as prompts, chunk/budget sets, canary seams, devices, PP stages,
and liveness regexes are explicit manifest fields. The generator validates
them and renders templates; it does not choose them.

### Onboarding runbook

Create `docs/ONBOARDING.md` with the ordered path:

1. freeze source/artifact identity;
2. inspect config, tensors, tokenizer, and template;
3. generate and review the geometry table;
4. implement/load only genuinely new semantics;
5. attach and validate any drafter;
6. generate standing gates before enabling broad dispatch;
7. prove kernel, argmax, spec, batch, serve, and invariance gates;
8. record raw receipts and update canonical docs.

The Qwen 3.8 material remains a worked example, but its executable details are
merged into the general runbook rather than duplicated.

## Audit verdict

The next architecture still needs human work for new math, template semantics,
drafter semantics, and honest canaries. It should not need humans to rewrite
layer geometry in five execution arms or hand-copy the same gate boilerplate
and fast-gate rows. Those are the two reusable components this lane extracts.

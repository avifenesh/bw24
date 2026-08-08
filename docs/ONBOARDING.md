# Architecture onboarding

This is the single maintained path from an artifact in hand to gates green. GGUF is memra's
primary runtime and delivery format. An explicitly supported safetensors backend follows the
same sequence, but it must preserve the source bytes and add format-specific classification
gates rather than silently converting the model.

Use a dedicated branch/worktree and a target-rig receipt directory. Do not enable a broad
dispatch door until the architecture's geometry, tokenizer/template, and standing canaries
are explicit. A same-family model may reuse existing semantics only after the artifact proves
that the relevant contracts are unchanged.

Canonical supporting documents:

- [Testing](TESTING.md) defines fast-gate versus the full merge/tag battery.
- [Flags](FLAGS.md) owns runtime parameters, rollback seams, and diagnostic flags.
- [Serving](SERVING.md) owns the public API and drafter attach syntax.
- [Performance](PERFORMANCE.md) owns rig authority and measurement protocol.

## Ordered checklist

| Phase | Work | Exit condition |
|---|---|---|
| 0 | Isolate the lane and freeze the source/artifact identity | branch, source SHA, artifact manifest, rig, and receipt directory recorded |
| 1 | Inspect metadata, tensor inventory, tokenizer, and raw chat template | every field classified as derived, reused, new semantic work, or STOP |
| 2 | Build one declarative per-layer geometry table | every trunk and MTP/draft execution row resolves heads, rotary geometry, window, and gate kind |
| 3 | Implement only genuinely new math and tensor ownership | loaders and kernels fail closed on unsupported shapes and doors |
| 4 | Prove tokenizer, chat, tools, and thinking behavior | parity/goldens cover exact bytes and the public thinking mapping |
| 5 | Attach and validate any drafter | interface, block topology, output head, K=1..8, and acceptance are proven |
| 6 | Generate standing gate scaffolds | chunk, tick, B>1 geometry, naked/canary rows, and map rows are reviewed |
| 7 | Enable broader execution arms in stages | eager first, then prefill/batch/spec/graph/PP only after their own liveness gates |
| 8 | Run the target-rig battery and close receipts | kernel, argmax, spec, invariance, batch, and serve gates are green with raw logs |
| 9 | Update canonical docs and release surfaces | no duplicate runbook, generated surfaces current, merge/tag only after target-rig approval |

## 0. Freeze the lane and artifact

Start from an isolated worktree. Record the exact source tree before the first model run:

```bash
export ARCH=<short-architecture-id>
export MODEL=/data/path/to/model.gguf
export R=research/${ARCH}-bringup-$(date +%Y%m%d)
mkdir -p "$R/raw"

git branch --show-current | tee "$R/branch.txt"
git rev-parse HEAD | tee "$R/source-commit.txt"
git status --short | tee "$R/worktree-status.txt"
sha256sum "$MODEL" | tee "$R/artifact.sha256"
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv | tee "$R/gpu-entry.csv"
```

For split GGUF, hash every shard and enter through any shard only after `gguf-inspect` confirms
the merged tensor count and per-shard ownership. For a directory artifact, hash the config,
tokenizer files, index, and every indexed tensor shard. Durable copies live under `/data`;
benchmark/eval copies on a remote research box are staged byte-identically to local NVMe and
their staged manifest hash is recorded.

Do not edit model metadata to make a parser accept it. If the artifact changes, start a new
manifest and preserve the old receipt.

## 1. Inspect before implementing

Build the header inspector and capture both metadata and tensor patterns:

```bash
cargo build --release -p memra-gguf --bin gguf-inspect
target/release/gguf-inspect "$MODEL" --all \
  2>&1 | tee "$R/raw/gguf-inspect-all.log"
target/release/gguf-inspect "$MODEL" --config \
  2>&1 | tee "$R/raw/gguf-inspect-config.log"
```

The audit is four inventories:

1. Config: architecture id, layer count, per-layer arrays, head dimensions, attention classes,
   RoPE parameters, window, MoE/router fields, normalization, clamp values, and MTP depth.
2. Tensors: naming patterns, shapes, dtypes/layouts, split-shard ownership, optional gates,
   position factors, expert banks, output heads, and companion draft blocks.
3. Tokenizer: tokenizer model, `tokenizer.ggml.pre`, normalizer/pre-tokenizer, vocabulary and
   special ids, BOS/EOS behavior, and exact encoding on a frozen parity corpus.
4. Template: raw template hash, roles, tools branch, generation prompt, whitespace, reasoning
   boundaries, `enable_thinking`, and any string-valued reasoning control.

Classify every difference:

| Class | Action |
|---|---|
| Mechanically derived | Parse it once and place it in `ModelConfig` or the geometry table. |
| Existing semantic contract | Reuse only after model-backed parity proves identical behavior. |
| New semantic contract | Add a narrow loader/renderer/kernel path plus confusion-guard tests. |
| Unsupported or ambiguous | STOP and keep the dispatch door closed. |

Tensor shape is evidence, not semantics. A shape-compatible output head, gate, or draft block
must not be borrowed without name/topology confirmation.

## 2. Declare per-layer geometry once

Architecture recognition and geometry live in `crates/memra-gguf/src/config.rs`.
`ArchGeometryTable` contains a compact set of `LayerGeometry` classes plus one class id per
layer. Each row declares:

- `mixer`: full or linear attention;
- query and KV head counts;
- K and V head dimensions;
- rotary width and base;
- optional sliding window;
- whether RoPE factors apply; and
- `AttentionGateKind::{None,FusedQ,SeparateHead}`.

Populate the table from authoritative artifact metadata. Map every retained trunk layer and
every appended MTP layer. A companion drafter resolves its rows from its own parsed config;
never fabricate an out-of-range row or borrow a trunk row because the tensor shape looks close.

All migrated execution arms read `layer_geometry()` or
`full_attention_geometry_at(layer)`. Do not reintroduce arch-specific head/rotary/window
match-arms in prefill, decode, verify, or batched code. New math, tensor names, clamps, router
semantics, and template behavior remain architecture-specific and do not belong in this table.

Minimum geometry tests:

- every artifact layer resolves the expected class;
- class counts and class transitions match metadata;
- MTP/draft rows resolve deliberately;
- full versus windowed rows differ in every field the artifact says differs;
- gate kind is mutually exclusive and matches tensor ownership; and
- missing/malformed arrays fail instead of falling back to projection-wide scalars.

## 3. Implement new semantics behind closed doors

Add only the pieces the audit marked as genuinely new:

- parser aliases and typed metadata;
- loader ownership and required tensor checks;
- attention/FFN/router math;
- quantization layout handling;
- tokenizer or chat renderer;
- drafter topology; and
- target-specific kernels.

Start with the most direct reference path. Keep graph, device-counter, pointer-table, resident,
grouped, fused, and batched doors closed until they consume the same declared geometry and have
an arm-specific exactness/liveness gate. A refusal with the architecture and missing contract
named is preferable to a plausible wrong answer.

Kernel tests need confusion guards, not only happy paths. Exercise the nearest wrong
interpretation: fused versus separate gates, full versus partial rotary, windowed versus full
attention, block-local versus file-level heads, uniform versus mixed quant layouts, and pruned
versus retained ids.

## 4. Prove tokenizer, template, tools, and thinking

Tokenizer and template work are separate gates:

- compare token ids against the authoritative tokenizer on a frozen corpus, including Unicode,
  whitespace, punctuation, code, and special-token boundaries;
- hash and archive the source template;
- render golden conversations for system/user/assistant/tool roles, generation prompts, and
  assistant continuations;
- verify tool calls and reasoning stripping on the live server surface; and
- map `reasoning_effort`/`reasoning` through the per-architecture behavior table.

Template marker detection is only capability triage. A template that mentions Qwen markers may
still have different roles, whitespace, tool semantics, or thinking control and require its own
renderer. The public gate is the rendered bytes and served response behavior.

## 5. Attach drafters explicitly

The served convention is `MEMRA_MODELS="alias=/path/trunk.gguf+/path/draft.gguf"`.
Direct `run-spec` work may use `MEMRA_MTP_DRAFT` where the binary documents it.

Before allocating the full model, validate:

- architecture and vocabulary compatibility;
- embedding width, attention geometry, token ids, and MTP depth;
- block sequence and state topology;
- exact block-local output-head ownership; and
- any architecture-specific transforms.

Then require both:

1. naked `run-spec` K=1..8 self-consistency; and
2. non-vacuous acceptance/e2e evidence on the served model-drafter-prompt regime.

Self-consistency proves verification correctness. It does not prove that a shape-compatible
head is semantically useful; 0% acceptance can still be self-consistent.

## 6. Generate standing gate scaffolds

`tools/generate-arch-gates.py` renders the repeated mechanics. It requires the architecture
name and default artifact path plus an explicit JSON spec:

```bash
python3 tools/generate-arch-gates.py "$ARCH" "$MODEL" \
  --spec "$R/gate-spec.json" \
  --out-dir "tools/generated-arch-gates/$ARCH"
python3 -m unittest tools.test_generate_arch_gates
shellcheck "tools/generated-arch-gates/$ARCH"/*.sh
```

The output contains chunk-invariance, tick-invariance, and B>1 geometry scripts; naked/canary
`models.tsv` rows; `map.tsv` rows; and a normalized spec with output hashes. The generator does
not modify the canonical registries. Review the fragments and merge them deliberately.

The spec shape is:

```json
{
  "id": "x",
  "artifact_env": "MEMRA_NEWARCH_GGUF",
  "chunk": {
    "label": "newarch-window",
    "prompts": ["research/newarch/prompt-past-window.txt"],
    "chunks": [4096, 513, 512, 256, 64],
    "steps": 24,
    "seam": "MEMRA_NEWARCH_CHUNK_LEGACY"
  },
  "tick": {
    "label": "newarch-tick",
    "prompts": ["research/newarch/prompt-past-window.txt"],
    "budgets": [0, 1024, 513, 512, 256, 64],
    "splits": [64, 256, 512],
    "steps": 24,
    "seam": "MEMRA_NEWARCH_CALLLOCAL"
  },
  "batch": {
    "model_alias": "newarch",
    "draft_path": null,
    "draft_env": "MEMRA_NEWARCH_DRAFT_GGUF",
    "canary_env": {"MEMRA_NEWARCH_BATCH": "0"},
    "required_gpus": 2,
    "pp_stages": 2,
    "pp_devices": [0, 1],
    "concurrency": [2, 4],
    "port": 8094,
    "receipt_dir": "research/newarch-batch/raw",
    "server_env": {
      "MEMRA_SERVE_SPEC": "0",
      "MEMRA_SERVE_B1FAST": "0"
    },
    "request": {
      "messages": [{"role": "user", "content": "Give one deterministic test response."}],
      "max_tokens": 48,
      "temperature": 0.0
    },
    "liveness": {
      "cap_regex": "newarch: decode chunk cap [0-9]+",
      "cap_min": 2,
      "walk_regex": "\\[newarch-batch\\] first B>1"
    }
  },
  "mapping": [
    {
      "path_regex": "^crates/memra-engine/src/(decode|decode_batch|forward|hybrid_forward|pp|prime_graph|graph_update|mla)\\.rs$",
      "kernel_scope": "synthetic",
      "base_probes": ["g12", "q9", "q35"],
      "base_spec_probes": ["q35spec"],
      "gate_families": ["chunk", "tick", "batch"]
    },
    {
      "path_regex": "^crates/memra-server/",
      "kernel_scope": "none",
      "base_probes": ["sstress", "accept"],
      "base_spec_probes": [],
      "gate_families": ["tick", "batch"]
    }
  ]
}
```

The manifest is explicit because the scientific inputs are not derivable:

- the prompt must cross the actual window/class boundary;
- chunk and tick sets must straddle the suspected segmentation boundary;
- the canary seam must change the runtime world and be shown to bite;
- PP topology must match the only valid placement;
- the concurrency widths must execute B>1; and
- liveness regexes must prove the intended architecture arm ran.

Re-copy the current base probe lists from `tools/fast-gate/map.tsv` before installing generated
rows. Do not make a canary pass by changing only the expected label, and do not treat `SKIP`
from a missing artifact or GPU as evidence.

## 7. Enable execution arms in gate order

Recommended promotion order:

1. parser plus loader;
2. eager prefill and tokenwise decode;
3. chunked and per-tick prefill;
4. MTP verify;
5. PP placement;
6. batched prefill/decode;
7. device-counter/graph/session paths; and
8. optimized fused or grouped kernels.

At each step, preserve a reference path and require both output identity and liveness evidence.
Do not use a faster arm's throughput to waive a model-quality or routing mismatch.

## 8. Target-rig battery and receipts

Run every command through the rig's shared GPU lock. Tee raw stderr/stdout before parsing it.
The minimum new-architecture battery is:

| Gate | Required proof |
|---|---|
| Config/parser tests | metadata arrays and geometry classes are exact |
| `kernel-check` | final `ALL GREEN`, including new confusion guards |
| `run-gen` | prefill/decode argmax `MATCH`, batched-prime/tokenwise `MATCH`, pinned tokens unchanged where refactoring |
| `run-spec` | K=1..8 `SELF-CONSISTENCY PASS` for every supported drafter |
| Chunk/tick gates | naked invariant result plus a canary that breaks the same assertion |
| B>1 geometry | c=2/c=4 equals c=1 and logs prove the B>1 arm executed |
| `serve-smoke` | real API, streaming, concurrency, deterministic output, and spec/plain identity green |
| Architecture surface | tokenizer/template/thinking/tools goldens and live requests green |

`tools/fast-gate/fast-gate.sh --tier 2` executes the standing correctness battery, but a new
architecture also needs its explicit artifact probes until the canonical battery owns them.
A clean-tree fast-gate run must name `--probes`; an empty diff otherwise runs nothing.

Record:

- source commit and staged tree hash used by remote builds;
- artifact and staged-copy manifests;
- exact commands and environment;
- GPU process state at entry and on any failure;
- raw logs, not summary-only rows;
- generated token arrays for before/after refactors;
- naked and canary logs; and
- server logs carrying liveness evidence.

An unexplained process death is “cause unknown, repro needed,” not OOM. Every median states N
and thermal regime; correctness receipts do not become performance claims.

## 9. Close the lane

Update only canonical docs. Add flags to [Flags](FLAGS.md), serving contracts to
[Serving](SERVING.md), and gate ownership to [Testing](TESTING.md). Do not copy those tables
into a per-model runbook.

If published performance numbers move, update `research/tune-data/current-board.json` and run
`tools/update-perf-board.py`; commit the JSON and every generated surface together. Never edit
PERF marker blocks by hand.

Before merge/tag:

- the target-rig battery is green;
- raw receipts are committed;
- unsupported doors still fail closed;
- generated perf surfaces pass `--check`;
- no unrelated work is staged; and
- the release follows [Releasing](RELEASING.md).

## What remains architecture-specific

| Surface | Why it cannot be generated honestly |
|---|---|
| New attention, FFN, router, or recurrence math | Metadata names dimensions; it does not define numerical semantics. |
| Tensor ownership and transforms | Shape-compatible tensors can have different roles or block locality. |
| Tokenizer/pre-tokenizer | Named or regex behavior must match the authoritative implementation byte-for-byte. |
| Chat/tools/thinking | Marker inventory cannot reproduce control flow, roles, whitespace, or reasoning boundaries. |
| Drafter state topology and head selection | K=1..8 exactness cannot identify the semantically correct head by itself. |
| Canary seams and prompts | They must be calibrated against a real failure mechanism and execution class. |
| Performance promotion | G7e/H100 evidence does not replace the owned deployment or default-flip rig. |

The kit removes repeated geometry threading and gate boilerplate. It does not automate away
architecture research.

## Worked example: Qwen 3.8 27B FP8-ST

Prepared 2026-08-08 for the expected Qwen 3.8 drop. Re-run the release lookup and preflight;
the assumptions below are not a substitute for current official metadata. This is an executable
same-architecture runbook, not a substitute for a bring-up lane when a STOP condition fires.

This worked example assumes the production candidate is the official Qwen FP8 safetensors
directory and forbids substituting Q8_0, GGUF, NVFP4, or a community requant for missing
official bytes. The frozen Qwen 3.6 GGUF appears below only as:

- the real-weight oracle for `kernel-check`, whose model-backed sections are GGUF-specific;
- an A/B performance reference; and
- a byte-verbatim MTP donor for a small external drafter, and only when the trunk interface is
  exactly compatible.

The shipped defaults are the desired path: `MEMRA_ST_E4M3` and `MEMRA_ST_E4M3_BLK` are on,
and block-128 prefill uses the native `MEMRA_FP8_MMQ` source. Do not set
`MEMRA_ST_E4M3=1`, `MEMRA_ST_E4M3_BLK=1`, or `MEMRA_FP8_MMQ=1`; naked behavior is the
production behavior. `MEMRA_FP8_MMQ=1` additionally admits a duplicate stash source and is
not this path.

### Before release

Run from the dedicated worktree in Bash; the snippets use Bash arrays and `PIPESTATUS`. Every GPU
command on the local 5090 uses the shared lock. Raw logs are written first and parsed second.

```bash
set -euo pipefail
cd /home/avifenesh/projects/wt-cx-38prep

export Q38_REPO=Qwen/Qwen3.8-27B-FP8
export Q38_DIR=/data/ai-ml/hf-models/qwen38-27b-fp8
export Q38_DERIVED_ROOT=/data/ai-ml/hf-models/qwen38-27b-derived
export Q38_VENV_ROOT=/data/ai-ml/venvs
export Q36_REF=/data/ai-ml/hf-models/qwen36-27b-hf-min
export Q36_ST=/data/ai-ml/hf-models/qwen36-27b-blk128fp8
export Q36_GGUF=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
export Q36_DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
export R=research/qwen38-bringup-$(date +%Y%m%d)
mkdir -p "$R"

# A copied shell must not turn a default-path gate into an inherited experiment.
unset MEMRA_ST_E4M3 MEMRA_ST_E4M3_BLK MEMRA_FP8_MMQ MEMRA_FP8_FOLD
unset MEMRA_FP8_BLK_GPU MEMRA_PP_FP8 MEMRA_MTP_DRAFT MEMRA_FRSPEC_TRIM
unset MEMRA_SPEC_K

tools/preflight-38.sh 2>&1 | tee "$R/preflight.log"
```

`PREFLIGHT-38: READY-WITH-WAITS` is green before release. A missing official repository and
target directory are expected `WAIT`s. Any `FAIL` is fixed before day one.

### Day-one clock

| Time | Work | Exit condition |
|---|---|---|
| H+00:00 | Bind the exact official repo/revision; fetch metadata | official FP8 repo and immutable revision recorded |
| H+00:15 | Config and tokenizer diff | same-architecture verdict; otherwise STOP |
| H+00:30 | Download FP8 weights; create manifest | every indexed shard present and hashed |
| H+01:00 | Build binaries and current HF reference environment | release binaries and reference imports succeed |
| H+01:30 | FP8 header census | every F8 weight is per-tensor or block-128 |
| H+02:00 | Model-backed `kernel-check` | final line `ALL GREEN` |
| H+02:30 | Naked memra first light and residency proof | native FP8 residency, no silent class fallback |
| H+03:15 | HF greedy-reference comparison | identical generated token ids |
| H+04:00 | Chunk invariance | invariant across the pinned chunk sizes |
| H+04:30 | Embedded-MTP `run-spec` | K=1..8 self-consistency passes |
| H+05:15 | ST serve battery and thinking surface | serve gate green; Qwen thinking switch detected and exercised |
| H+06:30 | Own-generation ranks and trim validation | corpus floor met; in-model trim A/B recorded |
| H+09:00 | External drafter build and gates | donor compatible, draft built, K=1..8 passes, e2e verdict recorded |
| H+10:00 | Close receipt | every publish gate green or an explicit STOP recorded |

Downloads, the HF reference environment, and the model-backed kernel battery can overlap.
Do not overlap two GPU jobs.

### H+00:00 - bind the official artifact

Search the official namespace instead of assuming the release name. An official BF16 repo is
source/reference material; the official FP8 sibling is the only production candidate.

```bash
curl -fsS \
  'https://huggingface.co/api/models?author=Qwen&search=Qwen3.8-27B&limit=100' \
  | tee "$R/hf-search.json"
jq -r '.[] | (.id // .modelId)' "$R/hf-search.json"

# Set this to the exact official FP8 id printed above if the release spelling differs.
export Q38_REPO=Qwen/Qwen3.8-27B-FP8
jq -e --arg repo "$Q38_REPO" \
  'any(.[]; (.id // .modelId) == $repo)' "$R/hf-search.json"

curl -fsS "https://huggingface.co/api/models/$Q38_REPO" \
  | tee "$R/hf-model.json"
export Q38_REV
Q38_REV=$(jq -er '.sha' "$R/hf-model.json")
printf 'repo=%s\nrevision=%s\n' "$Q38_REPO" "$Q38_REV" \
  | tee "$R/artifact-source.txt"
export A="$Q38_DERIVED_ROOT/$Q38_REV"
mkdir -p "$A" "$Q38_VENV_ROOT"
printf 'derived_artifacts=%s\n' "$A" | tee -a "$R/artifact-source.txt"
```

STOP if the FP8 sibling is absent. Do not fill the gap with Q8_0, a local conversion, or a
community FP8 requant.

Fetch the small files first:

```bash
mapfile -t META_FILES < <(
  jq -r '.siblings[].rfilename' "$R/hf-model.json" \
  | grep -E '(^README\.md$|\.json$|chat_template.*\.jinja$)'
)
hf download "$Q38_REPO" "${META_FILES[@]}" \
  --revision "$Q38_REV" --local-dir "$Q38_DIR" \
  2>&1 | tee "$R/hf-metadata-download.log"
```

### H+00:15 - same-architecture gate

Run the mechanized diff before downloading the weight shards:

```bash
set +e
python3 research/qwen38-prep-20260803/arch-diff-fields.py --expect-fp8 \
  "$Q38_DIR/config.json" "$Q36_REF/config.json" \
  2>&1 | tee "$R/config-diff.log"
CONFIG_RC=${PIPESTATUS[0]}
set -e

case "$CONFIG_RC" in
  0) echo "same-architecture config; continue" ;;
  2) echo "architecture matches; FP8 metadata waits for tensor-header proof" ;;
  *) echo "STOP: architecture/config contract changed"; exit 1 ;;
esac
```

Verify the tokenizer and thinking dialect. Qwen 3.8 must remain the Qwen ChatML class:

```bash
python3 - "$Q38_DIR/tokenizer_config.json" "$Q36_REF/tokenizer_config.json" <<'PY' \
  2>&1 | tee "$R/tokenizer-contract.log"
import json
import sys

new = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))
hard = ("tokenizer_class", "pretokenize_regex", "add_bos_token")
bad = [(key, ref.get(key), new.get(key)) for key in hard if new.get(key) != ref.get(key)]
template = new.get("chat_template") or ""
markers = ("enable_thinking", "add_generation_prompt", "<think>")
missing = [marker for marker in markers if marker not in template]
print("hard-field diffs:", bad)
print("missing Qwen thinking markers:", missing)
if bad or missing:
    raise SystemExit(1)
PY

jq -S '.model.type, .pre_tokenizer' "$Q38_DIR/tokenizer.json" \
  > "$R/tokenizer38-structure.json"
jq -S '.model.type, .pre_tokenizer' "$Q36_REF/tokenizer.json" \
  > "$R/tokenizer36-structure.json"
diff -u "$R/tokenizer36-structure.json" "$R/tokenizer38-structure.json" \
  | tee "$R/tokenizer-structure.diff"
test "${PIPESTATUS[0]}" -eq 0
```

Any tokenizer-class, pre-tokenizer, regex, or thinking-template change is a bring-up lane,
not a runbook edit.

On a true same-architecture result, `ModelConfig` must resolve the existing Qwen3.5 geometry
classes, including the explicit appended MTP row. Record the class count and transitions in the
receipt. Any new per-layer head array, rotary class, window, or attention-gate layout is not a
same-architecture fast path: STOP and return to the declarative geometry phase above.

### Config-diff checklist

These are the frozen Qwen 3.6 values and memra's current interpretation. “GO with gates” means
the parser consumes the value generically; it does not waive model-backed, HF-reference, or
serve gates.

| Config field | Frozen 3.6 value | Day-one classification |
|---|---|---|
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` | any change: STOP; current HF mapping does not recognize a new Qwen 3.8 class |
| top/text `model_type` | `qwen3_5` / `qwen3_5_text` | any change: STOP; only these names map to the Qwen35 engine |
| `num_hidden_layers` | `64` | GO with gates; parsed size |
| `hidden_size` | `5120` | GO with gates; parsed size, but a change forbids the 3.6 MTP donor |
| `intermediate_size` | `17408` | GO with gates; parsed size |
| `num_attention_heads` | `24` | any change: STOP |
| `num_key_value_heads` | `4` | any change: STOP |
| `head_dim` | `256` | any change: STOP; FA dispatch and MTP interface are head-dimension keyed |
| `full_attention_interval` | `4` | any change: STOP |
| `layer_types` | 64 entries repeating `linear, linear, linear, full` | new type, changed cycle, interval mismatch, or count mismatch with `num_hidden_layers`: STOP |
| `linear_num_key_heads` | `16` | any change: STOP |
| `linear_num_value_heads` | `48` | any change: STOP |
| `linear_key_head_dim` / `linear_value_head_dim` | `128` / `128` | any change: STOP |
| `linear_conv_kernel_dim` | `4` | any change: STOP |
| `attention_bias` / `attention_dropout` | `false` / `0.0` | any change: STOP; these are not generic runtime knobs |
| `attn_output_gate` / `output_gate_type` | `true` / `swish` | any change: STOP; the Qwen35 forward assumes this gate contract |
| `hidden_act` | `silu` | any change: STOP |
| `tie_word_embeddings` | `false` | any change: STOP; tensor ownership and output-head loading change |
| `rope_parameters.rope_type` | `default` | any change, including YaRN: STOP |
| `rope_parameters.rope_theta` | `10000000` | any change: STOP |
| `rope_parameters.partial_rotary_factor` | `0.25` | any change: STOP |
| `rope_parameters.mrope_interleaved` | `true` | any change: STOP |
| `rope_parameters.mrope_section` | `[11,11,10]` | any change: STOP |
| `vocab_size` | `248320` | GO for plain loading if tokenizer gates pass; change forbids the 3.6 MTP donor |
| `bos_token_id` / `eos_token_id` | `248044` / `248044` | review chat stops; tokenizer/template gates must explain any change |
| `image_token_id` / `video_token_id` | `248056` / `248057` | text-only load may continue if tokenizer contract is unchanged |
| `mtp_num_hidden_layers` | `1` | any change: publish/spec STOP |
| `mtp_use_dedicated_embeddings` | `false` | any change: publish/spec STOP |
| `max_position_embeddings` | `262144` | GO with gates; parsed limit |
| `rms_norm_eps` | `1e-6` | GO with gates; parsed numeric value |
| `num_experts` | absent | any value: STOP; this is no longer the dense-hybrid architecture |
| `quant_method` / `fmt` | `fp8` / `e4m3` | explicit different value: STOP |
| `weight_block_size` | `[128,128]` | block-128 direct class; a different explicit block: STOP |
| `activation_scheme` | `dynamic` | explicit different value: STOP |

The config is not authoritative for scale granularity. The tensor sibling shape is:

- one scale value: `MEMRA_ST_E4M3`, per-tensor direct;
- `[out,1]` or `out` values: per-row, no direct kernel, STOP;
- `[ceil(out/128),ceil(in/128)]`: `MEMRA_ST_E4M3_BLK`, block-128 direct;
- anything else: unsupported, STOP.

### H+00:30 - full download and immutable manifest

```bash
hf download "$Q38_REPO" --revision "$Q38_REV" --local-dir "$Q38_DIR" \
  2>&1 | tee "$R/hf-full-download.log"

while read -r shard; do
  test -f "$Q38_DIR/$shard" || {
    echo "missing indexed shard: $shard"
    exit 1
  }
done < <(jq -r '.weight_map[]' "$Q38_DIR/model.safetensors.index.json" | sort -u)

find "$Q38_DIR" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > "$R/q38-files.sha256"
du -sh "$Q38_DIR" | tee "$R/q38-size.txt"
```

Do not mutate the downloaded config to make a diff pass. A RoPE override creates a different
model configuration and belongs in a separate lane.

### H+01:00 - build and HF reference environment

```bash
cargo build --release \
  --bin kernel-check \
  --bin run-gen \
  --bin run-spec \
  --bin concat-prime-probe \
  --bin frspec-owngen \
  --bin memra-server \
  2>&1 | tee "$R/build.log"

export HFV="$Q38_VENV_ROOT/q38-hf-$Q38_REV"
uv venv "$HFV"
uv pip install --python "$HFV/bin/python" --upgrade \
  torch transformers accelerate safetensors \
  2>&1 | tee "$R/hf-reference-install.log"
"$HFV/bin/python" -c \
  'import torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__)' \
  | tee "$R/hf-reference-versions.txt"
```

If the released model card names a newer minimum or an additional package, install that exact
requirement and record it. Do not use an older environment merely because it imports.

### H+01:30 - FP8 artifact classification

```bash
python3 tools/inspect-fp8-st.py "$Q38_DIR" --require-direct \
  2>&1 | tee "$R/fp8-header-census.log"
```

This check is header-only. It must report at least one FP8 weight and zero per-row/unsupported
weights. Runtime still has to prove finite scales, no E4M3 NaN refusal, supported transforms,
native residency, and native prefill dispatch.

### H+02:00 - model-backed kernel battery

`kernel-check` cannot use an ST directory as its real-weight oracle. Pass the frozen Qwen 3.6
GGUF so the 27B shape sections are model-backed, while the synthetic FP8 cells cover per-tensor
and block-128 arithmetic:

```bash
flock /tmp/gpu5090.lock \
  target/release/kernel-check "$Q36_GGUF" \
  2>&1 | tee "$R/kernel-check.log"
grep -q 'ALL GREEN' "$R/kernel-check.log"
```

A new Qwen 3.8 shape that is not represented by the existing synthetic cells is a kernel lane,
not a reason to accept a skipped section.

### H+02:30 - naked direct-path proof

Use a prompt longer than the prefill GEMM threshold so block-128 MMQ can dispatch:

```bash
export PROBE=tools/fast-gate/prompts/probe.txt
FP8_DEFAULT_ENV=(
  -u MEMRA_ST_E4M3
  -u MEMRA_ST_E4M3_BLK
  -u MEMRA_FP8_MMQ
  -u MEMRA_FP8_FOLD
  -u MEMRA_FP8_BLK_GPU
  -u MEMRA_PP_FP8
)

flock /tmp/gpu5090.lock env "${FP8_DEFAULT_ENV[@]}" \
  MEMRA_CHAT=1 \
  MEMRA_NGEN=32 \
  MEMRA_PROMPT_FILE="$PROBE" \
  MEMRA_RESIDENCY_CENSUS=1 \
  target/release/run-gen "$Q38_DIR" \
  2>&1 | tee "$R/run-gen-naked.log"

grep -q 'argmax=.*MATCH' "$R/run-gen-naked.log"
grep -Eq 'F8_E4M3(_BLK)?:[[:space:]]+[1-9]' "$R/run-gen-naked.log"
if grep -Eq 'block-128[[:space:]]*:[[:space:]]*[1-9]' "$R/fp8-header-census.log"; then
  flock /tmp/gpu5090.lock env "${FP8_DEFAULT_ENV[@]}" \
    MEMRA_CHAT=1 \
    MEMRA_PROMPT_FILE="$PROBE" \
    MEMRA_PP_ONLY=1 \
    MEMRA_PP_WARMUP=1 \
    MEMRA_PP_REPS=1 \
    target/release/run-gen "$Q38_DIR" \
    2>&1 | tee "$R/run-gen-pp-direct.log"
  grep -Eq 'fp8-mmq dispatches: [1-9]' "$R/run-gen-pp-direct.log"
fi
```

Run one diagnostic rollback census. This is proof of the seam, not a production arm:

```bash
flock /tmp/gpu5090.lock env "${FP8_DEFAULT_ENV[@]}" \
  MEMRA_ST_E4M3=0 \
  MEMRA_CHAT=1 \
  MEMRA_NGEN=1 \
  MEMRA_PROMPT_FILE="$PROBE" \
  MEMRA_RESIDENCY_CENSUS=1 \
  target/release/run-gen "$Q38_DIR" \
  2>&1 | tee "$R/run-gen-q8-rollback-census.log"
! grep -Eq 'F8_E4M3(_BLK)?:[[:space:]]+[1-9]' "$R/run-gen-q8-rollback-census.log"
grep -Eq 'Q8_0:[[:space:]]+[1-9]' "$R/run-gen-q8-rollback-census.log"
```

The naked log must contain native `F8_E4M3` and/or `F8_E4M3_BLK` residency. The rollback log
must remove those native bytes and increase Q8_0 residency. Residual small Q8_0 tensors are
allowed; routing the FP8 projection bank through Q8_0 is not. Any native decline, NaN refusal,
bad grid, or zero block-MMQ dispatch is STOP.

Never set `MEMRA_FP8_FOLD=1`; it is lossy. Never set `MEMRA_FP8_MMQ=0`; that reintroduces
dequant-per-call. Never use `MEMRA_FP8_MMQ=1`; that enables the duplicate stash source.

### H+03:15 - Hugging Face greedy-reference gate

Generate the reference from the same official FP8 directory, same prompt, same chat template,
thinking enabled, and greedy decoding. The comparator checks the rendered prompt ids before it
checks the generated ids:

```bash
flock /tmp/gpu5090.lock \
  "$HFV/bin/python" tools/hf-greedy-reference.py "$Q38_DIR" \
  --prompt-file "$PROBE" \
  --tokens-out "$R/hf-greedy.json" \
  --max-new-tokens 32 \
  2>&1 | tee "$R/hf-greedy.log"

python3 tools/compare-greedy-tokens.py \
  "$R/hf-greedy.json" "$R/run-gen-naked.log" \
  2>&1 | tee "$R/hf-vs-memra.log"
```

The gate is exact prompt-token and generated-token identity. A mismatch is not waived as “close
logits”; prompt-token failure is tokenizer/template work, while a generated-token failure after
prompt parity is model arithmetic.

### H+04:00 - chunk invariance

The probe and wrapper accept an HF safetensors directory:

```bash
MEMRA_CHUNKINV_LOG="$R/chunk-invariance.raw.log" \
flock /tmp/gpu5090.lock \
  tools/chunk-invariance-gate.sh "$Q38_DIR" \
  --chunks 2048,64,32 --steps 48 --expect-invariant \
  2>&1 | tee "$R/chunk-invariance.gate.log"
grep -q 'chunk-invariance-gate: PASS' "$R/chunk-invariance.gate.log"
```

This same-family example reuses the standing Qwen chunk contract and its calibrated canary. If
the config audit found a new window/class boundary, do not reuse the Qwen prompt or seam; generate
a dedicated chunk/tick/B>1 scaffold from the real mechanism as described above.

Any chunk-dependent result is STOP. Do not tune the chunk list or switch the expectation to make
the release pass.

### H+04:30 - embedded MTP gate

If the checkpoint carries its own MTP tensors, run without `MEMRA_SPEC_K` so the binary executes
its K=1..8 self-consistency battery:

```bash
if jq -e 'any(.weight_map | keys[]; startswith("mtp."))' \
  "$Q38_DIR/model.safetensors.index.json" >/dev/null; then
  flock /tmp/gpu5090.lock env \
    -u MEMRA_SPEC_K \
    -u MEMRA_MTP_DRAFT \
    -u MEMRA_FRSPEC_TRIM \
    MEMRA_CHAT=1 \
    MEMRA_NGEN=64 \
    MEMRA_PROMPT_FILE="$PROBE" \
    target/release/run-spec "$Q38_DIR" \
    2>&1 | tee "$R/run-spec-embedded.log"
  grep -q 'SELF-CONSISTENCY PASS' "$R/run-spec-embedded.log"
else
  printf 'WAIT: checkpoint carries no embedded MTP tensors; run-spec waits for a compatible draft\n' \
    | tee "$R/run-spec-embedded.WAIT"
fi
```

Plain first-light evidence may continue without a head, but publication does not: `run-spec`
waits until either the embedded head or the compatible external draft below exists.

### H+05:15 - serving and thinking surface

Run the existing ST battery:

```bash
flock /tmp/gpu5090.lock \
  tools/serve-st-gate.sh "$Q38_DIR" \
  2>&1 | tee "$R/serve-st-gate.log"
grep -q 'serve-st-gate: 0 failed' "$R/serve-st-gate.log"
```

Pin the per-architecture thinking mapping in unit tests:

```bash
cargo test -p memra-tokenizer qwen_think_mode_covers_all_three_directions \
  2>&1 | tee "$R/test-qwen-thinking.log"
cargo test -p memra-server reasoning_effort_maps_to_think_switch \
  2>&1 | tee "$R/test-reasoning-effort.log"
```

Then exercise the actual Qwen 3.8 template:

```bash
export ADDR=127.0.0.1:8188
export BASE=http://$ADDR
flock -F /tmp/gpu5090.lock env MEMRA_COMPAT=openai MEMRA_SERVE_SPEC=0 \
  MEMRA_MODELS="q38=$Q38_DIR" MEMRA_ADDR="$ADDR" \
  target/release/memra-server > "$R/thinking-server.log" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true' EXIT
for _ in $(seq 600); do
  curl -sf "$BASE/health" >/dev/null && break
  sleep 2
done
curl -sf "$BASE/health" >/dev/null
grep -Eq 'q38: template caps .*think=true think_switch=true chat_ok=true' \
  "$R/thinking-server.log"

for effort in absent none high; do
  extra=
  [ "$effort" != absent ] && extra=",\"reasoning_effort\":\"$effort\""
  curl -fsS "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"q38\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue?\"}],
         \"max_tokens\":64,\"temperature\":0$extra}" \
    > "$R/thinking-$effort.json"
done

python3 - "$R" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = {
    name: json.load(open(root / f"thinking-{name}.json"))["choices"][0]["message"]
    for name in ("absent", "none", "high")
}
for name, message in records.items():
    emitted = (message.get("reasoning") or "") + (message.get("content") or "")
    assert emitted.strip(), f"{name}: empty response"
assert not (records["none"].get("reasoning") or ""), "none did not close thinking"
assert (records["absent"].get("reasoning") or ""), "default did not use Qwen thinking"
assert (records["high"].get("reasoning") or ""), "high did not enable Qwen thinking"
print("THINKING-SURFACE PASS: default/high on, none off")
PY

kill "$SPID"
wait "$SPID" || true
trap - EXIT
```

For the Qwen binary switch, `low`, `medium`, and `high` all mean thinking on; `none` and
`minimal` mean off; absent preserves the template default, which is on.

### H+06:30 - own-generation trim regime

Ranks always come from Qwen 3.8's own generations with chat templating on. Use bounded chunks
so the shared GPU lock is released between batches:

```bash
export PACK=research/gemma4-bringup/corpus-prompts
export RANKS="$A/q38-own-ranks-32768.gguf"
export CORPUS="$A/q38-own-corpus-ids.txt"
test "$(find "$PACK" -type f -name '*.txt' | wc -l)" -eq 254

while [ ! -f "$RANKS" ]; do
  flock /tmp/gpu5090.lock \
    target/release/frspec-owngen "$Q38_DIR" "$RANKS" 32768 \
    --ngen 1024 \
    --corpus-out "$CORPUS" \
    --limit 32 \
    --validate \
    "$PACK" \
    2>&1 | tee -a "$R/frspec-owngen.log"
done

awk '
  { total += NF }
  END {
    print "own-generated tokens:", total
    if (total < 131072) exit 1
  }
' "$CORPUS" | tee "$R/own-corpus-count.txt"
sha256sum "$CORPUS" "$RANKS" "$RANKS.txt" | tee "$R/own-ranks.sha256"
```

This is the daily-drafter regime's canonical 254-prompt pack, greedy decoding, chat template on,
and bounded 32-prompt lock holds. The 1,024-token window is deliberately larger than the older
512-token receipts so the current four-times-top-N floor is enforceable rather than theoretical.
The floor is at least 131,072 own-generated tokens for a 32,768-row head. If the count misses it,
STOP and expand the frozen prompt pack; do not build from the warning-sized corpus. The final
`--validate` run records embedded-MTP baseline versus runtime-trimmed e2e throughput. Acceptance
explains the result; e2e tokens/s decides it.

### H+09:00 - build the external trimmed drafter

The current standalone builder extracts MTP bytes from a GGUF donor. It does not extract from
the ST directory. The approved donor variant is executable only when the Qwen 3.8 trunk/MTP
interface is exactly the frozen Qwen 3.6 interface:

```bash
python3 - "$Q38_DIR/config.json" "$Q36_REF/config.json" <<'PY' \
  2>&1 | tee "$R/donor-interface.log"
import json
import sys

new = json.load(open(sys.argv[1])).get("text_config")
ref = json.load(open(sys.argv[2])).get("text_config")
keys = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
)
bad = [(key, ref.get(key), new.get(key)) for key in keys if new.get(key) != ref.get(key)]
print("donor interface diffs:", bad)
if bad:
    raise SystemExit(1)
PY

export DRAFT="$A/draft-q38-owntrim-nvfp4head-q4blk.gguf"
tools/make-trimmed-draft.sh \
  "$Q36_GGUF" "$RANKS.txt" "$DRAFT" 32768 \
  2>&1 | tee "$R/make-trimmed-draft.log"
sha256sum "$DRAFT" "$RANKS" "$RANKS.txt" \
  | tee "$R/drafter.sha256"

flock /tmp/gpu5090.lock env \
  -u MEMRA_SPEC_K \
  -u MEMRA_MTP_DRAFT \
  -u MEMRA_FRSPEC_TRIM \
  MEMRA_MTP_DRAFT="$DRAFT" \
  MEMRA_CHAT=1 \
  MEMRA_NGEN=64 \
  MEMRA_PROMPT_FILE="$PROBE" \
  target/release/run-spec "$Q38_DIR" \
  2>&1 | tee "$R/run-spec-owntrim.log"
grep -q 'SELF-CONSISTENCY PASS' "$R/run-spec-owntrim.log"

mkdir -p "$R/drafter-prompts"
cp "$PROBE" "$R/drafter-prompts/probe.txt"
cp research/e2e/prompts/p3-agentic-long.txt "$R/drafter-prompts/p3-agentic-long.txt"

run_draft_arm() {
  local arm=$1 rep=$2 draft=${3:-}
  local envs=(
    MEMRA_SPEC_K=3
    MEMRA_NGEN=256
    MEMRA_CHAT=1
    MEMRA_PROMPT_DIR="$R/drafter-prompts"
  )
  [ -n "$draft" ] && envs+=(MEMRA_MTP_DRAFT="$draft")
  flock /tmp/gpu5090.lock env \
    -u MEMRA_MTP_DRAFT \
    -u MEMRA_FRSPEC_TRIM \
    "${envs[@]}" \
    target/release/run-spec "$Q38_DIR" \
    2>&1 | tee "$R/drafter-ab-$arm-r$rep.log"
}

# Adjacent, alternating order: embedded/trimmed, then trimmed/embedded.
run_draft_arm embedded 1
run_draft_arm owntrim 1 "$DRAFT"
run_draft_arm owntrim 2 "$DRAFT"
run_draft_arm embedded 2

grep -hE '^\[SWEEP\]|SELF-CONSISTENCY' "$R"/drafter-ab-*.log \
  | tee "$R/drafter-ab-summary.txt"

flock /tmp/gpu5090.lock env \
  MEMRA_MTP_DRAFT="$Q36_DRAFT" \
  MEMRA_SPEC_K=3 \
  MEMRA_NGEN=256 \
  MEMRA_CHAT=1 \
  MEMRA_PROMPT_DIR="$R/drafter-prompts" \
  target/release/run-spec "$Q36_GGUF" \
  2>&1 | tee "$R/q36-daily-reference.log"
```

The donor supplies only byte-verbatim NextN/MTP block bytes; ranks are from Qwen 3.8, and the
serving model supplies token embeddings. Verify-based speculation remains exact, but donor drift
can reduce acceptance. The four A/B runs are one adjacent, alternating N=2 session over short and
agentic-long prompt classes. Adopt the external draft only when both per-prompt evidence and the
aggregate e2e tokens/s justify it; acceptance is diagnostic.

The Qwen 3.6 daily run is a frozen operational reference, not a Qwen 3.8 acceptance threshold.
Keep its number separate from the embedded-versus-owntrim Qwen 3.8 decision.

If the donor-interface check fails, do not create a full-model GGUF bridge. Keep the embedded ST
MTP path and open a narrowly scoped ST-MTP extraction lane before publishing an external draft.

### Completion and STOP matrix

Day one is complete only when the receipt directory contains:

- official repo id and immutable revision;
- full local file hashes;
- config and tokenizer verdicts;
- FP8 header census;
- model-backed `kernel-check` raw log;
- naked and rollback residency logs;
- HF and memra token streams plus exact comparison;
- chunk-invariance raw log;
- embedded and external-draft `run-spec` logs;
- ST serve battery;
- thinking default/off/on responses;
- own-generation corpus/rank/draft paths under `/data`, their hashes, and the e2e verdict.

Hard architecture STOP, open a bring-up lane:

- new `architectures`/`model_type`;
- changed full-attention cycle, attention/KV head counts, head dimension, attention gate/bias,
  GDN geometry, RoPE contract, tokenizer class/pre-tokenizer, or dense-to-MoE change;
- new tensor names or shapes the current Qwen35 mapping cannot resolve.

Artifact/direct-path STOP:

- no official FP8 release;
- explicit non-E4M3 or non-128 block metadata;
- per-row or unsupported FP8 scale layouts;
- native FP8 residency absent, block-MMQ dispatch absent for a block artifact, or any FP8 bank
  silently landing in Q8_0;
- HF greedy token mismatch.

Correctness/publish STOP:

- `kernel-check`, internal argmax, chunk invariance, or K=1..8 self-consistency failure;
- no embedded or interface-compatible external MTP head that passes `run-spec`;
- missing `enable_thinking`, `think_switch=false`, or live default/off/on mapping failure;
- external donor interface mismatch or a trimmed draft that loses e2e throughput.

Do not merge, tag, or publish support from this runbook until every applicable gate is green.

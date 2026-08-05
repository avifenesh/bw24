# Qwen 3.8 27B — day-one bring-up runbook

Written 2026-08-03 (lane/qwen38-prep), before the release; revised 2026-08-04 on the
train (FP8-ST leg added, dry-run fixes — see
`research/qwen38-prep-20260803/DRYRUN-20260804.md`). Target: HF repo appears →
"published as supported" in one day, IF the arch-diff is clean. Companion files:
`research/qwen38-prep-20260803/AUDIT.md` (what we know about 3.6-27B),
`baseline-36-27b.jsonl` (the frozen numbers every 3.8 cell diffs against),
`WATCH.md` (where the release lands).

Two serving legs run in parallel after the arch-diff (owner direction 2026-08-04:
FP8-ST serving on the exact arm is the prod direction):

- **Leg A — FP8-ST** (§3b): the official FP8 block-128 checkpoint served as a
  safetensors DIR by memra-server. Exact by construction (block-128 → Q8_0 host
  dequant, or the byte-identical `MEMRA_FP8_BLK_GPU` device dequant). No spec
  day-one — dir checkpoints are spec-QUARANTINED (research/serve-st-20260803).
- **Leg B — GGUF** (§3): the house NVFP4+MTP conversion, parallel/fallback leg and
  the ONLY leg with spec/drafter rows day-one.

Ground rules that apply to every step:

- Every GPU command on the 5090 goes through `flock /tmp/gpu5090.lock` (rig is shared).
- Raw logs are part of the deliverable — `tee` first, parse second (CLAUDE.md evidence
  discipline). Work dir for receipts: `research/qwen38-bringup-<date>/`.
- The deployment bar is >=1.1x e2e vs llama.cpp + own-gen trimmed drafter, per model,
  before "published as supported" (owner rule, see AUDIT.md §4). Gates green = bring-up
  floor, not the finish.

## 0. Branch + workspace

```bash
cd ~/projects/bw24-unified
git checkout restructure/public-split && git pull
git checkout -b lane/qwen38-bringup
mkdir -p research/qwen38-bringup-$(date +%Y%m%d)
```

## 1. Download (first hour; no GPU)

Grab safetensors + config FIRST (arch-diff needs only config.json + tokenizer, ~20 MB):

```bash
D=/data/ai-ml/hf-models/qwen38-27b
huggingface-cli download Qwen/Qwen3.8-27B config.json generation_config.json \
    tokenizer.json tokenizer_config.json chat_template.jinja --local-dir $D-hf-min
# full BF16 weights in the background while the arch-diff runs:
nohup huggingface-cli download Qwen/Qwen3.8-27B --local-dir $D > $D-dl.log 2>&1 &
# FP8-ST leg artifact: the official FP8 sibling (3.6 precedent: Qwen/Qwen3.6-27B-FP8
# was created the same day as the BF16 repo, block-128 e4m3, dynamic activations,
# vision tower excluded). This DIR is what memra-server serves in leg A:
nohup huggingface-cli download Qwen/Qwen3.8-27B-FP8 --local-dir $D-fp8 \
    > $D-fp8-dl.log 2>&1 &
```

If the FP8 sibling does NOT appear day-one, leg A waits (do not substitute a community
FP8 requant — official checkpoint only, it is the "exact arm" premise) and leg B carries
the day. If Qwen ships an official GGUF repo (they did not for 3.6; unsloth/NVIDIA filled
that gap — see WATCH.md), download it too but treat it as a cross-check, not the artifact:
our shipped 3.6 artifact is a house conversion (NVFP4 trunk + Q4_K_M/Q5_K head + baked MTP).

## 2. Arch-diff checklist (minutes; blocks everything downstream)

Reference = `/data/ai-ml/hf-models/qwen36-27b-hf-min/config.json`. Mechanized:

```bash
python3 research/qwen38-prep-20260803/arch-diff-fields.py \
    $D-hf-min/config.json /data/ai-ml/hf-models/qwen36-27b-hf-min/config.json
# exit 0 = fast-path (numeric-only diffs), exit 1 = STOP-CLASS diff or missing field.
# Self-tested 2026-08-04 on the 3.6 config (0 diffs) and the 3.6-FP8 sibling (4
# quantization_config diffs only): research/qwen38-prep-20260803/arch-diff-selftest-36.txt
```

3.6-27B values listed so a diff is readable without the reference file open:

| Field | 3.6-27B value | Divergence class |
|---|---|---|
| `architectures` / `model_type` | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | new string = check converter support first (§3) |
| `text_config.num_hidden_layers` | 64 | numeric-only diff = FAST PATH |
| `text_config.hidden_size` / `intermediate_size` | 5120 / 17408 | numeric-only = fast path |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 4 / 256 | numeric = fast path; NEW head_dim value needs an FA-arm check (fa dispatch is head_dim-keyed) |
| `layer_types` + `full_attention_interval` | linear_attention x3 : full_attention, interval 4 | any change in the pattern or a new layer type = STOP |
| `linear_*` (GDN block) | key_heads 16, value_heads 48, k/v head_dim 128, conv_kernel 4 | numeric = fast path; missing block entirely = STOP |
| `rope_parameters` | theta 1e7, partial_rotary_factor 0.25, mrope_interleaved, section [11,11,10] | theta/factor numeric = fast path; new rope_type or scaling block (yarn etc.) = STOP |
| `vocab_size` | 248320 | changed size with same tokenizer family = fast path (embed/head re-shape rides conversion); new tokenizer.json model class = STOP |
| special tokens | bos/eos 248044, image 248056, video 248057 | additions fine; EOS semantics change = re-check chat stops |
| `chat_template.jinja` | 3.6 template on disk | diff it; template-only change = fast path (re-derive ranks with template ON) |
| `mtp_num_hidden_layers` / `mtp_use_dedicated_embeddings` | 1 / false | more MTP layers or dedicated embeds = drafter pipeline change, STOP on drafter step only |
| `max_position_embeddings` | 262144 | shrink/grow = fine |
| vision tower present | yes (we serve text-only) | ignored, same as 3.6 |
| FP8 repo `quantization_config` | 3.6-FP8: `quant_method fp8`, `fmt e4m3`, `weight_block_size [128,128]`, `activation_scheme dynamic` | block-128 = the merged loader's class, leg A proceeds; per-tensor scalar = also fine (`MEMRA_ST_E4M3` class); any OTHER block size or a non-e4m3 fmt = leg A STOP (loader assumes 128×128), leg B unaffected |

**Same-arch fast path** = every diff is numeric-only (sizes, counts, theta) with the same
`model_type`, same layer_types pattern, same tokenizer class, same rope type, MTP >= 1 layer.
Proceed to §3.

**Divergent-arch STOP rule** — any of the following means STOP, do not force day-one publish;
open a real bring-up lane (per-model plan file, gemma4-bringup style) instead:

- new attention variant (layer_types pattern change, sliding-window fields appearing, new
  rope_type/scaling, attn gating semantics change)
- MoE-ification (any `num_experts` field — 27B going MoE changes the whole dispatch class)
- tokenizer swap (different tokenizer.json model class or pretokenizer regex — goldens,
  ranks, and every prompt-id file die)
- `model_type` unknown to the converter fork (§3) with structural tensor-name changes
- MTP head removed entirely (drafter falls back to donor-block variant,
  docs/DRAFT-REGIME.md §"Targets that ship no NextN head" — 3.6-27B is the natural donor
  ONLY if trunk interface matches: n_embd, head_dim, n_head, n_head_kv, identical vocab)

Record the diff verdict in `research/qwen38-bringup-<date>/arch-diff.md` before touching GPU.

## 3. GGUF conversion + the quant lanes we ship

Converter = the house llama.cpp fork (`~/projects/llama.cpp`, conversion/qwen.py —
`Qwen3_5TextModel` handles MRoPE + MTP + NVFP4 repack; `convert_hf_to_gguf.py` accepts
mixed NVFP4+FP8 groups since bb090d1f1). If `model_type` moved to `qwen3_8`, the first code
task is a subclass in conversion/qwen.py (usually a registration + any new hparam mapping —
small if fast-path).

3.6-27B ships two quant lanes (README.md:86): **NVFP4 trunk + Q4_K_M-class MTP block baked**
(the daily) — that is the ONE artifact to reproduce day-one. The recipe that produced the
daily-equivalent unsloth artifact (convert.log/mtp-convert.log in
`/data/ai-ml/hf-models/unsloth-qwen36-27b-nvfp4-gguf/`):

```bash
cd ~/projects/llama.cpp    # interpreter = .venv/bin/python3 — system python3 has no
                           # torch (dry-run finding 2026-08-04; venv now carries torch
                           # 2.13 cpu + numpy + transformers + safetensors)
# free pre-flight, before any weights write (validated on the 3.6 NVFP4 source:
# 1252 tensors / 28.2G main, 18 tensors / 2.5G mtp):
.venv/bin/python3 convert_hf_to_gguf.py $D --dry-run --outfile /tmp/dryrun-q38.gguf
# main model: NVFP4 experts/trunk repack straight from the modelopt/NVFP4 checkpoint if
# Qwen or NVIDIA publish one; from BF16 source, quantize via llama-quantize after an
# f16/bf16 convert. Embed/head land Q5_K (the "q5h" in the artifact name), imatrix-guided:
.venv/bin/python3 convert_hf_to_gguf.py $D --outfile $OUT/qwen38-27b-trunk.gguf [--outtype ...]
# MTP block sidecar (bf16 in the checkpoint -> q8_0; the acceptance-proven form):
.venv/bin/python3 convert_hf_to_gguf.py $D --mtp --outtype q8_0 --outfile $OUT/mtp-qwen38-27b-q8_0.gguf
```

Exact flags: mirror the 3.6 logs (they are the receipts — `convert.log` line 1 onward).
If only BF16 ships day-one, convert + quantize BF16→NVFP4 with the 3.6 imatrix procedure
(new imatrix from the new model, NOT the 3.6 one) and note the artifact is house-quantized.

Sanity gate before GPU: header dump via
`~/projects/llama.cpp/.venv/bin/python3 ~/projects/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py --no-tensors $OUT/qwen38-27b.gguf`
— n_layers/vocab/head counts match config.json; MTP block present as
`blk.<n_layers>.nextn.*` (3.6: blk.64, rig5090.jsonl:185).

## 3b. FP8-ST leg — the prod direction (parallel to §3, no conversion)

The official FP8 block-128 DIR is served as-is; "conversion" is zero steps. What gates it:

1. **Config check** (part of §2): `quantization_config` must be
   `fp8 / e4m3 / weight_block_size [128,128]` (the 3.6-FP8 shape). Anything else = leg A
   STOP, note it in arch-diff.md, continue leg B.
2. **Loader gates, block-128 class** — already merged and green on synthetic + real
   block-128 checkpoints (v0.68 lanes), re-run in-config on the new model:
   ```bash
   # kernel-check carries the fp8-blk-gpu bit-parity section (ARM B' device dequant
   # byte-equal to host, kernel_check.rs:3415) — runs in the §4 full kernel-check, no
   # separate invocation. The load-time dequant path this section gates is now the
   # FALLBACK, not the default (see below), but it must still be green: it is where every
   # native-residency precondition falls through.
   ```
   **NAKED IS THE FAST PATH ON THIS CLASS since lane/fp8-blk128-decode (2026-08-05) —
   day-one needs NO flags to get it.** If 3.8-FP8 ships `weight_block_size [128,128]`
   like 3.6, the checkpoint's e4m3 bytes are the ONE resident copy (decode dequants
   per-k128 in `qmatvec_e4m3_blk_mmvq`) and prefill rides the per-block MMQ tile
   directly. Measured on the 3.6-27B block-128 artifact: decode +1.69%, VRAM −430 MiB,
   pp512 +0.83% over the Q8_0-slab floor. Rollback seams if the new model misbehaves:
   `MEMRA_FP8_MMQ=0` (prefill route only), `MEMRA_ST_E4M3_BLK=0` (this class's residency),
   `MEMRA_ST_E4M3=0` (all native e4m3 residency).
   Still do NOT reach for the other perf doors day-one: `MEMRA_FP8_FOLD` is LOSSY (greedy
   MISMATCH pos 20, research/fp8st receipts) and `MEMRA_FP8_MMQ=1`'s extra meaning — the
   duplicate-copy STASH source — buys nothing on a natively-resident checkpoint. Both stay
   off. (The old "MEMRA_FP8_MMQ measured 0.81-0.94x the floor, stays OFF" line was correct
   for the stash denominator and wrong for this one; `research/fp8blk-20260805/VERDICT.md`
   has the arithmetic.)
3. **First light, CLI arm** (same §4 step-2 form, ST dir instead of GGUF):
   ```bash
   flock /tmp/gpu5090.lock env MEMRA_NGEN=20 MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt \
       target/release/run-gen $D-fp8 2>&1 | tee research/qwen38-bringup-*/first-light-fp8st.log
   ```
   Expected: argmax MATCH lines, coherent text.
4. **ST serving battery**:
   ```bash
   flock /tmp/gpu5090.lock tools/serve-st-gate.sh $D-fp8 \
       2>&1 | tee research/qwen38-bringup-*/serve-st-gate-fp8.log
   ```
   Green = 4 gates: /models lists it, chat coherent through the checkpoint's own
   template, CLI-vs-server greedy token streams IDENTICAL, spec quarantine holds.
   (Dry-run 2026-08-04 on the merged state, qwen35-4b BF16 ckpt: 0 failed —
   research/qwen38-prep-20260803/serve-st-gate-dryrun-20260804.log.)
5. **ST serving cells** (perf record, self-competition protocol per §7): decode tok/s
   N=5 medians on pp512.txt / p3-agentic-long.txt via run-gen on the DIR, interleaved
   against the NV-27B-ST 3.6 standing config re-measured same-session (rig5090.jsonl
   lines 180-289 lineage; every historical 27B ST pairing carries a correction row —
   keep the same-session law). Serve-side smoke: `tools/serve-smoke.sh` with
   `MEMRA_MODELS="q38fp8=$D-fp8"`.
6. **Known limits day-one**: spec/drafter rows are leg-B-only (dir ckpts
   spec-quarantined until the ST serve-spec exactness gate goes green — root cause open,
   research/serve-st-20260803/RESULTS.md); `MEMRA_PP_FP8` prefill GEMM on this class
   rides per-tensor scalar only (cuBLASLt refuses block-128 on sm_120,
   research/fp8st-20260803/P1-VERDICT.md) — prefill stays on the exact Q8_0-resident
   floor unless a leg-specific lane says otherwise.

## 4. Gate battery (order matters: cheap first)

All commands from repo root, binaries in `target/release/` (rebuild first: `cargo build
--release`). `M=$OUT/qwen38-27b.gguf`.

1. **fast-gate tier-0** (seconds, catches build/link + scoped kernel oracles):
   ```bash
   tools/fast-gate/fast-gate.sh --tier 0
   ```
   Green = compile + scoped kernel-check pass. (New-model bring-up mostly doesn't touch
   kernels; this guards whatever loader/dispatch edits §3 forced.)
2. **First light, run-gen argmax** (the bring-up moment of truth):
   ```bash
   flock /tmp/gpu5090.lock env MEMRA_NGEN=20 MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt \
       target/release/run-gen $M 2>&1 | tee research/qwen38-bringup-*/first-light.log
   ```
   Expected green: `MATCH` on both prefill-vs-decode argmax lines, no MISMATCH-STRUCTURED,
   coherent text. A MISMATCH here = loader/kernel bug, stop and bisect (the 3.6 bring-up
   history: load-tail OOM via TENSOR-map F8 resolution, rig5090.jsonl:184 — check there first).
3. **kernel-check ALL GREEN** (~4.5 min full):
   ```bash
   flock /tmp/gpu5090.lock target/release/kernel-check 2>&1 | tee .../kernel-check.log
   ```
   Note: the `nvfp4-27b-shape` section pins the 3.6 file path (kernel_check.rs:1877 at
   2026-08-04 HEAD; grep the section name, the line drifts) — it keeps gating 3.6
   regardless. If 3.8 introduces a new weight shape class, add a section; same-arch
   same-shapes needs nothing. The FP8-ST leg's `fp8-blk-gpu` bit-parity section
   (kernel_check.rs:3415) rides the same full run.
4. **run-gen argmax at depth** (the deep-fa arms): repeat step 2 with
   `MEMRA_PROMPT_FILE=research/e2e/prompts/p3-agentic-long.txt` (or the board 6257-tok
   prompt). Expected: MATCH.
5. **run-spec K=1..8 self-consistency** (needs the drafter from §6 — run K=1..8 sweep after
   the draft exists; a pre-drafter smoke with the baked MTP head at K=3 is legitimate):
   ```bash
   for K in 1 2 3 4 5 6 7 8; do
     flock /tmp/gpu5090.lock env MEMRA_SPEC_K=$K MEMRA_DRAFT=$DRAFT MEMRA_NGEN=64 \
       target/release/run-spec $M 2>&1 | tee -a .../run-spec-sweep.log
   done
   ```
   Expected green: `SELF-CONSISTENCY PASS` every K, acceptance > 0.
6. **prime-gate + serve-smoke** (serving surface): `tools/serve-smoke.sh` with the model, and
   a prime-gate run mirroring tools/local-ci.sh:79-84 form. Expected: PASS lines, exit 0.
   Day-one serve-feature smokes on the NEW model (features merged post-runbook, v0.68/0.69
   train — each is model-load-sensitive, so smoke them on the bring-up model, not just the
   models they were gated on):
   - **Constrained decoding**: one `/v1/chat/completions` with
     `response_format {"type":"json_schema", ...}` → valid JSON matching the schema, and
     one `{"type":"json_object"}` → valid JSON (docs/SERVING.md constrained section;
     plain constrained-greedy gated at 99.4% of unconstrained on q9). Draft-side grammar
     masking is default-on — spec+constrained needs no flag (leg B only; leg A has no spec).
   - **/v1/models OR-schema**: `curl $BASE/v1/models` → the model listed with the
     OR-schema enrichment fields (serve-tail lane form,
     research/serve-tail-20260804/v1-models.json is the reference shape).
   - **Graceful drain**: SIGTERM mid-stream → in-flight stream completes, new request
     503s, exit 0 (research/serve-tail-20260804/run-battery.sh item 3 is the ready-made
     probe — run it with the new model path as $1).
7. **Full battery on the final tree** (the merge/tag gate, unchanged):
   ```bash
   tools/local-ci.sh 2>&1 | tee .../local-ci.log
   ```

## 5. Register the model in the gate tables (same commit as bring-up)

- `tools/fast-gate/models.tsv`: add a `k38` argmax probe row (copy the k27 row, new path).
- `tools/fast-gate/map.tsv`: extend the NVFP4 rows (27, 30, 32) with `k38` if the artifact
  is NVFP4+Q45K-class like 3.6.
- Goldens: ONLY at a full-battery green point —
  `tools/fast-gate/fast-gate.sh --refresh-goldens` (docs/TESTING.md:78-90).
- Consider a q38 cell in `research/tune-data/perf-cells.json` — NOTE: 3.6-27B never got one
  (audit §3), which is a standing gap; adding both q27 and q38 plain cells closes it.

## 6. Drafter build — REQUIRED before publish, not optional (leg B / GGUF only)

Leg A note: the drafter and every spec row belong to the GGUF leg — dir checkpoints are
spec-quarantined in serve (§3b item 6). Leg A publishes plain serving cells only.

Own-gen trimmed drafter per docs/DRAFT-REGIME.md (laws 1-3). ~30-60 min GPU for ranks; use
the bounded-chunk mode on the shared rig:

```bash
# 1. own-gen ranks (chat template ON — we serve chat), chunked to release the GPU lock:
flock /tmp/gpu5090.lock target/release/frspec-owngen $M ranks38.gguf 32768 \
    --corpus-out corpus-ids.txt --limit 64        # rerun until the final chunk writes ranks
# 2. extract + trim + quantize (byte-verbatim from the SERVING gguf's own MTP block):
tools/make-trimmed-draft.sh $M ranks38.gguf.txt draft-qwen38-owntrim.gguf 32768 [imatrix.gguf]
# 3. validate (GOOD/WASH/BAD verdict, baseline-vs-trimmed A/B):
flock /tmp/gpu5090.lock target/release/frspec-owngen $M out.gguf --validate
```

Never reuse 3.6 ranks (law 1: foreign ranks measured −12 acceptance pts on an identical
tokenizer). Never convert the draft from the HF checkpoint (law 2: converter drafts
collapsed to ~35-39% acceptance). Adopt on e2e tok/s only (law 3). Then complete the
regime checklist (DRAFT-REGIME.md bottom): draft + ranks to the HF bench repo
(huggingface.co/Avifenesh/memra-bench) with provenance.

## 7. Board rows + the self-competition pairing (the publish gate)

**Measurement doctrine (owner, 2026-08-03): llama benching is STOPPED.** The llama
denominators recorded through 2026-08-03 (board + rig5090.jsonl) are frozen reference
points — no fresh llama builds, no new llama-bench runs, no interleaved llama arms.
All pairing is self-competition: memra-vs-memra, interleaved against OUR baseline binary.

Protocol: memra `run-gen MEMRA_NGEN=128` on pp512.txt / p3-agentic-long.txt, and the
3.8 arms interleaved round-robin against the **3.6-27B rows re-measured in the same
session with the same binary** (that pairing is the verdict). N=5 medians, flock'd,
idle-gated, temps recorded per rep. Spec row: K-sweep for the best class config, N>=2,
same window (the 07-18 q27 K-sweep shape, rig5090.jsonl:344, minus the llama arm).

**Re-pair the 3.6-27B rows in the same session** — the frozen baseline notes the spec row
is stale vs HEAD (baseline-36-27b.jsonl STALENESS row); the 3.8-vs-3.6 story needs both
measured in one window.

Publish decision: judged against the FROZEN llama row for the 27B class (recorded
2026-07/08, non-interleaved caveat stated) — >=1.1x vs that frozen number or the lane
stays open — plus absolute tok/s tracked in jsonl as the ongoing self-competition
record. Do not run new llama sessions to manufacture a fresh denominator.

Board mechanics (CLAUDE.md perf-board rule):

```bash
$EDITOR research/tune-data/current-board.json   # add plain/depth/spec/supported_models rows
python3 tools/update-perf-board.py              # regenerates README/PERFORMANCE/SVGs
git add research/tune-data/current-board.json README.md docs/PERFORMANCE.md docs/perf-card*.svg
# + the rig5090.jsonl evidence row + raw logs dir, same commit
```

Never hand-edit PERF-* marker blocks; never push with --no-verify.

## 8. Release (docs/RELEASING.md)

```bash
bash tools/changelog.sh          # preview
git tag vX.Y.0 && git push origin vX.Y.0    # minor bump: model lane landed
```

Tag only after the full battery is green on the tagged commit on the rig
(docs/RELEASING.md:11-19). Commit prefixes: bring-up lands as `feat:`, board move as
`perf:`/`data:` per the changelog split.

## Wall-clock estimate, same-arch fast path

| Step | Wall | GPU-holding |
|---|---|---|
| download (~55 GB BF16 safetensors — 3.6 precedent is 15 shards — + ~28 GB FP8 sibling; config first) | 0.5-2 h (bg) | none |
| arch-diff | 15 min | none |
| FP8-ST leg (first light + serve-st-gate + serving cells + serve smokes) | ~1.5 h, overlaps leg B's conversion window | short bursts |
| conversion + quant + header sanity | 1-2 h | none/negligible |
| gates 1-6 (first light → spec smoke) | ~1 h | short bursts |
| drafter (ranks + trim + validate) | 1.5-2.5 h | chunked bursts |
| run-spec K=1..8 + full local-ci | ~1 h | yes |
| board head-to-head (plain N=5 x2 depths + spec re-pair, incl. 3.6 re-pair) | 2-3 h | yes, serialized |
| board JSON + regenerate + release tag | 0.5 h | none |
| **Total** | **~8-11 h wall, one working day** | |

Assumes: fast-path arch, converter accepts the checkpoint with <1 h of code, no gate
failures. First structural surprise moves this to the divergent-arch lane — that is the
stop rule working, not the plan failing.

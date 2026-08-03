# Qwen 3.6 27B — current-state audit (2026-08-03)

Lane `lane/qwen38-prep` (base `restructure/public-split` @ 464c9da7). Purpose: freeze what the
repo knows about the 3.6 27B ("q27" / "k27") as the diff-anchor for the expected Qwen 3.8 27B
bring-up. Every claim cited file:line. No new measurements were taken for this audit.

## 1. Model files on disk

Daily serving artifact + drafters, `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/` (dir listing
2026-08-03):

| File | Size | Role |
|---|---|---|
| `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` | 16 GB | THE daily artifact: NVFP4 trunk + Q5_K embed/head + baked MTP block (provenance line: research/verify-economics-20260802/RESULTS.md:7-8) |
| `draft-daily-owntrim-nvfp4head-q4blk.gguf` | 1.2 GB | own-gen trimmed drafter, the board spec-row config (research/verify-economics-20260802/RESULTS.md:8-9) |
| `owngen-ranks-32768.gguf` + `.txt` | 131k/186k | own-gen rank file feeding the trim (docs/DRAFT-REGIME.md:44-50 recipe) |
| `mtp-Qwen3.6-27B-hfbf16-q8_0.gguf` | 3.2 GB | HF-bf16 MTP block @ q8_0 (acceptance arc, research/tune-data/g7e-rtx6000.jsonl:25) |
| `mtp-Qwen3.6-27B-NVFP4.gguf`, `mtp-*Q4_K_M*.gguf` x4 | 1-6 GB | legacy pre-regime drafts/trims (superseded 2026-07-18, rig5090.jsonl line 344) |

Other 27B trees under `/data/ai-ml/hf-models/`:

- `nvidia-qwen36-27b-nvfp4/` — NVIDIA-official NVFP4 safetensors (3 shards, 22 GB) + config.json
  + chat_template.jinja + tokenizer.json. The ST-serving lane artifact (rig5090.jsonl lines
  180-289, NV-27B ST standing config).
- `unsloth-qwen36-27b-nvfp4-gguf/` — `Qwen3.6-27B-unsloth-NVFP4-w4attn-im-q5h-mtp.gguf` (16 GB)
  + its own drafter + ranks + imatrix + convert logs (the conversion command record:
  `convert.log`, `mtp-convert.log` — converter = the house llama.cpp fork's
  `convert_hf_to_gguf.py`, arch `Qwen3_5ForConditionalGeneration`).
- `qwen36-27b-hf-min/` — config.json + tokenizer only (the arch-diff reference; §5 of the
  runbook uses it).
- `eagle3-qwen36-27b/` — effectively empty (24 KB cache stub), dead lane.
- `qwen36-27b-bf16-src/` — referenced by `/data/ai-ml/hf-models/27b_dl.log` but the directory
  is GONE from disk (deleted; only the dl log remains). BF16 source re-download would be
  needed for any from-scratch requant.

## 2. Published board rows

`research/tune-data/current-board.json` (updated field: "2026-08-02", line 2):

- Plain decode d512 (current-board.json:14-16): memra 47.6 vs llama 43.7 → **1.09x**
  (rendered docs/PERFORMANCE.md:72).
- Plain decode depth d6257 (current-board.json:225 block, `plain_decode_depth`): memra 46.2
  vs llama 41.8 → **1.11x** (docs/PERFORMANCE.md "Depth behavior" §, lines 102-107).
- Spec K=3 + own-gen trimmed draft (current-board.json:42 block): memra 116.4/101.2/86.0 vs
  llama 91.7/93.3/81.5 → **1.27x/1.08x/1.06x** short/medium/long-agentic
  (docs/PERFORMANCE.md:83).
- H100 board (current-board.json `h100_board`, updated 2026-08-01): memra e2e 96 vs vLLM 0.26
  FP8 73 → **1.31x** (docs/PERFORMANCE.md:210).
- Supported-models row (current-board.json:96 block; README.md:86): "Qwen3.6-27B | dense |
  NVFP4, Q4_K_M MTP-baked | MTP + own-gen trimmed draft | since v0.1.0".

Raw provenance:

- Plain rows: rig5090.jsonl line 361 (2026-08-02T14:45Z, commit d14d7d8d+lane/board-remeasure)
  — N=5 process reps, same-session interleaved, fresh llama denominator b… build c818263f2,
  flock'd, per-rep receipts `research/board-remeasure-20260802/board-remeasure.jsonl` +
  `mem-q27-*.log` x10 + `llama-q27-rep*.log` x5. q27 argmax MATCH on all reps.
- Spec row: rig5090.jsonl line 344 (2026-07-18T06:30, "BOARD MOVE — the daily itself gets the
  new draft recipe") — K=3 interleaved N=2, same-window llama re-pair 91.7/93.3/81.5.
- H100 row: current-board.json h100_board (protocol field: N=5 medians, same-session
  interleaved, argmax gate green); defense note HANDOVER.md:90 ("q27 STANDS 1.31x").

## 3. Gate coverage

- **kernel-check**: dedicated weight-oracle section `nvfp4-27b-shape` loads the daily GGUF
  directly (crates/memra-engine/src/bin/kernel_check.rs:1616; section named at
  kernel_check.rs:23 and docs/TESTING.md:42).
- **fast-gate tier-1**: probe id `k27` (tools/fast-gate/models.tsv:32) — argmax, 20-token
  golden on the daily GGUF, golden pinned at d9e45d86 2026-08-02 (tools/fast-gate/goldens/
  k27.tokens:1; perf smoke reference 48.38 tok/s in k27.perf). Map rows routing diffs to k27:
  tools/fast-gate/map.tsv:27 (`qmatvec_gemm.cu` → nvfp4-gemm,nvfp4-27b-shape → q9,k27),
  map.tsv:30 (`mmq_q45k.cu` → o35,k27), map.tsv:32 (`mmq_fp4|mmq_nvfp4_*|cutlass_fp4_sm120`
  → q9,k27).
- **run-gen argmax**: gates every board rep (board-remeasure.jsonl `argmax_match_lines: 2`
  per q27 rep; rig5090.jsonl line 361 "MATCH 30/30 memra runs"). Release gate names the 27B
  explicitly: docs/RELEASING.md:16 (`kernel-check <27B.gguf>` ALL GREEN).
- **run-spec K=1..8**: standing battery member — most recent full-sweep receipts on the q27
  tree: research/verify-economics-20260802/RESULTS.md:29-31 ("run-spec K=1..8
  self-consistency PASS on every battery (72/72 baseline + 72/72 after + 12 A/B cells)").
- **NOT covered**: `tools/local-ci.sh` correctness stage runs kernel-check (which loads the
  27B via nvfp4-27b-shape) but its named run-gen models are gemma 31B/12B + q35 prime-gate
  only (tools/local-ci.sh:77-123) — no direct q27 run-gen line. The perf-CI cell battery
  (research/tune-data/perf-cells.json, 9 cells) has **no q27 cell** — gemma x8 +
  qwen9b-plain-short only. q27 perf-drift detection currently rides fast-gate's --smoke
  tripwire (not evidence-class) and manual board re-measures.

## 4. Drafter / MTP status vs the deployment bar

Bar (owner memory `delivery-bar-e2e-1p1x`): ship only at >=1.1x e2e vs llama on EVERY
supported model + every deployed model carries the own-gen trimmed drafter.

- Drafter: **compliant**. Own-gen ranks + byte-verbatim extract + NVFP4-head/Q4_K_M-block
  trim built 2026-07-18 (rig5090.jsonl line 344; recipe docs/DRAFT-REGIME.md:44-50;
  builder tools/make-trimmed-draft.sh:1-14 cites the 27B measurement "101 tok/s @ 85.2%
  acceptance (K=3)").
- Spec economics: q27 spec pays 2.00x (prose) / 2.18x (code) vs own plain at K=3
  (research/verify-economics-20260802/RESULTS.md:21-24, 63-73).
- **vs-llama bar, honest reading**: q27 is the WEAKEST supported qwen model against the bar.
  - Plain d512 1.09x — below 1.1x by one rounding step (47.6/43.7; raw medians 47.64/43.67 =
    1.091, rig5090.jsonl:361).
  - Spec p2/p3 1.08x/1.06x — below 1.1x (docs/PERFORMANCE.md:83). Only spec-p1 (1.27x) and
    the H100 e2e (1.31x) clear it cleanly.
  - Prefill: 5090 pp trails llama 0.59-0.78x fleet-wide, root-caused as the W4A4 numeric-class
    refusal (docs/PERFORMANCE.md:105-107); 27B pp512 clean-wall ratio 0.58x at the July
    diagnosis (rig5090.jsonl line 240).
  The model is published as supported since v0.1.0 (predates the 1.1x codification), but by
  the letter of the bar the q27 5090 rows sit at 1.06-1.09x on three of four published cells.
  Any 3.8-27B publish will be measured against the same bar — expect the same grind unless
  3.8 numbers land differently.

## 5. Known open issues touching q27

- GitHub issues: none open (gh issue list 2026-08-03: empty).
- **Board staleness vs HEAD** — see baseline-36-27b.jsonl staleness row. Summary: plain rows
  measured at d14d7d8d (2026-08-02); since then HEAD gained the ladder-3072 merge (216419d6 —
  board impact recorded "none published: d512 plain rows flat (+-0.2%)", rig5090.jsonl line
  362) and the verify-tier/verify-economics spec-side merges (c041f70e dual gate+up twin,
  "+0.8% q27 spec e2e at K=3", research/verify-economics-20260802/RESULTS.md:25-27). The SPEC
  row (2026-07-18) is now ~2 weeks and several spec-path merges old; it predates the dual
  twin and the verify-tier fixes and has NOT been re-paired since. Staleness flagged, not
  re-measured (spec re-pair is a board-moving act, out of this lane's scope).
- Verify-tier finding parked with q27 relevance: K=4 re-opens at +7-9% on code with fixes
  1+2+3 (merge c28502d6 message) — an open spec-row upside, not a regression.
- rig5090.jsonl line 274 correction discipline note: every historical 27B ST cell pairing has
  a correction row — when re-pairing 3.8, keep the same-session law.

## 6. Architecture facts (3.6-27B, the 3.8 diff reference)

`/data/ai-ml/hf-models/qwen36-27b-hf-min/config.json` (full field dump in the runbook §2):
arch `Qwen3_5ForConditionalGeneration` / `model_type qwen3_5`, text_config: 64 layers,
hidden 5120, inter 17408, heads 24 / kv 4, head_dim 256, vocab 248320, hybrid
linear-attention with `full_attention_interval 4` (3 GDN linear layers per full-attn layer),
linear heads 16K/48V @ dim 128, partial_rotary_factor 0.25, rope_theta 1e7 (mrope_interleaved,
section [11,11,10]), mtp_num_hidden_layers 1, max_position_embeddings 262144, vision tower
attached (image_token_id 248056). MTP head maps blk.64 in GGUF (rig5090.jsonl line 185).

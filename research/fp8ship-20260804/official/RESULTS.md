# fp8-ship item B — official-checkpoint e2e battery (Qwen/Qwen3.6-27B-FP8)

Box: vast.ai 2x RTX 5090 32GB (rig2x5090-vast), CUDA 13.0.1 driver stack, **nvcc 13.0.88**
(`/usr/local/cuda/bin/nvcc`), 256-thread EPYC. Train HEAD **ac99e675** (#68 fixed, ST spec
quarantine lifted), rsync'd worktree. Checkpoint: `Qwen/Qwen3.6-27B-FP8` downloaded fresh to
`/root/models/qwen36-27b-fp8-official` — 29 GB, 80 files, all 66 index shards verified present;
`quantization_config: fp8 e4m3 weight_block_size [128,128]`, 407 `weight_scale_inv` block-128
grids, **zero** per-tensor scalar scales (layers-0 dtypes: 14 BF16 + 6 F8_E4M3). This is the
first OFFICIAL-artifact battery for the block-128 loader (all prior ARM B' gates ran on the
2.65 GB synthetic 1.7B ckpt, research/fp8st-20260803/armb/).

Scope: item B = PLAIN-serving battery + (post-ac99e675 course correction) spec-on-ST serve
cells. Item A (#68 root cause) was another agent's lane and landed as ac99e675 mid-battery.

## FINDING FIRST — nvcc 13.0.88 miscompiles the ARM B' dequant kernel (FIXED)

The kernel-check `[fp8-blk-gpu]` bit-parity arm **FAILED on this box** at the first run
(3968/680/8 bad bytes on the three shapes) — the same commit's arm is green on the laptop
(nvcc 13.1). Byte-in-block histogram localized every bad byte to **offset 0**: the LOW byte
of the Q8_0 `d` f16. A 40-line isolated repro (`miscompile/halfstore_repro.cu`) reproduces it
8/8 blocks: nvcc 13.0.88 at -O3/sm_120a **loses the first of two adjacent byte stores**
(`dst[0]=lo; dst[1]=hi;` under `if (lane==0)`) — dst[0] reads back as the cudaMemset poison
value's slot written as 0x00 (got == ref & 0xff00 every time). The same kernel with ONE
aligned u16 store (`*(uint16_t*)dst = bits`) is correct (`miscompile/halfstore_fix.cu`,
0/8 bad). Fix applied to `crates/memra-engine/cu/fp8_blk_dequant.cu` (single u16 store +
comment naming the toolchain); post-fix `kernel-check` is **ALL GREEN** including all three
fp8-blk-gpu shapes (`kernel-check-postfix-gpu1.log`), and the byte-level probe
(`miscompile/fp8_blk_probe.rs`) reports 0 bad bytes (`miscompile/parity-probe-postfix-gpu1.log`).

Consequence chain, captured pre-fix on the real checkpoint (raw logs lost to a mid-battery
rsync --delete — transcript-preserved numbers, repro on demand by reverting the one-line fix):
the miscompiled arm's first-token logits vs the CPU path showed max_abs 3.04, rms(diff)/rms(ref)
0.164, greedy stream divergence at token 14 — i.e. the bit-identity gate did exactly its job.
LAW RELEARNED: the sm_90a lane's "kernels are form-sensitive per toolchain — re-gate when the
toolchain moves" applies to nvcc 13.0 vs 13.1 on sm_120a too; the parity gate must run
per-box, not per-commit.

## 1. Load gates — first official-artifact gates for the block-128 loader (post-fix)

`run-gen <dir>` (ST branch), GPU0, flock'd, MEMRA_NGEN=32, interleaved default/blkgpu pairs.
Gates per rep: verify-prefill argmax (decode_step_t vs decode_step), 32-token greedy stream,
prefill-logit vector dump (`MEMRA_PREFILL_LOGITS`, 248320 f32).

| gate | default (CPU dequant → Q8_0) | MEMRA_FP8_BLK_GPU=1 (ARM B') |
|---|---|---|
| pp512 argmax | 365==365 MATCH maxdiff 0.0 (x3 reps) | 365==365 MATCH maxdiff 0.0 (x3 reps) |
| p1 argmax | 760==760 MATCH maxdiff 0.0 | 760==760 MATCH maxdiff 0.0 |
| CLI decode (32 tok, single run) | 52.79 tok/s | 52.50 tok/s |

**BIT-IDENTITY (ARM B' contract): HOLDS on the official artifact.** All 4 interleaved pairs
(pp512 r1-r3 + p1 r1): greedy token streams IDENTICAL, argmax lines IDENTICAL, prefill logit
vectors BIT-IDENTICAL (993280/993280 bytes, `cmp` clean). Within-arm: both arms produce
identical token streams across all 3 pp512 reps.

**Load wall-clock** (whole-process wall incl. gates + 32-tok gen — same protocol as the
laptop synthetic rows; N=3 interleaved pairs, warm page cache, `loadtime.log`):

| rep | default (s) | blkgpu (s) |
|---|---|---|
| 1 | 843.876 | 291.556 |
| 2 | 843.539 | 294.961 |
| 3 | 844.090 | 287.022 |
| **median** | **843.9** | **291.6** = **2.89x** |

(p1 pair: 822.7 → 281.3 = 2.93x.) First official-artifact datapoint for the load-time claim:
the laptop synthetic 2.65 GB ckpt gave 3.87x (35.7s → 9.2s); the 29 GB 27B on this box gives
2.89x, shedding ~9.2 min of host dequant+re-encode per load — the difference class is this
box's ratio of fixed cost (engine init + gates + 27B VRAM uploads) to dequant work, not a
regression. A cold 14-min default load per server restart vs <5 min with the flag is the
operational takeaway.

## 2. Serving gates (memra-server, official dir in MEMRA_MODELS, GPU1)

Server boot to /health: **847.6s** (default env — CPU dequant dominates; same class as the
CLI default load wall).

- `/models` lists the model: **PASS**
- `/v1/chat/completions` correctness through the checkpoint's own `chat_template.jinja`:
  **PASS** — coherent, "Paris" present, usage populated (`chat-correctness.json`).
- Greedy determinism x3 (same prompt, distinct `cache_salt` so each request recomputes):
  **PASS** — three identical texts (`greedy-det-default.txt`).
- `tools/serve-st-gate.sh <official dir>`: **0 failed** (`serve-st-gate.log`). Items: /models
  PASS; chat coherent PASS; CLI-vs-server greedy token streams IDENTICAL (64 ids); **item 4
  spec-ON** (ac99e675 rewrite): default (spec) server text prefix-matches the tokenwise serve
  oracle **624/624 chars** at the 400-token window; "no quarantine notice" PASS. One harness
  fix landed: the gate's server-wait was 240s, calibrated on the 4B ckpt — a 27B ST dir needs
  ~14 min, bumped to 1200s (`tools/serve-st-gate.sh`, in this change set).
- MTP: the ST loader picks up the checkpoint's own `mtp.safetensors` — the default server
  spec-bursts out of the box ([spec-acc] lines in `server-default.log`, cum acceptance
  0.57-0.64 on the gate prompt).

## 3. Serve-path perf cells (the ST self-competition rows)

Protocol: `/v1/chat/completions` **streaming SSE**, greedy, max_tokens=128, pp512-class prompt
(`research/e2e/prompts/pp512.txt` → 521 rendered prompt tokens), N=5, fresh `cache_salt` per
request (cached_tokens=0 verified every rep), single card (GPU1), medians. Client-side clocks:
TTFT = POST→first content delta; decode = (ct−1)/(last−first delta). Harness
`serve-perf.py` (in this dir); raw rows `serve-perf.jsonl`; per-arm logs `perf-st-*.log`.

| arm | decode tok/s (median, N=5) | TTFT s | prefill tok/s (TTFT-derived) | greedy stable x5 |
|---|---|---|---|---|
| ST plain (`MEMRA_SERVE_SPEC=0`) | **48.99** | 0.170 | 3064 | yes |
| ST spec, embedded MTP (default env) | **128.06** (2.61x plain) | 0.466 | 1118 | yes |
| ST spec + q27 own-trim drafter (`+draft-owntrim-nvfp4head-q4blk.gguf`) | **136.75** (2.79x plain) | 0.463 | 1125 | yes |
| ST e4m3 arm (`MEMRA_ST_E4M3=1`, spec off) | 48.99 | 0.171 | 3054 | yes |
| GGUF Q8_0 reference (rig2x5090.jsonl 2026-08-03) | 53.63 | — | 4151 (pp512) | — |

**Protocol caveat, stated plainly:** the GGUF reference row is a **CLI** cell (run-gen tg128
process-median, pure kernel prefill timing) measured 2026-08-03 in a different session; the ST
rows are **serve-path SSE** cells measured today — cross-protocol AND cross-day, so the 48.99
vs 53.63 gap is NOT a clean ST-vs-GGUF statement. The in-protocol anchors: CLI ST decode this
session = 52.5-52.9 tok/s (32-tok window, same resident Q8_0 format as the GGUF), so the ST
loader lands within ~1.5% of the GGUF CLI number, and the serve path costs ~7% vs CLI decode
(event streaming + per-token detok + session overhead). TTFT-derived "prefill" includes
tokenize+template+queue+prime, not comparable to the 4151 kernel pp512.

- **Spec rows:** first spec-on-ST serve numbers anywhere in the repo (quarantine lifted at
  ac99e675 mid-battery — course-corrected to include them). Embedded MTP head 128.1 tok/s,
  own-trim regime drafter 136.8 tok/s (+6.8% over embedded). Spec TTFT is ~0.30s higher than
  plain: the spec session's prime+first-burst happens before the first Token event. Note
  `spec-vs-plain greedy text` on the 128-tok chat probe DIVERGES at char 45 (`phase2-driver.log`)
  — that comparator is the BATCHED plain arm, which carries the accepted decode-config near-tie
  FP class (decode-batch-gate's jurisdiction, per the serve-st-gate item-4 comment); the
  exactness-proper comparator (tokenwise oracle, serve-st-gate item 4) PASSED 624/624 on this
  same box+ckpt. Spec arms are greedy-stable x5 within themselves.
- **e4m3 arm is FLAT by construction on this checkpoint, as predicted:** the QT_F8_E4M3
  resident arm requires `blk.is_none()`, and this ckpt has zero scalar-scale F8 tensors — all
  407 block-128 tensors fall through to the same Q8_0 re-encode. Decode identical (48.99),
  greedy text byte-identical to plain. The +4%-class e4m3-direct win measured on the NVIDIA
  NVFP4 ckpt (per-tensor scales) does NOT transfer to the official block-128 artifact until a
  per-block e4m3 mmvq twin exists (DECISION.md B1 second half).

## 4. Receipt files

- `phase1-driver.log`, `loadtime.log`, `load-{pp512,p1}-{default,blkgpu}-r*.log`,
  `logits-*.bin` — load gates + bit-identity raw.
- `miscompile/` — halfstore_repro.cu (+fix variant), halfstore-repro-out.log (8/8 fail →
  0/8 fixed), fp8_blk_probe.rs + parity-probe-postfix-gpu1.log, kernel-check-postfix-gpu1
  logs at top level.
- `phase2-driver.log`, `server-*.log`, `chat-correctness.json`, `greedy-*.txt`,
  `serve-st-gate.log` — serving gates raw.
- `serve-perf.jsonl`, `perf-st-*.log`, `serve-perf.py` — perf cells raw (per-rep rows).
- `fp8ship-phase1.sh`, `fp8ship-phase2.sh` — the drivers, literals baked.
- `rig2x5090.jsonl` — the summary rows (rig2x5090-serve).

Code changes riding this battery (uncommitted, in the local worktree for the orchestrator):
`crates/memra-engine/cu/fp8_blk_dequant.cu` (miscompile fix), `tools/serve-st-gate.sh`
(27B load-wait bump), `crates/memra-engine/src/bin/fp8_blk_probe.rs` would live in
`miscompile/` only (diagnostic, not shipped as a bin).

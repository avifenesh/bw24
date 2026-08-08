# step37-p2 — Step-3.7-Flash phase 2: model support + first PP-2 boot

Lane branch `lane/step37-p2` (off `origin/restructure/public-split`). Phase-1 dossier is
`research/step37-bringup-20260802/PLAN.md` — binding, not re-derived here.

SKU doctrine (owner, 2026-08-06): **Step-3.7-Flash serving on BOTH RTX PRO 6000 cards over PP-2.**
Honest artifact = StepFun official IQ4_XS (105.0 GB). Honest ctx target = **128K** (memra's KV
allocator sizes SWA layers at max_ctx today; 256K needs the SWA ring-buffer item).

Perf is NOT this lane's job. Bring-up + exactness only; the PP-2 perf story belongs to the
pp2-hardening lane sharing the box.

---

## Increment 1 — artifact staged + `step35` config parse arm (2026-08-06)

### Artifact: DONE, byte-verified

StepFun official GGUF on the box at `~/step37/models/step-3.7-flash/` (persistent root, NOT the
3.5T ephemeral NVMe at `/opt/dlami/nvme` — that is lost on a spot stop, and a re-download after
interruption is the thing worth avoiding).

| file | bytes | sha256 (head) |
|---|---:|---|
| `IQ4_XS/...-00001-of-00003.gguf` | 46,483,327,296 | `b940497a` |
| `IQ4_XS/...-00002-of-00003.gguf` | 46,999,941,600 | `e7e0caaa` |
| `IQ4_XS/...-00003-of-00003.gguf` | 11,510,293,728 | `ccbd3df8` |
| `Step3.7-flash-mtp-BF16.gguf` | 6,973,549,696 | `380f0793` |
| `Step3.7-flash-mtp-Q8_0.gguf` | 3,707,276,416 | `469a8166` |
| `Step-3.7.imatrix.gguf` | 465,998,112 | `7f94ca21` |

IQ4_XS total = **104,993,562,624 B (105.0 GB)** — matches the phase-1 HF manifest exactly, all
three splits byte-for-byte. Download 12:36:55Z → 12:43:17Z (6m22s, ~500 MB/s via hf_transfer).
Receipts: `raw/download-20260806.log`, `raw/artifact-sizes-20260806.txt`,
`raw/artifact-sha256-20260806.txt`.

**105.0 GB > 96 GB per card. This model boots PP-2 or not at all** — there is no single-card
resident configuration to fall back to, so the split loader is on the critical path for *any*
exactness result, not an optimization.

MTP status: the trunk GGUF carries **no** nextn tensors (754 tensors, 45 blocks, no
`nextn_predict_layers` key). The MTP head ships **standalone** as
`Step3.7-flash-mtp-{BF16,Q8_0}.gguf` with `nextn_predict_layers = 3` and 48-block numbering
(blocks 45/46/47), DeepSeek-style `eh_proj`/`enorm`/`hnorm`/`shared_head`. So memra's MTP path
needs an **external-file** arm — the drafter is a second GGUF, not extra blocks in the main one.
Both quantizations are staged; Q8_0 (3.7 GB) is the serving candidate.

### Code: `step35` arch recognized and parsed

`crates/memra-gguf/src/config.rs`:
- `Arch::Step35` + `"step35"` parse; `is_hybrid()`/`is_moe()`/`is_step35()`.
- `Step35Config` — per-layer `head_count`/`head_count_kv`, `swa_pattern`, dual RoPE base,
  `rope_dims_full`/`_swa`, the two `swiglu_clamp_*` arrays, and the sigmoid-router block. Accessors
  `is_swa(il)`, `n_head(il)`, `n_head_kv(il)`, `n_rot(il)`, `rope_base(il)`, `clamp_exp(il)`,
  `clamp_shexp(il)`, `n_full_attn(n_trunk)`.
- `ModelConfig::n_head_at(il)` / `n_head_kv_at(il)` / `is_swa_at(il)` — the per-layer entry points
  the forward pass will use (the latter two also fold in gemma4's existing arrays).

Three things worth naming because each is a live footgun, not boilerplate:

1. **`attention.head_count` is an ARRAY on this arch** (64 on full-attn layers at `il % 4 == 0`, 96
   on SWA). `MetaValue::as_u64` returns `None` on an `Array`, so the pre-existing
   `u("attention.head_count").expect("head_count")` was a **guaranteed panic** on this artifact.
   The global scalar now falls back to the **max** over layers (96, not 64): it sizes shared
   scratch/workspace buffers, so a min/first-value would under-size them. Per-layer shapes come
   from `step35.n_head(il)`.

2. **`attn_out_gate()` is a deny-list** (`m3.is_none() && hy3.is_none() && ...`), so any new arch
   defaults to `true` and the fused `q_gate_split` path reads **2x out of bounds** on step35's wq.
   Added `&& self.step35.is_none()`, plus an explicit positive predicate
   **`attn_gate_separate()`** for step35's form: a separate `blk.N.attn_gate.weight
   [n_embd, n_head_l]` tensor, one pre-sigmoid scalar per head broadcast over head_dim, applied to
   attn_out before wo, computed from the **post-attn_norm** hidden state
   (upstream `step35.cpp:267-285`). This is a different mechanism from qwen35's fused per-dim gate.

3. **Half rotary on FULL layers only.** Upstream's generic loader seeds `n_rot_swa` from
   `n_rot_full` (= `attention.key_length` = 128) and *then* `step35.cpp` halves `n_rot_full` to 64.
   Net: **SWA layers rotate 128 dims, full-attn layers rotate 64.** The artifact carries no
   `rope.dimension_count` key at all, so `head_dim_k` is the only source. Getting the order
   backwards silently rotates the wrong width on 33 of 45 layers.

`crates/memra-gguf/src/micro_gguf.rs`:
- `MetaW::ArrU32` / `ArrF32` + encoders (step35 is the first arch whose fixture needs per-layer
  u32/f32 arrays).
- `write_step35_meta_only()` — metadata-only GGUF pinned to the **real** artifact KV set from
  `research/step37-bringup-20260802/raw/gguf-header-stepfun-iq4xs-shard1-20260802.txt`, including
  its *absences* (no `vocab_size`, no `rope.dimension_count`, no `nextn_predict_layers`).
- `parse_step35_pinned_metadata` — asserts all 45 layers' swa flag / n_head / n_head_kv / n_rot /
  rope base, the clamp arrays (unset everywhere but 43→7.0 and 44→16.0), the router
  (sigmoid, scale 3.0, norm, 3 leading dense), and both gate predicates
  (`attn_out_gate() == false`, `attn_gate_separate() == true`).

`cargo test -p memra-gguf --lib`: **70 passed, 0 failed.**

---

## Increment 2 — `deepseek-v3` pre-tokenizer (2026-08-06)

`tokenizer.ggml.pre = deepseek-v3` was a confirmed real gap: the dispatch in
`crates/memra-tokenizer/src/lib.rs` handled only `qwen35`/`qwen2` and **silently fell through to
`split_qwen35`** for everything else. Now implemented and wired.

Upstream has **no** custom splitter for this pre-type — it runs three regexes through generic
`std::regex` (`llama-vocab.cpp:318-325`, `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`):

```
1. \p{N}{1,3}
2. [一-龥぀-ゟ゠-ヿ]+
3. [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
   |[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
   | ?[\p{P}\p{S}]+[\r\n]* |\s*[\r\n]+ |\s+(?!\S) |\s+
```

`split_deepseek_v3` in `crates/memra-tokenizer/src/unicode.rs` is a hand-written equivalent:
llama.cpp's collapsed-text representation (one byte per codepoint; non-ASCII → 0x0B whitespace
stand-in or a 0xD0-0xD5 category byte) plus the three ordered passes, each subdividing the
previous pass's offsets with `unicode_regex_split_stl`'s gap-emission behavior. Two upstream
details that are load-bearing and easy to miss: pass 3 runs on the **collapsed** text so `\s` is
std::regex's ASCII-only `\s` (every non-ASCII whitespace codepoint is already 0x0B), and pass 2
runs on the **codepoint** text (non-ASCII literals, no `\p{}` class → upstream takes the wregex
branch, not the collapsed one).

**How it is verified.** Not by eyeballing: `research/step37-p2-20260806/pretok-ref-deepseek-v3.py`
is an independent implementation of the same upstream algorithm executed by a **different regex
engine** (Python `re`), reading the codepoint-class table out of memra's own `unicode_data.rs` so
classification is identical by construction and only the *split algorithm* is under test. The
67-case corpus (English, contractions, digit runs of every length mod 3, CJK/hiragana/katakana
incl. the exact `龥`/`぀` range bounds, Cyrillic/Greek/Arabic, combining marks, NBSP, emoji,
punct/symbol runs, whitespace/newline/CRLF runs, end-of-string lookahead edges) round-trips
byte-exact through the Rust port. Reference output committed at
`raw/pretok-deepseek-v3-reference-20260806.txt`.

**A bug found in the reference while building it** (worth recording because it would have
produced a *confidently wrong* ground truth): the first version regex-scraped all `(0x..,0x..)`
pairs out of `unicode_data.rs`, which also swept `UNICODE_MAP_LOWERCASE` and overwrote the flag
table with lowercase-map entries — every non-ASCII letter mis-classified, `naïve` splitting as
`['na','ïve']`. Now bounded per array with length assertions (2273 ranges, 25 whitespace cpts).

The measured divergences from `split_qwen35` — i.e. exactly what the fall-through was getting
wrong on Step-3.7-Flash input — are pinned per mechanism in
`deepseek_v3_differs_from_qwen35_per_mechanism`:

| input | deepseek-v3 (correct) | qwen35 (what we were doing) |
|---|---|---|
| `12345678901234` | `123 456 789 012 34` | one token per digit |
| `日本語のテスト、カタカナ` | `日本語のテスト` `、` `カタカナ` | `日本語のテスト` `、カタカナ` |
| ` 中文` | ` ` `中文` | ` 中文` |
| `▁escaped▁space` | `▁` `escaped` `▁` `space` | `▁escaped` `▁space` |
| ` symbols ~ ^` | ` symbols` ` ` `~` ` ^` | ` symbols` ` ~` ` ^` |
| `-abc1` | `-abc` `1` | (no alt-1 counterpart) |

Digit grouping alone changes the tokenization of essentially every numeric span, so this was not
a corner case — it would have shifted ids across ordinary agentic/code traffic.

Also: the dispatch now **warns once** on any unknown `tokenizer.ggml.pre` instead of silently
falling back (`warn_unsupported_pre`). Silence is how an unimplemented pre-tokenizer masquerades
as a model-quality problem.

`cargo test -p memra-tokenizer`: **14 lib + 3 llama-parity tests pass** (the parity suite,
which checks memra's ids against `llama-tokenize` on a real model, is unaffected).

Still open on the tokenizer: byte-check Step-3.7-Flash ids against HF `apply_chat_template` on
the box once the artifact is loadable, and the StepFun chat-template dialect
(reasoning_effort, forced-open `<think>\n`, `<function=X><parameter=Y>` tools, tool_response
role) in `chat.rs`.

---

## Increment 3 — loader: the separate head-wise attention gate (2026-08-06)

`crates/memra-engine/src/hybrid.rs`: `FullAttnLayer` gains `attn_gate: Option<GpuTensor>`, and
`load_mixer_kind` gains a `sep_gate: bool` parameter (passed, not read off a cfg, so the MTP/draft
call sites that build a synthetic cfg opt in explicitly). All three call sites forward
`attn_gate_separate()`; gemma4's shared-KV literal is `attn_gate: None`.

Loaded with **`load_t`, not `load_opt`** — deliberate. If the arch says the gate exists and the
tensor is missing, the correct outcome is a load failure, not a forward pass that silently skips a
per-head sigmoid and returns plausible-but-wrong logits.

**A plan assumption corrected by reading the artifact header.** Going in (phase-1 notes) the gate
was described as belonging to "the full-attn path", implying a subset of layers. The real header
has `blk.N.attn_gate.weight` on **all 45 blocks**, with the width varying per layer because it is
that layer's query-head count:

```
blk.0.attn_gate.weight  [4096,   64]      blk.0.attn_q.weight  [4096,  8192]   (full attn, 64 heads)
blk.1.attn_gate.weight  [4096,   96]      blk.1.attn_q.weight  [4096, 12288]   (SWA,       96 heads)
```

So the gate is universal on this arch and it is the *per-layer wq/wo widths* that vary — the
forward must take both from `n_head_at(il)`, never from the global scalar (which is the max, 96,
and would over-read wq on the 12 full-attn layers).

**Half-rotary needs no kernel work.** `rope_neox2_f32` (`crates/memra-engine/cu/kernels.cu:1416`)
already takes `head_dim` and `n_dims` as *separate* arguments and does `int half = n_dims / 2; if
(j >= half) return;`. Passing `n_dims = 64` with `head_dim = 128` rotates the first 64 dims and
leaves the tail untouched — exactly upstream's half-rotary on the full-attn layers. One less
kernel on the critical path than the plan budgeted.

`cargo build -p memra-engine`: exit 0, no warnings.

---

## Increment 4 — the two new math kernels + kernel-check cells (2026-08-06)

step35 needs exactly two pieces of math memra did not have. In both cases memra already contains a
kernel that *looks* like the right one, and substituting it compiles, runs, and produces plausible
logits — so each cell's real job is the confusion guard, not the maxdiff.

| new kernel | the lookalike | how they differ |
|---|---|---|
| `attn_head_gate_f32` | `sig_mul_f16out_f32` | new: ONE sigmoid per (token, head), broadcast over head_dim. Lookalike: full width, one gate value per (head, dim) element (qwen35 packs it in wq). Same signature shape, adjacent name. |
| `swiglu_clamped_mul_scaled_f32` | `swigluoai_mul_scaled_f32` | new: `min(silu(gate*gs), limit) * clamp(up*us, ±limit)`. oai: clamps gate BEFORE swish, then `* (1 + clamp(up))`. |

step35's clamp form is verbatim from `llama-graph.cpp:2146-2165` (routed experts,
`swiglu_clamp_exp`) and `:1751-1770` (shared expert, `swiglu_clamp_shexp`), non-DEEPSEEK4 branch —
step35 is not DEEPSEEK4 and has no `dsv4_hc_mult`, so it takes the `ggml_silu` then
`ggml_clamp(-INF, limit)` path, not the `ggml_swiglu_split` one. Callers must gate on
`limit > 1e-6` (upstream's eps check): at limit=0 the kernel clamps every positive activation to
zero. Only layers 43 (7.0) and 44 (16.0) have a live limit on this artifact.

Cells (synthetic, CPU oracle, `raw/kernel-check-step35-cells-5090-20260806.log`):

- `attn_head_gate` at n_head **64 and 96** — the query-head count is per layer, so both widths are
  real geometry — head_dim 128, T=1 (decode) and T=7 (prefill, non-power-of-2 for the grid tail).
  Gate values spread over ±6 so sigmoid spans ~0.002..0.998; a wrong broadcast cannot hide inside a
  flat ~0.5. Asserts maxdiff, that the fp16 twin is `f32_to_f16_bits` of the f32 the same launch
  stored, and that per-head values genuinely vary across the tensor (the confusion guard). Plus a
  `dst16=None` cell for the nullable-pointer skip.
- `swiglu_clamped` at both real limits × (gs,us) ∈ {(1,1), (0.75,1.25)}, inputs spanning ±3·limit.
  The engaged-clamp count is asserted (>10% of elements), because a cell whose inputs stayed
  in-range would pass identically against plain `silu_mul` and prove nothing. Plus the divergence
  guard vs `swigluoai_mul_scaled`, asserted by **named mechanism** at four hand-picked points.

Result on the 5090 (release build, `raw/kernel-check-step35-cells-5090-20260806.log`, exit 0):
**ALL GREEN**, whole battery, 555 lines. The 10 step35 cells: `attn_head_gate` maxdiff 1.19e-7 to
2.38e-7 with 0 f16 mismatches at all four (n_head, T) combinations; `swiglu_clamped` maxdiff 1.9e-6
to 7.6e-6 with 3700-3914 of 4096 elements clamp-engaged; all four divergence mechanisms true.

**A bad test caught and fixed, not a bad kernel.** The divergence guard first demanded ">50% of
elements differ from swigluoai" and read **39%** — FAIL. The cause was the input distribution, not
the math: `pr()` returns **[-1, 1]** (`memra-validate/src/lib.rs:29`), so my `(pr - 0.5) * 6 * limit`
expression produced [-63, +21], and most elements sat at deep-negative gate where `silu(gate) → 0`
and *both* kernels correctly agree at ~0. Two fixes: the inputs became symmetric (`pr * 3 * limit`),
and the count threshold was replaced with four named-mechanism probes that do not depend on a
distribution at all —

| probe | why the formulas must disagree |
|---|---|
| `up = -1` exactly | oai's `1 + up` factor vanishes → oai is 0 for any gate; step35 is `-silu(5)` |
| `up = -0.99` | oai stays ~0.05, step35 ~ -4.92 — two orders apart |
| `gate = 12 > limit = 7`, `up = 2` | the clamp-ORDER difference: oai `swish(min(12,7)) × 3 ≈ 20.98`, step35 `min(silu(12),7) × 2 = 14` |
| `up = 0` | step35's product is exactly 0; oai's `1 + 0` leaves the whole swish term |

This is the second time this lane has shipped a threshold-over-random-inputs assertion that failed
for distribution reasons (the first was `deepseek_v3_differs_from_qwen35`, increment 2). Both are
now mechanism assertions. The lesson is worth stating once: *a divergence threshold is only as
meaningful as the input distribution behind it*, and when the point is "these two functions are not
the same function", the honest assertion is a point where they provably differ.

## Increment 5 — KV budget and q27 co-residence: PLAN.md §3.4 does not transfer to this box

Receipt: `raw/kv-budget-pp2-20260806.txt` (inputs sourced line-by-line; card capacity from
`nvidia-smi` on the box, weights from the sha-verified byte counts).

**PLAN.md §3.4's "256K does not fit" was priced for a 4× RTX 5090 box (4 × 32 = 128 GiB), which is
not the SKU any more.** Phase 1 was written against the sku-repick premise (PLAN.md:5); the owner's
2026-08-06 call moved Step to 2× RTX PRO 6000 Server 96GB = **191.19 GiB measured**. Headroom after
weights + MTP is **89.95 GiB, not the 26.8 GiB** §3.4 compares against — 3.4× more.

Two of §3.4's inputs also disagree with the code:

1. It prices FP16 and FP8 KV. memra's **default with no env set** is q8_0 K / q5_1 V
   (`kv_blk_bytes()`, `memra-kv/src/lib.rs:37-42`) = 1856 B/tok/layer — 45% of the FP16 row.
2. It compares against a whole-box aggregate, but KV is allocated **per stage** by the device that
   owns the layer (`memra-kv/src/lib.rs:253`), so the binding constraint is per-card.

| KV format | 128K at-max | 128K ring | 256K at-max | 256K ring |
|---|---:|---:|---:|---:|
| q8_0/q5_1 (memra default) | 10.20 | 2.75 | 20.39 | 5.47 |
| fp8 both planes | 11.25 | 3.03 | 22.50 | 6.03 |
| fp16 (§3.4's row) | 22.50 | 6.06 | 45.00 | 12.06 |

(GiB. "at-max" = today, all 45 layers at `max_ctx`. "ring" = after work item F.)

At the pp.rs default even cut (23/22), 256K in the default format is 10.42 GiB of KV on stage 0 —
stage 0 lands ~59.3 of 95.59 GiB. **So on this box 256K fits the allocator as it stands, and the
SWA ring buffer (F) is a memory optimization here rather than the precondition it was on 4×32 GiB.**

What that does **not** establish, and why the honest listing context stays 128K anyway:

- Activation + graph-pool footprint at 256K is unmeasured (PLAN.md:187 already flagged this). At 45
  layers × 96 SWA heads the prefill activation peak is not a rounding error, and the 93.5:1
  prefill-heavy profile makes long prefills the common case, not the tail.
- Even layer count ≠ even byte split: 3 dense + 42 MoE means stage 0's layers are cheaper. Real
  per-stage weight bytes need the loader's tensor→layer map — a first-boot measurement.
- Whether the model is *correct* at 256K is an exactness question this arithmetic is silent on.

The cap stays at 128K; what changed is the **reason** — no longer "the allocator cannot express it".

## Increment 6 — the attention mixer (commit `b13738ed`)

Three arms (`hybrid_forward.rs`, node-for-node vs `llama.cpp src/models/step35.cpp:216-300`):
`step35_attn` (pure prefill, no cache), `step35_attn_prime` (append into the resident quantized
cache and attend through the cache view), `step35_decode_attn` (T=1, incl. the pre-quantized
norm-fusion arm). `step35_geom(il)` returns `(head_dim, n_kv, n_head, rope_base, scale, is_swa)`.

### Why this is a dedicated mixer family and not a few branches in `full_attn*`

Five reasons. Each one produces plausible-but-wrong logits rather than a crash, which is the whole
argument for not trying to generalize the existing chain:

| # | mechanism | what the generic path does instead |
|---|---|---|
| 1 | `attention.head_count` is an ARRAY (64 full / 96 SWA) | reads the `cfg.n_head` scalar → wrong wq/wo/attn_gate shape and FA head count on 33 of 45 layers |
| 2 | FULL layers rotate 64 dims, SWA 128 | `cfg.rope_dim_count` is 128 for this arch (upstream halves `n_rot_full` *after* the generic loader defaults it to `n_embd_head_k`) → rotates 128 on the full layers |
| 3 | dual rope base 5e6/1e4 + `rope_freqs` on FULL only | one base, factors everywhere |
| 4 | SWA 3:1, window 512 | unwindowed attention on 33 layers |
| 5 | separate `attn_gate.weight [n_embd, n_head_l]`, per-HEAD scalar, pre-`wo`, fed by the post-attn_norm hidden | no gate at all (`attn_out_gate()` correctly denies step35 — the fused split would read wq 2× out of bounds — but nothing supplied the mechanism it *does* need) |

Attention scale is the default `1/sqrt(head_dim_k)` (step35.cpp:255), **not** gemma4's 1.0. That
one is worth naming because gemma4 is the nearest template in the repo and its 1.0 is the
exception, not the rule — copying the template wholesale would have left token 0 exact (softmax
over one element) and every later position drifting, which is exactly how the gemma4 bring-up bug
presented.

### SWA masking: the convention already matched, verified verbatim on both sides

| side | mask predicate | source |
|---|---|---|
| upstream | `p1 - p0 >= (int32_t) n_swa` | `llama-hparams.h:359-395`, `is_masked_swa` under `LLAMA_SWA_TYPE_STANDARD` |
| memra | `t < q_pos - (window - 1)`, `q_pos = (T_kv - T) + qt` | `cu/kernels.cu:1795-1836`, `sdpa_naive_w_f32` |

Identical, and `step35.cpp:6` sets `hparams.swa_type = LLAMA_SWA_TYPE_STANDARD` — so memra's
existing windowed kernels are the correct semantics and **no new mask math was needed**. Worth
recording as a negative result: the plausible-looking work item "write a step35 window mask" does
not exist.

**Decode SWA is free**: a token-aligned view offset into the quantized cache (the gemma4 R6
pattern). Keys carry absolute rope and the mask is purely positional, so the single query attending
the last `win` rows IS the windowed result — no mask kernel on the decode path at all.

**Prefill SWA needed one new primitive.** `sdpa_naive_w_quantized_view` (`lib.rs`), the windowed
twin of `sdpa_naive_quantized_view`: the same `fa_dequant_kv_ws_f32` launch into f32 workspaces,
then `sdpa_naive_w` instead of `sdpa_naive`. `window == 0` is bit-identical to the unwindowed
function, so it is a strict superset.

Two things forced it, and the second is the subtle one:

1. **Every windowed FlashAttention *prefill* stamp in `flash_attn.cu` is head_dim-256 only**
   (`fa_prefill_w_f32` == `fa_prefill_f32_body<256>`, and the quantized-view windowed twins
   likewise). step35 is hd128. Note this is specifically the *windowed* family —
   `fa_prefill_qw_hd128` (:4632) and `fa_prefill_qw_db_hd128` (:4892) DO exist, so the unwindowed
   quantized-view prefill is stamped at hd128 and the full-attn layers ride it. Likewise
   `fa_decode_vec_q_rows_smem_w` (:5749) exists, so windowed *decode-rows* is stamped. The gap is
   windowed prefill at hd128, and only that.
2. **Trimming the view is not sufficient — the mask is still required.** The view is trimmed to the
   oldest key any query in the chunk can reach (`off = base_len - (win-1)`), which bounds it at
   `win-1+t` rows. But *inside* the chunk, query `qt` may only see view keys `[qt, qt+win-1]`, so
   the earlier keys the trimmed view still contains have to be masked per query. A trim-only
   implementation would let early queries attend past their window — silently, and only on
   continuation chunks.

Same cache bytes and same numeric class as the unwindowed quantized-view fallback, so the
chunk-invariance contract holds on both arms. One constraint to remember: the f32 floor's shared
memory is `t_kv * 4` bytes, so it needs `t_kv = win-1+t <= 12287` — it relies on chunked prefill
(`MEMRA_PRIME_CHUNK`, default 4096). A monolithic 32K prime would exceed the 48 KiB dynamic-smem
default. SWA layers still inside the window, and all full-attn layers, take the existing hd128
dequant-once `fa_prefill_view_ws`.

### One gate-related fix in shared code

`mixer_in_q8_1_fast` (`decode.rs`) now also requires `attn_gate` on the q8_1 fast path when the
layer has one. step35 projects its head-wise gate from the same attn-normed input as q/k/v, and the
fused arm passes a **zero-length `h`** — so admitting a layer whose gate is off the fast path would
hand the gate matmul an empty activation. No behavior change for any other arch (`None` ⇒ true).

### Refusals rather than silent generic geometry

The generic paths that cannot serve step35 now return a named error instead of running the wrong
geometry. Naming each missing twin is the point — a silent wrong answer here is unfalsifiable:

| site | missing twin |
|---|---|
| `full_attn_decode_dc_inner` (dc + graph capture) | SWA needs an **offset** KV view, which the `len_d`-derived dc kernels cannot express; plus per-layer-`n_head` capture |
| `full_attn_decode_batched` (m-stream) | per-layer n_head, partial rope, offset view |
| `decode_step_batch` at B>1 | same (B=1 already routes to the shared eager trunk) |
| `prime_cache_batch` | feeds the GENERIC attn core (`full_attn_prime_core_inner`) |
| `full_attn_verify` (spec) | a verify computing different attention than decode defeats the K=1..8 self-consistency gate outright |
| `MEMRA_PRIME_SEG` core-split segments | same generic-core reason (excluded via `use_seg`) |

**B=1 serve needs no step35 arm** — `decode_batch.rs`'s B=1 fast path calls
`decode_layers_eager`, the same trunk `decode_step_h` and the PP-N stages use, so it inherits the
step35 mixer through `full_attn_decode_pre` for free. That shared trunk is also the PP-2 stage
walker, which is what makes the first PP-2 boot reachable from this increment.

`full_attn` gains an `il` parameter (step35 needs it; every other arch ignores it) — 3 call sites
updated (`forward`, `forward_last`, `t2probe`).

### Still open on the forward path

`spec.rs`'s `mtp_full_attn_dc` is a 17th `attn_out_gate()`-keyed site and will need a step35 arm
when the MTP external-file arm lands. The dc/graph and batched twins are deliberately deferred:
bring-up correctness first, and the eager path is what the exactness gates measure.

---

## Increment 7 — the windowed-prefill primitive is now guarded (kernel-check)

`b13738ed` shipped `sdpa_naive_w_quantized_view` with a documented contract and **no test**. The
lane's own sequence item 2 requires a kernel-check cell for every new mapping, so the primitive now
has one: `kernel_check.rs`, immediately after the ARC B `fa_prefill_view_ws` bit-identity cell, at
step35's real attention shape (**head_dim 128**, GQA 8/2 — the hd256 cells above it never reach the
hd128 stamp).

Four assertions per case, and the last two are the ones that catch a real bug:

| assertion | why it exists |
|---|---|
| `window == 0` vs `sdpa_naive_quantized_view`: **bitdiff must be 0** | the commit's literal strict-superset claim; `sdpa_naive_w_f32` treats `window <= 0` as no mask |
| `window >= t_kv`: **bitdiff must be 0** | no key can be older than `q_pos-(window-1)`, so a live window that still changes the answer is a mask-arithmetic bug |
| `window < t_kv` vs a CPU windowed oracle fed the **GPU-dequanted** K/V | isolates mask semantics from the quantized cache bytes: both sides see identical f32 operands, so only the mask predicate is under test |
| the windowed output must **differ** from the unwindowed one | a dropped or ignored `window` argument passes assertions 1 and 2 trivially; this is the assertion that fails if the arg never reaches the kernel |

Cases `(T, T_kv, window)` = `(64, 192, 32)`, `(100, 100, 48)`, `(37, 297, 64)` — a continuation
chunk whose window is smaller than the chunk (the SWA chunk-prime shape where an early query must
not see keys a trimmed view still holds), a fresh chunk, and a BK-unaligned tail.

Result (RTX 5090 Laptop, release build, N=1 per case — a correctness gate, not a perf number;
raw: `raw/kernel-check-step35-swaview-5090-20260806.log`):

```
sdpa_naive_w_quantized_view(window=0) vs unwindowed T=64 Tkv=192: bitdiff=0 OK
sdpa_naive_w_quantized_view(window>=Tkv) vs unwindowed T=64 Tkv=192: bitdiff=0 OK
sdpa_naive_w_quantized_view window=32 vs CPU windowed oracle T=64 Tkv=192: rel=4.84e-7 OK
sdpa_naive_w_quantized_view window=32 differs from unwindowed T=64 Tkv=192: changed=65536/65536 OK
```
plus the same four at `(100,100,48)` (rel 1.12e-7, changed 53248/102400) and `(37,297,64)`
(rel 8.53e-7, changed 37888/37888). **Full battery: ALL GREEN, 0 FAIL** (`grep -c FAIL` = 0 over
the whole log, 540 lines).

The `rel < 1e-4` band is deliberately tight, not an oracle band: both sides compute the same f32
dot in the same order, so only FMA contraction separates them — the measured 1e-7 confirms it. The
`changed` counts read correctly too: at `T == T_kv == 100, win=48` only the later queries have keys
old enough to mask, so 52% of elements move; where the chunk is a continuation (`T < T_kv`) every
query has a maskable past and 100% move.

---

## Increment 8 — the chat template: a ChatML *dialect*, and reusing the qwen arm would corrupt it

The step35 template's `tokenizer.chat_template` (5723 chars, byte-identical to the HF repo's
`chat_template.jinja` dumped in phase 1) shares `<|im_start|>role\n…<|im_end|>\n` with the
qwen3.5/3.6 class **and nothing else**. It also contains every marker memra's dispatcher keys the
qwen arm on — `<tools>`, `<think>`, `add_generation_prompt` — so before this increment a loaded
Step-3.7-Flash would have silently rendered through the qwen arm: right generation tail, wrong
turn bodies. The step35 check now precedes it, keyed on `render_message_content` (the macro no
other committed template defines).

### The oracle came first

memra ships no jinja engine, so the goldens are rendered from the shipped jinja itself:
`render_step35_template.py` (committed) renders 19 cases and writes
`raw/step35-template-goldens.txt`. Every `expected` string in the Rust tests is copied from that
file. One correctness detail in the harness is load-bearing: **`trim_blocks=True,
lstrip_blocks=True`** — what HF transformers' `_compile_jinja_template` uses and what llama.cpp's
minja parses with. With them false, the newline after this template's `{% endmacro %}` leaks into
every render as a leading `\n`; the first draft of the goldens had exactly that artifact and it
would have been baked into the Rust as a phantom prefix byte.

### Eight divergences from the qwen arm, each one a corrupted prompt

| | qwen3.5/3.6 | step35 |
|---|---|---|
| reasoning level | `enable_thinking` bool | `Reasoning: {low,medium,high}\n\n` **string inside the system turn** |
| `<think>` tail | switchable | **unconditional** (no `enable_thinking` at all) |
| prior assistant turns | content only | turns AFTER the last real user query also carry `<think>\n{reasoning}\n</think>\n` |
| tool results | grouped into a `user` turn, `\n<tool_response>\n…\n</tool_response>` | own **`tool_response` role**, `<tool_response>…</tool_response>`, **no inner newlines** |
| content | `\|trim`med | **not trimmed** |
| tools header | `following functions:` | `following functions in JSONSchema format:` |
| instruction block | 4 Reminder bullets, `<function=...></function>` | **2 bullets**, literal `\n...\n` inside the example tags |
| leading system + tools | appended AFTER the instruction block | folded in **before** `# Tools` |
| call separators | `\n\n` after content, `\n` between calls | **none** |

Two of these are not cosmetic. The **reasoning boundary** (`last_query_index`) means a prior
assistant turn's rendering depends on whether a later *real* user query exists — and a user turn
whose content is itself a `<tool_response>…</tool_response>` wrapper does **not** count, so a
client replaying tool output as a user turn must not reset it. And the **think tail is
unconditional**: `ThinkMode::NoThink` is a documented no-op here (the template has no
`enable_thinking`, so `ModelCaps::think_switch` is already false and the existing switchless
contract applies unchanged). A NoThink that emitted the qwen `<think>\n\n</think>\n\n` would be a
prompt this model has never seen.

`reasoning_effort` is the model's headline three-level control (low/medium/high, per the StepFun
card) and it is a **string in the system turn**, so a bool cannot carry it. It is a parameter of
`apply_step35_template` and tested directly, but nothing on the serve path supplies it yet —
`worker::Request` has no field, and `main.rs`'s `parse_think` currently collapses
low/medium/high into two `ThinkMode` values. Both call sites pass `None` (the template's own
default: no `Reasoning:` line). Plumbing it is a serve-surface change, tracked below, not
smuggled into a bring-up commit.

### Checked, not assumed

- **`ModelCaps` probe needs no change.** Measured against the real template: `tools_branch` true
  (has `<tools>`, no hy3/gemma4 markers), `qwen_think` true (**correct** — the tail really is
  `<think>\n`), `think_switch` false (**correct** — no `enable_thinking`, so NoThink is honestly
  reported as unavailable), `instruct_type` "chatml" (defensible: it is a ChatML dialect).
- **The tool-call parser needs no change.** step35 emits `</think>\n` where qwen emits
  `</think>\n\n`; `toolcall.rs` swallows *up to* two separator newlines (`postthink_nl`), so one
  is consumed correctly and nothing is lost. The `<tool_call>` / `<function=` / `<parameter=`
  emission grammar is identical between the two templates.
- **One deliberate divergence.** The jinja's body loop has no `else`: a role outside
  {system, user, assistant, tool} renders as **nothing at all** and the turn silently vanishes.
  memra renders it as a generic turn instead, matching every other arm in `chat.rs`. A dropped
  turn is the worse failure, and the branch cannot fire from the serve surface (OpenAI roles are
  exactly those four, all reproduced byte-for-byte).
- **Not reproduced, and cannot fire from an OpenAI client:** the `name == "observation"` alias
  that renames a non-leading `system` turn's role (`Turn` carries no `name`), and the
  `<im_patch>` image path (this is a VLM; memra is text-only here).

Tests: 8 new `step35_*` cases in `chat.rs` (22 pass in `memra-tokenizer`, 75 in `memra-server`,
0 fail). They include the dispatch-order guard — a body-shaped assertion (`" pad "` vs `"pad"`),
because a tail-shaped one would pass even if the qwen arm won.

**q27 co-residence (owner's open question): yes on bytes, with wide margin.** Step 97.78 + Step MTP
3.45 + q27 14.63 (`Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`, `/scratch-models`, measured on the box) =
115.86 GiB of 191.19, leaving 75.32 GiB for both models' KV and activations; Step's own 256K KV is
20.39 of that. The unanswered part is not capacity but **scheduling** — co-resident q27 shares the
SMs Step is using, so the cost is throughput interference, which is the pp2-hardening lane's
measurement, not a fit calculation.

---

## Ledger

| item | state |
|---|---|
| artifact staged + sha256 verified | DONE |
| MTP head is a separate GGUF (needs external-file arm) | CONFIRMED |
| `step35` config parse + per-layer accessors + tests | DONE |
| `attention.head_count` array panic | FIXED |
| `attn_out_gate()` deny-list mis-split | FIXED |
| `deepseek-v3` pretokenizer | DONE (cross-checked vs an independent engine) |
| unknown-`pre` silent fallback | now warns once |
| loader: `attn_gate.weight` (all 45 blocks, per-layer width) | DONE |
| half-rotary needs a new kernel | NO — `rope_neox2_f32` already splits `head_dim`/`n_dims` |
| forward: `step35_geom`, SWA 3:1, dual rope, gate epilogue, clamp | DONE (`b13738ed`; clamp shipped `b5c8450a`) |
| SWA mask convention vs upstream `LLAMA_SWA_TYPE_STANDARD` | MATCHES verbatim — no new mask math |
| windowed prefill at hd128 (`sdpa_naive_w_quantized_view`) | DONE (every windowed FA *prefill* stamp is hd256-only) |
| kernel-check cell for `sdpa_naive_w_quantized_view` (4 assertions x 3 shapes) | DONE — battery ALL GREEN, 0 FAIL |
| dc/graph, batched, varlen, spec-verify step35 twins | deferred — refuse with a named cause |
| `mtp_full_attn_dc` step35 arm | open (lands with the MTP external-file arm) |
| chat template (StepFun ChatML dialect) | DONE — 19 jinja-rendered goldens, 8 tests; dispatch precedes the qwen arm |
| `reasoning_effort` (low/medium/high) on the serve surface | open — renderer takes it, `Request` has no field yet |
| PP-2 split boot | open |
| KV @128K under PP-2 + q27 co-residence arithmetic | open |

## Increment 9 — split (multi-shard) GGUF support: the first PP-2 boot's real blocker

The first PP-2 boot attempt on the 2x RTX PRO 6000 box got further than expected and then died on
something that has nothing to do with step35. Quoted from
`raw/pp2boot-20260806T193536Z.log`:

```
[pp] cross-device transport: stage0=dev0 stage1=dev1 (cudaMemcpyPeerAsync per cross boundary;
     peer + default-pool access granted all pairs over [0, 1]; weight home: per-stage (sharded loader))
[moe] resident-experts decision: experts 44.12GB + trunk 2.36GB vs free 100.88GB
      (expert budget 96.52GB) -> RESIDENT
thread 'main' (68214) panicked at crates/memra-engine/src/hybrid.rs:963:22:
need post_attention_norm or ffn_norm
run-gen exit=101 after 21s
```

Two facts in that log, in order of importance:

1. **PP-2 transport came up.** Peer access granted on all pairs over [0,1], the sharded loader
   picked per-stage weight homes, and the run reached the trunk-layer loop. The pipeline plumbing
   is not the blocker.
2. **`experts 44.12GB` is wrong for this model** and is the same bug as the panic. The IQ4_XS
   artifact is 97.78 GiB; 44.12GB is what you see when you only count the experts in **shard 1 of
   3**. The loader then walked to `blk.22`, whose `ffn_norm` lives in shard 2, found nothing, and
   the `.expect("need post_attention_norm or ffn_norm")` fired.

### Root cause: memra's GGUF reader was single-file only

`GgufFile::open` mmap'd exactly one file and built one tensor table from its header. Nothing in
`crates/memra-gguf/` ever read `split.no` / `split.count` / `split.tensors.count`. Confirmed
against the real headers dumped in phase 1
(`research/step37-bringup-20260802/raw/gguf-header-stepfun-iq4xs-shard1-20260802.txt`):

```
KV split.no (u16) = 0
KV split.tensors.count (i32) = 754
KV split.count (u16) = 3
```

and the block ranges per shard (`grep -oE "blk\.[0-9]+\."` over the phase-1 header dumps):

| shard | `split.no` | blocks present | note |
|---|---|---|---|
| 1 of 3 | 0 | 0..21 | also carries arch + tokenizer KVs |
| 2 of 3 | 1 | 22..44 (unsloth twin: 0..24 boundary differs) | split keys only |
| 3 of 3 | 2 | 24..44 | split keys only |

This is a **general capability gap, not a step35 quirk** — every 100GB+ GGUF ships split, so the
fix lives in the shared reader, not in a step35 arm.

### What landed

`GgufFile` now holds `shards: Vec<Shard>` (mmap + inode + path + **its own** `data_start`) instead
of one mmap, and `TensorInfo` gains `shard: usize`. `open()` reads `split.count`; when > 1 it
rebuilds every sibling name from the count via the standard `-%05d-of-%05d.gguf` form
(`llama-gguf-split`'s `SPLIT_PATH_FORMAT`), parses each, and presents ONE merged tensor table.
Callers cannot tell a split model from a single-file one.

Deliberate design calls, each with a reason:

- **Names are rebuilt from `split.count`, not parsed out of the filename.** A shard whose name
  disagrees with its own `split.no` therefore cannot silently map to the wrong bytes — the
  per-shard `assert_eq!(no, i, ...)` catches it and names both values.
- **Shard 0's metadata wins the merge; later shards may only ADD keys.** Only shard 0 carries
  architecture/tokenizer KVs, so this makes *any* shard a valid entry point: opening shard 3
  yields the same model as opening shard 1 (tested).
- **`split.tensors.count` is checked** against the merged total (754 for this artifact). A
  truncated download that loses a shard fails loudly at open instead of at `blk.N`.
- **A split shard with a hand-renamed filename is a clear error**, not a partial load: the message
  names the declared `split.count` and says sibling shards could not be found.
- **The entry shard is re-parsed in the merge loop** rather than kept from the probe. One extra
  header parse + lazy mmap against a 105 GB model is free, and it keeps the loop uniform.

### The second bug the split fix exposed: the spill tier's mmap/offset pairing

`SpillCtx` held ONE `file_map` (shard 0's) and `place_expert` paired it with the absolute offsets
from `tensor_file_range`. On a split model those offsets are **relative to the owning shard**, so
that pairing would have read the wrong bytes for every expert in shards 2+ — silently, for any
shard whose experts fit inside shard 0's length, and as an out-of-bounds slice otherwise. Fixed by
making it per-shard: `file_maps: Vec<Arc<Mmap>>` / `files: Vec<Arc<File>>` indexed by
`TensorInfo::shard`, and `place_expert(.., shard)` selects the matching pair. This is a correctness
fix on the disk-spill path, found by reading the seam rather than by a failing run — the boot never
reached it (`MEMRA_SPILL_DISK` was unset, and this model went RESIDENT).

### Gates

`cargo test -p memra-gguf`: **75 passed, 0 failed** (70 pre-existing + 5 new). The 5 new tests
serialize REAL 2-shard GGUF pairs — full header, KV table, tensor infos, aligned data blob — with
per-tensor fill bytes, so a wrong-shard read is caught **by value**, not merely by length:

| test | what it pins |
|---|---|
| `split_model_presents_one_merged_tensor_table` | all 3 tensors visible from shard 0; the shard-1 tensor (the `blk.22` analogue) reads its own bytes; arch KV survives the merge |
| `any_shard_is_a_valid_entry_point` | opening the LAST shard yields identical `(shard, offset, n_bytes)` and identical bytes for every tensor |
| `tensor_file_range_is_relative_to_the_owning_shard` | the shard-1 range reproduces shard 1's bytes when applied to shard 1's file; the two shard inodes are distinct `Arc`s |
| `single_file_gguf_is_unchanged_one_shard` | `n_shards()==1`, `shard==0`, and `tensor_file_range` still equals `data_start + offset` — the historical contract |
| `split_shard_with_a_nonstandard_filename_is_a_clear_error` | a renamed shard fails at open with a message naming `split.count` and the missing siblings |

`cargo check --workspace --all-targets`: clean. `gguf-inspect` now prints `shards=N` with a
per-shard tensor count and path, and its data-bounds check runs per shard against that shard's real
file size (one global max would be meaningless once offsets are per-shard).

## Increment 10 — FIRST PP-2 BOOT: Step-3.7-Flash generates on both cards

Three boots, each failing on a different real bug, the third clean. All three logs are committed
(`raw/pp2boot-20260806T{193536,195322,195954}Z.log`) because the first two are the evidence for
the two fixes.

### Boot 2 (after the split-GGUF fix): the whole model loads

```
[moe] resident-experts decision: experts 101.07GB + trunk 3.92GB vs free 100.88GB
      (expert budget 94.96GB) -> SLRU cache
[fa] v4 decode family disabled: gqa 12 > fa_v4_smem capacity 8 (v3 lane serves)
loaded step35 (45 trunk layers; optional MTP skipped)
prefill argmax=6776  decode argmax=6776  logit maxdiff=3.702e-1  MATCH
Error: "step35 has no device-counter/graph decode arm (SWA needs an offset KV view the dc
        kernels cannot express) — use the eager decode"
```

`experts 101.07GB` (was 44.12GB) confirms the split fix at the byte level, and the prefill/decode
argmax already MATCHED. The failure was **this lane's own refusal, reached by a door that never
checked the arch.**

### The second bug: two DC-eager doors with no arch gate

`b13738ed`/earlier work put a deliberate, loud refusal in `full_attn_decode_dc_inner`
(decode.rs:1899) — step35's SWA layers read a token-OFFSET KV view that the dc kernels' `len_d`-
derived `t_kv` cannot express, so running the generic geometry would be silently wrong. Correct
refusal. But the two QWEN DC-EAGER doors that *call* it — `generate` (decode.rs:2173) and
`generate_with` (decode.rs:2452) — gate on `MEMRA_QWEN_DC != 0` + greedy + no-penalty and
**nothing else**. Every greedy step35 generation walked straight into the refusal, so a correct
internal guard surfaced as a user-visible `generate()` error *after* a clean load and an argmax
MATCH.

Fixed by adding `self.cfg.step35.is_none()` to both doors, which routes step35 to the host-logits
eager loop at the bottom of `generate_with` (`decode_step` -> `full_attn_decode_pre` ->
`step35_decode_attn`, decode.rs:2631) — the supported decode for this arch. Note what this is NOT:
it is not a flag and not a widening of the refusal. Opening the dc door for step35 still requires a
windowed dc `fa_decode` plus a per-layer-`n_head` capture.

### Boot 3: PASS

```
loaded step35 (45 trunk layers; optional MTP skipped)
prompt tokens: [0, 65106, 1205, 260, 28499, 6632, 344, 16]
prefill argmax=6776  decode argmax=6776  logit maxdiff=3.702e-1  MATCH
generated 32 tokens in 1.589s = 20.14 tok/s (Stage-B int8 dp4a decode, gen-only; prime 0.134s)
MoE cache STEADY-STATE (32 tokens after warmup): hit-rate=89.0% | 133.5 MB/decode-token
      (vs 2678 MB/token Stage-1 => 20.1x less PCIe)
run-gen exit=0 after 42s
```

Output text, coherent and on-topic:

> `# 2.2.2 Pipeline Hazards` / `A pipeline stage is said to have a hazard when the instruction in
> the stage cannot proceed in the normal`

**The mission-critical claim is now measured, not assumed: this model boots PP-2 and generates.**
97.78 GiB of weights across two 96GB cards, `stage0=dev0 stage1=dev1`, peer + default-pool access
granted on all pairs, per-stage weight homes from the sharded loader.

Two facts worth carrying, neither of them a perf claim (perf is the pp2-hardening lane's job —
these are single runs, N=1, stated as such):

- The model does **not** go resident even on 2x96GB: `101.07GB experts + 3.92GB trunk vs 100.88GB
  free` -> SLRU cache. So the PP-2 serving path for this SKU is a *spill* path, and the MoE cache
  is load-bearing rather than incidental. Its steady state on this 32-token run was 89.0% hit-rate
  / 133.5 MB per decode token.
- `[fa] v4 decode family disabled: gqa 12 > fa_v4_smem capacity 8` — step35's GQA ratio (96/8 on
  SWA layers) exceeds the v4 decode family's smem capacity, so the v3 lane serves. A named
  capability gap for whoever tunes this SKU, recorded here so it is not rediscovered.

## Increment 11 — the exactness battery on the box: PP-2 is BIT-IDENTICAL, and the base artifact has no drafter

`raw/gates-pp2-20260806T200323Z.log`, one bounded `flock -w 1800` window on the shared box
(`lock acquired 2026-08-06T20:03:23Z` … `lock released 2026-08-06T20:05:21Z`, 118 s), against
`fdca2df3+dcgate`. Three gates, all output quoted below verbatim.

### G1 — `ppn-gate <model> 2 8 16`: PASS, both arms

```
stage fence: [0, 22, 45] over 45 layers
ppn gate PASS [serial]: 24 steps (8 prime + 16 gen) BIT-IDENTICAL logits (n_vocab=128896,
      fence=[0, 22, 45]; stages=2 streams=per-stage overlap=0 devices=0,1
      splits=default(even) shard=per-stage)
ppn gate PASS [pipelined]: 24 steps (8 prime + 16 gen) BIT-IDENTICAL logits (n_vocab=128896, …)
ppn-gate exit=0
```

This is the gate that makes the PP-2 boot a *correctness* result and not just a "it ran". Both the
serial `decode_step` walk and the `decode_step_h_ppn_deferred` pipelined arm (3 tokens in flight,
overlap forced) reproduce the door-OFF reference's 128896 f32 logits **bit for bit** at all 24
steps. Per ppn_gate.rs's own header the reference runs the unsplit walk against the SAME sharded
placement, so this is specifically a test of the *seam* — the `cudaMemcpyPeerAsync` handoff at
layer 22 — not of placement.

Read together with the split-GGUF fix: the merged tensor table routes every read to the owning
shard, and PP-2's cross-device carrier moves activations across the fence, and the composition is
exact. Both of this session's fixes are now covered by a bit-identity gate, not just by a boot.

### G2 — `run-gen`, 19-token prompt, 64 tokens: MATCH, both argmax gates

```
prompt tokens: [0, 21750, 260, 3107, 15363, 26131, 1192, 260, 25454, 28499, 28232, 12740,
      62027, 14, 305, 6731, 834, 23616, 16]
prefill argmax=6776  decode argmax=6776  logit maxdiff=1.444e0  MATCH
batched-prime argmax=6776  tokenwise argmax=6776  logit maxdiff=1.097e0  MATCH
prefill 19 tok in 0.4956s = 38.3 tok/s (pp19)
generated 64 tokens in 3.961s = 16.16 tok/s (Stage-B int8 dp4a decode, gen-only; prime 0.499s)
run-gen exit=0
```

The second line is new relative to the boot (which ran an 8-token prompt): **batched-prime vs
tokenwise** also MATCHes, i.e. the windowed-SWA prefill path and the one-token-at-a-time path agree
on this artifact at pp19. Output text stayed on-task and clean through 64 tokens:

> `A CPU pipeline breaks instruction execution into stages (fetch, decode, execute, memory,
> write-back) so that several instructions are processed simultaneously, increasing throughput.
> One hazard is a data hazard, …`

Note the token id `0` leading the prompt: `add_bos_token=true` and BOS is `<|begin of sentence|>`
id 0, which is what the deepseek-v3 pre-tokenizer path is supposed to emit — consistent with
`raw/pretok-deepseek-v3-reference-20260806.txt`.

The two throughput numbers above are **N=1 bring-up receipts on a shared box, not perf claims**
(and the box's other tenant is the pp2-hardening lane — cross-run comparison would be invalid
anyway). They are recorded only because a gate log that omits its own timings is harder to
reproduce. The steady-state MoE line differs from the boot's (96.1% / 47.3 MB per decode token
here vs 89.0% / 133.5 MB on the 32-token boot) for the obvious reason — a longer run amortizes the
cold-stage — which is itself the argument that this SKU's serving story is dominated by the spill
tier.

### G3 — `run-spec` K=1..8: refused, and the refusal is correct

```
loaded step35 (45 layers, nextn=0)
ERROR: model has no MTP/NextN head (nextn_predict_layers=0, no blk.N.nextn.eh_proj).
      generate_spec is unavailable for this file.
run-spec exit=2
```

Not a failure of the lane — a fact about the artifact. StepFun ships the MTP head as a **separate
file**, so the base `Step-3.7-flash-IQ4_XS-*.gguf` carries `nextn_predict_layers=0` and there is
nothing to draft with. The K=1..8 self-consistency gate is therefore **not yet applicable** and
becomes applicable exactly when sequence item 5 (drafter wiring) lands; it is not being skipped.

The drafter that must attach (`raw/gguf-header-stepfun-mtp-q8-20260802.txt`,
`Step3.7-flash-mtp-Q8_0.gguf`, 3,707,276,416 B, sha `469a8166…`): `step35.block_count=48`,
`nextn_predict_layers=3`, three chained NextN blocks numbered **45, 46, 47** — each a full
`{enorm, hnorm, eh_proj[8192,4096], attn_*, ffn_*, shared_head_norm, shared_head_head[4096,128896]}`
set. Two structural facts already visible in the header that the wiring must handle:

1. `MtpHead::load_draft` computes its block index as `n = dcfg.n_layer - dcfg.nextn_predict_layers`
   = 48 - 3 = **45**, which is exactly the first NextN block. So the existing external-draft path
   addresses the right block with no arithmetic change — but it loads **one** head, ignoring 46/47.
   Depth >1 for this SKU is a separate decision, not a free consequence of attaching the file.
2. The head names the shared head `blk.N.nextn.shared_head_head.weight`, while `load_draft` reads
   the draft's own top-level `output.weight` (present here, `[4096, 128896]` Q8_0) — and the
   *embedded* loader at hybrid.rs:1078 looks for `nextn.shared_head.weight`, a name this file does
   **not** use. Recorded so the mapping is checked against the header, not assumed.

Battery bookkeeping: `=== battery rc=0`, and `nvidia-smi` at lock release reported `0, 0 MiB` /
`1, 0 MiB` — the process exited clean and left the cards free for the co-tenant lane.

---

## Sequence item 5 — MTP drafter wiring: CLOSED (gate PASSES both contracts)

Two bugs stood between the drafter file and a working spec path. Both were found on the box, both
have committed receipts, and the second one is the interesting one.

### Bug 1 — the attach panicked (`348a5787`)

The very first `MEMRA_MTP_DRAFT` attach died inside cudarc:

```
thread 'main' (82647) panicked at cudarc-0.19.8/src/driver/safe/core.rs:1917:32:
called Option::unwrap() on a None value
```

`RUST_BACKTRACE=full` gave the frames (`raw/mtp-bt-20260806T212127Z.log`):
`copy_into` → `full_attn_verify` → `decode_step_t_core_stream` → `generate_spec_inner2` →
`generate_spec`. Cause: `step35_verify` sized its output buffer from the per-layer **head**
geometry (`t * n_head_at(il) * head_dim` = 8192 on full-attn, 12288 on SWA) while
`step35_decode_attn` returns rows **post-`wo`**, i.e. `[n_embd]` = 4096 — the same contract the
generic arm's `matmul_decode_exact(&fa.wo, &attn_g, t)` return has. The first row copy overran
and `CudaView::slice` returned `None`. Fixed by sizing/striding on `n_embd` plus a
`debug_assert_eq!` that pins the contract. Blast radius is step35 only.

### The geometry work from phase 2's first half was correct

Worth stating plainly, because it was the thing most likely to be wrong: the drafter resolved its
per-layer geometry correctly on the **first** attach, and both of `load_draft`'s tensor-shape
witnesses passed silently.

```
[mtp-draft] step35 MTP geometry blk.45: n_head=96 n_head_kv=8 n_rot=128 rope_base=10000 swa=true window=512
```

That is the two-file geometry trap handled: 96 heads (not the trunk array's out-of-range `.last()`
of 64), SWA, 128 rotary dims, base 1e4 — all read from the **drafter's own** arrays.

### Bug 2 — the draft lm_head was the TRUNK's lm_head (`9f9d8321`)

With the panic gone, the gate produced the failure that no exactness gate can see:

```
[generate_spec K=1] 32 tok in 5.917s = 5.24 tok/s (0.32x vs generate)
  acceptance: 0/31 = 0.0%   self-consistency: PASS (identical to generate)
  WARNING: acceptance == 0 with identical output — MTP head is likely forwarded wrong
           (bonus-token masking). Speedup will be absent.
```

0/31, 0/62, 0/93, 0/124, 0/155, 0/186, 0/217, 0/248 — **exactly zero** at every K, with
self-consistency PASS throughout (`raw/mtp-draft-20260806T212902Z.log`). Exact zero across 248
drafts is structural, not a quality problem.

`load_draft` read the drafter file's **top-level `output.weight`** as the draft lm_head. Upstream's
`graph_mtp` (llama.cpp `src/models/step35.cpp:553`) instead prefers the block's own head:

```cpp
ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
```

Both tensors exist in this file at identical `[4096, 128896]` Q8_0, so **no shape check can tell
them apart**. So: hash the payload bytes (`raw/draft-head-tensor-hashes-20260807.txt`).

| tensor | sha256 (head) |
|---|---|
| drafter `output.weight` | `3eec5831…` |
| `blk.45.nextn.shared_head_head.weight` | `c90b907b…` |
| `blk.46.nextn.shared_head_head.weight` | `a22d2957…` |
| `blk.47.nextn.shared_head_head.weight` | `4b21e137…` |
| drafter `output_norm.weight` | `d7526f44…` |
| **trunk** `output_norm.weight` | `d7526f44…` |

Three distinct per-block heads, none equal to `output.weight`. And the tell: the drafter's
top-level `output_norm` is **byte-identical to the trunk artifact's**, while blk.45's own
`shared_head_norm` differs (`405dbb0d…`). The drafter's top level is a re-quantized (Q6_K→Q8_0)
**copy of the trunk's output stack**, shipped so the draft GGUF stands alone. It is not the MTP
head. (`token_embd` also differs from `output.weight` here, so this is not a tied-embedding model
either.) The old code was projecting the MTP block's hidden through the *trunk's* lm_head under the
*MTP block's* norm — fluent-looking drafts with near-zero agreement against the trunk's real
next-token distribution.

Fixed as a **preference**, not a replacement: prefer `blk.{n}.nextn.shared_head_head.weight`, fall
back to top-level `output.weight`. FR-Spec drafts publish their *trimmed* head (+ `d2t`) as the
file-level `output.weight` and carry no `shared_head_head`, so they keep the old path. The
`[mtp-draft]` line now prints which source was taken.

**Second bug, same root, found by the same probe.** The *embedded* MTP loader read
`blk.{n}.nextn.shared_head.weight` — a name no artifact and no upstream tensor mapping uses
(upstream: `LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD` → `"blk.%d.nextn.shared_head_head"`). That
`load_opt` was silently **always-None**, so every embedded-MTP model fell back to the trunk
`self.output` in `mtp_head_forward_dev` op 12. Benign for heads genuinely tied to the trunk head,
wrong for any artifact shipping its own. This is precisely the mapping the section above flagged
as "recorded so it is checked against the header, not assumed" — it was checked, and it was wrong.

### The gate, after the fix (`raw/mtp-draft-PASS-20260806T215132Z.log`)

```
[mtp-draft] external draft head: blk.45, source=nextn.shared_head_head, head_vocab=128896 (full)
[generate_spec K=1] acceptance: 14/18 = 77.8%   self-consistency: PASS (identical to generate)
```

| K | accepted / drafted | acceptance |
|---|---|---|
| 1 | 14/18 | **77.8%** |
| 2 | 15/34 | 44.1% |
| 3 | 15/51 | 29.4% |
| 4 | 15/68 | 22.1% |
| 5 | 15/85 | 17.6% |
| 6 | 15/102 | 14.7% |
| 7 | 15/119 | 12.6% |
| 8 | 15/136 | 11.0% |

Both contracts hold at every K: token-identical to `generate`, and acceptance > 0. K=1 went from
**0/31 = 0.0% → 14/18 = 77.8%**. One bounded `flock -w 1800` window (21:51:32Z → 21:53:45Z, 133s);
`nvidia-smi` at release reported `0, 0 MiB` / `1, 0 MiB`, cards returned to the co-tenant lane.

Sequence item 5 is closed. N=1 per K on a shared box; the tok/s figures in that log are bring-up
receipts, **not** perf claims — and note spec is still *slower* than plain generate here (0.56x at
K=1), which is expected while each draft costs a full MoE trunk-block forward against a 20 tok/s
baseline. The gate is acceptance, not speed; PP-2 spec throughput belongs to the pp2-hardening lane.

### What the acceptance curve says next (recorded, NOT fixed)

Read the table by column, not by row: **accepted is flat at 15** while drafted grows 18 → 136.
Slot 0 accepts ~78%; slots 1+ accept ~nothing. Raising K from 1 to 8 buys **one** extra accepted
token for 7.5x the draft work.

The cause candidate is in upstream's own comment (`step35.cpp:378`):

```cpp
// Multi-block MTP: the DECODER_MTP graph runs the MTP head selected by
// cparams.nextn_layer_offset (0 = first trained head). The speculative driver
// bumps the offset per draft step to chain heads 45->46->47.
const int il = hparams.n_layer() + cparams.nextn_layer_offset;
```

Upstream uses a **different head per draft step**. memra loads one block
(`n = n_layer - nextn_predict_layers` = 45) and reuses it for every step `j`. This artifact ships
all three heads with genuinely different weights (the byte hashes above). A head trained for the
+1 position, reused at +2/+3, drafts near-noise past slot 0 — which is exactly the flat-15 curve
measured. Not fixed here: multi-block chaining is a new code path (`MtpHead` becomes indexed per
draft step, each block needs its own scratch KV), not a bring-up fix.

**Served implication for this SKU today: K=1 is the correct depth.**

---

## Sequence item 4 (remainder) — admission ladder

`raw/admission-ladder-step37-20260807.txt`. Constants quoted from `worker.rs` @ `9f9d8321`
(`MAX_ACTIVE=4`, `MEMRA_CTX` floor 8192, `SPEC_SHRINK_SLACK=64`, `SPEC_SHRINK_RESERVE=1.5 GiB`,
gate `free >= cost + reserve`); bytes carried from `raw/kv-budget-pp2-20260806.txt`.

The binding constraint is **per-card, not pair-aggregate** — KV is allocated by the stage that owns
the layer (`memra-kv/src/lib.rs:253`) and weights are per-stage (`pp.rs:518`). At the even 23/22
cut with memra-default KV, stage-0 occupancy runs 54.1 / 59.3 / 64.5 / **69.7** GiB for 1-4
concurrent 128K sessions against 95.59 GiB. **At the honest 128K target, `MAX_ACTIVE=4` is not
KV-bound on this box** (~25.9 GiB spare on stage 0). 4×256K lands at 90.6 GiB — byte-plausible,
but that leaves 4.99 GiB for activations plus the 1.5 GiB reserve plus MoE staging, and that is
unmeasured.

Two serving findings fell out, both recorded and neither fixed in this lane:

**(A) This SKU is silently spec-ineligible when served.** `spec_eligible` (`worker.rs:2461`)
requires `lm.model.mtp.is_some()`, but the MTP head is a separate GGUF and the trunk declares
`nextn_predict_layers=0` — so a served trunk has `mtp == None` and every request takes plain
decode, with no log line saying so. The server's load path does not consult `MEMRA_MTP_DRAFT` for
this two-file shape. Nothing breaks (the plain-path `reserve = cost` branch takes over
consistently), but the model forgoes the MTP win this lane just proved works. Wiring the server's
`+draft` path to accept a standalone step35 drafter is the named follow-up — a serve-surface
change, deliberately not bundled into bring-up.

> **CLOSED 2026-08-07 by `lane/step-draft`** (`research/step-draft-20260807/RESULTS.md`). One
> correction to the finding above: the server load path *does* reach `MEMRA_MTP_DRAFT` —
> `load_from_source_impl` reads it (`hybrid.rs:1277`) and `HybridModel::load` funnels through
> there — and the `+draft` per-model spelling already worked for the two-file shape too, since
> `MtpHead::load_draft` resolves step35 geometry from the drafter file's own arrays. So the attach
> needed no new spelling. **The silence was the whole defect**, and it is now impossible: a step35
> model loaded without a drafter WARNS with the exact attach string (verified on the real 45-layer
> trunk, `raw/box-armE-warn-20260807T000837Z.log`), a `+draft` path that is missing or unloadable
> refuses to start with the cause quoted, and drafter + armed spec over sharded cross-device PP-2
> refuses before `Engine::new` with the #87 pointer. What remains for spec-*served* Step is only
> #87 — Step needs PP-2 to fit at all, and spec over PP-2 stays quarantined.

**(B) The MoE residency decision is PP-blind in its numerator.** `build_dev_exps`
(`hybrid.rs:244-280`) resolves `free` from the owning stage's engine — correct, `layer_engine`
hands it the per-stage engine (`pp.rs:755`) — but projects `exps` by summing **every**
`blk.*_exps.*` tensor in the GGUF header (`hybrid.rs:254-260`), i.e. the whole model's expert
bytes including the layers living on the *other* card. Hence the boot line
`experts 101.07GB + trunk 3.92GB vs free 100.88GB`: whole-model bytes against one card's free.
The verdict is right on this SKU anyway, and the SLRU path it chose is measurably healthy (89.0%
steady-state hit rate, 133.5 MB/decode-token vs the 2678 MB/token Stage-1 baseline = 20.1x less
PCIe). But it would wrongly spill a bank that *fits* per-stage on a wider PP split. Not fixed:
changing residency selection is perf-affecting and belongs behind the pp2-hardening lane's A/B.

---

## Regression guard for the draft-head fix (`3dad4f01`)

The 0%-acceptance bug had a property worth naming: **no gate in the repo could have caught it.**
`kernel-check` is model-free, `run-gen` argmax was MATCH, and `run-spec` self-consistency PASSED at
every K=1..8 — because the verify arbitrates, a wrong draft head produces *correct output* and only
loses speed. The single line that flagged it was `run_spec.rs`'s own
`WARNING: acceptance == 0 with identical output`, which is a warning, not a failure.

So the fix is pinned by construction rather than by a gate. `draft_head_tensor(has, n)` factors the
name preference out of `MtpHead::load_draft` (CUDA device + 3.5 GB artifact) into a pure function
over a tensor-presence predicate, with five GPU-free unit tests: the real drafter's tensor set
(transcribed from `raw/draft-head-tensor-hashes-20260807.txt`) resolving to
`blk.45.nextn.shared_head_head.weight`; per-block head selection across 45/46/47 (three distinct
matrices by sha256, so the name must be index-built — this is the seam multi-block chaining will
use); the FR-Spec fallback to file-level `output.weight`; the legacy `nextn.shared_head` probe
losing to the real name; and a different block's head never being borrowed.

`cargo test -p memra-engine --lib` = 46 passed / 0 failed (5 new), no GPU. The
`[mtp-draft] external draft head: source=` receipt now derives from the resolved name rather than
re-probing, so the log and the load cannot disagree.

Note on repo hygiene, unrelated to this lane: `cargo fmt --check` reports drift in ~100 files
across `memra-engine` (rustfmt 1.9.0-stable, no `rust-toolchain.toml` pin), and neither
`.github/workflows/ci.yml` nor `tools/hooks/pre-push` runs fmt or clippy. The lines this commit
adds are fmt-clean; the surrounding drift predates it and is a rustfmt-version artifact, not a
regression. Recorded, not touched.

---

## Sequence item 6 — kernel-check (model-backed) + chunk-invariance on PP-2

One flock window on the box, 2026-08-06T22:15:50Z -> 22:18:52Z, cards verified `0, 0 MiB` at
release. Raw: `raw/kc-chunkinv-20260806T221550Z.log` plus the two probe logs.

**kernel-check: ALL GREEN, and now actually model-backed.** `kc_model` resolves oracle artifacts by
exact basename, and the `iq4xs-mmq` section named only KAT-Coder — which this box does not have
(`find / -name '*IQ4_XS*.gguf'` returns the step artifact and nothing else). So that section
KC-SKIPped on a SKU whose entire trunk ships IQ4_XS: the oracle was skipping the one dtype it will
decode in production. `e3e8577a` adds the step artifact as an `.or_else` fallback (the 3-shard split
resolves fine — `GgufFile::open` finds siblings, `tensor_data` is shard-relative), so the arm now
runs on the SKU's own bytes:

    iq4xs-mmq [Step-3.7-flash-IQ4_XS-00001-of-00003.gguf token_embd.weight]
      T=16 rel=5.67e-7 | T=64 rel=1.63e-4 | T=128 rel=1.88e-4 | T=512 rel=2.04e-4   all OK
    ALL GREEN: kernels match CPU reference.   exit=0

Same commit fixes two log-honesty holes in that section: the label now names the artifact (trunk
tensor names collide across models), and a resolved-but-unusable artifact now prints KC-SKIP instead
of falling through in silence that reads exactly like a pass.

Worth noting for anyone sizing this: kernel-check is single-device by construction
(`Engine::new(0)`) and mmaps ONE tensor rather than the model, so a 105 GB artifact runs the battery
without PP at all.

**chunk-invariance: G6a PASS — and it is the first chunkinv assertion that crosses a device
boundary** (105 GB > 96 GB/card means PP-2 or nothing). Both pinned prompts, bit-identical logits,
hidden rows (`first_div=-1`), and 32-step greedy streams at chunks {2048, 64, 32}.

**Then the gate's own teeth-check fired, and it was right to.** Two gaps, both written up in
`raw/chunkinv-step35-findings-20260807.txt`:

**GAP 1 — the canary is inert on this arch.** `MEMRA_PRIME_F32CHUNK0` is read only inside
`full_attn_prime_fa_dispatch` (`hybrid_forward.rs:1417`), but `full_attn_prime` diverts step35 to
`step35_attn_prime` two lines earlier (`:1289-1291`). The canary sets an env var no code on this path
reads, so both arms ran the identical configuration and `CANARY UNEXPECTEDLY MATCHED` is the honest
report. There is no equivalent seam to flip because `step35_attn_pre_wo` was written grain-free from
the start — no `base_len == 0` f32 special case ever existed to roll back. The gate's header already
warns that a label-flipping canary is vacuous; the corollary this run establishes is that a
*real-seam* canary is equally vacuous on any arch that does not route through that seam. Practical
consequence: **the step35 arm of this gate has no demonstrated teeth today, so its PASS must not be
read as regression-proof.**

**GAP 2 — the more consequential one: the pinned prompts are shorter than the SWA window, so the
gate compared one kernel against itself.** step35's prime attention selects between two different
kernels, and the selector depends on chunk size (`hybrid_forward.rs:6820-6844`, `win=512`):
`t_kv > win` takes `sdpa_naive_w_quantized_view` (the f32 floor, windowed mask required — no
windowed FA stamp exists at head_dim 128), otherwise `fa_prefill_view_ws`. At T=96 and T=147, `t_kv`
never exceeds 512, so **every chunk at every tested chunk size took `fa_prefill_view_ws`.**
Byte-identity was close to guaranteed, and the property most at risk on this arch — that the two
kernels agree — went untested. The split lives at T≈2000, where chunk 4096/2048 are pure `naive_w`
while 512/64/32 are mixed. That is a chunk-dependent numeric *class*, the same family as the finding
this gate was built for, and it is load-bearing here: 93.5:1 prefill-heavy traffic at a 128K ctx
target makes multi-chunk prefill past 512 tokens the common case, not the tail. The code comment at
`:6829-6830` asserting "same numeric class" is supported by G6a only *within* the window.

Named next chunkinv work item: sweep prompts ≥ ~1.5K tokens over chunks {4096, 2048, 512, 64} and
read the per-row maxdiff razor already built into `chunkinv` (a kernel-class edge shows an
order-of-magnitude step at the boundary; GEMM fold noise is a flat band). Not a bring-up blocker —
G6a is a real pass and kernel-check is ALL GREEN — but the coverage claim must not be overstated
until it runs.

---

## GAP 2 closed — and it is a real defect, reduced to a closed form

The named next work item ran, and the prediction was right for a worse reason than expected.
step35 prefill is **chunk-dependent**: for any prompt past the 512 SWA window, `MEMRA_PRIME_CHUNK`
changes the logits, the hidden rows, and the generated text.

The control and the defect, same code and same prompt family (`raw/chunkinv-long-20260806T222721Z.log`):

| prompt | ref | 2048 | 512 | 64 | verdict |
|---|---|---|---|---|---|
| T=402 (below window) | 4096 | EXACT | EXACT | EXACT | CHUNK-INVARIANT |
| T=4883 | 4096 | EXACT | DIFFER 1.813e0, greedy diverges step 6 | DIFFER 1.813e0, step 6 | **CHUNK-DEPENDENT** |

T=402 being clean is what makes this a real finding rather than a broken probe: the defect requires
T past the window, which is exactly why the pinned T=96/147 prompts could never have reached it.

**The mechanism is kernel selection, and it collapses to arithmetic.** A chunk `[b,e)` computes
`off = max(0, b-(win-1))` and `t_kv = e-off`; `t_kv > win` takes the f32 windowed floor
`sdpa_naive_w_quantized_view`, otherwise the dequant-once `fa_prefill_view_ws`. So a chunk starting
below `win` has `off=0` and is FA iff `e <= win`, while any later chunk has `t_kv = t+511 > win`.
The FA rows are therefore always a contiguous **prefix** `[0,P)` with

```
P = c * floor(win/c)   for c <= win ;   P = 0   for c > win
```

and **the verdict depends only on P.** Verified by enumerating the real loop — including the
`PRIME_MIN_T=16` tail merge at `hybrid_forward.rs:470` — for every `c` in [2,700] plus
{768,1024,2048,4096}. This is why `P(512)=P(64)=512` are **byte-identical to each other** (10 chunks
vs 77) while both diverge from the `P=0` family, and why `P(1024)=P(768)=P(600)=0` are all exact.

A **pre-registered** battery then tried to break it (predictions committed in
`chunkinv-knife-step35.sh` before the run; `raw/chunkinv-knife-20260806T224947Z.log`) — 4/4:

| pair | P | predicted | measured |
|---|---|---|---|
| 4096 vs **513** | 0 vs 0 | EXACT | EXACT, 0.000e0 |
| 4096 vs **512** | 0 vs 512 | DIFFER | DIFFER, div\@0, 1.813e0, step 6 |
| 512 vs **384** | 512 vs 384 | DIFFER | DIFFER, div\@**384**, 1.417e0, step 13 |
| 512 vs **256** | 512 vs 512 | EXACT | EXACT, 0.000e0 |

Two of these are load-bearing. `513` vs `512` is a **one-token flip of the verdict** in a single
process on a single clock at an identical 10-chunk count — nothing in reduction order or tile shape
is discontinuous there, only the arm predicate. And `256` runs **twenty** chunks against the
reference's ten, double the partial sums, yet is bit-identical because P matches; a fold-order
account required that pair to differ. The model also predicted first divergence at
`min(P_ref,P_arm)`, and PRED-3 diverges at exactly row 384 — a number only this model produces.

Consequently the comment at `hybrid_forward.rs:6829-6830` ("Same cache bytes, same numeric class")
is **false as written**: same bytes, different numeric class.

Why it matters here rather than academically: `MEMRA_PRIME_CHUNK` is documented as a
machine-config/OOM knob, so two rigs serving this SKU with different values return different text —
the exact class `research/chunk-invariance-20260805` was built to eliminate, re-entering through a
different door on a new arch. P differs across {64,512,1024,2048,4096} for **95.7%** of prompt
lengths under 12000, starting at T=513. The default 4096 has P=0, so a naked single-rig run is
self-consistent and the bug surfaces as a field nondeterminism report — the worst way to find it.

**Not fixed here, deliberately**: this is kernel selection on the launch SKU's served prefill path,
so it needs before/after prefill numbers per `research/benchmarks.md`, not a bring-up commit. The
arithmetic does narrow it: forcing `naive_w` on SWA layers whenever `T > win` makes `P ≡ 0` for
every chunk size, is correct-by-construction, respects the `t_kv <= 12287` smem ceiling, and — since
the default already has `P=0` — **cannot move the shipped default's numbers at all**, paying only
where the knob is turned down. Full option set and the arbitrating measurement are in
`raw/chunkinv-step35-GAP2-CONFIRMED-20260807.txt`.

This also resolves GAP 1 differently than expected. The canary is inert because step35 never had a
grain seam, and after the fix there is still nothing to flip — the correct step35 assertion is naked
chunk-invariance (`chunkinv` at T=4883 over {4096,513,512,256,64} returning CHUNK-INVARIANT), which
must land in the **same commit as the fix**, since today it is legitimately red.

Two of my own intermediate claims were refuted along the way and are retracted in the receipt: that
512 and 64 should differ pairwise (section B measured EXACT — chasing this produced the closed form),
and a suspected "tail hazard" on the logits row (enumeration over T in (512,40000] found zero cases;
the `PRIME_MIN_T` merge rules it out). The `--profile` razor's "step at the boundary" framing also
does not fit this defect shape and is documented as inapplicable rather than forced.

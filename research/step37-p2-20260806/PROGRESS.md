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
| chat template (StepFun ChatML dialect) | open |
| PP-2 split boot | open |
| KV @128K under PP-2 + q27 co-residence arithmetic | open |

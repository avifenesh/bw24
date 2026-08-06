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

### Confirmed GAP, not yet fixed: `deepseek-v3` pretokenizer

`tokenizer.ggml.pre = deepseek-v3`. `crates/memra-tokenizer/src/lib.rs:578-589` dispatches only
`"qwen35"`/`"qwen2"` and **silently falls through to `split_qwen35`** for anything else. The
upstream deepseek-v3 regex set is materially different from qwen2's (CJK character class,
`\p{N}{1,3}` digit grouping, a distinct punctuation+letter class) — `llama-vocab.cpp:318-326` vs
`:373-381`. Phase-1 listed this as "VERIFY"; it is a real gap and will produce wrong token ids.

---

## Ledger

| item | state |
|---|---|
| artifact staged + sha256 verified | DONE |
| MTP head is a separate GGUF (needs external-file arm) | CONFIRMED |
| `step35` config parse + per-layer accessors + tests | DONE |
| `attention.head_count` array panic | FIXED |
| `attn_out_gate()` deny-list mis-split | FIXED |
| `deepseek-v3` pretokenizer | GAP, open |
| loader: `attn_gate.weight` on the full-attn path | open |
| forward: `step35_geom`, SWA 3:1, dual rope, half-rotary, gate epilogue, clamp | open |
| chat template (StepFun ChatML dialect) | open |
| PP-2 split boot | open |
| KV @128K under PP-2 + q27 co-residence arithmetic | open |

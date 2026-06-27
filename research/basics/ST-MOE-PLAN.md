# Safetensors MoE + Hybrid-SSM Name Map (the deferred half of `safetensors_loader`)

The dense safetensors path already works end-to-end: `SafetensorsSource` (`crates/bw24-gguf/src/source.rs:52-97`)
maps ggml names → HF names via `ggml_to_hf` (`crates/bw24-gguf/src/hf_mapping.rs:17-61`), reverses the
row-major shape (`safetensors.rs:58-60`), and `GpuTensor::load_from_source` (`model.rs:33-71`) dequants
F32/F16/BF16 → f32 or keeps quant blocks packed. `Model::load_dense_from_source` (`model.rs:140-172`)
loops entirely in ggml names. **Two things are still unwired**, and they are independent:

1. **MoE stacked-experts gather**: GGUF stores 256 experts as ONE 3D tensor `blk.N.ffn_*_exps.weight`
   `ne=[in_f,out_f,n_expert]`; HF stores them as N separate 2D tensors
   `model.layers.N.mlp.experts.{e}.{gate,up,down}_proj.weight`. `ggml_to_hf` returns `None` for the
   `*_exps` names on purpose (`hf_mapping.rs:53-56`); `HostExps::load` is GGUF-only (`model.rs:256`,
   takes `g: &GgufFile`, not a `TensorSource`).
2. **Hybrid SSM name map**: `ggml_to_hf` returns `None` for `attn_qkv`, `attn_gate`, `ssm_*`
   (`hf_mapping.rs:57`); `HybridModel::load` and its helpers are GGUF-only (`hybrid.rs:11-16,142-148`).

---

## 0. Reality reset (what is actually on disk — the prompt's "no checkpoint to test" worry, resolved)

The prompt is honest-cautious about test data. The disk inventory changes the GATE plan materially:

| Claim under test | Disk reality | Source |
|---|---|---|
| "no safetensors MoE checkpoint to validate against" (`hf_mapping.rs:65`) | **FALSE for the generic MoE gather**: `OLMoE-1B-7B-0924` is a real on-disk safetensors MoE — 64 experts, 8/tok, 16 layers, experts as separate 2D `mlp.experts.{e}.{gate,up,down}_proj.weight` | `models--allenai--OLMoE-1B-7B-0924/.../model.safetensors.index.json` (3219 tensors), `config.json` (`model_type=olmoe`, `num_experts=64`, `num_experts_per_tok=8`) |
| "the target is Qwen3.6-35B-A3B MoE" | **TRUE but GGUF-only on disk** (`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, `unsloth--Qwen3.6-35B-A3B-MTP-GGUF`, `Ornith-1.0-35B-GGUF`). No safetensors shards for the 35B exist here. | `ls ~/ai-ml/hf-models/qwen36-35b-moe/` |
| hybrid SSM has no safetensors test model (`hf_mapping.rs:10-11,57`) | **FALSE**: `qwen35-9b-hf` (4 shards, 775 tensors) and `qwen35-4b-hf` (2 shards) are on disk, with `linear_attn.in_proj_qkv/_z/_a/_b`, `A_log`, `conv1d`, `norm`, `out_proj` names. Plus a **GGUF twin** for argmax-match: `qwen35-9b-judge-f16.gguf` / `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`. | `qwen35-9b-hf/model.safetensors.index.json`, `qwen35-9b-judge-f16.gguf` |

**Consequences that shape the plan:**

- The **MoE-gather GATE uses OLMoE**, not the 35B. OLMoE is the *cleanest possible* MoE gather test:
  no shared expert, no qwen3.5 norm+1, no SSM, no MTP, dense attention, `tie_word_embeddings=false`.
  It exercises exactly the new code (gather N 2D → 1 HostExps) and nothing else.
- OLMoE has **no GGUF twin on disk** (only safetensors), so its GATE is **finite-logit + self-consistency**
  (HF reference logits or top-1 stability), NOT argmax-vs-GGUF-twin. Be honest about this in the gate.
- The **SSM-name-map GATE uses qwen35-9b-hf vs its GGUF twin** for a true argmax match — the hybrid forward
  graph already exists (`HybridModel`), so once names resolve, the existing kernels run.
- **`olmoe` arch is not mapped**: `Arch::parse` / `from_hf_model_type` (`config.rs:18-41`) fall through to
  `Arch::Other("olmoe")`. `is_moe()`/`is_hybrid()` (`config.rs:43-49`) return false for it. This must be
  fixed or the MoE gather has no arch to dispatch on. `from_hf` keys MoE off `arch.is_moe() || num_experts.is_some()`
  (`config.rs:181`), so `num_experts=64` alone *does* populate `cfg.moe` — but the per-expert HF name layout
  differs by arch, so we still want the arch tagged. See §3.

---

## 1. The expert gather: N HF 2D tensors → 1 stacked HostExps (the core of part 1)

### 1.1 What the consumer requires (do not break the byte contract)

`HostExps` (`model.rs:242-250`) is consumed by the per-token H2D stager: `expert_bytes(e)` returns the
**contiguous** block `bytes[e*expert_stride .. (e+1)*expert_stride]` (`model.rs:299-302`), and the staged-expert
qmatvec kernels read `row_bytes` per output row. The invariant asserted at load is
`expert_stride == out_f * row_bytes` (`model.rs:280-281`). So the gathered buffer must be **C-contiguous with
the expert axis slowest** — expert 0's full `[in_f,out_f]` block, then expert 1's, etc. The 2D-per-expert HF
layout, reversed to ggml `ne=[in_f,out_f]` (`safetensors.rs:58-60`), is *already* `out_f` rows of `row_bytes`
each, so concatenating expert blocks in order **0..n_expert** yields exactly that layout — no transpose needed
for gate/up. For `down_exps` the HF tensor is `down_proj` `[out=in_f_model, in=ff]`; reversed ne is
`[ff, n_embd]` i.e. `in_f=ff(512), out_f=n_embd` — same rule, the per-expert block is already row-major in
the slow→fast order HostExps wants. **Verify this against the stride assertion at load (§5 step A) rather than
trusting the comment.**

### 1.2 dtype: HF is F16/BF16, HostExps wants packed quant — the honest fork

`HostExps` today only accepts the 8 GGUF quant types (`model.rs:261-271`, explicit `panic!` on `other`).
OLMoE/qwen ST shards are **BF16** (`torch_dtype=bfloat16`). Two viable paths, pick **A** for the gate:

- **Path A (gate, simplest, correct): F16/BF16 HostExps variant.** Add a non-quant storage mode to `HostExps`
  so each gathered expert stays BF16 (or is dequantized to f32) host-resident, and teach the staged-expert
  qmatvec to take a BF16/f32 weight path (the Stage-A f32 path the dp4a fast-path falls back to already
  exists per `lib.rs` MoE dispatch). This is **load-time only** — no on-the-fly quantization, no accuracy
  question. VRAM is irrelevant (experts are host-resident by design, EDGE-1, `model.rs:186-188`). This is the
  minimal change that makes the gather *correct and testable*.
- **Path B (later, for the 35B target): quantize-on-load to Q8_0/Q6_K.** Matches the GGUF twin's packing,
  reuses the existing dp4a staged-expert kernels unchanged, smaller host footprint. But it introduces a
  quantization step (round-trip error vs the GGUF twin's own quantizer) and is only needed once a safetensors
  35B MoE is actually on disk. **Defer.** Note it explicitly in the gate as "Path B unverified — no ST 35B MoE present."

Decision: **ship Path A**, gate on it, leave a one-paragraph Path-B stub. The whole point of this capability
is "load a MoE safetensors model and produce correct logits," not "match the GGUF quantizer bit-for-bit."

### 1.3 The gather function (new code, ggml-name-driven)

Add to `source.rs` a method on `SafetensorsSource` that builds one expert tensor by gathering, OR — cleaner —
add a `HostExps::load_from_source` that takes `&dyn TensorSource` + the ggml `*_exps` name + `n_expert` +
`expert_used` from `cfg.moe`. It:

1. Strips the ggml name to recover `il` and which proj (`ffn_gate_exps` → "gate", `ffn_up_exps` → "up",
   `ffn_down_exps` → "down").
2. For `e in 0..n_expert`: builds the HF name. **`hf_expert_name(il, e, proj)` already exists**
   (`hf_mapping.rs:67-75`) and emits `model.layers.{il}.mlp.experts.{e}.{gate,up,down}_proj.weight` — exactly
   OLMoE's layout (verified against the index). Call `SafetensorsSource::raw_hf(hf_name)` (`source.rs:83-86`)
   to get `TensorView { bytes, ggml_type, ne=[in_f,out_f] }`.
3. **Shape check per expert**: assert `ne.len()==2` and that `in_f,out_f` match expert 0 (catches a layer/arch
   mismatch early). Accumulate `out_f`, derive `row_bytes` from expert 0, and append the bytes
   (Path A: BF16 verbatim, or dequant→f32 then re-emit as f32 bytes).
4. After the loop: set `n_expert`, `expert_stride = total_len/n_expert`, `row_bytes = total_len/(out_f*n_expert)`,
   and **run the same `expert_stride == out_f*row_bytes` assertion** (`model.rs:280-281`) so the gathered buffer
   is held to the identical invariant as the GGUF path.

Router + shared expert are **already mapped** in `ggml_to_hf`: `ffn_gate_inp.weight → mlp.gate.weight`,
`ffn_*_shexp → mlp.shared_expert.*` (`hf_mapping.rs:48-52`). OLMoE has **no** shared expert
(`shared_expert_intermediate_size` absent) so `load_ffn` must `load_opt` the shexp tensors (it currently
`load_t`s them unconditionally — `hybrid.rs:58-60`). That is a real bug for OLMoE; make shexp optional.

### 1.4 Per-expert HF naming differs by arch — handle it, don't hardcode

`hf_expert_name` hardcodes `mlp.experts.{e}.{p}_proj.weight`. OLMoE matches. Qwen3-MoE matches. But the
function takes no arch, so if a future arch uses `block_sparse_moe.experts.*` (Mixtral) or `feed_forward.experts.*`
it breaks silently. For the gate (OLMoE + qwen) the current literal is correct — **do not over-engineer**; add a
one-line comment that the literal is the qwen/olmoe layout and a future arch needs a branch. (Matches the house
rule: realistic scope, not a speculative multi-arch table.)

---

## 2. The hybrid SSM name map (the core of part 2) — table-driven, cited against llama.cpp

`HybridModel`/`load_mixer_kind` ask for these ggml names (`hybrid.rs:33-43`); each needs an HF target plus, for
three of them, a value transform. The HF prefix is `model.language_model.layers.{il}.linear_attn.` for
qwen35-9b-hf (multimodal wrapper — verified: `model.language_model.layers.14.linear_attn.in_proj_qkv.weight`).
**Prefix normalization must run first**: strip `model.language_model.` → `model.` so the existing dense map and
the new SSM map share one namespace (SAFETENSORS-DECISION.md §2.0). Add this strip inside `SafetensorsSource::find`
(or `raw_hf`) before lookup, plus a fallback that tries both prefixed and stripped names.

| ggml name (asked by `hybrid.rs`) | HF name (`linear_attn.*`) | transform | llama.cpp ref |
|---|---|---|---|
| `blk.N.attn_qkv.weight` | `in_proj_qkv.weight` | dim-reverse; V-reorder rows if `num_v_heads≠num_k_heads` (9b: 32≠16) | tensor_mapping.py:251 |
| `blk.N.attn_gate.weight` | `in_proj_z.weight` | dim-reverse (+V-reorder rows) | tensor_mapping.py:386 |
| `blk.N.ssm_beta.weight` | `in_proj_b.weight` | dim-reverse | tensor_mapping.py:909 |
| `blk.N.ssm_alpha.weight` | `in_proj_a.weight` | dim-reverse | tensor_mapping.py:885 |
| `blk.N.ssm_a` (bare, no `.weight`) | `A_log` | **`-exp(A_log)` elementwise, F32** | tensor_mapping.py:846 |
| `blk.N.ssm_dt.bias` | `dt_bias` | rename only | tensor_mapping.py:831 |
| `blk.N.ssm_conv1d.weight` | `conv1d.weight` | **squeeze singleton dim** `[C,1,K]→[C,K]` | tensor_mapping.py:816 |
| `blk.N.ssm_norm.weight` | `norm.weight` | dim-reverse, **NO +1** (the one norm excluded) | tensor_mapping.py:871 |
| `blk.N.ssm_out.weight` | `out_proj.weight` | dim-reverse | tensor_mapping.py:880 |

Plus the full-attn layers (every 4th, `full_attention_interval=4`) reuse the **already-mapped** dense names
(`attn_q/k/v/output`, `attn_q_norm/k_norm`) via `ggml_to_hf` (`hf_mapping.rs:32-37`) — **but qwen3.5 norms get
`+1`** added to every `*norm.weight` EXCEPT `linear_attn.norm` (SAFETENSORS-DECISION.md §3.2). The dense map
returns names only; the `+1` and `-exp` and squeeze are **value transforms** that the current `find` cannot
express (it returns a zero-copy `&[u8]` view). See §2.1.

### 2.1 Transforms break zero-copy — the seam consequence (be honest)

`TensorView` borrows mmap bytes (`source.rs:14-18`); `find` returns `Option<TensorView<'_>>`. Three SSM tensors
(`ssm_a = -exp`, every `+1` norm, `conv1d` squeeze) and the V-reorder need an **owned** buffer. Options:

- **Minimal:** add an `owned: Option<Vec<u8>>` arm to a `TensorView`-like return (or a parallel `find_owned`
  that returns `Cow<[u8]>`). Pass-through tensors stay zero-copy; transformed ones materialize. This is the
  smallest seam change and keeps the dense fast path untouched.
- Compute transforms in `ggml_to_hf`'s caller: have `SafetensorsSource::find` detect the transform-requiring
  ggml names (`ssm_a`, `*norm.weight` under qwen35, `ssm_conv1d.weight`) and build the owned bytes there.

`GpuTensor::load_from_source` already dequants F32/F16/BF16 → f32 for the `Float` arm (`model.rs:64-69`), so a
materialized f32 `ssm_a` flows straight through. The norm `+1` must happen **on the dequantized f32** (add 1.0
to each element) before upload — fold it into the owned-buffer producer.

**V-reorder** (group→tile reorder of V heads when `linear_num_value_heads(32) ≠ linear_num_key_heads(16)`,
SAFETENSORS-DECISION.md §3.6, qwen.py:296-303) is the single subtlest transform: it touches `in_proj_qkv` rows,
`in_proj_z`, `in_proj_a/b` rows, the conv1d V channels, and `out_proj` cols. Get it right against the GGUF twin
(the twin already has the reorder baked in by llama's converter), which is exactly what the argmax gate checks.

---

## 3. Config + arch wiring (small, but blocks everything)

1. **Map `olmoe`** in `Arch` (`config.rs:18-41`): add `"olmoe" => Arch::Olmoe` (new variant) or, minimal,
   keep `Arch::Other("olmoe")` but make `is_moe()` also return true when `cfg.moe.is_some()`. The cleaner fix is
   a dedicated `Arch::Olmoe` so the dense-attention + MoE-FFN + no-shexp shape is explicit. `from_hf` already
   populates `cfg.moe` from `num_experts` regardless of arch (`config.rs:181-189`), so MoE metadata is fine; the
   gather dispatch just needs `cfg.moe` to be `Some` (it is).
2. **`load_ffn` shexp must be optional** (`hybrid.rs:58-60`): wrap `gate_shexp/up_shexp/down_shexp` in
   `load_opt`. Models with `shared_expert_intermediate_size` absent (OLMoE) have no shexp tensors; the current
   `load_t` would panic with "missing tensor". Same for `ffn_gate_inp_shexp` (`hybrid.rs:54`).
3. **OLMoE is dense-attention MoE, not hybrid** → it should NOT go through `HybridModel` (which assumes
   `attn_qkv`/`ssm_*`). It needs a dense-attention + MoE-FFN path. Either extend `Model` (dense) with an MoE FFN
   arm, or relax `HybridModel` to handle `full_attention_interval==0` (all-full-attn) + MoE. The dense `Model`
   already loops in ggml names and asserts `cfg.moe.is_none()` (`model.rs:143`) — lift that assert and give the
   dense FFN an MoE branch mirroring `load_ffn`. **This is the larger of the wiring tasks; scope it honestly as
   the MoE-forward-on-dense-attention plumbing, not a one-liner.**

---

## 4. Wire into the TensorSource seam (the deferred-part deliverable)

The seam is `TensorSource::find(ggml_name) -> Option<TensorView>` (`source.rs:21-30`). Today only
`GpuTensor::load_from_source` (`model.rs:33`) consumes it; `HostExps::load` and `HybridModel::load` are GGUF-only.
Concrete edits:

1. **`HostExps::load_from_source(e, src: &dyn TensorSource, ggml_exps_name, cfg)`** — the gather (§1.3). Keep the
   existing `HostExps::load(e, g, name)` as a thin wrapper over `GgufSource` (mirrors `GpuTensor::load` →
   `load_from_source`, `model.rs:27-29`), so GGUF behavior is byte-identical and only the ST path is new.
2. **`HybridModel::load_from_source(e, src: &dyn TensorSource)`** — replace the `g: &GgufFile` plumbing in
   `load`/`load_mixer_kind`/`load_ffn`/`load_t`/`load_opt` (`hybrid.rs:11-69,142-148`) with `&dyn TensorSource`.
   Keep `HybridModel::load(e, g)` as the GGUF wrapper. The forward graph is untouched (it already runs on the
   loaded `GpuTensor`/`HostExps`).
3. **`SafetensorsSource::find`** — extend `ggml_to_hf` (or branch in `find`) to (a) resolve the SSM names
   (§2 table), (b) apply prefix-strip, (c) emit owned buffers for the `-exp`/`+1`/squeeze/V-reorder transforms
   (§2.1). MoE `*_exps` names stay `None` from `find` (they are gathered out-of-band via §4.1, not a single lookup).
4. **A `run_safetensors` bin** (clone `run_dense.rs`, swap `GgufFile::open` → `SafetensorsSource::open(dir)`,
   dispatch `Model` vs `HybridModel` on `cfg.arch`) so the gate is runnable: `cargo run --bin run-safetensors <hf_dir>`.

---

## 5. GATE (concrete, honest about the no-twin case)

**Gate MoE (OLMoE, no GGUF twin → finite-logit + self-consistency):**
- **A. Gather invariant:** load OLMoE via `HostExps::load_from_source` for `blk.0.ffn_{gate,up,down}_exps`;
  assert `n_expert==64`, `expert_stride==out_f*row_bytes`, and that `expert_bytes(63)` is in-bounds and distinct
  from `expert_bytes(0)`. This proves the gather/stack is byte-correct independent of the forward.
- **B. Finite logits:** `run-safetensors <olmoe_dir> <toks>` → all logits finite (mirrors `run_dense.rs:38-40`,
  the existing finite-check), argmax is a real token id, top-5 are plausible (not uniform/degenerate).
- **C. Self-consistency (the honest substitute for an argmax-twin):** same prompt twice → identical argmax
  (determinism), and a 4-token continuation that is locally coherent. If a tiny OLMoE GGUF twin can be obtained
  later, upgrade C to a true argmax match. **State plainly in the gate output: "no GGUF twin on disk; MoE gate is
  finite-logit + self-consistency, not argmax-equality."**

**Gate Hybrid SSM (qwen35-9b-hf, GGUF twin present → true argmax match):**
- **D. Name resolution:** every ggml name `HybridModel::load` asks for resolves through `find` (no
  "missing tensor" panic) for all 32 layers — proves the SSM map + prefix-strip + transforms are complete.
- **E. Argmax match vs twin:** `run-safetensors qwen35-9b-hf <toks>` argmax == `run-hybrid qwen35-9b-judge-f16.gguf <toks>`
  argmax on a fixed ≥8-token prompt. This is the strong gate: it catches a wrong V-reorder, a missing `+1`, a
  bad `-exp(A_log)`, or a transposed projection, because any of those diverge the logits within one layer.
- **F. Transform spot-checks:** assert `ssm_a` (from `-exp(A_log)`) is all-negative and finite; assert a qwen35
  `attn_norm.weight` ST value == GGUF-twin value (proves the `+1` is applied exactly once, on the right tensors).

**Honesty clauses to include verbatim in the gate report:**
- Path B (quantize-on-load Q8_0/Q6_K experts) is **unverified — no safetensors 35B-MoE checkpoint on disk**;
  only Path A (BF16/f32 host-resident experts) is gated.
- The 35B Qwen3.6-A3B target is **GGUF-only on disk**; the ST MoE path is validated on OLMoE (different arch,
  same gather mechanism) + the hybrid path on qwen35-9b-hf. The two halves are gated on different models because
  no single on-disk ST checkpoint is both MoE and hybrid.

---

## 6. Order of work (dependency-sorted, smallest correct slice first)

1. `Arch::Olmoe` + `is_moe` fix + `load_ffn`/`gate_inp_shexp` optional-shexp (`config.rs:18-41`, `hybrid.rs:54-60`). [small]
2. `HostExps::load_from_source` gather, Path A BF16/f32 + stride assertion (§1.3). [core MoE]
3. Dense-attention + MoE-FFN forward path for OLMoE (§3.3 — the real plumbing, not a one-liner). [medium]
4. `run-safetensors` bin + Gate A/B/C on OLMoE. [gate-1]
5. SSM name map table + prefix-strip in `find` (§2). [core hybrid, names only]
6. Owned-buffer transforms: `-exp(A_log)`, norm `+1`, conv1d squeeze, V-reorder (§2.1). [core hybrid, values]
7. `HybridModel::load_from_source` plumbing (§4.2). [wiring]
8. Gate D/E/F on qwen35-9b-hf vs GGUF twin. [gate-2]

Steps 1-4 (MoE) and 5-8 (hybrid) are independent and can land in either order; each has its own runnable gate.

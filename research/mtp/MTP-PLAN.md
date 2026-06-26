# Qwen3.5 MTP Greedy Speculative Decode — Build Plan (researched, reviewed)

The naming is settled: `blk.{N}.nextn.*` where N = the MTP block index = `n_trunk` (the block dropped by the load loop). The standard attention/FFN tensors of that block use the plain `blk.{N}.attn_q.weight` etc. naming (same as trunk layers). All facts confirmed. Producing the plan.

---

# Build Plan: Qwen3.5 MTP Greedy Speculative Decode for bw24

**Scope reality check:** This is a focused engine feature, not a multi-week effort. It is ~6 new functions and 2 new structs against an already-validated decode path. The hard parts are (1) the T=K batched verify launcher and (2) the hybrid-cache rollback. Everything else is mechanical.

## 0. Resolved contradictions (settle these before writing code)

**0.1 — eh_proj concat order: NO contradiction. Order is `[norm(e); norm(h)]` (embedding first, hidden second), concat along the feature dim.**
- llama.cpp `step35.cpp:414-420`: `concat = ggml_concat(e_norm, h_norm, dim=0)` → eh_proj `[2*n_embd, n_embd]`.
- SGLang/vLLM `qwen3_5_mtp.py:171-173`: `torch.cat([inputs_embeds, hidden_states], dim=-1)` after `pre_fc_norm_embedding(e)` and `pre_fc_norm_hidden(h)`.
- All 5 sources (Reader-1, Reader-2 across 4 impls) agree. The earlier "discrepancy" was a phantom — Reader-2 explicitly states "Zero variation in order." **Use `[e_norm, h_norm]`.** In bw24 memory layout, `e_norm` occupies elements `[0, n_embd)` and `h_norm` occupies `[n_embd, 2*n_embd)` of the concat buffer.

**0.2 — Rollback approach: the readers describe two *different engines*, not a contradiction. vLLM/SGLang use block-table indexing (Reader-4 strategy A) because they pre-allocate physical KV/state blocks; bw24 has NO block table — its cache is a single resident growing buffer (`cache.rs:12-32`). Therefore bw24 must use the OTHER strategy: snapshot + replay (strategy B). This is forced by the architecture, not a preference.** See §C.

**0.3 — GGUF tensor naming: settled.** MTP tensors are `blk.{N}.nextn.{eh_proj,enorm,hnorm,shared_head_head,shared_head_norm}` and the MTP block's transformer tensors are plain `blk.{N}.{attn_q,attn_k,...,ffn_*}.weight`, where **N = n_trunk** (`= cfg.n_layer - cfg.nextn_predict_layers`, i.e. block 40 for the 35B-MoE; block 32 for the 9B if it had one). Confirmed `llama-arch.cpp:465-470` + `:783` ("NextN/MTP tensors stored per-block blk.%d.nextn.*"). The earlier "nextn.* vs blk.32.*" open question is answered: it is BOTH — `blk.{N}.nextn.eh_proj`.

**0.4 — T>1 kernel support: a T>1 path ALREADY EXISTS.** `fa_prefill` (`lib.rs:493-513`) is the FA-2 prefill kernel for T>1, GQA, causal, head_dim 256. `fa_decode` (`lib.rs:516-538`) is T=1-only (split-K over t_kv). The verify pass needs causal attention of K query rows over `[0..pos+K)` KV — exactly what `fa_prefill` does. **No new attention kernel is needed; reuse `fa_prefill`.** RoPE, rms_norm, q_gate_split, silu_mul all already take a `T` argument (e.g. `rope_neox(..., n_head, T, ...)` at `decode.rs:119`). The genuinely-new work is plumbing T=K through `decode_step` (currently hardcoded T=1) and a multi-token KV append.

---

## A. NextN head forward — exact numbered ops with shapes

Notation: `D = n_embd` (2048 for 9B-class; check cfg), `T` = number of query tokens in this forward (T=1 when drafting one token; the MTP head emits exactly 1 draft token per call — Reader-2: `num_mtp_layers=1`, `forward` called once per draft token). Let `H_head` = `head_dim_k` (256), `n_head`, `n_head_kv`, `n_ff`. The MTP block is loaded from `blk.{N}.*` with `N = n_trunk`.

Inputs to the MTP head:
- `h_seed`: the **pre-final-norm** hidden state `[D, T]` of the *trunk's last layer* at the relevant position(s). Capture this from `decode_step` as `x` immediately **before** `output_norm` is applied (`decode.rs:58`, the value of `x` after the layer loop, BEFORE line 59). Reader-1 `step35.cpp:539`.
- `e`: the token embedding `[D, T]` of the **token being predicted-from** (for greedy spec: the last committed token), via `embd.gather` (`decode.rs:20`).

Ops:

1. `e_norm = RMSNorm(e, blk.N.nextn.enorm, eps)` → `[D, T]`. (`rms_norm(&e, enorm, ..., D, T, eps)`.) Reader-1 `step35.cpp:415`.
2. `h_norm = RMSNorm(h_seed, blk.N.nextn.hnorm, eps)` → `[D, T]`. Reader-1 `step35.cpp:414`.
3. `concat = [e_norm ; h_norm]` → `[2*D, T]`. Element layout per column: `[0,D)=e_norm`, `[D,2D)=h_norm`. (See §0.1.) Reader-1 `step35.cpp:420`.
4. `inpSA = eh_proj @ concat`, eh_proj is `[2*D, D]` → output `[D, T]`. This is a plain matmul on the concat buffer. Reader-1 `step35.cpp:423`, Reader-2 vLLM `qwen3_5_mtp.py:89-97`.
5. `a_norm = RMSNorm(inpSA, blk.N.attn_norm, eps)` → `[D, T]`. Reader-1 `step35.cpp:432`.
6. Attention (identical to `full_attn_decode` body, `decode.rs:88-147`, but T=K-aware):
   - 6a. `qf = wq@a_norm` `[2*H_head*n_head, T]` (fused q|gate), `k = wk@a_norm` `[H_head*n_head_kv, T]`, `v = wv@a_norm` `[H_head*n_head_kv, T]`. Reader-1 `qwen35.cpp:559`.
   - 6b. `q_gate_split(qf -> q[H_head*n_head,T], gate[H_head*n_head,T])` (`decode.rs:109`).
   - 6c. `q = RMSNorm(q, q_norm, eps)` per-head over `H_head`; `k = RMSNorm(k, k_norm, eps)` (`decode.rs:113-117`).
   - 6d. `rope_neox(q, pos_d, H_head, rope_dim_count, n_head, T, freq_base)`; same for `k` (`decode.rs:119-120`). For T>1, `pos_d` is the **vector** `[pos, pos+1, …, pos+T-1]` (rope_neox already takes a `T` arg and a pos buffer).
   - 6e. Append `k,v` (T columns) into the MTP block's own KV cache slot, then attend: `attn = softmax(q@kᵀ / sqrt(H_head)) @ v` over `[0..pos+T)`, **causal**. T=1 → `fa_decode`; T>1 → `fa_prefill` (causal=true). Reader-1 `qwen35.cpp` SDPA.
   - 6f. `attn_g = attn * sigmoid(gate)` (`decode.rs:142-145`). Reader-1 `qwen35.cpp:600` head-wise sigmoid gate.
   - 6g. `attn_out = wo @ attn_g` → `[D, T]`.
7. `x1 = inpSA + attn_out` (residual) → `[D, T]`. Reader-1 `step35.cpp:496`.
8. `z = RMSNorm(x1, blk.N.post_attention_norm, eps)` → `[D, T]` (the pre-FFN norm). Reader-1 `qwen35.cpp:611` `attn_post_norm`.
9. FFN: Dense SwiGLU or MoE (whichever the block carries). Mechanically identical to `decode.rs:36-52` / `moe_ffn`. → `ffn_out [D, T]`. Reader-1 `step35.cpp:503-535`.
10. `h_nextn = x1 + ffn_out` (residual) → `[D, T]`. Reader-1 `step35.cpp:537`.
11. `final = RMSNorm(h_nextn, blk.N.nextn.shared_head_norm OR output_norm, eps)` → `[D, T]`. Reader-1 `step35.cpp:543`. shared_head_norm is optional; fall back to `model.output_norm`.
12. `draft_logits = (blk.N.nextn.shared_head_head OR model.output) @ final` → `[n_vocab, T]`. Reader-1 `step35.cpp:551`.
13. Greedy draft token per column: `draft[t] = argmax(draft_logits[:, t])`.

For greedy multi-draft (K>1): the MTP head emits 1 token per call (Reader-2 `qwen3_5_mtp.py:73,146`). To draft K tokens, call the head K times, each time feeding `e = embed(prev_draft_token)` and `h_seed =` the MTP block's own hidden output from the prior draft step. Each draft step advances the MTP block's KV cache by 1 and its linear-attn state (if any) by 1.

---

## B. Greedy verify loop algorithm

This is a faithful port of `speculative-simple.cpp:217-289` + `sampling.cpp:624-651`, specialized to greedy (temperature 0), where spec decode is provably exact.

```
state: cache (target, dual: KV + recur), pos = #committed tokens, last_token
loop until EOS or max_new:
  # --- 0. snapshot (see §C) ---
  snap = cache.snapshot()                      # KV lens + conv_state + ssm_state per layer

  # --- 1. DRAFT K tokens with the MTP head (autoregressive, T=1 each) ---
  draft = []
  e_tok = last_token
  h_seed = last trunk hidden (saved from the step that produced last_token)
  for j in 0..K:
      logits_j = mtp_head_forward(e_tok, h_seed)   # §A, T=1, advances MTP-block cache+state
      d = argmax(logits_j)
      draft.push(d)
      e_tok = d
      h_seed = mtp_block_hidden_j                  # h_nextn from step j (§A op 10)
  # NOTE the MTP-block cache/state is scratch; only the TARGET cache must be exact.

  # --- 2. VERIFY: one batched target forward over [last_token, draft[0..K-1]] ---
  verify_tokens = [last_token, draft[0], ..., draft[K-1]]   # T = K+1
  verify_pos    = [pos,        pos+1,    ..., pos+K]
  tlogits = target_forward_T(verify_tokens, verify_pos, cache)  # [n_vocab, K+1]
  #  ^ runs the FULL trunk (not the MTP head) with T=K+1; appends K+1 cols to every KV
  #    layer and advances every recur layer by K+1 steps (in scratch — see §C).

  # --- 3. GREEDY ACCEPT (walk prefix, stop at first mismatch) ---
  tgt_argmax[t] = argmax(tlogits[:, t])  for t in 0..K        # column t predicts token at pos+t+1
  n_acc = 0
  for j in 0..K:
      if tgt_argmax[j] == draft[j]: n_acc += 1
      else: break
  # bonus token: the target's own prediction at the first non-accepted slot
  bonus = tgt_argmax[n_acc]            # always valid: index n_acc in [0..K]

  # --- 4. COMMIT + advance ---
  committed = draft[0..n_acc] ++ [bonus]          # n_acc + 1 tokens
  emit(committed)
  pos += n_acc + 1
  last_token = bonus
  h_seed = trunk_hidden_at_column(n_acc)          # trunk hidden of the accepted/bonus position

  # --- 5. ROLLBACK cache to exactly `pos` (§C) ---
  if n_acc < K:                                   # we over-ran by (K - n_acc) tokens
      cache.rollback(snap, accept_len = n_acc + 1)
  # if n_acc == K (full accept), the K+1 verify columns are exactly the committed prefix;
  # NO rollback needed (commit count == verify count). Reader-3 full-accept = K+1.
```

Commit counts (Reader-3 `speculative-simple.cpp:286-289`, `sampling.cpp:643-648`):
- **Full accept** (n_acc=K): commit **K+1** tokens. No rollback (verify already advanced cache by K+1, all correct).
- **Partial accept** (n_acc=j<K): commit **j+1** tokens (j drafts + 1 bonus). Roll cache back from `pos+K+1` to `pos+j+1`.

Greedy correctness invariant: because both draft-accept and bonus use **argmax**, every committed token equals what plain greedy `decode_step` would have produced. This is the validation hook (§E).

---

## C. THE ROLLBACK RECIPE for the hybrid dual cache

bw24's cache (`cache.rs:12-32`) is a single resident growing buffer per layer — there is no block table — so vLLM's index-into-precomputed-block strategy (Reader-4 strategy A) is **not applicable**. bw24 must use **snapshot + truncate/restore (strategy B)**, which Reader-4 also identifies as correct and cheap (~100KB/layer state). The two cache families need different handling:

### C.1 Full-attention KV layers → truncation (cheap, exact)
The verify pass appended K+1 columns at offset `len*kv_dim` (`decode.rs:123-127`). KV cache is **append-only and position-addressed** — column t is the K/V for absolute position `pos+t`, never overwritten. So rollback = **decrement `len`**:
```
kvl.len = snap.kv_len[il] + accept_len      # = pos_before + n_acc + 1
```
No data copy. The stale columns beyond `len` are simply never read (attention views `0..len*kv_dim`, `decode.rs:131`). **Why correct:** accepted positions `0..pos+n_acc` hold exactly the K/V the non-spec path would have written, because they were computed from the same (accepted) tokens at the same positions with the same RoPE. Reader-5 `cache.rs:10-68`; Reader-3 `speculative-simple.cpp:266` `llama_memory_seq_rm`.

### C.2 Linear-attention conv_state / ssm_state layers → snapshot + replay
This is the crux. GDN/conv state is **recurrent**: the verify pass mutated `conv_state` and `ssm_state` in place across all K+1 steps (`decode.rs:185` conv_assemble_and_roll, `decode.rs:212` `rl.ssm_state = state_scratch`). On partial accept, those buffers reflect K+1 absorbed tokens but only `n_acc+1` are committed — the state is **corrupt** for `n_acc < K`. Truncation cannot fix recurrent state (it has no position index to truncate). The fix:

**Snapshot before draft+verify; on partial accept, restore the snapshot and replay exactly `accept_len` tokens.**

```
snapshot:  for each linear-attn layer il:
    snap.conv[il] = device-copy of rl.conv_state   # [conv_dim*(d_conv-1)] ≈ 96KB
    snap.ssm[il]  = device-copy of rl.ssm_state     # [d_state*d_state*num_v] ≈ 2MB
  (CudaSlice clone = Arc refcount, NOT a copy — must alloc fresh + memcpy_dtod; Reader-5 decode.rs:210-212, lib.rs:66-71)

rollback(snap, accept_len):
    truncate every KV layer (C.1)
    for each linear-attn layer: rl.conv_state = snap.conv[il]; rl.ssm_state = snap.ssm[il]
    cache.pos = pos_before
    # replay the committed tokens through the FULL trunk to rebuild recurrent state:
    for t in committed[0..accept_len-1 except the bonus already implied]:
        decode_step(committed[t])   # T=1, advances KV + recur exactly as non-spec greedy
```

**Why this is correct and why it's the right strategy for bw24:**
- Replaying `accept_len` tokens through the real `decode_step` (T=1) reconstructs **bit-identical** recurrent state to the non-spec greedy path — by construction, since it *is* the non-spec path. Reader-4: "re-computing (replay)… in neither case does the final state get corrupted."
- Cost is bounded: `accept_len ≤ K+1` (K≈4-5 typical), and replay only runs on **partial accept**. On full accept (the common, high-throughput case) there is **zero** replay and zero state copy needed beyond freeing the snapshot.
- Snapshot size is tiny (Reader-4: ~100KB conv + ~2MB ssm per linear-attn layer; for ~36 linear layers that's ~75MB, trivially affordable on the 24GB target — and only one snapshot is live at a time).
- The alternative (vLLM block-indexing) would require rearchitecting the resident cache into block tables — large, unjustified work for single-sequence edge inference (Reader-4 strategy A is "more aligned with vllm infrastructure" but bw24 has none of it).

**Optimization (optional, defer):** instead of replaying through the *full* trunk, replay only the **linear-attn layers** for `accept_len` tokens (conv + GDN are the only stateful ops; KV is already truncated correctly in C.1). This avoids recomputing attention/FFN. But the simple correct version replays full `decode_step` — implement that first, validate (§E), then optimize.

---

## D. Precise bw24 code changes

### D.1 New struct: `MtpHead` (in `hybrid.rs`, after `HybridLayer`)
```rust
pub struct MtpHead {
    pub enorm: GpuTensor,            // blk.N.nextn.enorm
    pub hnorm: GpuTensor,            // blk.N.nextn.hnorm
    pub eh_proj: GpuTensor,          // blk.N.nextn.eh_proj  [2*n_embd, n_embd]
    pub attn_norm: GpuTensor,        // blk.N.attn_norm
    pub post_attn_norm: GpuTensor,   // blk.N.post_attention_norm
    pub mixer: Mixer,                // Full(FullAttnLayer) — MTP block is full-attn in qwen35
    pub ffn: Ffn,                    // Dense or Moe (same loader as trunk)
    pub shared_head_norm: Option<GpuTensor>,  // blk.N.nextn.shared_head_norm (fallback output_norm)
    pub shared_head_head: Option<GpuTensor>,  // blk.N.nextn.shared_head_head (fallback output)
}
```
Add field to `HybridModel`: `pub mtp: Option<MtpHead>` (`hybrid.rs:63-69`).

### D.2 Stop dropping the MTP block in `load()` (`hybrid.rs:83-136`)
The trunk loop is correct as-is (`n_trunk` excludes the MTP block). **Add** after the loop, before `Ok(HybridModel{...})`:
```rust
let mtp = if cfg.nextn_predict_layers > 0 {
    let n = n_trunk as u32;                  // MTP block index = first dropped block (40 for 35B)
    let p = |s: &str| format!("blk.{n}.{s}");
    Some(MtpHead {
        enorm:  load_t(e, g, &p("nextn.enorm"))?,
        hnorm:  load_t(e, g, &p("nextn.hnorm"))?,
        eh_proj: load_t(e, g, &p("nextn.eh_proj"))?,
        attn_norm: load_t(e, g, &p("attn_norm.weight"))?,
        post_attn_norm: load_opt(e, g, &p("post_attention_norm.weight"))?
            .or(load_opt(e, g, &p("ffn_norm.weight"))?).expect("..."),
        mixer: /* Mixer::Full, same 6-tensor load as trunk full-attn block */,
        ffn:   /* same Dense/Moe branch as trunk */,
        shared_head_norm: load_opt(e, g, &p("nextn.shared_head_norm"))?,
        shared_head_head: load_opt(e, g, &p("nextn.shared_head_head"))?,
    })
} else { None };
```
Naming confirmed against `llama-arch.cpp:465-470` (`blk.%d.nextn.eh_proj` etc.) and `:783-791`. Guard on `nextn_predict_layers > 0` answers Reader-5's open question (the dense 9B with nextn=0 loads no MTP head).

### D.3 T=K batched verify path — reuse existing kernels, add a `T` parameter
**Do NOT write a new attention kernel** (resolves Reader-5 open question). `fa_prefill` (`lib.rs:493-513`) already does causal T>1 GQA attention; RoPE/rms_norm/q_gate_split/silu_mul already take `T`. Add:

1. `Engine::copy_into` already supports a contiguous multi-column append (`lib.rs:66-71`) — verify appends K+1 columns: `copy_into(&mut kvl.k, len*kv_dim, &k_TK, (K+1)*kv_dim)`. **The KV cache stores `[token, kv_dim]` row-major (`cache.rs:16`); the projection produces `[kv_dim, T]` channel-major** — so the multi-token append needs a transpose-on-write OR write per-column in a loop of K+1 `copy_into` calls. Simplest correct version: loop `copy_into` per column (K+1 tiny D2D copies, negligible). Flag: confirm the K/V projection column-stride matches the cache row-stride before the loop.

2. New `HybridModel::decode_step_t(&self, e, tokens: &[u32], pos0: usize, cache: &mut Cache) -> Result<Vec<f32>>` — a T=K generalization of `decode_step` (`decode.rs:12-64`):
   - embed K tokens → `[D, T]` (gather K rows).
   - `pos_d = htod_i32(&[pos0, pos0+1, …, pos0+T-1])`.
   - per layer: `rms_norm(..., D, T, ...)`, then `Mixer::Full` → new `full_attn_verify` using `fa_prefill` (causal); `Mixer::Linear` → **K sequential T=1 GDN/conv steps** (recurrent, cannot batch — Reader-5 confirms linear-attn is inherently sequential).
   - FFN with `T` (silu_mul/moe_ffn already take a count arg, `decode.rs:48,51`).
   - returns all T logit columns (caller needs every column for verify argmax).

3. **Linear-attn in verify is the K sequential T=1 path** (answers Reader-5: not batched). This is exactly why §C.2 replay works — the verify pass already advances recur state one token at a time; rollback just restores the pre-verify snapshot and replays the accepted prefix.

### D.4 Cache snapshot/restore (`cache.rs`)
Add to `Cache`:
```rust
pub struct CacheSnapshot {
    kv_len: Vec<usize>,                       // per full-attn layer
    conv: Vec<Option<CudaSlice<f32>>>,        // per linear-attn layer (D2D copy)
    ssm:  Vec<Option<CudaSlice<f32>>>,
    pos: usize,
}
impl Cache {
    pub fn snapshot(&self, e:&Engine) -> Result<CacheSnapshot,_> { /* memcpy_dtod each recur buf; record each kvl.len */ }
    pub fn rollback(&mut self, e:&Engine, snap:&CacheSnapshot, accept_len:usize) { /* C.1 + C.2 */ }
}
```
**Critical (Reader-5 confirmed): `CudaSlice::clone()` is an `Arc` refcount, NOT a buffer copy.** Snapshot must `e.zeros(n)` then `memcpy_dtod` (same pattern as `copy_into`, `lib.rs:69`). Restore reassigns `rl.conv_state = snap.conv[il].clone()` (here the Arc-clone is *fine* because we want both snapshot and live to share until next snapshot — or alloc+copy to keep snapshot reusable across the K-loop; alloc+copy is safer).

### D.5 New orchestrator: `HybridModel::generate_spec(&self, e, prompt, max_new, k) -> Vec<u32>`
Implements §B. Reuses `decode_step` for prompt prime and for replay; calls `decode_step_t` for verify; calls a new `mtp_head_forward` (T=1) for drafting; snapshots/rolls via D.4. Must also save the trunk's pre-`output_norm` hidden (`h_seed`) — add an out-param or a variant of `decode_step` that returns `(logits, h_seed_device)` (capture `x` at `decode.rs:58` before line 59).

### D.6 MTP block needs its own cache slots
The MTP block is full-attn and has its own KV (and, if the arch places linear layers in it, its own recur state). Allocate **separate scratch KV/recur for the MTP head** (not part of the trunk `Cache.kv/recur`) since drafting K tokens advances MTP-block state that is discarded after verify. Simplest: give `MtpHead` forward its own small `Cache`-like scratch sized `K+context`, reset per verify round. The MTP head only attends to the *committed* context echoed through it during draft — confirm whether the MTP block shares the trunk KV or has independent KV (Reader-1 `step35.cpp` treats it as a full layer with its own attn; bw24 should give it independent scratch KV reset each draft round).

---

## E. Validation strategy

Greedy spec decode is **mathematically exact** — the accepted+bonus sequence MUST be token-for-token identical to plain greedy. This gives a hard, binary oracle.

1. **Self-consistency (primary gate):** run `generate(prompt, max_new)` (existing greedy, `decode.rs:68-84`) and `generate_spec(prompt, max_new, k)` on the SAME model + prompt. Assert the two token sequences are **identical** (`assert_eq!`). Any divergence = bug in draft logits, verify argmax, commit count, or (most likely) rollback. Test across K ∈ {1,2,4,8} and multiple prompts. K=1 isolates the verify/commit path; K>1 exercises rollback.

2. **Rollback isolation test:** force partial acceptance by feeding a draft model that deliberately mispredicts at known position j (or compare against a draft that's the target itself — should always full-accept). Verify recur state after rollback equals state from the non-spec path: after `generate_spec`, dump `ssm_state`/`conv_state` (dtoh) and compare to `generate`'s state at the same `pos`. Must match to f32 bit-equality (replay is the same code path).

3. **llama.cpp argmax cross-check (existing project oracle):** the repo already validates argmax against llama.cpp (commit `02af8fc`: "35B-A3B argmax=1178 == llama.cpp"). Run `generate_spec` and confirm first-token argmax and the full greedy continuation match llama.cpp's `speculative-simple` output for the same Qwen3.5 MTP GGUF. Build the reference with the existing `tools/` harness pattern (`tools/ggml_list_tensors` already present for tensor-name verification).

4. **Acceptance-rate sanity (perf, not correctness):** log `n_acc/K` per round. For a well-matched MTP head this should be high (>0.6); near-zero acceptance with still-correct output means the head is loaded/forwarded wrong (e.g. eh_proj concat order flipped — §0.1) even though the verify path masks it by always falling back to the bonus token. **This is the trap: a wrong MTP head still produces correct greedy output via the bonus token, but with 0 speedup.** So gate on BOTH exactness (test 1) AND acceptance rate > 0.

---

## Open items to confirm at implementation time (not blockers)
- **K/V column-stride vs cache row-stride** for the multi-token append (D.3 step 1) — verify the projection layout before choosing loop-copy vs transpose-write.
- **Does the MTP block share trunk KV or use independent KV?** (D.6) — Reader-1 implies independent (full layer with own attn); confirm from the GGUF/llama.cpp graph (`llama-graph.cpp:1085-1187`) which sequence positions the MTP attention sees.
- **Replay scope** (C.2 optimization) — ship full-`decode_step` replay first; narrow to linear-only after test 2 passes.

**Relevant files (absolute):**
`/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid.rs` (structs, load, D.1/D.2/D.6), `/home/avifenesh/projects/bw24/crates/bw24-engine/src/decode.rs` (decode_step, full_attn_decode, linear_attn_decode → D.3/D.5), `/home/avifenesh/projects/bw24/crates/bw24-engine/src/cache.rs` (snapshot/rollback → D.4), `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` (fa_prefill:493, fa_decode:516, copy_into:66, view:74 → D.3), `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/config.rs:84,127,152` (nextn_predict_layers, n_layer_total).

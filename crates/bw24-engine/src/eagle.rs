//! EAGLE3.1 greedy-chain speculative decode (research/basics/EAGLE-PLAN.md, N1-N7).
//!
//! Greedy spec decode is MATHEMATICALLY EXACT: the accepted+bonus token stream is token-for-token
//! identical to plain greedy `generate` (decode.rs). EAGLE differs from MTP (spec.rs) ONLY in the
//! DRAFT step: instead of the trunk-coupled NextN head, EAGLE drafts with a SEPARATE 1-layer model
//! (own vocab, own RoPE, untied lm_head) fed the trunk's hidden states from 3 aux layers [1,15,28]
//! fused through an encoder `fc`. The verify / accept-prefix / snapshot / rollback are REUSED
//! VERBATIM from spec.rs (decode_step_t, the greedy accept walk, cache.snapshot/rollback).
//!
//! On-disk draft (`eagle3-qwen35-9b/model.safetensors`, bf16, ground-truthed at impl time):
//!   fc.weight                            [4096, 12288]  (3*n_embd -> n_embd encoder)
//!   midlayer.input_layernorm.weight      [4096]         (RMSNorm of the prev-token EMBED)
//!   midlayer.hidden_norm.weight          [4096]         (RMSNorm of the recurrent hidden g)
//!   midlayer.self_attn.{q,k,v}_proj      q[4096,8192] k/v[1024,8192]  (in = 2*n_embd!)
//!   midlayer.self_attn.o_proj            [4096, 4096]
//!   midlayer.post_attention_layernorm    [4096]
//!   midlayer.mlp.{gate,up}_proj          [12288,4096]   down [4096,12288]
//!   norm.weight                          [4096]         (final RMSNorm before lm_head)
//!   lm_head.weight                       [32000, 4096]  (DRAFT vocab)
//!   d2t                                  [32000] i64    target_id = draft_id + d2t[draft_id]
//!   t2d                                  [248320] bool  (unused on the chain-greedy decode path)
//!
//! Op-sequence (authoritative: vLLM `llama_eagle3.py` LlamaDecoderLayer layer_idx==0, this ckpt's
//! flags norm_before_residual=false, norm_before_fc=false, fc_norm=false, norm_output=false):
//!   ENCODE (once/round): g = fc @ concat(aux[1], aux[15], aux[28])                 -> [n_embd]
//!   DRAFT step (T=1):
//!     e   = embed(prev_tok)                          (TARGET embedding; EAGLE3 shares it)
//!     eN  = RMSNorm(e,  input_layernorm)
//!     res = g                                         (_norm_after_residual: residual is PRE-norm g)
//!     gN  = RMSNorm(g,  hidden_norm)
//!     cat = [eN ; gN]                                 -> [2*n_embd]
//!     attn= o_proj @ SDPA( q,k,v = {q,k,v}_proj @ cat ; partial RoPE 64/256 @ theta 1e7 ; GQA16:4 )
//!     x1  = attn + res
//!     z   = RMSNorm(x1, post_attention_layernorm)
//!     mlp = down @ silu(gate @ z) * (up @ z)
//!     gsum= mlp + x1                                  (the model's final fused-add residual)
//!     dl  = lm_head @ RMSNorm(gsum, norm)             -> draft_logits[32000]
//!     g_next = gsum                                   (EAGLE recurrence: pre-norm residual)

use cudarc::driver::CudaSlice;
use std::path::Path;
use bw24_gguf::dequant;
use bw24_gguf::safetensors::StModel;
use crate::Engine;
use crate::model::GpuTensor;
use crate::hybrid::HybridModel;
use crate::cache::{Cache, KvLayer};
use crate::forward::argmax;

/// The EAGLE3 draft model: encoder `fc` + ONE Llama-style decoder layer + untied lm_head + d2t.
/// All weights are bf16 -> dequant to f32 GpuTensor::Float (the draft is ~0.8 GB; the matmuls go
/// through cuBLASLt `linear`). The draft attention is PLAIN Llama (no QK-norm, no output gate),
/// distinct from the trunk's gated/QK-normed full-attn.
pub struct Eagle3Draft {
    pub fc: GpuTensor,                 // [3*n_embd, n_embd]  encoder
    pub input_layernorm: GpuTensor,    // [n_embd]  norm of prev-token embedding
    pub hidden_norm: GpuTensor,        // [n_embd]  norm of recurrent g
    pub q_proj: GpuTensor,             // [2*n_embd, n_head*head_dim]
    pub k_proj: GpuTensor,             // [2*n_embd, n_head_kv*head_dim]
    pub v_proj: GpuTensor,             // [2*n_embd, n_head_kv*head_dim]
    pub o_proj: GpuTensor,             // [n_head*head_dim, n_embd]
    pub post_attention_layernorm: GpuTensor,
    pub gate_proj: GpuTensor,
    pub up_proj: GpuTensor,
    pub down_proj: GpuTensor,
    pub norm: GpuTensor,               // [n_embd]  final RMSNorm before lm_head
    pub lm_head: GpuTensor,            // [n_embd, draft_vocab]
    pub d2t: Vec<i64>,                 // [draft_vocab]  target_id = draft_id + d2t[draft_id]

    // shape / rope params (from the draft config.json, NOT the trunk cfg)
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub draft_vocab: usize,
    pub rope_dim_count: usize,         // partial_rotary_factor * head_dim  (0.25 * 256 = 64)
    pub rope_theta: f32,               // 1e7
    pub eps: f32,
    pub aux_layers: Vec<usize>,        // [1, 15, 28]
}

/// Load a single bf16 (or f32) tensor from the draft safetensors into a GpuTensor::Float.
/// `name` is the raw HF/EAGLE name in the file (e.g. "fc.weight", "midlayer.self_attn.q_proj.weight").
fn load_float(e: &Engine, m: &StModel, name: &str) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    let (info, bytes) = m.raw(name).ok_or_else(|| format!("EAGLE3 draft missing tensor {name}"))?;
    let ne = info.ne();                              // inner-fastest (ne[0]=in_features for a weight)
    let n: u64 = ne.iter().product();
    let f32v = dequant::dequantize(info.ggml_type(), bytes, n as usize);
    Ok(GpuTensor::Float { data: e.htod(&f32v)?, ne })
}

impl Eagle3Draft {
    /// Load the EAGLE3 draft from a checkpoint directory (config.json + model.safetensors) or a
    /// direct path to the .safetensors. Reads the geometry/rope params from the sibling config.json.
    /// `aux_layers` is the trunk layer-id list from `eagle_config.eagle_aux_hidden_state_layer_ids`.
    pub fn load(e: &Engine, path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = if path.is_file() { path.parent().unwrap_or(Path::new(".")) } else { path };
        let cfg = EagleConfig::from_json(&dir.join("config.json"))?;
        let m = StModel::open(path)?;

        let d2t = read_i64(&m, "d2t")?;
        assert_eq!(d2t.len(), cfg.draft_vocab, "d2t len != draft_vocab_size");

        let draft = Eagle3Draft {
            fc: load_float(e, &m, "fc.weight")?,
            input_layernorm: load_float(e, &m, "midlayer.input_layernorm.weight")?,
            hidden_norm: load_float(e, &m, "midlayer.hidden_norm.weight")?,
            q_proj: load_float(e, &m, "midlayer.self_attn.q_proj.weight")?,
            k_proj: load_float(e, &m, "midlayer.self_attn.k_proj.weight")?,
            v_proj: load_float(e, &m, "midlayer.self_attn.v_proj.weight")?,
            o_proj: load_float(e, &m, "midlayer.self_attn.o_proj.weight")?,
            post_attention_layernorm: load_float(e, &m, "midlayer.post_attention_layernorm.weight")?,
            gate_proj: load_float(e, &m, "midlayer.mlp.gate_proj.weight")?,
            up_proj: load_float(e, &m, "midlayer.mlp.up_proj.weight")?,
            down_proj: load_float(e, &m, "midlayer.mlp.down_proj.weight")?,
            norm: load_float(e, &m, "norm.weight")?,
            lm_head: load_float(e, &m, "lm_head.weight")?,
            d2t,
            n_embd: cfg.hidden_size,
            n_head: cfg.n_head,
            n_head_kv: cfg.n_head_kv,
            head_dim: cfg.head_dim,
            n_ff: cfg.intermediate_size,
            draft_vocab: cfg.draft_vocab,
            rope_dim_count: ((cfg.partial_rotary_factor * cfg.head_dim as f32).round() as usize).max(2),
            rope_theta: cfg.rope_theta,
            eps: cfg.rms_eps,
            aux_layers: cfg.aux_layers,
        };
        // shape sanity (catches a wrong checkpoint / mapping):
        assert_eq!(draft.fc.in_features(), 3 * draft.n_embd, "fc in != 3*n_embd");
        assert_eq!(draft.fc.out_features(), draft.n_embd, "fc out != n_embd");
        assert_eq!(draft.q_proj.in_features(), 2 * draft.n_embd, "q_proj in != 2*n_embd");
        assert_eq!(draft.q_proj.out_features(), draft.n_head * draft.head_dim, "q_proj out");
        assert_eq!(draft.lm_head.out_features(), draft.draft_vocab, "lm_head out != draft_vocab");
        Ok(draft)
    }

    /// Map a DRAFT-vocab id to a TARGET-vocab id (d2t is a DELTA: target = draft + d2t[draft]).
    #[inline]
    pub fn d2t_map(&self, draft_id: u32) -> u32 {
        (draft_id as i64 + self.d2t[draft_id as usize]) as u32
    }

    /// ENCODE (once per round, EAGLE-PLAN N3): g = fc @ concat(aux0, aux1, aux2). `aux` are the 3
    /// trunk residual hiddens of the just-committed token (decode_step_aux / decode_step_t_aux),
    /// in ascending-layer order. Returns the recurrent draft hidden `g` [n_embd].
    pub fn encode(&self, e: &Engine, aux: &[CudaSlice<f32>]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert_eq!(aux.len(), self.aux_layers.len(), "aux count != #aux layers");
        let n = self.n_embd;
        let mut cat = e.zeros(self.aux_layers.len() * n)?;
        for (i, a) in aux.iter().enumerate() {
            e.copy_into(&mut cat, i * n, a, n)?;
        }
        e.matmul(&self.fc, &cat, 1)               // [3*n_embd] @ fc[3n_embd,n_embd] -> [n_embd]
    }

    /// One DRAFT-token forward (EAGLE-PLAN N4, T=1). `prev_tok` = the TARGET token id to predict
    /// from (last committed or previous draft). `g` = the recurrent draft hidden (encode() output
    /// on round entry, then the previous step's g_next). Returns (draft_logits[draft_vocab] host,
    /// g_next dev). Mirrors the vLLM op-sequence documented at the top of this file.
    pub fn draft_token(&self, e: &Engine, target: &HybridModel, prev_tok: u32, g: &CudaSlice<f32>,
                       scratch: &mut Eagle3Scratch, pos: usize)
                       -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n = self.n_embd;
        let eps = self.eps;
        let pos_d = e.htod_i32(&[pos as i32])?;

        // e = TARGET embedding of prev_tok (EAGLE3 shares the target's token embedding).
        // eN = input_layernorm(e); gN = hidden_norm(g); residual = PRE-norm g (norm_after_residual).
        let e_emb = e.htod(&target.embd.gather(n, &[prev_tok]))?;
        let mut e_norm = e.zeros(n)?;
        e.rms_norm(&e_emb, self.input_layernorm.float_data(), &mut e_norm, n, 1, eps)?;
        let res = e.clone_dtod(g)?;
        let mut g_norm = e.zeros(n)?;
        e.rms_norm(g, self.hidden_norm.float_data(), &mut g_norm, n, 1, eps)?;
        // cat = [eN ; gN] -> [2*n_embd]  (vLLM llama_eagle3: torch.cat([embeds, hidden_states])).
        let mut cat = e.zeros(2 * n)?;
        e.copy_into(&mut cat, 0, &e_norm, n)?;
        e.copy_into(&mut cat, n, &g_norm, n)?;

        // attention from the 2*n_embd concat (plain Llama: no QK-norm, no output gate).
        let attn = self.attn(e, &cat, &pos_d, scratch)?;
        // x1 = attn + residual(g)
        let mut x1 = e.zeros(n)?;
        e.add(&attn, &res, &mut x1, n)?;
        // z = post_attention_layernorm(x1)
        let mut z = e.zeros(n)?;
        e.rms_norm(&x1, self.post_attention_layernorm.float_data(), &mut z, n, 1, eps)?;
        // mlp = down @ (silu(gate@z) * (up@z))
        let gate = e.matmul(&self.gate_proj, &z, 1)?;
        let up = e.matmul(&self.up_proj, &z, 1)?;
        let mut act = e.zeros(self.n_ff)?;
        e.silu_mul(&gate, &up, &mut act, self.n_ff)?;
        let mlp = e.matmul(&self.down_proj, &act, 1)?;
        // g_next = mlp + x1  (final fused-add residual; this is the aux_output recurrence)
        let mut g_next = e.zeros(n)?;
        e.add(&mlp, &x1, &mut g_next, n)?;
        // dl = lm_head @ norm(g_next)
        let mut hn = e.zeros(n)?;
        e.rms_norm(&g_next, self.norm.float_data(), &mut hn, n, 1, eps)?;
        let logits = e.matmul(&self.lm_head, &hn, 1)?;
        let host = e.dtoh(&logits)?;
        Ok((host, g_next))
    }

    /// Plain Llama attention over the [2*n_embd] concat input, T=1, on the draft's own scratch KV.
    /// q/k/v project from 2*n_embd; partial RoPE (rope_dim_count of head_dim) at the draft theta;
    /// GQA broadcast in fa_decode; o_proj back to n_embd. No QK-norm, no output gate.
    fn attn(&self, e: &Engine, cat: &CudaSlice<f32>, pos_d: &CudaSlice<i32>, scratch: &mut Eagle3Scratch)
            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (nh, nhkv, hd) = (self.n_head, self.n_head_kv, self.head_dim);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut q = e.matmul(&self.q_proj, cat, 1)?;    // [nh*hd]
        let mut k = e.matmul(&self.k_proj, cat, 1)?;    // [nhkv*hd]
        let v = e.matmul(&self.v_proj, cat, 1)?;        // [nhkv*hd]

        // partial RoPE: rope_dim_count = partial_rotary_factor * head_dim (= 64 of 256), draft theta.
        e.rope_neox(&mut q, pos_d, hd, self.rope_dim_count, nh, 1, self.rope_theta, 1.0)?;
        e.rope_neox(&mut k, pos_d, hd, self.rope_dim_count, nhkv, 1, self.rope_theta, 1.0)?;

        let kv = &mut scratch.kv;
        e.append_kv_quantized(&k, &v, &mut kv.k, &mut kv.v, kv.len,
                              kv.kv_dim_k, kv.kv_dim_v, kv.k_tok_bytes, kv.v_tok_bytes)?;
        kv.len += 1;
        let t_kv = kv.len;
        let (ktb, vtb) = (kv.k_tok_bytes, kv.v_tok_bytes);
        let k_view = e.view_u8(&kv.k, t_kv * ktb);
        let v_view = e.view_u8(&kv.v, t_kv * vtb);
        let mut attn = e.zeros(nh * hd)?;
        e.fa_decode(&q, &k_view, &v_view, &mut attn, hd, nh, nhkv, t_kv, scale, ktb, vtb)?;
        e.matmul(&self.o_proj, &attn, 1)
    }
}

/// Tiny scratch KV for the EAGLE3 draft layer (one full-attn layer). Reset each draft round. Uses
/// the SAME q8_0-K / q5_1-V quantized layout as the trunk KV (head_dim%32==0 holds: 256).
pub struct Eagle3Scratch {
    pub kv: KvLayer,
}
impl Eagle3Scratch {
    pub fn new(e: &Engine, draft: &Eagle3Draft, cap: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let (nhkv, hd) = (draft.n_head_kv, draft.head_dim);
        assert!(hd % 32 == 0, "KVQUANT requires head_dim%32==0 (EAGLE3 scratch)");
        let kv_dim_k = hd * nhkv;
        let kv_dim_v = hd * nhkv;
        let k_tok_bytes = (kv_dim_k / 32) * 34;
        let v_tok_bytes = (kv_dim_v / 32) * 24;
        Ok(Eagle3Scratch { kv: KvLayer {
            k: e.alloc_u8(cap * k_tok_bytes)?, v: e.alloc_u8(cap * v_tok_bytes)?,
            kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes, len: 0,
        } })
    }
    pub fn reset(&mut self) { self.kv.len = 0; }
}

impl HybridModel {
    /// Greedy EAGLE3 speculative decode (EAGLE-PLAN N6). Token-identical to `generate(prompt,n)`
    /// but drafts K tokens with the separate EAGLE3 draft, then verifies them in ONE batched target
    /// forward. Verify/accept/snapshot/rollback are REUSED from the MTP path (decode_step_t,
    /// cache.snapshot/rollback). Returns (tokens, total_drafted, total_accepted).
    pub fn generate_spec_eagle(&self, e: &Engine, draft: &Eagle3Draft, prompt: &[u32],
                               max_new: usize, k: usize)
                               -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        assert!(k >= 1, "k must be >= 1");
        assert!(!prompt.is_empty(), "prompt must be non-empty");
        let n_vocab = self.output.out_features();
        let n_embd = self.cfg.n_embd as usize;
        assert_eq!(n_embd, draft.n_embd, "draft n_embd != target n_embd");
        let aux = &draft.aux_layers;
        let max_ctx = prompt.len() + max_new + k + 8;
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;

        // prime: feed the prompt; capture the LAST token's aux hiddens (seed for round-1 encode).
        let mut prime_logits = Vec::new();
        let mut prime_aux: Vec<CudaSlice<f32>> = Vec::new();
        for &tok in prompt {
            let (l, a) = self.decode_step_aux(e, tok, &mut cache, aux)?;
            prime_logits = l; prime_aux = a;
        }

        let mut scratch = Eagle3Scratch::new(e, draft, k + 1)?;
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        let mut total_drafted = 0usize;
        let mut total_accepted = 0usize;

        // EAGLE3 token/hidden alignment (vLLM `llama_eagle3.py`/`cnets.py`): the draft pairs the
        // aux hidden of position p with the EMBEDDING of the token at position p+1 (input_ids are the
        // target tokens shifted left by one). So drafting the token after `last_token` (at pos p)
        // uses g = encode(aux of the token BEFORE last_token, at pos p-1) and embed(last_token).
        // BW24_EAGLE_ALIGN=0 forces the un-shifted MTP-style pairing (aux & embed both = last_token)
        // for A/B comparison; default (1) is the EAGLE shift. The prime loop already gave us the
        // aux of the prompt's last token (= the predecessor of `last_token`), so we keep it as
        // `prev_aux` and roll it forward by one each round.
        let shift = std::env::var("BW24_EAGLE_ALIGN").ok().map(|s| s != "0").unwrap_or(true);
        let mut last_token = argmax(&prime_logits) as u32;
        out.push(last_token);
        // prev_aux = aux of the token at the position whose forward predicted `last_token`
        // (= the prompt's last token for round 1). g_aux = aux of `last_token` itself.
        let mut prev_aux = prime_aux;
        let (mut last_logits, mut g_aux) = self.decode_step_aux(e, last_token, &mut cache, aux)?;

        while out.len() < max_new {
            let pos = cache.pos;
            let snap = cache.snapshot(e)?;

            // --- 1. ENCODE once: g0 = fc @ concat(aux). With the EAGLE shift, the seed aux is the
            //        PREDECESSOR token's (paired with embed(last_token)); else last_token's own. ---
            let seed_aux = if shift { &prev_aux } else { &g_aux };
            let g0 = draft.encode(e, seed_aux)?;

            // --- 2. DRAFT k tokens with the EAGLE3 draft (autoregressive, T=1 each) ---
            scratch.reset();
            let mut draft_toks: Vec<u32> = Vec::with_capacity(k);
            let mut prev = last_token;
            let mut g = g0;
            for j in 0..k {
                let (dl, g_next) = draft.draft_token(e, self, prev, &g, &mut scratch, pos + j)?;
                let d_draft = argmax(&dl) as u32;
                let d_target = draft.d2t_map(d_draft);   // map draft-vocab id -> target-vocab id
                draft_toks.push(d_target);
                prev = d_target;
                g = g_next;
            }

            // --- 3. VERIFY: one batched target forward over draft_toks (T=k). REUSED from MTP. ---
            let tlogits = self.decode_step_t(e, &draft_toks, pos, &mut cache)?;

            // --- 4. GREEDY ACCEPT (walk prefix, stop at first mismatch). REUSED logic. ---
            let t_pred = |j: usize| -> u32 {
                if j == 0 { argmax(&last_logits) as u32 }
                else { argmax(&tlogits[(j - 1) * n_vocab..j * n_vocab]) as u32 }
            };
            let mut n_acc = 0usize;
            for j in 0..k {
                if t_pred(j) == draft_toks[j] { n_acc += 1; } else { break; }
            }
            let bonus = t_pred(n_acc);
            total_drafted += k;
            total_accepted += n_acc;

            // --- 5. COMMIT draft[0..n_acc] then bonus ---
            for j in 0..n_acc {
                if out.len() >= max_new { break; }
                out.push(draft_toks[j]);
            }
            let bonus_emitted = out.len() < max_new;
            if bonus_emitted { out.push(bonus); }
            last_token = bonus;

            // --- 6. ROLLBACK + advance to pos + n_acc + 1 committed tokens (REUSED from MTP). The
            //        next round's EAGLE seed needs TWO auxs: g_aux = aux(bonus) and prev_aux =
            //        aux(bonus's predecessor). bonus's predecessor is the last committed token BEFORE
            //        bonus = draft[n_acc-1] if n_acc>=1, else this round's `last_token` (its aux is
            //        the CURRENT g_aux). We always replay [committed-tail.. , bonus] aux-capturing so
            //        the predecessor's aux is the second-to-last column; this keeps both exact.
            let pred_is_prev_round = n_acc == 0;       // bonus's predecessor = old last_token
            let old_g_aux = std::mem::take(&mut g_aux); // = aux(old last_token)
            // Unified exact path (also covers full-accept n_acc==k): restore the pre-round snapshot
            // then replay the committed prefix draft[0..n_acc] ++ [bonus] as ONE T=(n_acc+1) aux-
            // capturing forward — single weight read, bit-identical to greedy (verify-all-columns
            // math). Captures aux at the last column (bonus) and, when the predecessor of bonus is a
            // replayed token (n_acc>=1), the second-to-last column.
            cache.rollback(e, &snap, 0)?;
            let mut replay: Vec<u32> = draft_toks[0..n_acc].to_vec();
            replay.push(bonus);
            let pred_col = if pred_is_prev_round { None } else { Some(replay.len() - 2) };
            let (rl, mut a_last, a_pred) =
                self.decode_step_t_aux2(e, &replay, pos, &mut cache, aux, pred_col)?;
            last_logits = rl[(replay.len() - 1) * n_vocab..replay.len() * n_vocab].to_vec();
            prev_aux = if pred_is_prev_round { old_g_aux } else { a_pred.unwrap() };
            g_aux = std::mem::take(&mut a_last);
        }
        out.truncate(max_new);
        Ok((out, total_drafted, total_accepted))
    }
}

// ============================ draft config.json (geometry + rope) ============================

struct EagleConfig {
    hidden_size: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    intermediate_size: usize,
    draft_vocab: usize,
    partial_rotary_factor: f32,
    rope_theta: f32,
    rms_eps: f32,
    aux_layers: Vec<usize>,
}

impl EagleConfig {
    fn from_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let txt = std::fs::read_to_string(path)?;
        // Minimal field extraction (avoid a serde dep here; the draft config.json is flat-ish).
        let num = |key: &str| -> Option<f64> {
            let pat = format!("\"{key}\"");
            let i = txt.find(&pat)? + pat.len();
            let rest = &txt[i..];
            let c = rest.find(':')? + 1;
            let tail = rest[c..].trim_start();
            let end = tail.find(|ch: char| ch == ',' || ch == '}' || ch == '\n').unwrap_or(tail.len());
            tail[..end].trim().parse::<f64>().ok()
        };
        let aux_layers: Vec<usize> = {
            // eagle_aux_hidden_state_layer_ids: [1, 15, 28]
            let pat = "\"eagle_aux_hidden_state_layer_ids\"";
            match txt.find(pat) {
                Some(i) => {
                    let rest = &txt[i + pat.len()..];
                    let lb = rest.find('[').ok_or("no [ after aux ids")?;
                    let rb = rest.find(']').ok_or("no ] after aux ids")?;
                    rest[lb + 1..rb].split(',').filter_map(|s| s.trim().parse::<usize>().ok()).collect()
                }
                None => vec![1, 15, 28],   // fall back to the known EAGLE3-qwen35-9b layers
            }
        };
        Ok(EagleConfig {
            hidden_size: num("hidden_size").ok_or("hidden_size")? as usize,
            n_head: num("num_attention_heads").ok_or("num_attention_heads")? as usize,
            n_head_kv: num("num_key_value_heads").ok_or("num_key_value_heads")? as usize,
            head_dim: num("head_dim").ok_or("head_dim")? as usize,
            intermediate_size: num("intermediate_size").ok_or("intermediate_size")? as usize,
            draft_vocab: num("draft_vocab_size").ok_or("draft_vocab_size")? as usize,
            partial_rotary_factor: num("partial_rotary_factor").unwrap_or(1.0) as f32,
            rope_theta: num("rope_theta").unwrap_or(10000.0) as f32,
            rms_eps: num("rms_norm_eps").unwrap_or(1e-6) as f32,
            aux_layers,
        })
    }
}

/// Read an i64 1-D tensor (d2t) from the draft safetensors.
fn read_i64(m: &StModel, name: &str) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let (info, bytes) = m.raw(name).ok_or_else(|| format!("EAGLE3 draft missing {name}"))?;
    assert_eq!(info.dtype, "I64", "{name} dtype != I64");
    let n = bytes.len() / 8;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()));
    }
    Ok(v)
}
